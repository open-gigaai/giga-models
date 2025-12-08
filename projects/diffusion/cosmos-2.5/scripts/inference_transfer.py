import json
import os
from typing import List

import imageio
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro
from accelerate.utils import set_seed
from decord import VideoReader
from giga_datasets import image_utils
from giga_datasets import utils as gd_utils
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from tqdm import tqdm

from giga_models import Cosmos25Pipeline, load_pipeline
from giga_models.acceleration import get_sequence_parallel_group, initialize_sequence_parallel_group
from giga_models.utils import find_free_port


class ControlPipeline:
    """A wrapper class to handle different types of control signal generation
    (e.g., edge detection, depth estimation) for the ControlNet."""

    def __init__(self, mode, device):
        """Initializes the appropriate pipeline based on the specified mode.

        Args:
            mode (str): The type of control signal to generate ('edge', 'depth', 'seg', 'blur').
            device: The torch device to run the pipeline on.
        """
        if mode == 'edge':
            self.pipe = load_pipeline('edge_detection/canny', lazy=True)
        elif mode == 'depth':
            self.pipe = load_pipeline('depth_estimation/video_depth_anything/small', lazy=True)
        elif mode == 'seg':
            self.pipe = load_pipeline('segmentation/grounded_sam2/gd_swint_ogc_sam21_hiera_l', lazy=True)
        elif mode == 'blur':
            self.pipe = load_pipeline('others/blur', lazy=True)
        else:
            raise ValueError(f'Invalid control mode: {mode}')
        self.mode = mode
        self.pipe.to(device)

    def __call__(self, images, det_labels=None):
        """Processes a list of images to generate control signals.

        Args:
            images (list): A list of PIL Images.
            det_labels (list, optional): Detection labels for segmentation. Defaults to None.

        Returns:
            list: A list of PIL Images representing the control signals.
        """
        if self.mode == 'edge':
            output_images = [self.pipe(image) for image in images]
        elif self.mode == 'depth':
            output_images = self.pipe(images)
        elif self.mode == 'seg':
            assert det_labels is not None, 'Detection labels are required for segmentation mode.'
            output_images = self.pipe(images, det_labels=det_labels, return_mode='image')[0]
            output_images = [output_images[i][0] for i in range(len(output_images))]
        elif self.mode == 'blur':
            output_images = self.pipe(images)
        else:
            raise ValueError(f'Invalid control mode: {self.mode}')
        return output_images


def _inference(
    device,
    data_path: str,
    save_dir: str,
    transformer_model_path: str,
    controlnet_model_path: str,
    text_encoder_model_path: str = None,
    vae_model_path: str = None,
    num_inference_steps: int = 35,
    fps: int = 28,
    num_frames: int = 93,
    height: int = 704,
    width: int = 960,
    mode: str = 'edge',
    det_labels: list | None = None,
    seed: int = 584845,
    dp_world_size: int = 1,
    dp_rank: int = 0,
    process_index: int = 0,
):
    """Internal function to run inference on a single process."""
    # Load the main inference pipeline with ControlNet
    pipe = Cosmos25Pipeline.from_pretrained(
        transformer_model_path=transformer_model_path,
        controlnet_model_path=controlnet_model_path,
        text_encoder_model_path=text_encoder_model_path,
        vae_model_path=vae_model_path,
    )
    pipe.to(device, dtype=torch.bfloat16)
    # pipe.set_attn_backend('sage')

    # Initialize the control signal generation pipeline
    cn_pipe = ControlPipeline(mode=mode, device=device)

    # Define a negative prompt to guide the generation away from undesirable attributes
    negative_prompt = (
        'The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents '
        'a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. '
        'The geometries are very primitive. The images are very pixelated and of poor CG quality. '
        'There are many subtitles in the footage. Overall, the video is unrealistic at all.'
    )
    # Load and partition data for distributed inference
    data_list = json.load(open(data_path, 'r'))
    data_list = gd_utils.split_data(data_list, dp_world_size, dp_rank)
    os.makedirs(save_dir, exist_ok=True)

    # Inference loop
    for n in tqdm(range(len(data_list))):
        set_seed(seed)
        data_dict = data_list[n]
        prompt = data_dict['prompt']
        video_path = data_dict['video']
        if not os.path.exists(video_path):
            video_path = os.path.join(os.path.dirname(data_path), video_path)

        # Load and sample frames from the source video
        video = VideoReader(video_path)
        indexes = np.arange(len(video), dtype=int)[:num_frames]
        images = video.get_batch(indexes)
        images = images.numpy() if isinstance(images, torch.Tensor) else images.asnumpy()
        images = [Image.fromarray(image) for image in images]

        # --- Resize and Crop ---
        image_width, image_height = images[0].width, images[0].height
        dst_width, dst_height = image_utils.get_image_size((image_width, image_height), (width, height), mode='area', multiple=16)
        if float(dst_height) / image_height < float(dst_width) / image_width:
            new_height = int(round(float(dst_width) / image_width * image_height))
            new_width = dst_width
        else:
            new_height = dst_height
            new_width = int(round(float(dst_height) / image_height * image_width))
        assert dst_width <= new_width and dst_height <= new_height
        x1 = (new_width - dst_width) // 2
        y1 = (new_height - dst_height) // 2

        # Generate control images and apply the same transformations
        cn_images = cn_pipe(images, det_labels=det_labels)
        for i in range(len(cn_images)):
            cn_images[i] = F.resize(cn_images[i], (new_height, new_width), InterpolationMode.BILINEAR)
            cn_images[i] = F.crop(cn_images[i], y1, x1, dst_height, dst_width)

        # Run the main pipeline
        output_images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=3,
            num_inference_steps=num_inference_steps,
            fps=fps,
            num_frames=num_frames,
            height=height,
            width=width,
            control_video=cn_images,
            cond_timestep=0,
            seed=seed,
            use_kerras_sigma=False,
            pad_mode='zero',
        )[0]

        # Save results
        if process_index == 0:
            vis_images = []
            for k in range(len(output_images)):
                # Concatenate control image and generated image for comparison
                vis_image = [cn_images[k], output_images[k]]
                vis_image = image_utils.concat_images_grid(vis_image, cols=2, pad=2)
                vis_images.append(vis_image)
            save_path = os.path.join(save_dir, f'{n}.mp4')
            imageio.mimsave(save_path, vis_images, fps=fps)


def _inference_sp(rank, gpu_ids, sp_size, port, kwargs):
    """Worker function for sequence parallel (SP) inference.

    Initializes the distributed process group and then calls the internal _inference function.
    """
    gpu_id = gpu_ids[rank]
    world_size = len(gpu_ids)
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{port}',
        world_size=world_size,
        rank=rank,
        device_id=torch.device(device),
    )
    initialize_sequence_parallel_group(sp_size)
    sp_group = get_sequence_parallel_group()
    sp_world_size = dist.get_world_size(sp_group)
    sp_rank = dist.get_rank(sp_group)
    dp_world_size = world_size // sp_world_size
    dp_rank = rank // sp_world_size
    assert sp_size == sp_world_size
    _inference(device, dp_world_size=dp_world_size, dp_rank=dp_rank, process_index=sp_rank, **kwargs)


def inference(
    data_path: str,
    save_dir: str,
    transformer_model_path: str,
    controlnet_model_path: str,
    text_encoder_model_path: str = None,
    vae_model_path: str = None,
    gpu_ids: List[int] = [0],
    num_inference_steps: int = 35,
    fps: int = 28,
    num_frames: int = 93,
    height: int = 704,
    width: int = 960,
    mode: str = 'edge',
    det_labels: list | None = None,
    seed: int = 584845,
):
    """Main function to start the ControlNet inference process.

    Handles both single-GPU and multi-GPU (via multiprocessing) inference.
    """
    kwargs = dict(
        data_path=data_path,
        save_dir=save_dir,
        transformer_model_path=transformer_model_path,
        controlnet_model_path=controlnet_model_path,
        text_encoder_model_path=text_encoder_model_path,
        vae_model_path=vae_model_path,
        num_inference_steps=num_inference_steps,
        fps=fps,
        num_frames=num_frames,
        height=height,
        width=width,
        mode=mode,
        det_labels=det_labels,
        seed=seed,
    )
    num_gpus = len(gpu_ids)
    assert num_gpus >= 1
    if num_gpus == 1:
        # Single GPU inference
        _inference(f'cuda:{gpu_ids[0]}', **kwargs)
    else:
        # Multi-GPU inference using multiprocessing
        port = find_free_port()
        mp.start_processes(
            _inference_sp,
            nprocs=num_gpus,
            args=(gpu_ids, num_gpus, port, kwargs),
        )


if __name__ == '__main__':
    tyro.cli(inference)

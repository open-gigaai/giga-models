import json
import os
from typing import List

import imageio
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro
from accelerate.utils import set_seed
from giga_datasets import image_utils
from giga_datasets import utils as gd_utils
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from tqdm import tqdm

from giga_models import Cosmos25Pipeline
from giga_models.acceleration import get_sequence_parallel_group, initialize_sequence_parallel_group
from giga_models.utils import find_free_port


def _inference(
    device,
    data_path: str,
    save_dir: str,
    transformer_model_path: str,
    text_encoder_model_path: str = None,
    vae_model_path: str = None,
    num_inference_steps: int = 35,
    fps: int = 28,
    num_frames: int = 93,
    height: int = 704,
    width: int = 1280,
    seed: int = 1,
    dp_world_size: int = 1,
    dp_rank: int = 0,
    process_index: int = 0,
):
    """Internal function to run inference on a single process.

    Handles both text-to-video and image-to-video generation.
    """
    # Load the main inference pipeline
    pipe = Cosmos25Pipeline.from_pretrained(
        transformer_model_path=transformer_model_path,
        text_encoder_model_path=text_encoder_model_path,
        vae_model_path=vae_model_path,
    )
    pipe.to(device, dtype=torch.bfloat16)
    # pipe.set_attn_backend('sage')

    # Define a comprehensive negative prompt to avoid common generation artifacts
    negative_prompt = (
        'The video captures a series of frames showing ugly scenes, static with no motion, '
        'motion blur, over-saturation, shaky footage, low resolution, grainy texture,'
        ' pixelated images, poorly lit areas, underexposed and overexposed scenes, '
        'poor color balance, washed out colors, choppy sequences, jerky movements, '
        'low frame rate, artifacting, color banding, unnatural transitions, '
        'outdated special effects, fake elements, unconvincing visuals, '
        'poorly edited content, jump cuts, visual noise, and flickering. '
        'Overall, the video is of poor quality.'
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
        image_path = data_dict.get('image', None)

        # --- Prepare Input Image (for Image-to-Video) ---
        if image_path is not None:
            if not os.path.exists(image_path):
                image_path = os.path.join(os.path.dirname(data_path), image_path)
            image = Image.open(image_path)
            # --- Resize and Crop ---
            image_width, image_height = image.width, image.height
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
            input_image = F.resize(image, (new_height, new_width), InterpolationMode.BILINEAR)
            input_image = F.crop(input_image, y1, x1, dst_height, dst_width)
        else:
            # Text-to-Video case
            input_image = []
            dst_height, dst_width = height, width

        # Run the main pipeline
        output_images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            num_inference_steps=num_inference_steps,
            fps=fps,
            num_frames=num_frames,
            height=dst_height,
            width=dst_width,
            cond_timestep=0.1,
            seed=seed,
            use_kerras_sigma=True,
            pad_mode='repeat',
        )[0]

        # Save results
        if process_index == 0:
            vis_images = []
            for k in range(len(output_images)):
                if image_path is not None:
                    # Concatenate input and output for comparison
                    vis_image = [input_image, output_images[k]]
                else:
                    vis_image = [output_images[k]]
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
    text_encoder_model_path: str = None,
    vae_model_path: str = None,
    gpu_ids: List[int] = [0],
    num_inference_steps: int = 35,
    fps: int = 28,
    num_frames: int = 93,
    height: int = 704,
    width: int = 1280,
    seed: int = 1,
):
    """Main function to start the inference process.

    Handles both single-GPU and multi-GPU (via multiprocessing) inference.
    """
    kwargs = dict(
        data_path=data_path,
        save_dir=save_dir,
        transformer_model_path=transformer_model_path,
        text_encoder_model_path=text_encoder_model_path,
        vae_model_path=vae_model_path,
        num_inference_steps=num_inference_steps,
        fps=fps,
        num_frames=num_frames,
        height=height,
        width=width,
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

import os
from glob import glob
from typing import List

import numpy as np
import torch
import tyro
from decord import VideoReader
from giga_datasets import Dataset, FileWriter, PklWriter, load_dataset
from PIL import Image
from tqdm import tqdm

from giga_models import load_pipeline
from giga_models.models.diffusion.cosmos2_5 import Reason1TextEncoder
from giga_models.utils import download_from_huggingface


class ControlPipeline:
    """A wrapper class to handle different types of control signal generation
    (e.g., depth estimation, segmentation) for preparing training data."""

    def __init__(self, mode, device):
        """Initializes the appropriate pipeline based on the specified mode.

        Args:
            mode (str): The type of control signal to generate ('depth', 'seg').
            device: The torch device to run the pipeline on.
        """
        if mode == 'depth':
            self.pipe = load_pipeline('depth_estimation/video_depth_anything/small', lazy=True)
        elif mode == 'seg':
            self.pipe = load_pipeline('segmentation/grounded_sam2/gd_swint_ogc_sam21_hiera_l', lazy=True)
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
        if self.mode == 'depth':
            output_images = self.pipe(images)
        elif self.mode == 'seg':
            assert det_labels is not None, 'Detection labels are required for segmentation mode.'
            output_images = self.pipe(images, det_labels=det_labels, return_mode='image')[0]
            output_images = [output_images[i][0] for i in range(len(output_images))]
        else:
            raise ValueError(f'Invalid control mode: {self.mode}')
        return output_images


def pack_data(
    video_dir: str,
    save_dir: str,
    text_encoder_model_path: str | None = None,
    transfer_mode: str | None = None,
    det_labels: list | None = None,
    device: str = 'cuda',
):
    """Processes a directory of videos and their corresponding text prompts to
    create a packed dataset for training. This involves encoding prompts,
    saving videos, and optionally generating and saving control signals (like
    depth maps or segmentation masks).

    Args:
        video_dir (str): Path to the directory containing .mp4 videos and .txt prompts.
        save_dir (str): Path to the directory where the packed dataset will be saved.
        text_encoder_model_path (str, optional): Path to the Reason1 text encoder model.
                                                 If None, it will be downloaded from Hugging Face.
        transfer_mode (str, optional): If specified, generates control signals.
                                       Supported modes: 'depth', 'seg'.
        det_labels (list, optional): Detection labels required for 'seg' transfer_mode.
        device (str, optional): The device to run models on. Defaults to 'cuda'.
    """
    if text_encoder_model_path is None:
        # Download the text encoder if a path is not provided
        text_encoder_model_path = download_from_huggingface('nvidia/Cosmos-Reason1-7B')

    # Load the Reason1 text encoder for prompt embedding
    text_encoder = Reason1TextEncoder(text_encoder_model_path)
    text_encoder.to(device)

    # Find all video files in the specified directory
    video_paths: List[str] = glob(os.path.join(video_dir, '*.mp4'))

    # Initialize writers for different components of the dataset
    label_writer = PklWriter(os.path.join(save_dir, 'labels'))
    video_writer = FileWriter(os.path.join(save_dir, 'videos'))
    prompt_writer = FileWriter(os.path.join(save_dir, 'prompts'))

    # Initialize control signal pipeline and writer if transfer_mode is enabled
    if transfer_mode is not None:
        assert transfer_mode in ['depth', 'seg'], "Transfer mode must be 'depth' or 'seg'."
        pipe = ControlPipeline(transfer_mode, device=device)
        transfer_writer = FileWriter(os.path.join(save_dir, f'{transfer_mode}_videos'))

    # Process each video and its corresponding prompt
    for idx in tqdm(range(len(video_paths))):
        # Assume the prompt file has the same name as the video file but with a .txt extension
        anno_file = video_paths[idx].replace('.mp4', '.txt')
        prompt = open(anno_file, 'r').read().strip()

        # Encode the prompt to get text embeddings
        prompt_embeds = text_encoder.encode_prompts(prompt)[0].to(torch.bfloat16).cpu()

        # Write data to the respective files
        label_dict = dict(data_index=idx, prompt=prompt)
        label_writer.write_dict(label_dict)
        video_writer.write_video(idx, video_paths[idx])
        prompt_writer.write_dict(idx, dict(prompt_embeds=prompt_embeds))

        # Generate and write control signal data if in transfer mode
        if transfer_mode is not None:
            video = VideoReader(video_paths[idx])
            indexes = np.arange(len(video), dtype=int)
            frames = video.get_batch(indexes)
            frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
            frames = [Image.fromarray(frame) for frame in frames]
            transfer_frames = pipe(frames, det_labels=det_labels)
            transfer_writer.write_video(idx, transfer_frames, fps=video.get_avg_fps())

    # Finalize and close all writers
    label_writer.write_config()
    video_writer.write_config()
    prompt_writer.write_config()
    label_writer.close()
    video_writer.close()
    prompt_writer.close()
    if transfer_mode is not None:
        transfer_writer.write_config(data_name=f'{transfer_mode}_video')
        transfer_writer.close()

    # Load the individual datasets and combine them into a single composite Dataset object
    label_dataset = load_dataset(os.path.join(save_dir, 'labels'))
    video_dataset = load_dataset(os.path.join(save_dir, 'videos'))
    prompt_dataset = load_dataset(os.path.join(save_dir, 'prompts'))
    datasets = [label_dataset, video_dataset, prompt_dataset]
    if transfer_mode is not None:
        transfer_dataset = load_dataset(os.path.join(save_dir, f'{transfer_mode}_videos'))
        datasets.append(transfer_dataset)

    # Save the final combined dataset
    dataset = Dataset(datasets)
    dataset.save(save_dir)


if __name__ == '__main__':
    tyro.cli(pack_data)

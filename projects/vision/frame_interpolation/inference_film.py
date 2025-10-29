import os

import imageio
import numpy as np
import torch
import tyro
from decord import VideoReader
from PIL import Image

from giga_models import load_pipeline


def load_video(video_path: str):
    video = VideoReader(video_path)
    indexes = np.arange(len(video), dtype=int)
    frames = video.get_batch(indexes)
    frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
    frames = [Image.fromarray(frame) for frame in frames]
    return frames, video.get_avg_fps()


def inference(video_path: str, save_path: str, interval: int = 1, save_fps: int = -1, device: str = 'cuda'):
    pipe = load_pipeline('frame_interpolation/film/film_net_fp16')
    pipe.to(device)
    frames, fps = load_video(video_path)
    new_frames = pipe(frames, interval=interval)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_fps = save_fps if save_fps > 0 else fps
    imageio.mimsave(save_path, new_frames, fps=save_fps)


if __name__ == '__main__':
    tyro.cli(inference)

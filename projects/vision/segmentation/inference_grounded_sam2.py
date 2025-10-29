import os
import sys

import imageio
import numpy as np
import torch
import tyro
from decord import VideoReader
from PIL import Image

from giga_models import load_pipeline
from giga_models.utils import git_clone, run_pip


def import_repo():
    try:
        import sam2  # noqa: F401
    except ImportError:
        repo_path = git_clone('https://github.com/IDEA-Research/Grounded-SAM-2.git')
        run_pip('install -e .', cwd=repo_path)
        run_pip('install --no-build-isolation -e grounding_dino', cwd=repo_path)
        sys.path.insert(0, repo_path)
    try:
        import grounding_dino  # noqa: F401
    except ImportError:
        repo_path = git_clone('https://github.com/IDEA-Research/Grounded-SAM-2.git')
        sys.path.insert(0, repo_path)


def load_video(video_path: str):
    video = VideoReader(video_path)
    indexes = np.arange(len(video), dtype=int)
    frames = video.get_batch(indexes)
    frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
    frames = [Image.fromarray(frame) for frame in frames]
    return frames, video.get_avg_fps()


def inference(video_path: str, det_labels: list, save_dir: str | None = None, device: str = 'cuda'):
    import_repo()
    pipe_names = [
        'segmentation/grounded_sam2/gd_swint_ogc_sam21_hiera_t',
        'segmentation/grounded_sam2/gd_swint_ogc_sam21_hiera_s',
        'segmentation/grounded_sam2/gd_swint_ogc_sam21_hiera_l',
    ]
    frames, fps = load_video(video_path)
    for pipe_name in pipe_names:
        pipe = load_pipeline(pipe_name)
        pipe.to(device)
        video_segments, boxes, labels = pipe(frames, det_labels=det_labels)
        if video_segments is not None and save_dir is not None:
            from giga_datasets import ImageVisualizer

            assert len(frames) == len(video_segments)
            for i, label in enumerate(labels, start=1):
                vis_images = []
                for j, frame in enumerate(frames):
                    masks, obj_ids = video_segments[j]
                    mask = masks[obj_ids == i]
                    vis_image = ImageVisualizer(frame)
                    vis_image.draw_masks(mask)
                    vis_images.append(vis_image.get_image())
                save_name = pipe_name.replace('/', '_')
                save_path = os.path.join(save_dir, f'{save_name}_{label}.mp4')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                imageio.mimsave(save_path, vis_images, fps=fps)


if __name__ == '__main__':
    tyro.cli(inference)

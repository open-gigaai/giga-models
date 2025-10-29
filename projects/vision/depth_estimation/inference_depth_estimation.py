import os

import tyro
from PIL import Image

from giga_models import load_pipeline


def inference(image_path: str, save_dir: str, device: str = 'cuda'):
    pipe_names = [
        'depth_estimation/depth_anything/v2_small_hf',
        'depth_estimation/depth_anything/v2_base_hf',
        'depth_estimation/depth_anything/v2_large_hf',
        'depth_estimation/dpt/hybrid_midas',
        'depth_estimation/dpt/large',
    ]
    image = Image.open(image_path)
    os.makedirs(save_dir, exist_ok=True)
    for pipe_name in pipe_names:
        pipe = load_pipeline(pipe_name)
        pipe.to(device)
        depth_image = pipe(image)
        save_name = pipe_name.replace('/', '_')
        save_path = os.path.join(save_dir, f'{save_name}.jpg')
        depth_image.save(save_path)


if __name__ == '__main__':
    tyro.cli(inference)

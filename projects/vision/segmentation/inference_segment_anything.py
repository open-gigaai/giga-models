import os

import tyro
from PIL import Image

from giga_models import load_pipeline


def inference(image_path: str, save_dir: str | None = None, device: str = 'cuda'):
    pipe_names = [
        'segmentation/segment_anything/vit_b_01ec64',
        'segmentation/segment_anything/vit_l_0b3195',
        'segmentation/segment_anything/vit_h_4b8939',
    ]
    image = Image.open(image_path)
    for pipe_name in pipe_names:
        pipe = load_pipeline(pipe_name)
        pipe.to(device)
        masks = pipe(image)
        if save_dir is not None:
            from giga_datasets import ImageVisualizer

            os.makedirs(save_dir, exist_ok=True)
            save_name = pipe_name.replace('/', '_')
            save_path = os.path.join(save_dir, f'{save_name}.jpg')
            vis_image = ImageVisualizer(image)
            vis_image.draw_masks(masks)
            vis_image.save(save_path)


if __name__ == '__main__':
    tyro.cli(inference)

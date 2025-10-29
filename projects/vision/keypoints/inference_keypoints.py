import os

import tyro
from PIL import Image

from giga_models import load_pipeline


def inference(image_path: str, save_dir: str, device: str = 'cuda'):
    pipe_names = [
        'keypoints/openpose/body_hand_face',
        'keypoints/rtmpose/performance',
    ]
    image = Image.open(image_path)
    os.makedirs(save_dir, exist_ok=True)
    for pipe_name in pipe_names:
        pipe = load_pipeline(pipe_name)
        pipe.to(device)
        if 'openpose' in pipe_name:
            new_image = pipe(image, include_hand=True, include_face=True)
        else:
            new_image = pipe(image)
        save_name = pipe_name.replace('/', '_')
        save_path = os.path.join(save_dir, f'{save_name}.jpg')
        new_image.save(save_path)


if __name__ == '__main__':
    tyro.cli(inference)

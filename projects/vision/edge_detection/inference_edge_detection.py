import os

import tyro
from PIL import Image

from giga_models import load_pipeline


def inference(image_path: str, save_dir: str, device: str = 'cuda'):
    pipe_names = [
        'edge_detection/canny',
        'edge_detection/hed/apache2',
        'edge_detection/lineart/sk_model',
        'edge_detection/mlsd/large_512_fp32',
        'edge_detection/pidinet/table5',
    ]
    image = Image.open(image_path)
    os.makedirs(save_dir, exist_ok=True)
    for pipe_name in pipe_names:
        pipe = load_pipeline(pipe_name)
        pipe.to(device)
        if 'lineart' in pipe_name:
            edge_image = pipe(image, coarse=True)
        elif 'pidinet' in pipe_name:
            edge_image = pipe(image, safe=True)
        else:
            edge_image = pipe(image)
        save_name = pipe_name.replace('/', '_')
        save_path = os.path.join(save_dir, f'{save_name}.jpg')
        edge_image.save(save_path)


if __name__ == '__main__':
    tyro.cli(inference)

import sys

import tyro
from PIL import Image

from giga_models import load_pipeline
from giga_models.utils import git_clone


def import_repo():
    try:
        import groundingdino  # noqa: F401
    except ImportError:
        repo_path = git_clone('https://github.com/IDEA-Research/GroundingDINO.git')
        # run_pip('install -e .', cwd=repo_path)
        sys.path.insert(0, repo_path)


def inference(image_path: str, det_labels: list, save_path: str = None):
    import_repo()
    pipe = load_pipeline('detection/grounding_dino/swint_ogc')
    image = Image.open(image_path)
    pred_boxes, pred_labels, pred_scores = pipe(image, det_labels)
    if save_path is not None:
        from giga_datasets import ImageVisualizer

        vis_image = ImageVisualizer(image)
        vis_image.draw_boxes(pred_boxes, texts=pred_labels)
        vis_image.save(save_path)


if __name__ == '__main__':
    tyro.cli(inference)

import tyro

from giga_models import load_pipeline


def inference(video_path: str, save_dir: str | None = None, device: str = 'cuda'):
    pipe = load_pipeline('shot_boundary_detection/transnetv2')
    pipe.to(device)
    scenes = pipe(video_path)
    if save_dir is not None:
        pipe.scenes_to_videos(video_path, scenes, save_dir)


if __name__ == '__main__':
    tyro.cli(inference)

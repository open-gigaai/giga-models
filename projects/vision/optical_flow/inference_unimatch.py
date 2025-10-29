import sys

import tyro

from giga_models import load_pipeline
from giga_models.utils import git_clone


def import_repo():
    try:
        import unimatch  # noqa: F401
    except ImportError:
        repo_path = git_clone('https://github.com/autonomousvision/unimatch.git')
        sys.path.insert(0, repo_path)


def inference(video_path: str, device: str = 'cuda'):
    import_repo()
    pipe_names = [
        'optical_flow/unimatch/gmflow_scale2_regrefine6_mixdata',
    ]
    for pipe_name in pipe_names:
        pipe = load_pipeline(pipe_name)
        pipe.to(device)
        optical_flow = pipe(video_path)
        print(optical_flow)


if __name__ == '__main__':
    tyro.cli(inference)

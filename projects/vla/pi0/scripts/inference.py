import json
import os

import numpy as np
import torch
import tyro
from giga_datasets import load_dataset

from giga_models import Pi0Pipeline


def inference(
    model_path: str,
    tokenizer_model_path: str,
    data_path: str,
    norm_stats_path: str,
    vis_output_path: str | None = None,
    device: str = 'cuda',
    action_chunk: int = 50,
    original_action_dim: int = 14,
):
    """Run inference with PI0 model on LeRobot dataset.

    Args:
        model_path: Path to the pretrained PI0 model checkpoint.
        tokenizer_model_path: Path to the tokenizer model.
        data_path: Path to the LeRobot dataset.
        norm_stats_path: Path to the normalization statistics JSON file.
        vis_output_path: Path to save visualization results.
        device: Device to run inference on (e.g., 'cuda:0', 'cpu').
        action_chunk: Number of action chunks for delta computation.
        original_action_dim: Dimension of the original action space.
    """
    torch.cuda.set_device(device)
    if vis_output_path is not None:
        os.makedirs(vis_output_path, exist_ok=True)

    # Load normalization statistics
    with open(norm_stats_path, 'r') as f:
        norm_stats_data = json.load(f)['norm_stats']

    # Initialize pipeline
    pipe = Pi0Pipeline(
        model_path=model_path,
        tokenizer_model_path=tokenizer_model_path,
        state_norm_stats=norm_stats_data['observation.state'],
        action_norm_stats=norm_stats_data['action'],
        original_action_dim=original_action_dim,
    )
    pipe.to(device)
    pipe.compile(fullgraph=True)
    # pipe.compile(fullgraph=True, mode='max-autotune')  # Faster inference speed, but requires longer compilation time.

    # Load dataset
    data_or_config = [
        dict(
            _class_name='LeRobotDataset',
            data_path=data_path,
            delta_info=dict(
                action=action_chunk,
            ),
            meta_name='meta',
        )
    ]
    dataset = load_dataset(data_or_config)

    # Run inference and visualize results
    for idx in range(0, 1000, 100):
        if idx >= len(dataset):
            break
        data = dataset[idx]
        images = {
            'observation.images.cam_high': data['observation.images.cam_high'],
            'observation.images.cam_left_wrist': data['observation.images.cam_left_wrist'],
            'observation.images.cam_right_wrist': data['observation.images.cam_right_wrist'],
        }
        task = data['task']
        state = data['observation.state']

        pred_action = pipe(images, task, state)

        if vis_output_path is not None:
            action_names = None
            if 'meta' in data:
                action_names = data['meta'].info['features']['action']['names']
            visualize_result(
                data['action'].numpy(),
                pred_action.numpy(),
                os.path.join(vis_output_path, f'{idx}.png'),
                action_names,
            )


def visualize_result(gt_action: np.ndarray, pred_action: np.ndarray, out_path: str, action_names: list[str] | None = None):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    num_ts, num_dim = gt_action.shape
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(10, 2 * num_dim))

    time_axis = np.arange(num_ts) / 30.0

    colors = plt.cm.viridis(np.linspace(0, 1, num_dim))

    # plot joint state
    if action_names is None or len(action_names) == 0:
        action_names = [str(i) for i in range(num_dim)]

    dim_list = range(num_dim)
    for ax_idx, dim_idx in enumerate(dim_list):
        ax = axs[ax_idx]

        ax.plot(time_axis, gt_action[:, dim_idx], label='GT', color=colors[ax_idx], linewidth=2, linestyle='-')
        ax.plot(time_axis, pred_action[:, dim_idx], label='Pred', color=colors[ax_idx], linewidth=2, linestyle='--')

        ax.set_title(f'Joint {ax_idx}: {action_names[ax_idx]}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (rad)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')

        if num_ts > 0:
            ax.scatter(time_axis[-1], gt_action[-1, dim_idx], color='red', s=50, zorder=5)
            ax.text(time_axis[-1], gt_action[-1, dim_idx], f' {gt_action[-1, dim_idx]:.3f}', verticalalignment='bottom', horizontalalignment='left')

            ax.scatter(time_axis[-1], pred_action[-1, dim_idx], color='blue', s=50, zorder=5)
            ax.text(time_axis[-1], pred_action[-1, dim_idx], f' {pred_action[-1, dim_idx]:.3f}', verticalalignment='top', horizontalalignment='left')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    tyro.cli(inference)

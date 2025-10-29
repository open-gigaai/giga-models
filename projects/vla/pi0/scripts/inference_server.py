import json
import types

import tyro

from giga_models import Pi0Pipeline
from giga_models.sockets import RobotInferenceServer


def get_policy(model_path: str, tokenizer_model_path: str, norm_stats_path: str, original_action_dim: int):
    """Initialize and return the policy pipeline.

    Args:
        model_path: Path to the model checkpoint directory.
        tokenizer_model_path: Path to the tokenizer model.
        norm_stats_path: Path to the normalization statistics JSON file.
        original_action_dim: Original action dimension.

    Returns:
        Configured Pi0Pipeline with inference method attached.
    """
    with open(norm_stats_path, 'r') as f:
        norm_stats_data = json.load(f)['norm_stats']

    pipe = Pi0Pipeline(
        model_path=model_path,
        tokenizer_model_path=tokenizer_model_path,
        state_norm_stats=norm_stats_data['observation.state'],
        action_norm_stats=norm_stats_data['action'],
        original_action_dim=original_action_dim,
    )
    pipe.to('cuda')
    pipe.compile(fullgraph=True)
    # pipe.compile(fullgraph=True, mode='max-autotune')  # Faster inference speed, but requires longer compilation time.

    def inference(self, data):
        """Inference method for the policy.

        Args:
            data: Dictionary containing observation images, task, and state.

        Returns:
            Predicted action.
        """
        images = {
            'observation.images.cam_high': data['observation.images.cam_high'],
            'observation.images.cam_left_wrist': data['observation.images.cam_left_wrist'],
            'observation.images.cam_right_wrist': data['observation.images.cam_right_wrist'],
        }
        task = data['task']
        state = data['observation.state']

        pred_action = pipe(images, task, state)

        return pred_action

    pipe.inference = types.MethodType(inference, pipe)

    return pipe


def run_server(
    model_path: str,
    norm_stats_path: str,
    original_action_dim: int,
    tokenizer_model_path: str = 'google/paligemma-3b-pt-224',
    host: str = '127.0.0.1',
    port: int = 8080,
):
    """Start the Pi0 inference server.

    Args:
        model_path: Path to the model checkpoint directory.
        norm_stats_path: Path to the normalization statistics JSON file.
        original_action_dim: Original action dimension.
        tokenizer_model_path: Path to the tokenizer model.
        host: Server host address.
        port: Server port number.
    """
    policy = get_policy(
        model_path=model_path,
        tokenizer_model_path=tokenizer_model_path,
        norm_stats_path=norm_stats_path,
        original_action_dim=original_action_dim,
    )
    server = RobotInferenceServer(policy, host=host, port=port)
    server.run()


if __name__ == '__main__':
    tyro.cli(run_server)

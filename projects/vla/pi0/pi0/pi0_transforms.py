import json
from typing import Any

from giga_train import TRANSFORMS

from giga_models.pipelines.vla.pi0.pi0_utils import (
    AlohaInputs,
    DeltaActions,
    ImageTransform,
    Normalize,
    PadStatesAndActions,
    PromptTokenizerTransform,
)


@TRANSFORMS.register
class Pi0Transform:
    """Dataset-side preprocessing for PI0 training.

    Handles state/action normalization, optional delta actions, prompt tokenization, image preprocessing and padding to the model's max dims.
    """

    def __init__(
        self,
        adapt_to_pi: bool = True,
        use_delta_joint_actions: bool = True,
        norm_stats_path: str | None = None,
        use_quantiles: bool = False,
        image_cfg: dict[str, Any] | None = None,
        prompt_cfg: dict[str, Any] | None = None,
    ) -> None:
        """Initialize transforms.

        Args:
            adapt_to_pi: Whether to adapt Aloha inputs to the pi0 runtime.
            use_delta_joint_actions: If True, convert absolute to delta joint actions.
            norm_stats_path: Path to JSON file containing normalization stats.
            use_quantiles: If True, use quantile-based normalization.
            image_cfg: Config for `ImageTransform`.
            prompt_cfg: Config for `PromptTokenizerTransform`.
        """
        self.adapt_to_pi = adapt_to_pi
        self.use_delta_joint_actions = use_delta_joint_actions

        self.aloha_inputs_transform = AlohaInputs(adapt_to_pi=self.adapt_to_pi)
        if self.use_delta_joint_actions:
            self.delta_action_transform = DeltaActions()

        self.pad_states_and_actions_transform = PadStatesAndActions(action_dim=32)

        self.norm_stats_path = norm_stats_path
        assert self.norm_stats_path is not None, 'norm_stats_path is required'
        with open(self.norm_stats_path, 'r') as f:
            norm_stats_data = json.load(f)['norm_stats']
        self.state_normalize_transform = Normalize(norm_stats_data['observation.state'], use_quantiles=use_quantiles)
        self.action_normalize_transform = Normalize(norm_stats_data['action'], use_quantiles=use_quantiles)

        assert prompt_cfg is not None, 'prompt_cfg is required'
        self.prompt_tokenizer_transform = PromptTokenizerTransform(**prompt_cfg)

        assert image_cfg is not None, 'image_cfg is required'
        self.image_transform = ImageTransform(**image_cfg)

    def __call__(self, data_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply all transforms and return a model-ready batch dict.

        Expected keys in input:
            - 'observation.state', 'action', 'task', image keys

        Returns:
            Dict with keys: 'images', 'image_masks', 'lang_tokens', 'lang_masks',
            'observation.state', 'action', 'action_loss_mask'.
        """
        data_dict = self.aloha_inputs_transform(data_dict)
        if self.use_delta_joint_actions:
            data_dict = self.delta_action_transform(data_dict)

        data_dict['observation.state'] = self.state_normalize_transform(data_dict['observation.state'])
        data_dict['action'] = self.action_normalize_transform(data_dict['action'])

        output_dict = {}
        output_dict['lang_tokens'], output_dict['lang_masks'] = self.prompt_tokenizer_transform(data_dict)

        data_dict = self.pad_states_and_actions_transform(data_dict)
        output_dict['observation.state'] = data_dict['observation.state']
        output_dict['action'] = data_dict['action']

        output_dict['images'], output_dict['image_masks'] = self.image_transform(data_dict)

        output_dict['action_loss_mask'] = ~data_dict['action_is_pad']

        return output_dict

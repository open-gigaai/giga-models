from typing import Any

import torch

from ....models import GigaBrain0Policy
from ...pipeline import BasePipeline
from .giga_brain_0_utils import AbsoluteActions, ImageTransform, Normalize, PadStatesAndActions, PromptTokenizerTransform, Unnormalize


class GigaBrain0Pipeline(BasePipeline):
    """Pipeline wrapper for GigaBrain0 policy inference and utilities.

    This pipeline handles input preprocessing (state normalization, image transformation, tokenization), calls the underlying policy to sample actions
    and optionally 2D trajectories, and postprocesses outputs back to the original scale and absolute action space.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_model_path: str,
        fast_tokenizer_path: str,
        embodiment_id: int,
        state_norm_stats: dict,
        action_norm_stats: dict,
        delta_mask: list[bool],
        original_action_dim: int,
        discrete_state_input: bool = True,
        autoregressive_inference_mode: bool = False,
        depth_img_prefix_name: str | None = None,
    ):
        """Initialize the GigaBrain0 pipeline.

        Args:
            model_path: Path to the model checkpoint directory.
            tokenizer_model_path: Path to the tokenizer model.
            fast_tokenizer_path: Path to the fast tokenizer model.
            embodiment_id: Embodiment identifier of the robot/task.
            state_norm_stats: Normalization stats for state.
            action_norm_stats: Normalization stats for action.
            delta_mask: Boolean mask indicating which action dimensions are delta-controlled.
            original_action_dim: Expected original action vector dimension.
            discrete_state_input: Whether to use discrete state input.
            autoregressive_inference_mode: Whether to use autoregressive inference mode.
            depth_img_prefix_name: Optional prefix for depth image keys when depth is enabled.
        """
        super().__init__()
        self.policy = GigaBrain0Policy.from_pretrained(model_path)
        self.policy.eval()
        self.embodiment_id = embodiment_id
        self.device = 'cpu'
        self.resize_imgs_with_padding = (224, 224)

        self.enable_depth_img = self.policy.vision_in_channels == 4

        # Input transforms
        self.state_normalize_transform = Normalize({embodiment_id: state_norm_stats}, use_quantiles=True)
        self.image_transform = ImageTransform(
            is_train=False,
            resize_imgs_with_padding=self.resize_imgs_with_padding,
            enable_image_aug=False,
            enable_depth_img=self.enable_depth_img,
            depth_img_prefix_name=depth_img_prefix_name,
        )
        self.prompt_tokenizer_transform = PromptTokenizerTransform(
            is_train=False,
            tokenizer_model_path=tokenizer_model_path,
            fast_tokenizer_path=fast_tokenizer_path,
            max_length=200,
            discrete_state_input=discrete_state_input,
            encode_action_input=False,
            encode_sub_task_input=True,
            autoregressive_inference_mode=autoregressive_inference_mode,
        )
        self.pad_states_and_actions_transform = PadStatesAndActions(action_dim=self.policy.max_action_dim)
        # Output transforms
        self.state_unnormalize_transform = Unnormalize({embodiment_id: state_norm_stats}, use_quantiles=True)
        self.action_unnormalize_transform = Unnormalize({embodiment_id: action_norm_stats}, use_quantiles=True)
        self.absolute_actions_transform = AbsoluteActions({embodiment_id: delta_mask})
        self.original_action_dim = original_action_dim

    def to(self, device: str | torch.device):
        self.device = device
        self.policy.to(device)
        self.state_normalize_transform.to(device)
        self.state_unnormalize_transform.to(device)
        self.action_unnormalize_transform.to(device)
        self.absolute_actions_transform.to(device)
        self.prompt_tokenizer_transform.to(device)
        return self

    def quantize(self) -> None:
        """Apply dynamic float8 quantization to the Paligemma blocks only."""
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_

        # Only quantize the paligemma part. Skip the action expert part.
        layers = self.policy.paligemma_with_expert.layers
        for i in range(len(layers)):
            quantize_(layers[i].mlps[0], Float8DynamicActivationFloat8WeightConfig())
            quantize_(layers[i].self_attn.q_proj[0], Float8DynamicActivationFloat8WeightConfig())
            quantize_(layers[i].self_attn.k_proj[0], Float8DynamicActivationFloat8WeightConfig())
            quantize_(layers[i].self_attn.v_proj[0], Float8DynamicActivationFloat8WeightConfig())
            quantize_(layers[i].self_attn.o_proj[0], Float8DynamicActivationFloat8WeightConfig())

    def compile(self, **kwargs: Any) -> None:
        """Compile the `sample_actions` method using `torch.compile` for
        improved runtime speed."""
        self.policy.sample_actions = torch.compile(self.policy.sample_actions, **kwargs)

    @torch.no_grad()
    def __call__(
        self,
        images: dict[str, torch.Tensor],
        task: str,
        state: torch.Tensor,
        enable_2d_traj_output: bool = False,
        autoregressive_mode_only: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run policy inference to get the predicted action and optional 2D
        trajectory.

        Args:
            images: Observation images keyed by camera name.
            task: The executing task description.
            state: The joint state tensor.
            enable_2d_traj_output: If True, also return predicted 2D trajectory.
            autoregressive_mode_only: If True, only use autoregressive mode to predict actions.

        Returns:
            If `enable_2d_traj_output` is False, returns the predicted action tensor.
            Otherwise, returns a tuple of (predicted action tensor, 2D trajectory tensor).
        """
        if autoregressive_mode_only:
            return self.predict_autoregressive_actions(images, task, state)

        # Input transforms
        ori_device = state.device
        state = state.to(self.device)
        for key in images:
            images[key] = images[key].to(self.device)

        images, img_masks, image_transform_params = self.image_transform(images)
        state = self.state_normalize_transform(state, embodiment_id=self.embodiment_id)
        lang_tokens, lang_masks, _, _, _, _ = self.prompt_tokenizer_transform({'task': task, 'observation.state': state})
        state = self.pad_states_and_actions_transform({'observation.state': state})['observation.state']
        emb_ids = torch.tensor(self.embodiment_id, dtype=torch.long, device=self.device)
        emb_ids = emb_ids[None, ...]
        for i in range(len(images)):
            images[i] = images[i][None, ...]
            img_masks[i] = img_masks[i][None, ...]
        lang_tokens = lang_tokens[None, ...]
        lang_masks = lang_masks[None, ...]

        # Inference
        outputs = self.policy.sample_actions(images, img_masks, lang_tokens, lang_masks, emb_ids, enable_2d_traj_output=enable_2d_traj_output)
        if enable_2d_traj_output:
            pred_action, traj_pred = outputs
        else:
            pred_action = outputs

        # Output transforms
        output_dict = {'action': pred_action[0], 'observation.state': state, 'embodiment_id': self.embodiment_id}
        output_dict['observation.state'] = self.state_unnormalize_transform(output_dict['observation.state'], embodiment_id=self.embodiment_id)
        output_dict['action'] = self.action_unnormalize_transform(output_dict['action'], embodiment_id=self.embodiment_id)
        output_dict = self.absolute_actions_transform(output_dict)
        pred_action = output_dict['action'][:, : self.original_action_dim].to(ori_device)
        if enable_2d_traj_output:
            traj_pred = traj_pred[0]
            if 'resize_with_pad' in image_transform_params:
                ratio = image_transform_params['resize_with_pad']['ratio']
                pad_x, pad_y = image_transform_params['resize_with_pad']['padding']
                traj_pred[:, ::2] = (traj_pred[:, ::2] * self.resize_imgs_with_padding[0] - pad_x) * ratio
                traj_pred[:, 1::2] = (traj_pred[:, 1::2] * self.resize_imgs_with_padding[1] - pad_y) * ratio
            traj_pred = traj_pred.to(ori_device)

            return pred_action, traj_pred

        return pred_action

    @torch.no_grad()
    def predict_current_subtask(self, images: dict[str, torch.Tensor], task: str) -> list[str]:
        """Predict the current subtask from images and task description.

        Args:
            images: Observation images keyed by camera name.
            task: The executing task description.

        Returns:
            List of predicted subtask strings, one per sample in the batch.
        """
        tokenizer = self.prompt_tokenizer_transform.paligemma_tokenizer

        for key in images:
            images[key] = images[key].to(self.device)

        images, img_masks, _ = self.image_transform(images)
        lang_tokens, lang_masks, _, _, _, _ = self.prompt_tokenizer_transform({'task': task})

        generated = self.generate_autoregressive_tokens(images, img_masks, lang_tokens, lang_masks)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return decoded

    @torch.no_grad()
    def predict_autoregressive_actions(
        self, images: dict[str, torch.Tensor], task: str, state: torch.Tensor, max_new_tokens: int = 200
    ) -> torch.Tensor:
        """Predict actions using autoregressive generation.

        This method generates actions by autoregressively generating tokens and extracting
        action sequences from the generated tokens.

        Args:
            images: Observation images keyed by camera name.
            task: The executing task description.
            state: The joint state tensor.
            max_new_tokens: Maximum number of new tokens to generate during autoregressive
                inference. Defaults to 200.

        Returns:
            Predicted action tensor.
        """
        ori_device = state.device
        state = state.to(self.device)
        for key in images:
            images[key] = images[key].to(self.device)

        images, img_masks, _ = self.image_transform(images)
        state = self.state_normalize_transform(state, embodiment_id=self.embodiment_id)
        lang_tokens, lang_masks, _, _, _, _ = self.prompt_tokenizer_transform({'task': task, 'observation.state': state})

        generated = self.generate_autoregressive_tokens(images, img_masks, lang_tokens, lang_masks, max_new_tokens=max_new_tokens)

        pred_action = self.prompt_tokenizer_transform.extract_actions(generated, self.policy.n_action_steps, self.original_action_dim)
        pred_action = pred_action.to(self.device)
        output_dict = {'action': pred_action[0], 'observation.state': state, 'embodiment_id': self.embodiment_id}
        output_dict['observation.state'] = self.state_unnormalize_transform(output_dict['observation.state'], embodiment_id=self.embodiment_id)
        output_dict['action'] = self.action_unnormalize_transform(output_dict['action'], embodiment_id=self.embodiment_id)
        output_dict = self.absolute_actions_transform(output_dict)
        pred_action = output_dict['action'].to(ori_device)
        return pred_action

    @torch.no_grad()
    def generate_autoregressive_tokens(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        max_new_tokens: int = 64,
    ) -> list[list[int]]:
        """Autoregressively generate language tokens until EOS or limit.

        Args:
            images: List of image tensors.
            img_masks: List of image mask tensors.
            lang_tokens: Language token tensors.
            lang_masks: Language mask tensors.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            List of text tokens, one per sample in the batch.
        """
        for i in range(len(images)):
            images[i] = images[i][None, ...]
            img_masks[i] = img_masks[i][None, ...]
        lang_tokens = lang_tokens[None, ...]
        lang_masks = lang_masks[None, ...]

        # Initialize: build prefix cache and get next-step logits
        next_logits, gen_state = self.policy.init_lang_generation(images, img_masks, lang_tokens, lang_masks)

        tokenizer = self.prompt_tokenizer_transform.paligemma_tokenizer
        eos_id = tokenizer.eos_token_id
        generated = []
        bsize = lang_tokens.shape[0]
        finished = torch.zeros(bsize, dtype=torch.bool, device=self.device)
        # Initialize per-sample generated token lists
        for _ in range(bsize):
            generated.append([])

        for _ in range(max_new_tokens):
            step_token = torch.argmax(next_logits, dim=-1).to(torch.long)  # (b,)
            # For finished samples, keep feeding eos to avoid shape mismatch
            step_token = torch.where(finished, torch.tensor(eos_id, device=step_token.device), step_token)
            # Update each sample
            for i in range(bsize):
                if not finished[i].item():
                    generated[i].append(step_token[i].item())
            # Update finished flags
            finished = finished | (step_token == eos_id)
            if torch.all(finished):
                break

            # Proceed to the next step
            input_token = step_token.view(bsize, 1)
            next_logits, gen_state = self.policy.next_lang_logits(gen_state, input_token)

        return generated

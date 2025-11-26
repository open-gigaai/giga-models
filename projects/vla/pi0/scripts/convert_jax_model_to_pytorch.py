import pathlib

import jax
import numpy as np
import orbax.checkpoint as ocp
import torch
import tyro
from jax.sharding import SingleDeviceSharding
from lerobot.policies.pi0.conversion_scripts.conversion_utils import get_gemma_config, get_paligemma_config

from giga_models import PI0Policy

PRECISIONS = {'bfloat16': torch.bfloat16, 'float32': torch.float32, 'float16': torch.float16}


def slice_and_remap_state_dict(state_dict, paligemma_config, gemma_config, pi05_enabled=False):
    suffix = '/value' if 'img/embedding/kernel/value' in state_dict else ''

    # fmt: off
    # patch embeddings
    state_dict['vision_tower.embeddings.patch_embedding.weight'] = state_dict.pop(f'img/embedding/kernel{suffix}').transpose(
        3, 2, 0, 1
    )
    state_dict['vision_tower.embeddings.patch_embedding.bias'] = state_dict.pop(f'img/embedding/bias{suffix}')
    # positional embeddings
    state_dict['vision_tower.embeddings.position_embedding.weight'] = state_dict.pop(f'img/pos_embedding{suffix}').reshape(
        -1, paligemma_config.vision_config.hidden_size
    )

    # extract vision layers to be sliced at index 0. There are 27 layers in the base model.
    encoderblock_layernorm0_scale = state_dict.pop(f'img/Transformer/encoderblock/LayerNorm_0/scale{suffix}')
    encoderblock_layernorm0_bias = state_dict.pop(f'img/Transformer/encoderblock/LayerNorm_0/bias{suffix}')
    encoderblock_layernorm1_scale = state_dict.pop(f'img/Transformer/encoderblock/LayerNorm_1/scale{suffix}')
    encoderblock_layernorm1_bias = state_dict.pop(f'img/Transformer/encoderblock/LayerNorm_1/bias{suffix}')

    encoderblock_mlp_dense0_kernel = state_dict.pop(f'img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel{suffix}')
    encoderblock_mlp_dense0_bias = state_dict.pop(f'img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias{suffix}')
    encoderblock_mlp_dense1_kernel = state_dict.pop(f'img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel{suffix}')
    encoderblock_mlp_dense1_bias = state_dict.pop(f'img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias{suffix}')

    encoderblock_attention_0_key_kernel = state_dict.pop(f'img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel{suffix}')
    encoderblock_attention_0_key_bias = state_dict.pop(f'img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias{suffix}')
    encoderblock_attention_0_value_kernel = state_dict.pop(f'img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel{suffix}')
    encoderblock_attention_0_value_bias = state_dict.pop(f'img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias{suffix}')
    encoderblock_attention_0_query_kernel = state_dict.pop(f'img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel{suffix}')
    encoderblock_attention_0_query_bias = state_dict.pop(f'img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias{suffix}')
    encoderblock_attention_0_out_kernel = state_dict.pop(f'img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel{suffix}')
    encoderblock_attention_0_out_bias = state_dict.pop(f'img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias{suffix}')

    for i in range(paligemma_config.vision_config.num_hidden_layers):
        state_dict[f'vision_tower.encoder.layers.{i}.layer_norm1.weight'] = encoderblock_layernorm0_scale[i].transpose()
        state_dict[f'vision_tower.encoder.layers.{i}.layer_norm1.bias'] = encoderblock_layernorm0_bias[i]
        state_dict[f'vision_tower.encoder.layers.{i}.layer_norm2.weight'] = encoderblock_layernorm1_scale[i].transpose()
        state_dict[f'vision_tower.encoder.layers.{i}.layer_norm2.bias'] = encoderblock_layernorm1_bias[i]

        state_dict[f'vision_tower.encoder.layers.{i}.mlp.fc1.weight'] = encoderblock_mlp_dense0_kernel[i].transpose()
        state_dict[f'vision_tower.encoder.layers.{i}.mlp.fc1.bias'] = encoderblock_mlp_dense0_bias[i]
        state_dict[f'vision_tower.encoder.layers.{i}.mlp.fc2.weight'] = encoderblock_mlp_dense1_kernel[i].transpose()
        state_dict[f'vision_tower.encoder.layers.{i}.mlp.fc2.bias'] = encoderblock_mlp_dense1_bias[i]
        state_dict[f'vision_tower.encoder.layers.{i}.self_attn.k_proj.weight'] = \
            encoderblock_attention_0_key_kernel[i].reshape(-1, paligemma_config.vision_config.hidden_size).transpose()
        state_dict[f'vision_tower.encoder.layers.{i}.self_attn.k_proj.bias'] = \
            encoderblock_attention_0_key_bias[i].reshape(-1, paligemma_config.vision_config.hidden_size).reshape(-1)
        state_dict[f'vision_tower.encoder.layers.{i}.self_attn.v_proj.weight'] = \
            encoderblock_attention_0_value_kernel[i].reshape(-1, paligemma_config.vision_config.hidden_size).transpose()
        state_dict[f'vision_tower.encoder.layers.{i}.self_attn.v_proj.bias'] = \
            encoderblock_attention_0_value_bias[i].reshape(-1, paligemma_config.vision_config.hidden_size).reshape(-1)
        state_dict[f'vision_tower.encoder.layers.{i}.self_attn.q_proj.weight'] = \
            encoderblock_attention_0_query_kernel[i].reshape(-1, paligemma_config.vision_config.hidden_size).transpose()
        state_dict[f'vision_tower.encoder.layers.{i}.self_attn.q_proj.bias'] = \
            encoderblock_attention_0_query_bias[i].reshape(-1, paligemma_config.vision_config.hidden_size).reshape(-1)
        state_dict[f'vision_tower.encoder.layers.{i}.self_attn.out_proj.weight'] = \
            encoderblock_attention_0_out_kernel[i].reshape(-1, paligemma_config.vision_config.hidden_size).transpose()
        state_dict[f'vision_tower.encoder.layers.{i}.self_attn.out_proj.bias'] = \
            encoderblock_attention_0_out_bias[i].reshape(-1, paligemma_config.vision_config.hidden_size).reshape(-1)

    state_dict['vision_tower.post_layernorm.weight'] = \
        state_dict.pop(f'img/Transformer/encoder_norm/scale{suffix}').transpose()
    state_dict['vision_tower.post_layernorm.bias'] = state_dict.pop(f'img/Transformer/encoder_norm/bias{suffix}')

    # multimodal projector

    state_dict['multi_modal_projector.linear.weight'] = state_dict.pop(f'img/head/kernel{suffix}').transpose()
    state_dict['multi_modal_projector.linear.bias'] = state_dict.pop(f'img/head/bias{suffix}')

    # text decoder (gemma)
    embedding_vector = state_dict.pop(f'llm/embedder/input_embedding{suffix}')
    state_dict['embed_tokens.weight'] = embedding_vector

    # pop the einsum attention + mlp representations. There are 18 layers in gemma-2b.
    llm_attention_attn_vec_einsum = state_dict.pop(f'llm/layers/attn/attn_vec_einsum/w{suffix}')
    llm_attention_kv_einsum = state_dict.pop(f'llm/layers/attn/kv_einsum/w{suffix}')
    llm_attention_q_einsum = state_dict.pop(f'llm/layers/attn/q_einsum/w{suffix}')

    llm_mlp_gating_einsum = state_dict.pop(f'llm/layers/mlp/gating_einsum{suffix}')
    llm_mlp_linear = state_dict.pop(f'llm/layers/mlp/linear{suffix}')

    llm_input_layernorm = state_dict.pop(f'llm/layers/pre_attention_norm/scale{suffix}')
    llm_post_attention_layernorm = state_dict.pop(f'llm/layers/pre_ffw_norm/scale{suffix}')

    # Expert part
    num_expert = 1
    llm_attention_attn_vec_einsum_expert = state_dict.pop(f'llm/layers/attn/attn_vec_einsum_{num_expert}/w{suffix}')
    llm_attention_kv_einsum_expert = state_dict.pop(f'llm/layers/attn/kv_einsum_{num_expert}/w{suffix}')
    llm_attention_q_einsum_expert = state_dict.pop(f'llm/layers/attn/q_einsum_{num_expert}/w{suffix}')

    llm_mlp_gating_einsum_expert = state_dict.pop(f'llm/layers/mlp_{num_expert}/gating_einsum{suffix}')
    llm_mlp_linear_expert = state_dict.pop(f'llm/layers/mlp_{num_expert}/linear{suffix}')

    if pi05_enabled:
        llm_input_layernorm_kernel_expert = state_dict.pop(f'llm/layers/pre_attention_norm_{num_expert}/Dense_0/kernel{suffix}')
        llm_input_layernorm_bias_expert = state_dict.pop(f'llm/layers/pre_attention_norm_{num_expert}/Dense_0/bias{suffix}')
        llm_post_attention_layernorm_kernel_expert = state_dict.pop(f'llm/layers/pre_ffw_norm_{num_expert}/Dense_0/kernel{suffix}')
        llm_post_attention_layernorm_bias_expert = state_dict.pop(f'llm/layers/pre_ffw_norm_{num_expert}/Dense_0/bias{suffix}')
    else:
        llm_input_layernorm_expert = state_dict.pop(f'llm/layers/pre_attention_norm_{num_expert}/scale{suffix}')
        llm_post_attention_layernorm_expert = state_dict.pop(f'llm/layers/pre_ffw_norm_{num_expert}/scale{suffix}')

    for i in range(paligemma_config.text_config.num_hidden_layers):
        # PaliGemma part
        q_proj_weight_reshaped = llm_attention_q_einsum[i].transpose(0, 2, 1).reshape(
            paligemma_config.text_config.num_attention_heads * paligemma_config.text_config.head_dim, paligemma_config.text_config.hidden_size
        )
        state_dict[f'layers.{i}.self_attn.q_proj.0.weight'] = q_proj_weight_reshaped

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f'layers.{i}.self_attn.k_proj.0.weight'] = k_proj_weight_reshaped
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f'layers.{i}.self_attn.v_proj.0.weight'] = v_proj_weight_reshaped

        o_proj_weight_reshaped = llm_attention_attn_vec_einsum[i].transpose(2, 0, 1).reshape(
            paligemma_config.text_config.hidden_size, paligemma_config.text_config.num_attention_heads * paligemma_config.text_config.head_dim
        )
        state_dict[f'layers.{i}.self_attn.o_proj.0.weight'] = o_proj_weight_reshaped

        # mlp layers
        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f'layers.{i}.mlps.0.gate_proj.weight'] = gate_proj_weight.transpose()
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f'layers.{i}.mlps.0.up_proj.weight'] = up_proj_weight.transpose()
        state_dict[f'layers.{i}.mlps.0.down_proj.weight'] = llm_mlp_linear[i].transpose()

        state_dict[f'layers.{i}.input_layernorms.0.weight'] = llm_input_layernorm[i]
        state_dict[f'layers.{i}.post_attention_layernorms.0.weight'] = llm_post_attention_layernorm[i]

        # Expert part
        q_proj_weight_reshaped_expert = llm_attention_q_einsum_expert[i].transpose(0, 2, 1).reshape(
            gemma_config.num_attention_heads * gemma_config.head_dim, gemma_config.hidden_size
        )
        state_dict[f'layers.{i}.self_attn.q_proj.1.weight'] = q_proj_weight_reshaped_expert

        k_proj_weight_reshaped_expert = llm_attention_kv_einsum_expert[i, 0, 0].transpose()
        state_dict[f'layers.{i}.self_attn.k_proj.1.weight'] = k_proj_weight_reshaped_expert
        v_proj_weight_reshaped_expert = llm_attention_kv_einsum_expert[i, 1, 0].transpose()
        state_dict[f'layers.{i}.self_attn.v_proj.1.weight'] = v_proj_weight_reshaped_expert

        o_proj_weight_reshaped_expert = llm_attention_attn_vec_einsum_expert[i].reshape(
            gemma_config.num_attention_heads * gemma_config.head_dim, gemma_config.hidden_size
        ).transpose(1, 0)
        state_dict[f'layers.{i}.self_attn.o_proj.1.weight'] = o_proj_weight_reshaped_expert

        # mlp layers
        gate_proj_weight_expert = llm_mlp_gating_einsum_expert[i, 0]
        state_dict[f'layers.{i}.mlps.1.gate_proj.weight'] = gate_proj_weight_expert.transpose()
        up_proj_weight_expert = llm_mlp_gating_einsum_expert[i, 1]
        state_dict[f'layers.{i}.mlps.1.up_proj.weight'] = up_proj_weight_expert.transpose()
        state_dict[f'layers.{i}.mlps.1.down_proj.weight'] = llm_mlp_linear_expert[i].transpose()

        if pi05_enabled:
            state_dict[f'layers.{i}.input_layernorms.1.dense.weight'] = llm_input_layernorm_kernel_expert[i].transpose()
            state_dict[f'layers.{i}.input_layernorms.1.dense.bias'] = llm_input_layernorm_bias_expert[i].transpose()
            state_dict[f'layers.{i}.post_attention_layernorms.1.dense.weight'] = llm_post_attention_layernorm_kernel_expert[i].transpose()
            state_dict[f'layers.{i}.post_attention_layernorms.1.dense.bias'] = llm_post_attention_layernorm_bias_expert[i].transpose()
        else:
            state_dict[f'layers.{i}.input_layernorms.1.weight'] = llm_input_layernorm_expert[i]
            state_dict[f'layers.{i}.post_attention_layernorms.1.weight'] = llm_post_attention_layernorm_expert[i]

    state_dict['norms.0.weight'] = state_dict.pop(f'llm/final_norm/scale{suffix}')
    if pi05_enabled:
        state_dict['norms.1.dense.weight'] = state_dict.pop(f'llm/final_norm_{num_expert}/Dense_0/kernel{suffix}').transpose()
        state_dict['norms.1.dense.bias'] = state_dict.pop(f'llm/final_norm_{num_expert}/Dense_0/bias{suffix}').transpose()
    else:
        state_dict['norms.1.weight'] = state_dict.pop(f'llm/final_norm_{num_expert}/scale{suffix}')

    # Cleanup remaining keys
    keys_to_delete = [key for key in state_dict if key.startswith('llm/')]
    for key in keys_to_delete:
        del state_dict[key]

    # Convert all to tensors
    final_state_dict = {}
    for key, value in state_dict.items():
        if value.dtype == 'bfloat16':
            value = value.astype(np.float32)
        if not isinstance(value, torch.Tensor):
            final_state_dict[key] = torch.from_numpy(value)
        else:
            final_state_dict[key] = value

    return final_state_dict


def flatten_for_memory(tree, parent_key=''):
    out = {}
    for k, v in tree.items():
        new_key = f'{parent_key}/{k}' if parent_key else k
        if isinstance(v, dict):
            out.update(flatten_for_memory(v, new_key))
        else:
            out[new_key] = np.array(v)  # Ensure conversion to np.array for consistency
    return out


def flatten_for_npz(tree, parent_key=''):
    out = {}
    for k, v in tree.items():
        new_key = f'{parent_key}/{k}' if parent_key else k
        if isinstance(v, dict):
            out.update(flatten_for_npz(v, new_key))
        else:
            # bf16/f32 here?
            out[new_key] = np.array(v)
    return out


def slice_initial_orbax_checkpoint(checkpoint_dir: str):
    params_path = pathlib.Path(checkpoint_dir).resolve()
    checkpointer = ocp.PyTreeCheckpointer()

    metadata = checkpointer.metadata(params_path)
    print('Metadata keys:', list(metadata.keys()))

    params_name = 'params'

    item = {params_name: metadata[params_name]}
    device = jax.local_devices()[0]  # Use the first local device
    sharding = SingleDeviceSharding(device)
    restored = checkpointer.restore(
        params_path,
        ocp.args.PyTreeRestore(
            item=item,
            restore_args=jax.tree_util.tree_map(
                lambda _: ocp.ArrayRestoreArgs(
                    restore_type=jax.Array,  # or np.ndarray, but bf16 is annoying about it
                    sharding=sharding,
                ),
                item,
            ),
            transforms={},
        ),
    )
    params = restored[params_name]

    # get params for PaliGemma
    pali_params = params['PaliGemma']
    del params['PaliGemma']
    pali_params_flat = flatten_for_npz(pali_params)
    return {'paligemma_params': pali_params_flat, 'projection_params': params}


def update_keys_with_prefix(d: dict, prefix: str) -> dict:
    """Update dictionary keys by adding a prefix."""
    return {f'{prefix}{key}': value for key, value in d.items()}


def convert_jax_model_to_pytorch(checkpoint_dir: str, precision: str, tokenizer_id: str, output_path: str, pi05_enabled: bool = False):
    """Convert PI0 checkpoint from JAX to PyTorch format.

    Args:
        checkpoint_dir: Path to the JAX checkpoint directory.
        precision: Model precision ('bfloat16', 'float32', or 'float16').
        tokenizer_id: Hugging Face tokenizer identifier.
        output_path: Path to save the converted PyTorch checkpoint.
        pi05_enabled: Whether to use PI0.5 configuration.
    """
    # Break down orbax ckpts - they are in OCDBT
    initial_params = slice_initial_orbax_checkpoint(checkpoint_dir=checkpoint_dir)
    # process projection params
    if pi05_enabled:
        keys = [
            'action_in_proj',
            'action_out_proj',
            'time_mlp_in',
            'time_mlp_out',
        ]
    else:
        keys = [
            'state_proj',
            'action_in_proj',
            'action_out_proj',
            'action_time_mlp_in',
            'action_time_mlp_out',
        ]

    projection_params = {}
    for key in keys:
        kernel_params = initial_params['projection_params'][key]['kernel']
        bias_params = initial_params['projection_params'][key]['bias']
        if isinstance(kernel_params, dict):
            weight = kernel_params['value']
            bias = bias_params['value']
        else:
            weight = kernel_params
            bias = bias_params
        weight = np.array(weight)
        bias = np.array(bias)
        if weight.dtype == 'bfloat16':
            weight = weight.astype(np.float32)
        if bias.dtype == 'bfloat16':
            bias = bias.astype(np.float32)
        projection_params[f'{key}.weight'] = torch.from_numpy(weight).T
        projection_params[f'{key}.bias'] = torch.from_numpy(bias)

    # Process PaliGemma weights
    paligemma_config = get_paligemma_config(precision)
    gemma_config = get_gemma_config(precision)

    merged_params = slice_and_remap_state_dict(
        initial_params['paligemma_params'], paligemma_config=paligemma_config, gemma_config=gemma_config, pi05_enabled=pi05_enabled
    )

    # gemma_config=gemma_config, paligemma_config=paligemma_config)
    model = PI0Policy(pi05_enabled=pi05_enabled)

    # Prefix all keys with `paligemma_with_expert.`
    merged_params = update_keys_with_prefix(merged_params, 'paligemma_with_expert.')

    # load state dict
    torch_dtype = PRECISIONS[precision]

    states = {**merged_params, **projection_params}

    model.load_state_dict(states, strict=True)

    model = model.to(torch_dtype)

    model.save_pretrained(output_path, safe_serialization=True)

    # assert that model loads properly
    del model
    PI0Policy.from_pretrained(output_path)


if __name__ == '__main__':
    tyro.cli(convert_jax_model_to_pytorch)

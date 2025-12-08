import os.path

import torch
import tyro
from giga_models import Cosmos25ControlNet3DModel, Cosmos25Transformer3DModel
from giga_models.utils import download_from_huggingface


def load_pretrained(pretrained):
    """
    Loads a pretrained model checkpoint from a file.

    This function loads a PyTorch state dictionary and filters out any keys
    that end with '_extra_state', which are often optimizer states or other
    non-model parameters.

    Args:
        pretrained (str): The path to the pretrained model file (.pt).

    Returns:
        dict: A state dictionary containing only the model weights.
    """
    all_state_dict = torch.load(pretrained, map_location='cpu', weights_only=False)
    new_state_dict = dict()
    for key in all_state_dict:
        if key.endswith('_extra_state'):
            continue
        new_state_dict[key] = all_state_dict[key]
    return new_state_dict


def convert_predict2_5(pretrained, save_path, mode):
    """
    Converts a pretrained Cosmos-Predict2.5 model to the giga-models format.

    This function defines the model architecture, loads the pretrained weights,
    maps the weight names from the original checkpoint to the new model's format,
    and saves the converted model.

    Args:
        pretrained (str): Path to the original pretrained model file.
        save_path (str): Path to save the converted model.
        mode (str): The mode of the model, either 'base' or 'action'.

    Returns:
        tuple: A tuple containing the new state dictionary and the remaining
               keys from the original state dictionary.
    """
    # Define the model architecture arguments
    kwargs = dict(
        in_channels=17,
        out_channels=16,
        num_attention_heads=16,
        attention_head_dim=128,
        num_layers=28,
        mlp_ratio=4.0,
        text_in_channels=100352,
        text_embed_dim=1024,
        adaln_lora_dim=256,
        max_size=(128, 240, 240),
        patch_size=(1, 2, 2),
        rope_scale=(1.0, 3.0, 3.0),
        concat_padding_mask=True,
    )
    if mode == 'action':
        kwargs.update(
            dict(
                action_dim=7,
            )
        )
    else:
        assert mode == 'base'

    # Initialize the giga-models transformer
    model = Cosmos25Transformer3DModel(**kwargs)
    model.to(torch.bfloat16)
    state_dict = model.state_dict()
    all_state_dict = load_pretrained(pretrained)
    new_state_dict = dict()

    # --- Weight Name Mapping ---
    # Map weights for the transformer blocks
    map_dict = {
        'norm1.linear_1': 'adaln_modulation_self_attn.1',
        'norm1.linear_2': 'adaln_modulation_self_attn.2',
        'norm2.linear_1': 'adaln_modulation_cross_attn.1',
        'norm2.linear_2': 'adaln_modulation_cross_attn.2',
        'norm3.linear_1': 'adaln_modulation_mlp.1',
        'norm3.linear_2': 'adaln_modulation_mlp.2',
        'attn1.to_q': 'self_attn.q_proj',
        'attn1.norm_q': 'self_attn.q_norm',
        'attn1.to_k': 'self_attn.k_proj',
        'attn1.norm_k': 'self_attn.k_norm',
        'attn1.to_v': 'self_attn.v_proj',
        'attn1.to_out.0': 'self_attn.output_proj',
        'attn2.to_q': 'cross_attn.q_proj',
        'attn2.norm_q': 'cross_attn.q_norm',
        'attn2.to_k': 'cross_attn.k_proj',
        'attn2.norm_k': 'cross_attn.k_norm',
        'attn2.to_v': 'cross_attn.v_proj',
        'attn2.to_out.0': 'cross_attn.output_proj',
        'ff.net.0.proj': 'mlp.layer1',
        'ff.net.2': 'mlp.layer2',
    }
    for i in range(28):
        for key in map_dict:
            key2 = map_dict[key]
            key = f'transformer_blocks.{i}.{key}.weight'
            key2 = f'net.blocks.{i}.{key2}.weight'
            val = state_dict.pop(key)
            val2 = all_state_dict.pop(key2)
            assert val.shape == val2.shape and val.dtype == val2.dtype
            new_state_dict[key] = val2

    # Map weights for other layers
    map_dict = {
        'patch_embed.proj': 'x_embedder.proj.1',
        'time_embed.t_embedder.linear_1': 't_embedder.1.linear_1',
        'time_embed.t_embedder.linear_2': 't_embedder.1.linear_2',
        'time_norm': 't_embedding_norm',
        'norm_out.linear_1': 'final_layer.adaln_modulation.1',
        'norm_out.linear_2': 'final_layer.adaln_modulation.2',
        'proj_out': 'final_layer.linear',
    }
    for key in map_dict:
        key2 = map_dict[key]
        key = f'{key}.weight'
        key2 = f'net.{key2}.weight'
        val = state_dict.pop(key)
        val2 = all_state_dict.pop(key2)
        assert val.shape == val2.shape and val.dtype == val2.dtype
        new_state_dict[key] = val2

    # Map weights for layers with both weight and bias
    map_dict = {
        'text_embed.0': 'crossattn_proj.0',
    }
    if mode == 'action':
        map_dict.update(
            {
                'action_embed.fc1': 'action_embedder_B_D.fc1',
                'action_embed.fc2': 'action_embedder_B_D.fc2',
                'action_embed_3d.fc1': 'action_embedder_B_3D.fc1',
                'action_embed_3d.fc2': 'action_embedder_B_3D.fc2',
            }
        )
    for key in map_dict:
        key2 = map_dict[key]
        key2 = f'net.{key2}'
        for suffix in ['weight', 'bias']:
            key_s = f'{key}.{suffix}'
            key2_s = f'{key2}.{suffix}'
            val = state_dict.pop(key_s)
            val2 = all_state_dict.pop(key2_s)
            assert val.shape == val2.shape and val.dtype == val2.dtype
            new_state_dict[key_s] = val2

    # Ensure all weights from the new model have been mapped
    assert len(state_dict) == 0

    # Load the new state dict and save the model
    if save_path is not None:
        model.load_state_dict(new_state_dict)
        model.save_pretrained(save_path, safe_serialization=True)
        # Verification step: reload and check for differences
        new_model = Cosmos25Transformer3DModel.from_pretrained(save_path)
        new_model.to(torch.bfloat16)
        new_state_dict_reloaded = new_model.state_dict()
        for key in new_state_dict.keys():
            val = new_state_dict[key]
            new_val = new_state_dict_reloaded[key]
            diff = torch.sum(torch.abs(val - new_val))
            if diff != 0:
                print(f"Difference found in key {key}: {diff}")

    return new_state_dict, all_state_dict


def convert_transfer2_5(pretrained, save_base_path, save_path):
    """
    Converts a pretrained Cosmos-Transfer2.5 model to the giga-models format.

    This function first converts the base transformer model and then extracts
    and converts the ControlNet weights.

    Args:
        pretrained (str): Path to the original pretrained model file.
        save_base_path (str): Path to save the converted base transformer model.
                              Can be None if the base model is already converted.
        save_path (str): Path to save the converted ControlNet model.
    """
    # Convert the base model first to get its state dict
    base_state_dict, all_state_dict = convert_predict2_5(pretrained, save_base_path, mode='base')

    # Define ControlNet architecture
    block_ids = [0, 7, 14, 21]
    kwargs = dict(
        in_channels=17,
        control_in_channels=130,
        num_attention_heads=16,
        attention_head_dim=128,
        block_ids=block_ids,
        mlp_ratio=4.0,
        text_in_channels=100352,
        text_embed_dim=1024,
        adaln_lora_dim=256,
        max_size=(128, 240, 240),
        patch_size=(1, 2, 2),
        rope_scale=(1.0, 3.0, 3.0),
        concat_padding_mask=True,
    )
    model = Cosmos25ControlNet3DModel(**kwargs)
    model.to(torch.bfloat16)
    state_dict = model.state_dict()
    new_state_dict = dict()

    # --- Weight Name Mapping for ControlNet ---
    map_dict = {
        'norm1.linear_1': 'adaln_modulation_self_attn.1',
        'norm1.linear_2': 'adaln_modulation_self_attn.2',
        'norm2.linear_1': 'adaln_modulation_cross_attn.1',
        'norm2.linear_2': 'adaln_modulation_cross_attn.2',
        'norm3.linear_1': 'adaln_modulation_mlp.1',
        'norm3.linear_2': 'adaln_modulation_mlp.2',
        'attn1.to_q': 'self_attn.q_proj',
        'attn1.norm_q': 'self_attn.q_norm',
        'attn1.to_k': 'self_attn.k_proj',
        'attn1.norm_k': 'self_attn.k_norm',
        'attn1.to_v': 'self_attn.v_proj',
        'attn1.to_out.0': 'self_attn.output_proj',
        'attn2.to_q': 'cross_attn.q_proj',
        'attn2.norm_q': 'cross_attn.q_norm',
        'attn2.to_k': 'cross_attn.k_proj',
        'attn2.norm_k': 'cross_attn.k_norm',
        'attn2.to_v': 'cross_attn.v_proj',
        'attn2.to_out.0': 'cross_attn.output_proj',
        'ff.net.0.proj': 'mlp.layer1',
        'ff.net.2': 'mlp.layer2',
    }
    for i in range(len(block_ids)):
        for key in map_dict:
            key2 = map_dict[key]
            key = f'transformer_blocks.{block_ids[i]}.{key}.weight'
            key2 = f'net.control_blocks.{i}.{key2}.weight'
            val = state_dict.pop(key)
            val2 = all_state_dict.pop(key2)
            assert val.shape == val2.shape and val.dtype == val2.dtype
            new_state_dict[key] = val2

    for i in range(len(block_ids)):
        key = f'control_blocks.{block_ids[i]}'
        key2 = f'net.control_blocks.{i}.after_proj'
        for suffix in ['weight', 'bias']:
            key_s = f'{key}.{suffix}'
            key2_s = f'{key2}.{suffix}'
            val = state_dict.pop(key_s)
            val2 = all_state_dict.pop(key2_s)
            assert val.shape == val2.shape and val.dtype == val2.dtype
            new_state_dict[key_s] = val2

    map_dict = {
        'input_block.weight': 'control_blocks.0.before_proj.weight',
        'input_block.bias': 'control_blocks.0.before_proj.bias',
        'control_patch_embed.proj.weight': 'control_embedder.proj.1.weight',
    }
    for key in map_dict:
        key2 = map_dict[key]
        key2 = f'net.{key2}'
        val = state_dict.pop(key)
        val2 = all_state_dict.pop(key2)
        assert val.shape == val2.shape and val.dtype == val2.dtype
        new_state_dict[key] = val2

    # Copy remaining weights from the base model
    keys = list(state_dict.keys())
    for key in keys:
        val = state_dict.pop(key)
        val2 = base_state_dict[key]
        assert val.shape == val2.shape and val.dtype == val2.dtype
        new_state_dict[key] = val2

    assert len(state_dict) == 0
    model.load_state_dict(new_state_dict)
    model.save_pretrained(save_path, safe_serialization=True)

    # Verification step
    new_model = Cosmos25ControlNet3DModel.from_pretrained(save_path)
    new_model.to(torch.bfloat16)
    new_state_dict_reloaded = new_model.state_dict()
    for key in new_state_dict.keys():
        val = new_state_dict[key]
        new_val = new_state_dict_reloaded[key]
        diff = torch.sum(torch.abs(val - new_val))
        if diff != 0:
            print(f"Difference found in key {key}: {diff}")


def main(save_dir: str, token: str = None):
    """
    Main function to download and convert all necessary models.

    Args:
        save_dir (str): The root directory to save all downloaded and converted models.
        token (str, optional): Hugging Face API token for downloading gated models.
    """
    # Download T5 text encoder model
    text_encoder_model_path = download_from_huggingface(
        'nvidia/Cosmos-Reason1-7B',
        local_dir=os.path.join(save_dir, 'text_encoder'),
    )
    print(f'download text_encoder model to {text_encoder_model_path}')

    # Download VAE model
    vae_model_path = download_from_huggingface(
        'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
        local_dir=save_dir,
        folders='vae',
    )
    print(f'download vae model to {vae_model_path}')

    # Download official Cosmos-Predict2.5 model
    cosmos_predict25_dir = download_from_huggingface(
        'nvidia/Cosmos-Predict2.5-2B',
        local_dir=os.path.join(save_dir, 'official', 'models--nvidia--Cosmos-Predict2.5-2B'),
        token=token,
    )
    print(f'download cosmos_predict2.5 model to {cosmos_predict25_dir}')

    # Download official Cosmos-Transfer2.5 model
    cosmos_transfer25_dir = download_from_huggingface(
        'nvidia/Cosmos-Transfer2.5-2B',
        local_dir=os.path.join(save_dir, 'official', 'models--nvidia--Cosmos-Transfer2.5-2B'),
        token=token,
    )
    print(f'download cosmos_transfer2.5 model to {cosmos_transfer25_dir}')

    # Define save directories for converted models
    save_cosmos_predict25_dir = os.path.join(save_dir, 'models--nvidia--Cosmos-Predict2.5-2B')
    save_cosmos_transfer25_dir = os.path.join(save_dir, 'models--nvidia--Cosmos-Transfer2.5-2B')

    # --- Convert Predict2.5 Models ---
    print("Converting Cosmos-Predict2.5 base model...")
    convert_predict2_5(
        os.path.join(cosmos_predict25_dir, 'base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt'),
        os.path.join(save_cosmos_predict25_dir, 'base/post-trained/transformer'),
        mode='base',
    )
    print("Converting Cosmos-Predict2.5 action model...")
    convert_predict2_5(
        os.path.join(cosmos_predict25_dir, 'robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt'),
        os.path.join(save_cosmos_predict25_dir, 'robot/action-cond/transformer'),
        mode='action',
    )

    # --- Convert Transfer2.5 (ControlNet) Models ---
    print("Converting Cosmos-Transfer2.5 depth model...")
    convert_transfer2_5(
        os.path.join(cosmos_transfer25_dir, 'general/depth/0f214f66-ae98-43cf-ab25-d65d09a7e68f_ema_bf16.pt'),
        os.path.join(save_cosmos_transfer25_dir, 'general/transformer'),
        os.path.join(save_cosmos_transfer25_dir, 'general/controlnet/depth'),
    )
    print("Converting Cosmos-Transfer2.5 edge model...")
    convert_transfer2_5(
        os.path.join(cosmos_transfer25_dir, 'general/edge/ecd0ba00-d598-4f94-aa09-e8627899c431_ema_bf16.pt'),
        None,  # Base transformer already converted and saved
        os.path.join(save_cosmos_transfer25_dir, 'general/controlnet/edge'),
    )
    print("Converting Cosmos-Transfer2.5 segmentation model...")
    convert_transfer2_5(
        os.path.join(cosmos_transfer25_dir, 'general/seg/fcab44fe-6fe7-492e-b9c6-67ef8c1a52ab_ema_bf16.pt'),
        None,
        os.path.join(save_cosmos_transfer25_dir, 'general/controlnet/seg'),
    )
    print("Converting Cosmos-Transfer2.5 blur model...")
    convert_transfer2_5(
        os.path.join(cosmos_transfer25_dir, 'general/blur/20d9fd0b-af4c-4cca-ad0b-f9b45f0805f1_ema_bf16.pt'),
        None,
        os.path.join(save_cosmos_transfer25_dir, 'general/controlnet/blur'),
    )
    print("All conversions finished.")


if __name__ == '__main__':
    tyro.cli(main)

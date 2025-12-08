import functools

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKLWan
from einops import rearrange
from giga_train import ModuleDict, Trainer, utils

from giga_models import Cosmos25ControlNet3DModel, Cosmos25Transformer3DModel


class Cosmos25Trainer(Trainer):
    """A specialized Trainer for Cosmos2.5 models, handling the setup of VAE,
    Transformer, optional ControlNet, and the training loop."""

    def get_models(self, model_config):
        """Initializes and returns the models required for training.

        This method sets up the VAE, the main Transformer model, and an optional
        ControlNet. It also configures the noise scheduler.

        Args:
            model_config: A configuration object containing paths to the pre-trained models.

        Returns:
            A ModuleDict containing the initialized models.
        """
        model = dict()
        # VAE setup
        vae_dtype = model.get('vae_dtype', self.dtype)
        vae = AutoencoderKLWan.from_pretrained(model_config.vae_model_path)
        vae.requires_grad_(False)
        vae.to(self.device, dtype=vae_dtype)
        self.vae = vae
        # Pre-calculated mean and std for latent space normalization
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(self.device, dtype=vae_dtype)
        self.latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(self.device, dtype=vae_dtype)

        # Transformer setup
        transformer_model_path = model_config.transformer_model_path
        transformer = Cosmos25Transformer3DModel.from_pretrained(transformer_model_path)

        # ControlNet setup (optional)
        controlnet_model_path = model_config.get('controlnet_model_path', None)
        if controlnet_model_path is None:
            # If no ControlNet, the transformer is the main trainable model.
            model.update(transformer=transformer)
        else:
            # If ControlNet is present, freeze the transformer and train the ControlNet.
            transformer.requires_grad_(False)
            transformer.to(self.device, dtype=self.dtype)
            self.transformer = transformer
            controlnet = Cosmos25ControlNet3DModel.from_pretrained(controlnet_model_path)
            model.update(controlnet=controlnet)

        # Scheduler setup
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=5)
        self.noise_scheduler.timesteps = self.noise_scheduler.timesteps.to(device=self.device, dtype=torch.float32)
        self.noise_scheduler.sigmas = self.noise_scheduler.sigmas.to(device=self.device, dtype=torch.float32)

        # Finalize model dictionary
        model = ModuleDict(model)
        model.to(self.dtype)
        model.train()
        return model

    def forward_step(self, batch_dict):
        """Performs a single forward pass and calculates the loss.

        Args:
            batch_dict: A dictionary containing the batch data, including images,
                        prompts, and conditioning masks.

        Returns:
            The calculated loss for the current step.
        """
        # Determine which transformer to use (trainable or frozen)
        if hasattr(self, 'transformer'):
            transformer = self.transformer
        else:
            transformer = functools.partial(self.model, 'transformer')

        # Extract training parameters and data from batch
        cond_timestep = self.kwargs['cond_timestep']
        timestep_scale = self.kwargs['timestep_scale']
        images = batch_dict['images']
        prompt_embeds = batch_dict['prompt_embeds']
        cond_masks = batch_dict['cond_masks']
        batch_size = images.shape[0]

        padding_mask = torch.zeros((batch_size, 1, images.shape[-2], images.shape[-1]), device=self.device, dtype=self.dtype)
        fps = batch_dict['fps'][0]

        # Sample timesteps and sigmas for the diffusion process
        indices = torch.sigmoid(torch.randn((batch_size,))).to(device=self.device, dtype=torch.float32)
        indices = (indices * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices]
        sigmas = self.noise_scheduler.sigmas[indices]
        timesteps = timesteps[:, None]
        sigmas = sigmas[:, None, None, None, None]

        # Prepare latents
        latents = self.forward_vae(images)
        noise = torch.randn(latents.shape, device=self.device, dtype=torch.float32)
        # Add noise to latents based on the sampled sigma
        input_latents = noise * sigmas + latents * (1 - sigmas)
        # Apply conditioning masks to keep some latents clean
        input_latents = latents * cond_masks + input_latents * (1 - cond_masks)
        input_masks = cond_masks.repeat(1, 1, 1, input_latents.shape[-2], input_latents.shape[-1])

        # Adjust timesteps based on conditioning
        if cond_timestep > 0:
            cond_timestep_masks = cond_masks[:, 0, :, 0, 0]
            timesteps = cond_timestep * cond_timestep_masks + timesteps * (1 - cond_timestep_masks)
        timesteps = timesteps * timestep_scale

        # Prepare arguments for the transformer model
        kwargs = dict(
            hidden_states=input_latents.to(self.dtype),
            timestep=timesteps.to(self.dtype),
            encoder_hidden_states=prompt_embeds.to(self.dtype),
            fps=fps,
            condition_mask=input_masks.to(self.dtype),
            padding_mask=padding_mask,
        )
        if 'actions' in batch_dict:
            kwargs['action'] = batch_dict['actions'].to(self.dtype)

        # Forward pass through ControlNet if it exists
        if 'cn_images' in batch_dict:
            controlnet = functools.partial(self.model, 'controlnet')
            cn_latents = self.forward_vae(batch_dict['cn_images'])
            control_samples = controlnet(control_cond=cn_latents, **kwargs)
            utils.to_dtype(control_samples, self.dtype)
            kwargs['control_hidden_states'] = control_samples

        # Forward pass through the main transformer
        pred_latents_vt = transformer(**kwargs)
        # Calculate the target velocity
        latents_vt = noise - latents
        # Apply conditioning masks to the predicted velocity
        pred_latents_vt = latents_vt * cond_masks + pred_latents_vt * (1 - cond_masks)
        # Calculate the mean squared error loss
        loss = torch.mean((pred_latents_vt - latents_vt) ** 2)
        return loss

    def forward_vae(self, images):
        """Encodes images into the latent space using the VAE.

        Args:
            images: A tensor of images with shape (b, t, c, h, w).

        Returns:
            A tensor of latents, normalized.
        """
        images = images.to(self.vae.dtype)
        with torch.no_grad():
            # VAE expects (b, c, t, h, w)
            images = rearrange(images, 'b t c h w -> b c t h w')
            latents = self.vae.encode(images).latent_dist.sample()
        # Normalize latents
        latents = (latents - self.latents_mean) * self.latents_std
        return latents

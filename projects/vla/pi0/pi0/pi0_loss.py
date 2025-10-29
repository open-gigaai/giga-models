import torch
import torch.nn as nn
import torch.nn.functional as F


class PI0Loss(nn.Module):
    """Diffusion-style training loss for PI0 actions."""

    def __init__(self) -> None:
        super().__init__()

    def sample_noise(self, shape: tuple[int, ...], device: torch.device | str) -> torch.Tensor:
        """Sample standard normal noise with the given shape on the device."""
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def _sample_beta(self, alpha: float, beta: float, bsize: int, device: torch.device | str) -> torch.Tensor:
        """Sample from Beta(alpha, beta) using the ratio of powered uniforms
        trick.

        Returns:
            A tensor of shape (bsize,) with samples in (0, 1).
        """
        gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
        gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
        return gamma1 / (gamma1 + gamma2)

    def sample_time(self, bsize: int, device: torch.device | str) -> torch.Tensor:
        """Sample diffusion times in (0.001, 1.0) biased toward later times."""
        time_beta = self._sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def add_noise(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Create noisy actions x_t and compute the target u_t for training.

        Args:
            actions: Ground-truth actions of shape (B, T, D).

        Returns:
            A tuple (x_t, time) where x_t has shape (B, T, D) and time has shape (B,).
        """
        noise = self.sample_noise(actions.shape, actions.device)
        time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        self.x_t = x_t
        self.u_t = u_t
        self.time = time

        return x_t, time

    def forward(self, model_pred: torch.Tensor, loss_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute v-prediction MSE loss.

        Args:
            model_pred: Predicted v_t with shape (B, T, D).
            loss_mask: Optional mask (B, T) to ignore padded steps.

        Returns:
            Per-sample loss with shape (B, T) if mask provided then masked accordingly.
        Note:
            `add_noise` must be called before forward to set `self.u_t`.
        """
        loss = F.mse_loss(self.u_t, model_pred, reduction='none').mean(dim=-1)
        if loss_mask is not None:
            loss = loss * loss_mask

        return loss

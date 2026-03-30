"""
Diffusion Process — DDPM and DDIM

Implements forward diffusion (adding noise) and reverse denoising
for action sequence generation. Supports both full DDPM sampling
and accelerated DDIM sampling for real-time inference.

Extracted and completed from the original dppo_model.py skeleton.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal positional embeddings for diffusion timestep encoding.
    Converts scalar timestep t to a dense vector representation.

    Reused directly from dppo_model.py.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionProcess:
    """
    Forward and reverse diffusion process for action sequences.

    Forward:  q(A_t | A_0) = N(sqrt(alpha_bar_t) * A_0, (1-alpha_bar_t) * I)
    Reverse:  p_theta(A_{t-1} | A_t, S)  via trained denoising network
    """

    def __init__(self,
                 num_timesteps: int = 100,
                 beta_schedule: str = 'cosine',
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02):
        self.num_timesteps = num_timesteps

        # Generate noise schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Precompute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # For DDPM p_sample
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule from "Improved DDPM" (Nichol & Dhariwal, 2021).
        Produces smoother noise schedule than linear.
        """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps) / timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def _extract(self, schedule: torch.Tensor, t: torch.Tensor,
                 x_shape: tuple) -> torch.Tensor:
        """Extract schedule values at timestep t and reshape for broadcasting."""
        batch_size = t.shape[0]
        out = schedule.to(t.device).gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    # ====================================================================
    # Forward Process
    # ====================================================================

    def q_sample(self, action_0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: q(A_t | A_0) = sqrt(alpha_bar_t) * A_0 + sqrt(1 - alpha_bar_t) * eps

        Args:
            action_0: (B, T_pred, action_dim) or (B, action_dim) clean actions
            t: (B,) timestep indices
            noise: optional pre-generated noise

        Returns:
            noisy_action: A_t
            noise: epsilon used
        """
        if noise is None:
            noise = torch.randn_like(action_0)

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, action_0.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, action_0.shape
        )

        noisy_action = sqrt_alpha * action_0 + sqrt_one_minus_alpha * noise
        return noisy_action, noise

    # ====================================================================
    # DDPM Reverse Process
    # ====================================================================

    @torch.no_grad()
    def p_sample(self, denoise_fn, action_t: torch.Tensor,
                 t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Single DDPM reverse step: sample A_{t-1} from p_theta(A_{t-1} | A_t, S).

        Args:
            denoise_fn: callable(noisy_action, timestep, condition) -> predicted_noise
            action_t: (B, ...) noisy action at timestep t
            t: (B,) current timestep (all same value)
            condition: conditioning features

        Returns:
            action_t_minus_1: denoised action one step closer to A_0
        """
        # Predict noise
        predicted_noise = denoise_fn(action_t, t, condition)

        # Compute mean of p_theta
        sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, t, action_t.shape)
        beta = self._extract(self.betas, t, action_t.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, action_t.shape
        )

        model_mean = sqrt_recip_alpha * (
            action_t - beta / sqrt_one_minus_alpha_cumprod * predicted_noise
        )

        # Add noise (except at t=0)
        if t[0] > 0:
            posterior_var = self._extract(self.posterior_variance, t, action_t.shape)
            noise = torch.randn_like(action_t)
            return model_mean + torch.sqrt(posterior_var) * noise
        else:
            return model_mean

    @torch.no_grad()
    def ddpm_sample(self, denoise_fn, condition: torch.Tensor,
                    shape: tuple) -> torch.Tensor:
        """
        Full DDPM sampling: denoise from T to 0.

        Args:
            denoise_fn: noise prediction network
            condition: conditioning features
            shape: output shape (B, T_pred, action_dim) or (B, action_dim)

        Returns:
            action_0: denoised action sequence
        """
        device = next(iter([p for p in [] if False]), condition).device \
            if hasattr(condition, 'device') else condition.device
        batch_size = shape[0]

        action_t = torch.randn(shape, device=device)

        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            action_t = self.p_sample(denoise_fn, action_t, t, condition)

        return action_t

    # ====================================================================
    # DDIM Sampling (Fast Inference)
    # ====================================================================

    @torch.no_grad()
    def ddim_sample(self, denoise_fn, condition: torch.Tensor,
                    shape: tuple, ddim_steps: int = 10,
                    eta: float = 0.0) -> torch.Tensor:
        """
        DDIM sampling with sub-stepped timesteps for fast inference.

        Args:
            denoise_fn: noise prediction network
            condition: conditioning features
            shape: output shape
            ddim_steps: number of denoising steps (< num_timesteps)
            eta: stochasticity (0 = deterministic DDIM)

        Returns:
            action_0: denoised action
        """
        device = condition.device
        batch_size = shape[0]

        # Create sub-sampled timestep sequence
        step_size = self.num_timesteps // ddim_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        # Start from pure noise
        action_t = torch.randn(shape, device=device)

        alphas_cumprod = self.alphas_cumprod.to(device)

        for i in range(len(timesteps)):
            t_cur = timesteps[i]
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0

            t_batch = torch.full((batch_size,), t_cur, device=device, dtype=torch.long)

            # Predict noise
            predicted_noise = denoise_fn(action_t, t_batch, condition)

            # Current and previous alpha_cumprod
            alpha_cur = alphas_cumprod[t_cur]
            alpha_prev = alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)

            # Predict x_0
            sqrt_alpha_cur = torch.sqrt(alpha_cur)
            sqrt_one_minus_alpha_cur = torch.sqrt(1.0 - alpha_cur)
            predicted_x0 = (action_t - sqrt_one_minus_alpha_cur * predicted_noise) / sqrt_alpha_cur

            # Clamp predicted x0 for stability
            predicted_x0 = torch.clamp(predicted_x0, -5.0, 5.0)

            # DDIM step
            sqrt_alpha_prev = torch.sqrt(alpha_prev)
            sqrt_one_minus_alpha_prev = torch.sqrt(1.0 - alpha_prev)

            # Compute sigma for stochastic DDIM
            sigma = eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha_cur) * (1 - alpha_cur / alpha_prev)
            )

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1.0 - alpha_prev - sigma**2) * predicted_noise

            # Compute x_{t-1}
            action_t = sqrt_alpha_prev * predicted_x0 + dir_xt

            if eta > 0 and t_prev > 0:
                action_t = action_t + sigma * torch.randn_like(action_t)

        return action_t

"""
Vision Diffusion Policy — Full Pipeline

Combines VisionEncoder + ConditionalUnet1d + DiffusionProcess into a
complete policy that maps image stacks to motor command sequences.

Training:  MSE loss on noise prediction (supervised from expert data)
Inference: DDIM sampling conditioned on FPV image stack
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

from .vision_encoder import VisionEncoder
from .conditional_unet1d import ConditionalUnet1d
from .diffusion_process import DiffusionProcess


class DemoDataset(torch.utils.data.Dataset):
    """
    Dataset of expert demonstrations with sliding window over episodes.

    Each sample:
        image_stack:    (T_obs * C, H, W)    — stacked past frames
        action_sequence: (T_pred, action_dim) — future action sequence
    """

    def __init__(self, hdf5_path: str, T_obs: int = 2, T_pred: int = 8):
        import h5py
        self.T_obs = T_obs
        self.T_pred = T_pred

        # Load all episodes into memory
        self.samples = []
        with h5py.File(hdf5_path, 'r') as hf:
            for ep_key in sorted(hf.keys()):
                if not ep_key.startswith('episode_'):
                    continue
                images = hf[ep_key]['images'][:]    # (T, C, H, W)
                actions = hf[ep_key]['actions'][:]  # (T, action_dim)

                ep_len = len(images)
                # Create sliding window samples
                for t in range(T_obs - 1, ep_len - T_pred):
                    self.samples.append((ep_key, t))

            # Store full data for indexing
            self._images = {}
            self._actions = {}
            for ep_key in sorted(hf.keys()):
                if not ep_key.startswith('episode_'):
                    continue
                self._images[ep_key] = hf[ep_key]['images'][:]
                self._actions[ep_key] = hf[ep_key]['actions'][:]

        print(f"Loaded {len(self.samples)} samples from {hdf5_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ep_key, t = self.samples[idx]

        # Stack T_obs consecutive images along channel dimension
        images = self._images[ep_key]  # (T, C, H, W)
        img_stack = []
        for i in range(self.T_obs):
            frame_idx = t - (self.T_obs - 1) + i
            img_stack.append(images[frame_idx])
        img_stack = np.concatenate(img_stack, axis=0)  # (T_obs * C, H, W)

        # Future action sequence
        actions = self._actions[ep_key]
        action_seq = actions[t:t + self.T_pred]  # (T_pred, action_dim)

        return (
            torch.from_numpy(img_stack).float(),
            torch.from_numpy(action_seq).float(),
        )


class VisionDiffusionPolicy(nn.Module):
    """
    End-to-end Vision Diffusion Policy.

    Pipeline:
        FPV images -> VisionEncoder -> features
        features + timestep -> ConditionalUnet1d -> predicted noise
        DiffusionProcess handles forward/reverse diffusion
    """

    def __init__(self,
                 action_dim: int = 4,
                 T_obs: int = 2,
                 T_pred: int = 8,
                 image_channels: int = 3,
                 image_size: int = 64,
                 feature_dim: int = 256,
                 time_embed_dim: int = 128,
                 down_dims: tuple = (256, 512),
                 num_diffusion_steps: int = 100,
                 beta_schedule: str = 'cosine',
                 ddim_steps: int = 10):
        super().__init__()

        self.action_dim = action_dim
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.ddim_steps = ddim_steps

        # Vision encoder: (B, T_obs*C, H, W) -> (B, feature_dim)
        self.vision_encoder = VisionEncoder(
            in_channels=T_obs * image_channels,
            feature_dim=feature_dim,
        )

        # Noise prediction network: (B, action_dim, T_pred) -> (B, action_dim, T_pred)
        self.noise_pred_net = ConditionalUnet1d(
            action_dim=action_dim,
            feature_dim=feature_dim,
            time_embed_dim=time_embed_dim,
            down_dims=down_dims,
        )

        # Diffusion process (not a nn.Module, just schedules)
        self.diffusion = DiffusionProcess(
            num_timesteps=num_diffusion_steps,
            beta_schedule=beta_schedule,
        )

    def compute_loss(self, image_stack: torch.Tensor,
                     action_sequence: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion training loss (MSE on noise prediction).

        Args:
            image_stack: (B, T_obs*C, H, W)
            action_sequence: (B, T_pred, action_dim) clean expert actions

        Returns:
            loss: scalar MSE loss
        """
        B = image_stack.shape[0]
        device = image_stack.device

        # Encode visual features (detach-free, trains encoder end-to-end)
        visual_features = self.vision_encoder(image_stack)  # (B, feature_dim)

        # Reshape actions for 1D conv: (B, T_pred, action_dim) -> (B, action_dim, T_pred)
        action_seq = action_sequence.permute(0, 2, 1)

        # Sample random diffusion timesteps
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device)

        # Forward diffusion: add noise
        noisy_actions, noise = self.diffusion.q_sample(action_seq, t)

        # Predict noise
        predicted_noise = self.noise_pred_net(noisy_actions, t, visual_features)

        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def compute_weighted_loss(self, image_stack: torch.Tensor,
                              action_sequence: torch.Tensor,
                              advantages: torch.Tensor,
                              beta: float = 1.0) -> torch.Tensor:
        """
        DPPO loss: advantage-weighted diffusion loss.

        L = E[ exp(beta * A_normalized) * ||eps_theta - eps||^2 ]

        Args:
            image_stack: (B, T_obs*C, H, W)
            action_sequence: (B, T_pred, action_dim)
            advantages: (B,) normalized advantages
            beta: advantage weighting coefficient

        Returns:
            loss: scalar weighted loss
        """
        B = image_stack.shape[0]
        device = image_stack.device

        visual_features = self.vision_encoder(image_stack)
        action_seq = action_sequence.permute(0, 2, 1)

        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device)
        noisy_actions, noise = self.diffusion.q_sample(action_seq, t)
        predicted_noise = self.noise_pred_net(noisy_actions, t, visual_features)

        # Per-sample MSE
        per_sample_loss = F.mse_loss(predicted_noise, noise, reduction='none')
        per_sample_loss = per_sample_loss.mean(dim=(1, 2))  # (B,)

        # Advantage weighting
        weights = torch.exp(beta * advantages)
        weights = torch.clamp(weights, 0.1, 10.0)  # stability clamp

        loss = (weights * per_sample_loss).mean()
        return loss

    @torch.no_grad()
    def predict_action(self, image_stack: torch.Tensor,
                       ddim_steps: Optional[int] = None) -> torch.Tensor:
        """
        Generate action sequence from image stack using DDIM sampling.

        Args:
            image_stack: (B, T_obs*C, H, W) or (T_obs*C, H, W)

        Returns:
            action_sequence: (B, T_pred, action_dim) predicted actions
        """
        if image_stack.dim() == 3:
            image_stack = image_stack.unsqueeze(0)

        B = image_stack.shape[0]
        device = image_stack.device
        steps = ddim_steps or self.ddim_steps

        # Encode visual features
        visual_features = self.vision_encoder(image_stack)

        # Define denoising function for the diffusion process
        def denoise_fn(noisy_action, timestep, condition):
            return self.noise_pred_net(noisy_action, timestep, condition)

        # DDIM sampling
        shape = (B, self.action_dim, self.T_pred)
        action_seq = self.diffusion.ddim_sample(
            denoise_fn, visual_features, shape, ddim_steps=steps
        )

        # Reshape: (B, action_dim, T_pred) -> (B, T_pred, action_dim)
        action_seq = action_seq.permute(0, 2, 1)

        # Clamp to valid range [-1, 1]
        action_seq = torch.clamp(action_seq, -1.0, 1.0)

        return action_seq

    def save(self, filepath: str):
        """Save full policy checkpoint."""
        torch.save({
            'vision_encoder': self.vision_encoder.state_dict(),
            'noise_pred_net': self.noise_pred_net.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        """Load policy checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        self.vision_encoder.load_state_dict(checkpoint['vision_encoder'])
        self.noise_pred_net.load_state_dict(checkpoint['noise_pred_net'])

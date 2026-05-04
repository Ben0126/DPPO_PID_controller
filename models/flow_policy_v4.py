"""
Flow Matching Policy v4.0

Replaces DDPM with Conditional Flow Matching (linear interpolant / OT-CFM):

  Training:
    x_0 = expert CTBR action sequence  (B, 4, T_pred)
    ε   ~ N(0, I)
    t   ~ U[0, 1]
    x_t = (1 - t) * x_0 + t * ε       (linear path)
    target velocity: v* = ε - x_0
    loss: ||v_θ(x_t, t, cond) - v*||²

  Inference (1-step Euler, pure OT):
    x_1 ~ N(0, I)
    x_0 = x_1 - v_θ(x_1, 1.0, cond)

  Inference (N-step Euler, higher quality):
    x_1 ~ N(0, I)
    dt = 1 / N
    for t in [1.0, 1-dt, ..., dt]:
        x_{t-dt} = x_t - dt * v_θ(x_t, t, cond)

No cosine schedule. Max velocity magnitude = ||ε - x_0|| = O(1).
Avoids 64× amplification of DDPM at t=99.

Architecture (same as v3.3):
  VisionEncoder (CNN, 6ch → 256D) + IMUEncoder (MLP, 6D → 32D)
  global_cond = cat([256D, 32D]) = 288D
  FlowNet = ConditionalUnet1d with cond_dim = 288 + 128 = 416D
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_encoder import VisionEncoder
from .conditional_unet1d import ConditionalUnet1d


class FlowMatchingPolicyV4(nn.Module):
    """
    Flow Matching policy for CTBR quadrotor control.

    Args:
        vision_feature_dim: VisionEncoder output dim (default 256)
        imu_feature_dim:    IMUEncoder output dim (default 32)
        time_embed_dim:     sinusoidal time embedding dim (default 128)
        down_dims:          UNet encoder channel dims (default [256, 512])
        T_obs:              number of stacked frames (default 2, → 6 channels)
        T_pred:             action prediction horizon (default 8)
        action_dim:         CTBR action dim (default 4)
        n_inference_steps:  Euler integration steps at inference (default 1)
        t_embed_scale:      scale t∈[0,1] → int for sinusoidal embed (default 999)
    """

    def __init__(
        self,
        vision_feature_dim: int = 256,
        imu_feature_dim: int = 32,
        time_embed_dim: int = 128,
        down_dims: tuple = (256, 512),
        T_obs: int = 2,
        T_pred: int = 8,
        action_dim: int = 4,
        n_inference_steps: int = 1,
        t_embed_scale: int = 999,
    ):
        super().__init__()
        self.T_pred = T_pred
        self.action_dim = action_dim
        self.n_inference_steps = n_inference_steps
        self.t_embed_scale = t_embed_scale

        global_cond_dim = vision_feature_dim + imu_feature_dim  # 288

        self.vision_encoder = VisionEncoder(
            in_channels=T_obs * 3,
            feature_dim=vision_feature_dim,
        )

        self.imu_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.Mish(),
            nn.Linear(64, imu_feature_dim),
        )

        self.flow_net = ConditionalUnet1d(
            action_dim=action_dim,
            feature_dim=global_cond_dim,
            time_embed_dim=time_embed_dim,
            down_dims=down_dims,
            kernel_size=5,
            n_groups=8,
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, images: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, T_obs*3, H, W) uint8 or float
            imu:    (B, 6) normalised physics IMU

        Returns:
            global_cond: (B, 288)
        """
        vis_feat = self.vision_encoder(images)      # (B, 256)
        imu_feat = self.imu_encoder(imu)             # (B, 32)
        return torch.cat([vis_feat, imu_feat], dim=-1)  # (B, 288)

    def _t_to_int(self, t: torch.Tensor) -> torch.Tensor:
        """Scale continuous t∈[0,1] → integer index for sinusoidal embedding."""
        return (t * self.t_embed_scale).long().clamp(0, self.t_embed_scale)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        images: torch.Tensor,
        imu: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute flow matching loss.

        Args:
            images:  (B, T_obs*3, H, W)
            imu:     (B, 6)
            actions: (B, action_dim, T_pred)  CTBR in [-1, 1]

        Returns:
            loss: scalar MSE between predicted and target velocity
        """
        B = actions.shape[0]
        device = actions.device

        global_cond = self._encode(images, imu)

        t = torch.rand(B, device=device)                 # (B,) ~ U[0,1]
        eps = torch.randn_like(actions)                   # (B, 4, T_pred)

        t_expand = t[:, None, None]
        x_t = (1.0 - t_expand) * actions + t_expand * eps  # linear interpolant
        v_target = eps - actions                             # target velocity

        t_int = self._t_to_int(t)
        v_pred = self.flow_net(x_t, t_int, global_cond)

        return F.mse_loss(v_pred, v_target)

    # ------------------------------------------------------------------
    # RL fine-tuning (ReinFlow)
    # ------------------------------------------------------------------

    def compute_weighted_loss(
        self,
        images: torch.Tensor,
        imu: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        beta: float,
        fixed_x1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Advantage-weighted flow matching loss for ReinFlow.

        Uses t=1.0 and the stored rollout noise x1 for stable gradient estimates.
        Only positive-advantage steps contribute to the loss.

        Args:
            images:     (B, T_obs*3, H, W) float32 [0, 1]
            imu:        (B, 6)
            actions:    (B, action_dim, T_pred) — rollout actions as x_0
            advantages: (B,) normalised GAE advantages
            beta:       temperature for exponential weighting
            fixed_x1:   (B, action_dim, T_pred) stored rollout noise (optional)

        Returns:
            loss: scalar
        """
        B = actions.shape[0]
        device = actions.device

        global_cond = self._encode(images, imu)

        if fixed_x1 is not None:
            # Use stored rollout noise at t=1.0 for stable gradient
            eps = fixed_x1
            t = torch.ones(B, device=device)
        else:
            t = torch.rand(B, device=device)
            eps = torch.randn_like(actions)

        t_expand = t[:, None, None]
        x_t = (1.0 - t_expand) * actions + t_expand * eps
        v_target = eps - actions

        t_int = self._t_to_int(t)
        v_pred = self.flow_net(x_t, t_int, global_cond)

        weights = torch.exp(beta * advantages).clamp(max=20.0).detach()  # (B,)

        mse = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=[1, 2])  # (B,)
        return (weights * mse).mean()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(
        self,
        images: torch.Tensor,
        imu: torch.Tensor,
        n_steps: Optional[int] = None,
        _fixed_x1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample action sequence via Euler integration.

        Args:
            images:  (B, T_obs*3, H, W)
            imu:     (B, 6)
            n_steps: override n_inference_steps if provided

        Returns:
            actions: (B, action_dim, T_pred)
        """
        B = images.shape[0]
        device = images.device
        n = n_steps if n_steps is not None else self.n_inference_steps

        global_cond = self._encode(images, imu)

        x = _fixed_x1 if _fixed_x1 is not None else \
            torch.randn(B, self.action_dim, self.T_pred, device=device)

        dt = 1.0 / n
        for i in range(n):
            t_val = 1.0 - i * dt                          # 1.0 → dt
            t_batch = torch.full((B,), t_val, device=device)
            t_int = self._t_to_int(t_batch)
            v = self.flow_net(x, t_int, global_cond)
            x = x - dt * v

        return x                                           # (B, 4, T_pred)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location='cpu'))

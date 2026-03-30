"""
Conditional 1D U-Net for Action Sequence Denoising

Operates on action sequences of shape (B, action_dim, T_pred).
Uses FiLM conditioning (scale + shift) from visual features + timestep embedding.

Architecture:
    Encoder:  ResBlock(4->256) -> down(8->4) -> ResBlock(256->512) -> down(4->2)
    Mid:      ResBlock(512->512)
    Decoder:  up(2->4) -> ResBlock(512+512->256) -> up(4->8) -> ResBlock(256+256->action_dim)
    Final:    Conv1d(action_dim, action_dim, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_process import SinusoidalPositionEmbeddings


class FiLMConditioningMLP(nn.Module):
    """Generates scale and shift parameters for FiLM conditioning."""

    def __init__(self, cond_dim: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
        )

    def forward(self, cond: torch.Tensor):
        """
        Args:
            cond: (B, cond_dim)
        Returns:
            scale: (B, out_channels, 1)
            shift: (B, out_channels, 1)
        """
        out = self.net(cond)
        scale, shift = out.chunk(2, dim=-1)
        return scale[:, :, None], shift[:, :, None]


class ConditionalResBlock1d(nn.Module):
    """
    1D residual block with FiLM conditioning.

    Features:
      - Two Conv1d layers with GroupNorm and Mish activation
      - FiLM conditioning (scale + shift) injected after first GroupNorm
      - Residual connection (with channel projection if needed)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 cond_dim: int, kernel_size: int = 5, n_groups: int = 8):
        super().__init__()

        # First convolution
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=kernel_size // 2)
        self.norm1 = nn.GroupNorm(min(n_groups, out_channels), out_channels)

        # FiLM conditioning
        self.film = FiLMConditioningMLP(cond_dim, out_channels)

        # Second convolution
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=kernel_size // 2)
        self.norm2 = nn.GroupNorm(min(n_groups, out_channels), out_channels)

        # Residual projection
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.activation = nn.Mish()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T) input features
            cond: (B, cond_dim) conditioning vector

        Returns:
            (B, C_out, T) output features
        """
        residual = self.residual_proj(x)

        # Conv1 + Norm
        h = self.conv1(x)
        h = self.norm1(h)

        # FiLM conditioning: scale and shift
        scale, shift = self.film(cond)
        h = h * (scale + 1.0) + shift

        h = self.activation(h)

        # Conv2 + Norm
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)

        return h + residual


class Downsample1d(nn.Module):
    """Downsample temporal dimension by factor 2."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """Upsample temporal dimension by factor 2."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalUnet1d(nn.Module):
    """
    1D U-Net for denoising action sequences.

    Input:  noisy action sequence  (B, action_dim, T_pred)
    Cond:   visual features (B, feature_dim) + timestep (B,)
    Output: predicted noise         (B, action_dim, T_pred)
    """

    def __init__(self,
                 action_dim: int = 4,
                 feature_dim: int = 256,
                 time_embed_dim: int = 128,
                 down_dims: tuple = (256, 512),
                 kernel_size: int = 5,
                 n_groups: int = 8):
        super().__init__()

        self.action_dim = action_dim
        cond_dim = feature_dim + time_embed_dim

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.Mish(),
        )

        # --- Encoder ---
        dims = [action_dim] + list(down_dims)
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(len(down_dims)):
            self.encoder_blocks.append(
                ConditionalResBlock1d(dims[i], dims[i + 1], cond_dim, kernel_size, n_groups)
            )
            self.downsamples.append(Downsample1d(dims[i + 1]))

        # --- Mid ---
        self.mid_block = ConditionalResBlock1d(
            down_dims[-1], down_dims[-1], cond_dim, kernel_size, n_groups
        )

        # --- Decoder ---
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        reversed_dims = list(reversed(down_dims))
        # Output dims for decoder: [down_dims[-1], down_dims[0], action_dim]
        decoder_out_dims = list(reversed(down_dims[:-1])) + [action_dim]

        for i in range(len(down_dims)):
            # Input has skip connection (concat), so double channels
            in_ch = reversed_dims[i] * 2 if i > 0 else reversed_dims[i] + reversed_dims[i]
            # For first decoder block: mid output + skip from last encoder = 2 * last_dim
            if i == 0:
                in_ch = reversed_dims[0] * 2
            else:
                in_ch = reversed_dims[i] + decoder_out_dims[i - 1]

            self.upsamples.append(Upsample1d(reversed_dims[i] if i == 0 else decoder_out_dims[i - 1]))
            self.decoder_blocks.append(
                ConditionalResBlock1d(in_ch, decoder_out_dims[i], cond_dim, kernel_size, n_groups)
            )

        # Final projection
        self.final_conv = nn.Conv1d(action_dim, action_dim, 1)

    def forward(self, noisy_action: torch.Tensor, timestep: torch.Tensor,
                visual_features: torch.Tensor) -> torch.Tensor:
        """
        Predict noise from noisy action sequence.

        Args:
            noisy_action: (B, action_dim, T_pred) noisy action sequence
            timestep: (B,) diffusion timestep indices
            visual_features: (B, feature_dim) from VisionEncoder

        Returns:
            predicted_noise: (B, action_dim, T_pred)
        """
        # Build conditioning vector
        t_emb = self.time_mlp(timestep)  # (B, time_embed_dim)
        cond = torch.cat([visual_features, t_emb], dim=-1)  # (B, cond_dim)

        # Encoder (save skip connections)
        skips = []
        h = noisy_action

        for enc_block, downsample in zip(self.encoder_blocks, self.downsamples):
            h = enc_block(h, cond)
            skips.append(h)
            h = downsample(h)

        # Mid
        h = self.mid_block(h, cond)

        # Decoder (use skip connections)
        for i, (upsample, dec_block) in enumerate(zip(self.upsamples, self.decoder_blocks)):
            h = upsample(h)
            skip = skips.pop()

            # Handle size mismatch from downsampling/upsampling
            if h.shape[-1] != skip.shape[-1]:
                h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))

            h = torch.cat([h, skip], dim=1)
            h = dec_block(h, cond)

        # Final projection
        return self.final_conv(h)

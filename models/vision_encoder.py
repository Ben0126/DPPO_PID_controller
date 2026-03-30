"""
Vision Encoder — Lightweight CNN

Maps a stack of T_obs FPV images to a fixed-size feature vector.
The feature vector conditions the 1D U-Net during diffusion denoising.

Architecture:
    Conv2d(in, 32, 3, 2) -> GroupNorm -> Mish
    Conv2d(32, 64, 3, 2)  -> GroupNorm -> Mish
    Conv2d(64, 128, 3, 2) -> GroupNorm -> Mish
    Conv2d(128, 256, 3, 2) -> GroupNorm -> Mish
    AdaptiveAvgPool2d(1) -> Flatten -> Linear(256, feature_dim)
"""

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """
    CNN encoder for FPV image stacks.

    Input:  (B, T_obs * C, H, W)  — stacked frames along channel dim
    Output: (B, feature_dim)       — conditioning feature vector
    """

    def __init__(self, in_channels: int = 6, feature_dim: int = 256):
        """
        Args:
            in_channels: T_obs * 3 (e.g., 2 frames * 3 RGB = 6)
            feature_dim: output feature dimension
        """
        super().__init__()

        self.encoder = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.Mish(),

            # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.Mish(),

            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.Mish(),

            # 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.Mish(),

            # Global average pooling -> (B, 256, 1, 1) -> (B, 256)
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.fc = nn.Linear(256, feature_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, T_obs * C, H, W) uint8 or float [0, 1]

        Returns:
            features: (B, feature_dim)
        """
        # Normalize to [0, 1] if uint8
        x = images.float()
        if x.max() > 1.0:
            x = x / 255.0

        x = self.encoder(x)
        x = self.fc(x)
        return x

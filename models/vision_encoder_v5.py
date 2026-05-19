"""
Vision Encoder v5 — exposes spatial feature map for cross-attention.

Same CNN as v4 (VisionEncoder) but `forward(images, return_spatial=True)` returns
both the pooled feature vector (B, feature_dim) AND the pre-pool spatial map
(B, 256, 4, 4) so an IMU-guided cross-attention module can attend to specific
spatial tokens instead of being restricted to a single global pool.

Conv weights are byte-identical to VisionEncoder, allowing weight transfer from
H4 BC checkpoints (`vision_encoder.encoder.*`, `vision_encoder.fc.*`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoderV5(nn.Module):
    """
    CNN encoder that exposes both pooled and spatial features.

    forward(images)                       -> (B, feature_dim)
    forward(images, return_spatial=True)  -> (pooled (B, feature_dim),
                                              spatial (B, 256, 4, 4))
    """

    def __init__(self, in_channels: int = 6, feature_dim: int = 256):
        super().__init__()

        # Convolutional trunk — identical kernels/strides/norms to v4
        # so state_dict keys (`encoder.0.weight`, etc.) match for transfer.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.Mish(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.Mish(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.Mish(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.Mish(),
        )

        self.fc = nn.Linear(256, feature_dim)

    def forward(self, images: torch.Tensor, return_spatial: bool = False):
        x = images.float()
        if x.max() > 1.0:
            x = x / 255.0

        spatial = self.encoder(x)                            # (B, 256, 4, 4)
        pooled  = F.adaptive_avg_pool2d(spatial, 1).flatten(1)  # (B, 256)
        pooled  = self.fc(pooled)                            # (B, feature_dim)

        if return_spatial:
            return pooled, spatial
        return pooled

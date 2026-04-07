"""
Architecture v3.1: IMU Late Fusion + FCN Auxiliary Depth

Extends the baseline VisionDiffusionPolicy with:
  1. IMU Late Fusion — 6D IMU (ω + a) encoded by a lightweight MLP and
     concatenated with the 256D vision feature before conditioning the UNet.
     cond_dim: 256(vision) + 32(IMU) + 128(time) = 416
  2. FCN Auxiliary Depth Decoder — training-only ConvTranspose2d branch that
     reconstructs a 64×64 depth map from the vision feature vector.
     Stripped completely before ONNX/TensorRT export.

Total loss (Phase 3c):
    L = L_diffusion_weighted + λ_disp × L_dispersive + λ_depth × MSE(depth)

Prerequisite: expert_demos_v31.h5 collected with scripts/collect_data.py --v31
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

from .vision_encoder import VisionEncoder
from .conditional_unet1d import ConditionalUnet1d
from .diffusion_process import DiffusionProcess
from .diffusion_policy import DemoDataset


# ============================================================================
# Dataset
# ============================================================================

class DemoDatasetV31(DemoDataset):
    """
    Extends DemoDataset to also load imu_data and depth_maps from v3.1 HDF5.

    Each sample:
        image_stack:     (T_obs * C, H, W)    — unchanged
        action_sequence: (T_pred, action_dim)  — unchanged
        imu:             (6,)                  — [ωx, ωy, ωz, ax, ay, az]
        depth:           (1, H, W)             — depth GT aligned to last obs frame
    """

    # Path convention: if hdf5_path is 'data/expert_demos_v31.h5', the memmap
    # cache lives at 'data/v31_mmap/'.  Build it with:
    #   cd DPPO_PID_controller && python -c "from models.vision_dppo_v31 import \
    #       DemoDatasetV31; DemoDatasetV31.build_mmap_cache('data/expert_demos_v31.h5')"
    MMAP_DIR = 'data/v31_mmap'

    @staticmethod
    def build_mmap_cache(hdf5_path: str, out_dir: str = 'data/v31_mmap'):
        """One-time conversion of HDF5 → flat numpy memmap files (~1 min)."""
        import h5py, os
        os.makedirs(out_dir, exist_ok=True)
        with h5py.File(hdf5_path, 'r') as hf:
            ep_keys   = sorted(k for k in hf.keys() if k.startswith('episode_'))
            n_steps   = sum(hf[k]['images'].shape[0] for k in ep_keys)
            img_shape = hf[ep_keys[0]]['images'].shape[1:]
            dep_shape = hf[ep_keys[0]]['depth_maps'].shape[1:]
            act_dim   = hf[ep_keys[0]]['actions'].shape[1]
            mm_img = np.memmap(f'{out_dir}/images.dat',  dtype='uint8',   mode='w+',
                               shape=(n_steps, *img_shape))
            mm_act = np.memmap(f'{out_dir}/actions.dat', dtype='float32', mode='w+',
                               shape=(n_steps, act_dim))
            mm_imu = np.memmap(f'{out_dir}/imu.dat',     dtype='float32', mode='w+',
                               shape=(n_steps, 6))
            mm_dep = np.memmap(f'{out_dir}/depths.dat',  dtype='uint8',   mode='w+',
                               shape=(n_steps, *dep_shape))
            ep_offsets, offset = {}, 0
            for ep_key in ep_keys:
                T = hf[ep_key]['images'].shape[0]
                mm_img[offset:offset+T] = hf[ep_key]['images'][:]
                mm_act[offset:offset+T] = hf[ep_key]['actions'][:]
                mm_imu[offset:offset+T] = hf[ep_key]['imu_data'][:]
                mm_dep[offset:offset+T] = hf[ep_key]['depth_maps'][:]
                ep_offsets[ep_key] = offset
                offset += T
            del mm_img, mm_act, mm_imu, mm_dep
            np.save(f'{out_dir}/ep_offsets.npy', ep_offsets)
            np.save(f'{out_dir}/ep_keys.npy', ep_keys)
            np.save(f'{out_dir}/meta.npy', {'n_steps': n_steps, 'img_shape': img_shape,
                                             'dep_shape': dep_shape, 'act_dim': act_dim})
        print(f'Memmap cache built: {out_dir}')

    def __init__(self, hdf5_path: str, T_obs: int = 2, T_pred: int = 8):
        torch.utils.data.Dataset.__init__(self)
        self.T_obs  = T_obs
        self.T_pred = T_pred

        mmap_dir = self.MMAP_DIR
        if not os.path.isfile(f'{mmap_dir}/images.dat'):
            raise FileNotFoundError(
                f"Memmap cache not found at '{mmap_dir}/'. "
                "Build it first:\n  python -m scripts.build_v31_cache"
            )

        # Load metadata and open read-only memmap files
        ep_offsets = np.load(f'{mmap_dir}/ep_offsets.npy', allow_pickle=True).item()
        meta       = np.load(f'{mmap_dir}/meta.npy',       allow_pickle=True).item()
        n          = meta['n_steps']

        self._img = np.memmap(f'{mmap_dir}/images.dat',  dtype='uint8',   mode='r',
                              shape=(n, *meta['img_shape']))
        self._act = np.memmap(f'{mmap_dir}/actions.dat', dtype='float32', mode='r',
                              shape=(n, meta['act_dim']))
        self._imu = np.memmap(f'{mmap_dir}/imu.dat',     dtype='float32', mode='r',
                              shape=(n, 6))
        self._dep = np.memmap(f'{mmap_dir}/depths.dat',  dtype='uint8',   mode='r',
                              shape=(n, *meta['dep_shape']))

        # Build samples: flat_t = episode offset + step index
        # Use sorted offsets to compute each episode's length correctly.
        self.samples = []
        sorted_keys    = sorted(ep_offsets.keys())
        sorted_offsets = [ep_offsets[k] for k in sorted_keys]
        sorted_offsets.append(n)   # sentinel: total steps

        for i, ep_key in enumerate(sorted_keys):
            off    = sorted_offsets[i]
            ep_len = sorted_offsets[i + 1] - off
            for t in range(T_obs - 1, ep_len - T_pred + 1):
                self.samples.append(off + t)   # flat index of the 'current' step

        print(f"Loaded {len(self.samples)} samples from {mmap_dir}")

    def __getitem__(self, idx: int):
        flat_t = self.samples[idx]

        img_stack  = self._img[(flat_t - self.T_obs + 1):(flat_t + 1)]  # (T_obs, C, H, W)
        img_stack  = img_stack.reshape(-1, *img_stack.shape[2:])          # (T_obs*C, H, W)
        action_seq = self._act[flat_t:flat_t + self.T_pred]               # (T_pred, 4)
        imu        = self._imu[flat_t]                                     # (6,)
        depth      = self._dep[flat_t]                                     # (1, H, W)

        return (
            torch.from_numpy(np.array(img_stack)).float(),
            torch.from_numpy(np.array(action_seq)).float(),
            torch.from_numpy(np.array(imu)).float(),
            torch.from_numpy(np.array(depth)).float(),
        )


# ============================================================================
# Sub-modules
# ============================================================================

class IMUEncoder(nn.Module):
    """
    Lightweight MLP encoder for 6D IMU measurements.

    Input:  (B, 6) — [ωx, ωy, ωz, ax, ay, az] in body frame
    Output: (B, out_dim=32)
    """

    def __init__(self, imu_dim: int = 6, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(imu_dim, 64),
            nn.Mish(),
            nn.Linear(64, out_dim),
        )

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        return self.net(imu)


class DepthDecoder(nn.Module):
    """
    FCN auxiliary depth decoder (training only — stripped before deployment).

    Reconstructs a 64×64 depth map from the 256D vision feature vector.
    Depth is normalised to [0, 1] (1.0 = 10 m+ in the environment).

    Input:  (B, 256) vision features
    Output: (B, 1, 64, 64) predicted depth in [0, 1]

    Spatial resolution path:
        (B,256,1,1)
        → CT(256→128, k=4, s=1)       → (B,128,4,4)
        → CT(128→64,  k=4, s=2, p=1)  → (B,64,8,8)
        → CT(64→32,   k=4, s=2, p=1)  → (B,32,16,16)
        → CT(32→16,   k=4, s=2, p=1)  → (B,16,32,32)
        → CT(16→1,    k=4, s=2, p=1)  → (B,1,64,64)
        → Sigmoid
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (feature_dim, 1, 1)),
            nn.ConvTranspose2d(feature_dim, 128, kernel_size=4, stride=1),
            nn.GroupNorm(8, 128),
            nn.Mish(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.Mish(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.Mish(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 16),
            nn.Mish(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, feature_dim)
        Returns:
            depth: (B, 1, 64, 64) in [0, 1]
        """
        return self.decoder(features)


# ============================================================================
# Main policy
# ============================================================================

class VisionDPPOv31(nn.Module):
    """
    Architecture v3.1: IMU Late Fusion + FCN Auxiliary Depth.

    Pipeline:
        FPV images   → VisionEncoder              → 256D vision_feat
        6D IMU       → IMUEncoder                 → 32D imu_feat
        cat([vision, imu])                         → 288D global_cond
        global_cond + timestep_embed(128)          → 416D cond
        ConditionalUnet1d(feature_dim=288)         → predicted noise ε_θ

        [training only]
        vision_feat  → DepthDecoder               → (1,64,64) depth_pred

    Loss (training):
        L = exp(β × A) × L_diffusion
          + λ_disp  × L_dispersive(vision_feat)
          + λ_depth × MSE(depth_pred, depth_gt)
    """

    IMU_DIM        = 6
    IMU_FEAT_DIM   = 32
    VISION_DIM     = 256
    GLOBAL_COND_DIM = VISION_DIM + IMU_FEAT_DIM   # 288

    def __init__(self,
                 action_dim: int = 4,
                 T_obs: int = 2,
                 T_pred: int = 8,
                 image_channels: int = 3,
                 image_size: int = 64,
                 time_embed_dim: int = 128,
                 down_dims: tuple = (256, 512),
                 num_diffusion_steps: int = 100,
                 beta_schedule: str = 'cosine',
                 ddim_steps: int = 10,
                 use_depth_decoder: bool = True):
        super().__init__()

        self.action_dim       = action_dim
        self.T_obs            = T_obs
        self.T_pred           = T_pred
        self.ddim_steps       = ddim_steps
        self.use_depth_decoder = use_depth_decoder

        # Vision encoder: (B, T_obs*C, H, W) → (B, 256)
        self.vision_encoder = VisionEncoder(
            in_channels=T_obs * image_channels,
            feature_dim=self.VISION_DIM,
        )

        # IMU encoder: (B, 6) → (B, 32)
        self.imu_encoder = IMUEncoder(
            imu_dim=self.IMU_DIM,
            out_dim=self.IMU_FEAT_DIM,
        )

        # Noise prediction UNet conditioned on 288D global_cond
        self.noise_pred_net = ConditionalUnet1d(
            action_dim=action_dim,
            feature_dim=self.GLOBAL_COND_DIM,   # 288 instead of 256
            time_embed_dim=time_embed_dim,
            down_dims=down_dims,
        )

        # FCN depth decoder (training only)
        self.depth_decoder = DepthDecoder(feature_dim=self.VISION_DIM) \
            if use_depth_decoder else None

        # Diffusion schedules
        self.diffusion = DiffusionProcess(
            num_timesteps=num_diffusion_steps,
            beta_schedule=beta_schedule,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, image_stack: torch.Tensor,
                imu_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image + IMU → (global_cond, vision_feat)."""
        vision_feat = self.vision_encoder(image_stack)         # (B, 256)
        imu_feat    = self.imu_encoder(imu_data)               # (B, 32)
        global_cond = torch.cat([vision_feat, imu_feat], dim=-1)  # (B, 288)
        return global_cond, vision_feat

    @staticmethod
    def _dispersive_loss(features: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        """
        Forces all feature vectors in a batch to repel each other.
        L_disp = -mean( log(||fi - fj|| + ε) )  for i ≠ j
        """
        B = features.shape[0]
        if B < 2:
            return features.sum() * 0.0
        diff = features.unsqueeze(1) - features.unsqueeze(0)    # (B,B,D)
        dist = torch.norm(diff, dim=-1)                          # (B,B)
        mask = 1 - torch.eye(B, device=features.device)
        loss = -torch.log(dist + 1e-6) * mask
        return loss.sum() / (B * (B - 1))

    # ------------------------------------------------------------------
    # Training: supervised (Phase 3a v3.1)
    # ------------------------------------------------------------------

    def compute_loss(self, image_stack: torch.Tensor,
                     action_sequence: torch.Tensor,
                     imu_data: torch.Tensor,
                     depth_gt: Optional[torch.Tensor] = None,
                     lambda_dispersive: float = 0.1,
                     lambda_depth: float = 0.1) -> Tuple[torch.Tensor, dict]:
        """
        Supervised diffusion loss + dispersive + optional depth.

        Args:
            image_stack:     (B, T_obs*C, H, W)
            action_sequence: (B, T_pred, action_dim)
            imu_data:        (B, 6)
            depth_gt:        (B, 1, H, W) float [0,1] or None
            lambda_dispersive: weight for dispersive loss
            lambda_depth:      weight for depth reconstruction loss

        Returns:
            total_loss, metrics_dict
        """
        B      = image_stack.shape[0]
        device = image_stack.device

        global_cond, vision_feat = self._encode(image_stack, imu_data)
        action_seq = action_sequence.permute(0, 2, 1)   # (B, action_dim, T_pred)

        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device)
        noisy_actions, noise = self.diffusion.q_sample(action_seq, t)
        predicted_noise = self.noise_pred_net(noisy_actions, t, global_cond)

        l_diff = F.mse_loss(predicted_noise, noise)
        l_disp = self._dispersive_loss(vision_feat) * lambda_dispersive

        l_depth = torch.tensor(0.0, device=device)
        if depth_gt is not None and self.depth_decoder is not None:
            depth_pred = self.depth_decoder(vision_feat)
            depth_gt_f = depth_gt.float() / 255.0   # normalise uint8 GT to [0,1]
            l_depth    = F.mse_loss(depth_pred, depth_gt_f) * lambda_depth

        total = l_diff + l_disp + l_depth
        return total, {
            'loss_diffusion': l_diff.item(),
            'loss_dispersive': l_disp.item(),
            'loss_depth': l_depth.item(),
        }

    # ------------------------------------------------------------------
    # Training: DPPO advantage-weighted (Phase 3c v3.1)
    # ------------------------------------------------------------------

    def compute_weighted_loss(self, image_stack: torch.Tensor,
                              action_sequence: torch.Tensor,
                              imu_data: torch.Tensor,
                              advantages: torch.Tensor,
                              beta: float = 0.1,
                              depth_gt: Optional[torch.Tensor] = None,
                              lambda_dispersive: float = 0.1,
                              lambda_depth: float = 0.1) -> Tuple[torch.Tensor, dict]:
        """
        DPPO advantage-weighted loss + dispersive + depth.

        L = E[ exp(β × A_norm) × ||ε_θ - ε||² ] + λ_disp × L_disp + λ_depth × L_depth
        """
        B      = image_stack.shape[0]
        device = image_stack.device

        global_cond, vision_feat = self._encode(image_stack, imu_data)
        action_seq = action_sequence.permute(0, 2, 1)

        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device)
        noisy_actions, noise = self.diffusion.q_sample(action_seq, t)
        predicted_noise = self.noise_pred_net(noisy_actions, t, global_cond)

        per_sample_loss = F.mse_loss(predicted_noise, noise, reduction='none')
        per_sample_loss = per_sample_loss.mean(dim=(1, 2))   # (B,)

        weights = torch.exp(beta * advantages)
        weights = torch.clamp(weights, 0.1, 10.0)
        l_diff  = (weights * per_sample_loss).mean()

        l_disp = self._dispersive_loss(vision_feat) * lambda_dispersive

        l_depth = torch.tensor(0.0, device=device)
        if depth_gt is not None and self.depth_decoder is not None:
            depth_pred = self.depth_decoder(vision_feat)
            depth_gt_f = depth_gt.float() / 255.0
            l_depth    = F.mse_loss(depth_pred, depth_gt_f) * lambda_depth

        total = l_diff + l_disp + l_depth
        return total, {
            'loss_diffusion': l_diff.item(),
            'loss_dispersive': l_disp.item(),
            'loss_depth': l_depth.item(),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(self, image_stack: torch.Tensor,
                       imu_data: torch.Tensor,
                       ddim_steps: Optional[int] = None) -> torch.Tensor:
        """
        Generate action sequence (DDIM).  Depth decoder is NOT called.

        Args:
            image_stack: (B, T_obs*C, H, W) or (T_obs*C, H, W)
            imu_data:    (B, 6) or (6,)

        Returns:
            action_sequence: (B, T_pred, action_dim) in [-1, 1]
        """
        if image_stack.dim() == 3:
            image_stack = image_stack.unsqueeze(0)
        if imu_data.dim() == 1:
            imu_data = imu_data.unsqueeze(0)

        B      = image_stack.shape[0]
        device = image_stack.device
        steps  = ddim_steps or self.ddim_steps

        global_cond, _ = self._encode(image_stack, imu_data)

        def denoise_fn(noisy_action, timestep, condition):
            return self.noise_pred_net(noisy_action, timestep, condition)

        shape      = (B, self.action_dim, self.T_pred)
        action_seq = self.diffusion.ddim_sample(
            denoise_fn, global_cond, shape, ddim_steps=steps
        )
        action_seq = action_seq.permute(0, 2, 1)
        return torch.clamp(action_seq, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        """Save full policy (including depth decoder if present)."""
        torch.save({
            'vision_encoder': self.vision_encoder.state_dict(),
            'imu_encoder':    self.imu_encoder.state_dict(),
            'noise_pred_net': self.noise_pred_net.state_dict(),
            'depth_decoder':  self.depth_decoder.state_dict()
                              if self.depth_decoder is not None else None,
        }, filepath)

    def load(self, filepath: str):
        """Load policy checkpoint."""
        ckpt = torch.load(filepath, map_location='cpu', weights_only=False)
        self.vision_encoder.load_state_dict(ckpt['vision_encoder'])
        self.imu_encoder.load_state_dict(ckpt['imu_encoder'])
        self.noise_pred_net.load_state_dict(ckpt['noise_pred_net'])
        if self.depth_decoder is not None and ckpt.get('depth_decoder') is not None:
            self.depth_decoder.load_state_dict(ckpt['depth_decoder'])

    def save_deployable(self, filepath: str):
        """
        Save inference-only checkpoint: depth_decoder is excluded.
        Use this before ONNX/TensorRT export for Jetson deployment.
        """
        torch.save({
            'vision_encoder': self.vision_encoder.state_dict(),
            'imu_encoder':    self.imu_encoder.state_dict(),
            'noise_pred_net': self.noise_pred_net.state_dict(),
            'depth_decoder':  None,   # explicitly absent in deploy model
        }, filepath)
        print(f"Deployable model saved (no depth decoder): {filepath}")

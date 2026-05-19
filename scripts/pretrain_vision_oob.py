"""
Stage B — Vision OOB (Out-of-Base) Pre-training.

Trains only VisionEncoderV5 + StatePredictor on OOD tilt images from
expert_demos_v4_recovery.h5, so that downstream distillation starts with a
vision encoder that already understands tilted-image physics.

Why this is different from v5 BC training:
  * v5 BC used hover-only episodes → vision encoder never saw swift perturbation
  * DAgger rollouts inject OOD tilt images → state predictor explodes (±5σ outlier)
  * This script fixes the coverage gap by pretraining on recovery episodes

Design:
  * Loads FlowMatchingPolicyV5 with v5 BC weights (flow/IMU/cross_attn already good)
  * Freezes: imu_encoder, cross_attn, flow_net, tilt_head
  * Trains:  vision_encoder + state_predictor ONLY
  * State target: 'partial' normalization — rot6D[3:9] kept raw (avoids ±5σ issue)
  * Saves full model state_dict → distillation can load directly via policy.load()

Usage:
    dppo/Scripts/python.exe -m scripts.pretrain_vision_oob \\
        --pretrained checkpoints/flow_policy_v5/20260518_072501/best_model.pt \\
        --flow-config configs/flow_policy_v5.yaml \\
        --recovery-h5 data/expert_demos_v4_recovery.h5 \\
        --recovery-episodes 500 \\
        --epochs 60
"""

import os
import sys
import argparse
import yaml
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.flow_policy_v5 import FlowMatchingPolicyV5
from models.ppo_expert import RunningMeanStd


# ---------------------------------------------------------------------------
# Normalization helper — mirrors train_distillation_v5._normalise_state
# State layout: [pos_error(0:3), rot_6d(3:9), vel(9:12), omega(12:15)]
# ---------------------------------------------------------------------------

def _normalise_state(state: np.ndarray, obs_rms: RunningMeanStd,
                     mode: str = 'partial') -> np.ndarray:
    if mode == 'raw':
        return state.astype(np.float32)
    full = obs_rms.normalize(state).astype(np.float32)
    if mode == 'full':
        return full
    if mode == 'partial':
        full[3:9] = state[3:9].astype(np.float32)
        return full
    raise ValueError(f"unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OOBStateDataset(Dataset):
    """
    Loads image stacks + partial-normalised states from recovery h5.
    Ignores actions entirely — we only care about image -> state mapping.
    """

    def __init__(self, h5_path: str, obs_rms: RunningMeanStd,
                 episode_indices: list, T_obs: int = 2,
                 state_norm: str = 'partial'):
        img_buf, imu_buf, state_buf = [], [], []

        print(f"Loading OOB dataset from {h5_path} ({len(episode_indices)} episodes)...")
        with h5py.File(h5_path, 'r') as f:
            for ep_idx in episode_indices:
                key = f'episode_{ep_idx}'
                if key not in f:
                    continue
                imgs   = f[key]['images'][:]       # (T, 3, H, W) uint8
                states = f[key]['states'][:]       # (T, 15) float32
                imus   = f[key]['imu_data'][:]     # (T, 6) float32
                T = imgs.shape[0]
                for t in range(T_obs - 1, T):
                    frames = imgs[t - T_obs + 1 : t + 1]         # (T_obs, 3, H, W)
                    img_buf.append(np.concatenate(frames, axis=0)) # (T_obs*3, H, W)
                    imu_buf.append(imus[t])
                    state_buf.append(_normalise_state(states[t], obs_rms, state_norm))

        self._imgs   = np.stack(img_buf)    # (N, T_obs*3, H, W) uint8
        self._imus   = np.stack(imu_buf)    # (N, 6)
        self._states = np.stack(state_buf)  # (N, 15)
        print(f"  {len(self._imgs):,} samples | "
              f"images: {self._imgs.nbytes/1e9:.2f}GB")

    def __len__(self):
        return len(self._imgs)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self._imgs[idx]),    # uint8
            torch.from_numpy(self._imus[idx]),
            torch.from_numpy(self._states[idx]),
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    with open(args.flow_config, 'r', encoding='utf-8') as f:
        flow_cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    vis_cfg  = flow_cfg['vision']
    act_cfg  = flow_cfg['action']
    unet_cfg = flow_cfg['unet']
    imu_cfg  = flow_cfg['imu']
    xattn    = flow_cfg.get('cross_attn', {})
    sp_cfg   = flow_cfg.get('state_predictor', {})

    # Build model and load v5 BC weights
    policy = FlowMatchingPolicyV5(
        vision_feature_dim     = vis_cfg['feature_dim'],
        imu_feature_dim        = imu_cfg['feature_dim'],
        time_embed_dim         = unet_cfg['time_embed_dim'],
        down_dims              = tuple(unet_cfg['down_dims']),
        T_obs                  = vis_cfg['T_obs'],
        T_pred                 = act_cfg['T_pred'],
        action_dim             = act_cfg['action_dim'],
        n_inference_steps      = flow_cfg['flow']['n_inference_steps'],
        t_embed_scale          = flow_cfg['flow']['t_embed_scale'],
        cross_attn_heads       = xattn.get('n_heads', 8),
        state_predictor_hidden = sp_cfg.get('hidden_dim', 256),
        state_dim              = sp_cfg.get('state_dim', 15),
    ).to(device)

    if args.pretrained:
        policy.load(args.pretrained)
        print(f"Loaded v5 BC weights: {args.pretrained}")

    # Freeze everything except vision_encoder + state_predictor
    for name, param in policy.named_parameters():
        if name.startswith('vision_encoder') or name.startswith('state_predictor'):
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in policy.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"(vision_encoder + state_predictor only)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    # obs_rms for state normalisation
    norm_data = np.load(args.obs_rms)
    obs_rms   = RunningMeanStd(shape=(15,))
    obs_rms.load_state_dict({
        'mean':  norm_data['mean'],
        'var':   norm_data['var'],
        'count': float(norm_data['count']),
    })

    # Dataset split
    all_eps = list(range(args.recovery_episodes))
    np.random.shuffle(all_eps)
    n_val  = max(1, int(len(all_eps) * 0.1))
    val_ep = all_eps[:n_val]
    trn_ep = all_eps[n_val:]

    trn_ds = OOBStateDataset(args.recovery_h5, obs_rms, trn_ep,
                              T_obs=vis_cfg['T_obs'],
                              state_norm=args.state_norm)
    val_ds = OOBStateDataset(args.recovery_h5, obs_rms, val_ep,
                              T_obs=vis_cfg['T_obs'],
                              state_norm=args.state_norm)

    # num_workers=0 on Windows: dataset is fully in-memory numpy arrays,
    # spawn-based multiprocessing can't pickle large (5GB+) arrays through the pipe.
    n_workers  = 0 if os.name == 'nt' else 4
    pin_mem    = n_workers > 0
    trn_loader = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=n_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=n_workers, pin_memory=pin_mem)

    # Logging
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/pretrain_vision_oob/oob_{ts}"
    sav_dir = f"checkpoints/pretrain_vision_oob/oob_{ts}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(sav_dir, exist_ok=True)
    writer  = SummaryWriter(log_dir)

    print(f"\n{'='*60}")
    print(f"Vision OOB Pretrain — vision_encoder + state_predictor only")
    print(f"Epochs: {args.epochs} | LR: {args.lr:.1e} | Batch: {args.batch_size}")
    print(f"Train episodes: {len(trn_ep)} | Val episodes: {len(val_ep)}")
    print(f"State norm: {args.state_norm}")
    print(f"Save: {sav_dir}")
    print(f"{'='*60}\n")

    best_val = float('inf')

    for epoch in range(args.epochs):
        # --- Train ---
        policy.train()
        trn_losses = []
        for imgs_u8, imus, states_gt in trn_loader:
            imgs = imgs_u8.to(device, dtype=torch.float32, non_blocking=True) / 255.0
            imus = imus.to(device, non_blocking=True)
            states_gt = states_gt.to(device, non_blocking=True)

            vis_pooled, _ = policy.vision_encoder(imgs, return_spatial=True)
            state_pred    = policy.state_predictor(vis_pooled)
            loss = F.mse_loss(state_pred, states_gt)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, policy.parameters()), 1.0
            )
            optimizer.step()
            trn_losses.append(loss.item())

        scheduler.step()

        # --- Val ---
        policy.eval()
        val_losses = []
        # Per-dimension val loss to see which dims are hardest
        per_dim = torch.zeros(15, device=device)
        n_val_batches = 0
        with torch.no_grad():
            for imgs_u8, imus, states_gt in val_loader:
                imgs = imgs_u8.to(device, dtype=torch.float32, non_blocking=True) / 255.0
                states_gt = states_gt.to(device, non_blocking=True)
                vis_pooled, _ = policy.vision_encoder(imgs, return_spatial=True)
                state_pred    = policy.state_predictor(vis_pooled)
                val_losses.append(F.mse_loss(state_pred, states_gt).item())
                per_dim += ((state_pred - states_gt) ** 2).mean(dim=0)
                n_val_batches += 1
        per_dim /= n_val_batches

        trn_mean = float(np.mean(trn_losses))
        val_mean = float(np.mean(val_losses))

        writer.add_scalar('pretrain/train_state_loss', trn_mean, epoch)
        writer.add_scalar('pretrain/val_state_loss',   val_mean, epoch)
        writer.add_scalar('pretrain/lr', scheduler.get_last_lr()[0], epoch)
        # Log per-dim: pos(0:3), rot6d(3:9), vel(9:12), omega(12:15)
        writer.add_scalar('pretrain/val_pos_loss',   per_dim[0:3].mean().item(),  epoch)
        writer.add_scalar('pretrain/val_rot6d_loss', per_dim[3:9].mean().item(),  epoch)
        writer.add_scalar('pretrain/val_vel_loss',   per_dim[9:12].mean().item(), epoch)
        writer.add_scalar('pretrain/val_omega_loss', per_dim[12:15].mean().item(),epoch)

        print(f"Epoch {epoch+1:>3}/{args.epochs}  "
              f"trn={trn_mean:.4f}  val={val_mean:.4f}  "
              f"[pos={per_dim[0:3].mean():.3f} "
              f"rot={per_dim[3:9].mean():.3f} "
              f"vel={per_dim[9:12].mean():.3f} "
              f"ω={per_dim[12:15].mean():.3f}]")

        if val_mean < best_val:
            best_val = val_mean
            policy.save(os.path.join(sav_dir, 'best_model.pt'))
            print(f"  --> Best val={best_val:.4f} saved")

        if (epoch + 1) % 10 == 0:
            policy.save(os.path.join(sav_dir, f'epoch_{epoch+1}.pt'))

    policy.save(os.path.join(sav_dir, 'final_model.pt'))
    writer.close()
    print(f"\nOOB pretrain complete. Best val state_loss: {best_val:.4f}")
    print(f"Saved to: {sav_dir}")
    print(f"\nNext step — restart distillation with:")
    print(f"  dppo/Scripts/python.exe -m scripts.train_distillation_v5 \\")
    print(f"      --pretrained {sav_dir}/best_model.pt \\")
    print(f"      --flow-config configs/flow_policy_v5.yaml \\")
    print(f"      --rl-config   configs/distillation_v5.yaml")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stage B — Vision OOB pretrain (vision_encoder + state_predictor)')
    parser.add_argument('--pretrained',    type=str,
                        default='checkpoints/flow_policy_v5/20260518_072501/best_model.pt',
                        help='v5 BC checkpoint to init from')
    parser.add_argument('--flow-config',   type=str,
                        default='configs/flow_policy_v5.yaml')
    parser.add_argument('--recovery-h5',   type=str,
                        default='data/expert_demos_v4_recovery.h5')
    parser.add_argument('--recovery-episodes', type=int, default=500)
    parser.add_argument('--obs-rms',       type=str,
                        default='checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz')
    parser.add_argument('--epochs',        type=int, default=60)
    parser.add_argument('--batch-size',    type=int, default=512)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--state-norm',    type=str, default='partial',
                        choices=['partial', 'full', 'raw'])
    args = parser.parse_args()
    train(args)

"""
Phase 3a v4.0: Flow Matching Policy Supervised Pre-training

Trains FlowMatchingPolicyV4 on CTBR expert demonstrations.

Flow Matching (OT-CFM):
  x_t = (1-t)*x_0 + t*ε,  t ~ U[0,1],  ε ~ N(0,I)
  Loss = ||v_θ(x_t, t, cond) − (ε − x_0)||²

No cosine schedule, no noise amplification.

Usage:
    python -m scripts.train_flow_v4 --config configs/flow_policy_v4.yaml
    python -m scripts.train_flow_v4 --config configs/flow_policy_v4.yaml --quick
"""

import os
import sys
import time
import argparse
import numpy as np
import yaml
import h5py
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.flow_policy_v4 import FlowMatchingPolicyV4


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FlowDatasetV4(Dataset):
    """
    Sliding-window dataset from one or more HDF5 files (CTBR + INDI).

    Accepts a list of (h5_path, episode_indices) pairs so that hover and
    recovery demonstrations can be mixed in a single dataset object.

    Each sample:
      images:  (T_obs*3, H, W) uint8 — stacked FPV frames
      imu:     (6,) float32          — normalised physics IMU at last obs frame
      actions: (action_dim, T_pred)  — CTBR target sequence
    """

    def __init__(self, sources: list,
                 T_obs: int = 2, T_pred: int = 8):
        """
        Args:
            sources: list of (h5_path, episode_indices) tuples.
                     Pass a single-element list for the standard single-file case.
        """
        self.T_obs  = T_obs
        self.T_pred = T_pred

        self._img_buf  = []
        self._imu_buf  = []
        self._act_buf  = []
        _tilt_buf      = []

        for h5_path, episode_indices in sources:
            print(f"  Loading {len(episode_indices)} episodes from {h5_path} ...")
            with h5py.File(h5_path, 'r') as f:
                for ep_idx in episode_indices:
                    key = f'episode_{ep_idx}'
                    if key not in f:
                        continue
                    imgs       = f[key]['images'][:]      # (T, 3, H, W) uint8
                    acts       = f[key]['actions'][:]     # (T, 4) float32
                    imus       = f[key]['imu_data'][:]    # (T, 6) float32
                    states_ep  = f[key]['states'][:]      # (T, 15) float32
                    T = acts.shape[0]
                    for start in range(T_obs - 1, T - T_pred):
                        frames = imgs[start - T_obs + 1 : start + 1]
                        self._img_buf.append(np.concatenate(frames, axis=0))
                        self._imu_buf.append(imus[start])
                        self._act_buf.append(
                            acts[start + 1 : start + 1 + T_pred].T)
                        # tilt_gt from rot_6d stored in state[3:9]
                        # R[:,0]=state[3:6], R[:,1]=state[6:9]
                        # R[2,2] = cross(R[:,0], R[:,1])[2] = s[3]*s[7] - s[4]*s[6]
                        s = states_ep[start]
                        r2z = float(s[3]*s[7] - s[4]*s[6])
                        _tilt_buf.append(np.float32(np.arccos(np.clip(r2z, -1.0, 1.0))))

        self._img_arr  = np.stack(self._img_buf)
        self._imu_arr  = np.stack(self._imu_buf)
        self._act_arr  = np.stack(self._act_buf)
        self._tilt_arr = np.array(_tilt_buf, dtype=np.float32)
        del self._img_buf, self._imu_buf, self._act_buf, _tilt_buf

        N = len(self._img_arr)
        print(f"  Total: {N:,} samples  "
              f"img={self._img_arr.nbytes/1e9:.2f}GB  "
              f"imu+act={(self._imu_arr.nbytes + self._act_arr.nbytes)/1e6:.1f}MB")

    def __len__(self):
        return len(self._img_arr)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self._img_arr[idx]),
            torch.from_numpy(self._imu_arr[idx]),
            torch.from_numpy(self._act_arr[idx]),
            torch.tensor(self._tilt_arr[idx]),
        )


# ---------------------------------------------------------------------------
# GPU augmentation (on-device, zero CPU overhead)
# ---------------------------------------------------------------------------

def gpu_augment(images: torch.Tensor) -> torch.Tensor:
    """
    Brightness + contrast jitter entirely on GPU.
    images: (B, C, H, W) float32 [0, 255]
    """
    B = images.shape[0]
    # Brightness ×[0.7, 1.3]
    bright = torch.empty(B, 1, 1, 1, device=images.device).uniform_(0.7, 1.3)
    images = images * bright
    # Contrast jitter: per-image mean shift
    contrast = torch.empty(B, 1, 1, 1, device=images.device).uniform_(0.8, 1.2)
    mean = images.mean(dim=[2, 3], keepdim=True)
    images = (images - mean) * contrast + mean
    return images.clamp(0, 255)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    if args.quick:
        cfg['training']['num_epochs'] = 5
        cfg['training']['batch_size'] = 64
        cfg['training']['num_workers'] = 0
        cfg['training']['_max_episodes'] = 20
        print("[Quick Test Mode] 5 epochs, 20 episodes, batch=64")

    vis_cfg    = cfg['vision']
    imu_cfg    = cfg['imu']
    unet_cfg   = cfg['unet']
    act_cfg    = cfg['action']
    flow_cfg   = cfg['flow']
    train_cfg  = cfg['training']
    log_cfg    = cfg['logging']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Dataset split — supports optional recovery h5 mix-in
    # ------------------------------------------------------------------
    h5_path = train_cfg['dataset_path']
    with h5py.File(h5_path, 'r') as f:
        n_ep = f.attrs['n_episodes']

    max_ep = args.hover_episodes if args.hover_episodes > 0 else train_cfg.get('_max_episodes', n_ep)
    all_ep = list(range(min(n_ep, max_ep)))
    n_val  = max(1, int(len(all_ep) * train_cfg['val_split']))
    n_train = len(all_ep) - n_val
    rng = np.random.default_rng(42)
    rng.shuffle(all_ep)
    hover_train_ep = all_ep[:n_train]
    hover_val_ep   = all_ep[n_train:]

    T_obs  = vis_cfg['T_obs']
    T_pred = act_cfg['T_pred']

    # Build source lists (each entry = (h5_path, episode_indices))
    train_sources = [(h5_path, hover_train_ep)]
    val_sources   = [(h5_path, hover_val_ep)]

    if args.recovery_h5:
        with h5py.File(args.recovery_h5, 'r') as f:
            n_rec = f.attrs['n_episodes']
        max_rec = args.recovery_episodes if args.recovery_episodes > 0 else n_rec
        rec_ep  = list(range(min(n_rec, max_rec)))
        n_rec_val   = max(1, int(len(rec_ep) * train_cfg['val_split']))
        n_rec_train = len(rec_ep) - n_rec_val
        rng2 = np.random.default_rng(99)
        rng2.shuffle(rec_ep)
        train_sources.append((args.recovery_h5, rec_ep[:n_rec_train]))
        val_sources.append(  (args.recovery_h5, rec_ep[n_rec_train:]))
        print(f"Recovery mix-in: {n_rec_train} train + {n_rec_val} val episodes "
              f"from {args.recovery_h5}")

    train_ds = FlowDatasetV4(train_sources, T_obs=T_obs, T_pred=T_pred)
    val_ds   = FlowDatasetV4(val_sources,   T_obs=T_obs, T_pred=T_pred)
    print(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    nw = train_cfg['num_workers']
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=nw,
        pin_memory=(nw > 0),
        persistent_workers=(nw > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=nw,
        pin_memory=(nw > 0),
        persistent_workers=(nw > 0),
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = FlowMatchingPolicyV4(
        vision_feature_dim = vis_cfg['feature_dim'],
        imu_feature_dim    = imu_cfg['feature_dim'],
        time_embed_dim     = unet_cfg['time_embed_dim'],
        down_dims          = tuple(unet_cfg['down_dims']),
        T_obs              = T_obs,
        T_pred             = T_pred,
        action_dim         = act_cfg['action_dim'],
        n_inference_steps  = flow_cfg['n_inference_steps'],
        t_embed_scale      = flow_cfg['t_embed_scale'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
    )

    num_epochs   = train_cfg['num_epochs']
    warmup_steps = len(train_loader) * train_cfg['warmup_epochs']
    total_steps  = len(train_loader) * num_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir  = os.path.join(log_cfg['tensorboard_log'], timestamp)
    save_dir = os.path.join(log_cfg['save_path'], timestamp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    recovery_tag = f" + recovery({args.recovery_h5})" if args.recovery_h5 else ""
    print(f"\n{'='*60}")
    print(f"Flow Matching Policy v4.0 — Supervised Pre-training{recovery_tag}")
    print(f"Epochs:     {num_epochs}   Batch: {train_cfg['batch_size']}   LR: {train_cfg['learning_rate']}")
    print(f"Train samples: {len(train_ds):,}  Val samples: {len(val_ds):,}")
    print(f"Save dir:   {save_dir}")
    print(f"{'='*60}\n")

    lambda_tilt = train_cfg.get('lambda_tilt', 0.1)
    best_val_loss = float('inf')
    global_step   = 0
    scaler = GradScaler('cuda')

    for epoch in range(1, num_epochs + 1):
        # ----------------------------------------------------------------
        # Train epoch
        # ----------------------------------------------------------------
        model.train()
        epoch_loss = 0.0
        t_start = time.time()

        for images, imu, actions, tilt_gt in train_loader:
            # uint8 → float32 on GPU in one transfer
            images  = images.to(device=device, dtype=torch.float32, non_blocking=True)
            imu     = imu.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            tilt_gt = tilt_gt.to(device, non_blocking=True)

            # GPU augmentation then normalize [0,255] → [0,1]
            images = gpu_augment(images) / 255.0

            optimizer.zero_grad()
            with autocast('cuda'):
                loss = model.compute_loss(images, imu, actions,
                                          tilt_gt=tilt_gt, lambda_tilt=lambda_tilt)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

        avg_train_loss = epoch_loss / len(train_loader)
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t_start

        # ----------------------------------------------------------------
        # Val epoch
        # ----------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, imu, actions, _ in val_loader:
                images  = images.to(device=device, dtype=torch.float32, non_blocking=True) / 255.0
                imu     = imu.to(device, non_blocking=True)
                actions = actions.to(device, non_blocking=True)
                with autocast('cuda'):
                    # val loss = pure flow matching only (comparable across runs)
                    val_loss += model.compute_loss(images, imu, actions).item()
        avg_val_loss = val_loss / len(val_loader)

        # ----------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------
        writer.add_scalar('train/loss', avg_train_loss, epoch)
        writer.add_scalar('val/loss',   avg_val_loss,   epoch)
        writer.add_scalar('train/lr',   lr_now,         epoch)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{num_epochs} | "
                  f"train={avg_train_loss:.6f} | val={avg_val_loss:.6f} | "
                  f"lr={lr_now:.2e} | {elapsed:.1f}s")

        # ----------------------------------------------------------------
        # Checkpointing
        # ----------------------------------------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save(os.path.join(save_dir, 'best_model.pt'))

        if epoch % log_cfg['save_freq'] == 0:
            model.save(os.path.join(save_dir, f'epoch_{epoch}.pt'))

    model.save(os.path.join(save_dir, 'final_model.pt'))
    writer.close()

    print(f"\nTraining complete!")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Saved to:      {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 3a v4.0: Flow Matching supervised pre-training')
    parser.add_argument('--config', type=str, default='configs/flow_policy_v4.yaml')
    parser.add_argument('--quick', action='store_true', help='5-epoch smoke test')
    parser.add_argument('--recovery-h5', type=str, default=None,
                        help='Path to recovery demo h5 to mix in (DAgger Step 1 output)')
    parser.add_argument('--recovery-episodes', type=int, default=0,
                        help='Max recovery episodes to use (0 = all)')
    parser.add_argument('--hover-episodes', type=int, default=0,
                        help='Max hover episodes to use (0 = all); use to cap RAM when mixing in recovery data')
    args = parser.parse_args()
    train(args)

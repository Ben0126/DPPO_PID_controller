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
    Sliding-window dataset from expert_demos_v4.h5 (CTBR + INDI).

    Pre-loads all data into RAM in a single pass (images ~6GB → too large;
    uses per-episode numpy arrays in a dict instead of repeated h5py opens).

    Each sample:
      images:  (T_obs*3, H, W) uint8 — stacked FPV frames
      imu:     (6,) float32          — normalised physics IMU at last obs frame
      actions: (action_dim, T_pred)  — CTBR target sequence
    """

    def __init__(self, h5_path: str, episode_indices: list,
                 T_obs: int = 2, T_pred: int = 8):
        self.T_obs  = T_obs
        self.T_pred = T_pred

        # Single-pass load: actions + imu into RAM; images into RAM as well
        # (per-episode uint8 images: 500 * 3 * 64 * 64 = 6.1 MB/ep → 950 ep ≈ 5.8 GB max)
        # Use a list-of-arrays approach to avoid one massive allocation.
        self._images  = {}
        self._actions = {}
        self._imu     = {}
        self.index    = []   # (ep_idx_in_dict, start_step)

        print(f"  Loading {len(episode_indices)} episodes from {h5_path} ...")
        # Precompute all samples during init so __getitem__ is a pure array lookup.
        # images_list:  list of (T_obs*3, H, W) uint8 arrays
        # imu_list:     list of (6,) float32 arrays
        # actions_list: list of (4, T_pred) float32 arrays
        self._img_buf  = []
        self._imu_buf  = []
        self._act_buf  = []

        with h5py.File(h5_path, 'r') as f:
            for ep_idx in episode_indices:
                key = f'episode_{ep_idx}'
                if key not in f:
                    continue
                imgs = f[key]['images'][:]      # (T, 3, H, W) uint8
                acts = f[key]['actions'][:]     # (T, 4) float32
                imus = f[key]['imu_data'][:]    # (T, 6) float32
                T = acts.shape[0]
                for start in range(T_obs - 1, T - T_pred):
                    frames = imgs[start - T_obs + 1 : start + 1]   # (T_obs, 3, H, W)
                    self._img_buf.append(
                        np.concatenate(frames, axis=0))              # (T_obs*3, H, W) uint8
                    self._imu_buf.append(imus[start])                # (6,) float32
                    self._act_buf.append(
                        acts[start + 1 : start + 1 + T_pred].T)     # (4, T_pred) float32

        # Stack into contiguous arrays for fast indexing
        self._img_arr = np.stack(self._img_buf)   # (N, T_obs*3, H, W)
        self._imu_arr = np.stack(self._imu_buf)   # (N, 6)
        self._act_arr = np.stack(self._act_buf)   # (N, 4, T_pred)
        del self._img_buf, self._imu_buf, self._act_buf

        N = len(self._img_arr)
        print(f"  {N:,} samples ready. "
              f"img={self._img_arr.nbytes/1e9:.2f}GB  "
              f"imu+act={( self._imu_arr.nbytes + self._act_arr.nbytes)/1e6:.1f}MB")

    def __len__(self):
        return len(self._img_arr)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self._img_arr[idx]),   # uint8 — convert to float32 on GPU
            torch.from_numpy(self._imu_arr[idx]),
            torch.from_numpy(self._act_arr[idx]),
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
    # Dataset split
    # ------------------------------------------------------------------
    h5_path = train_cfg['dataset_path']
    with h5py.File(h5_path, 'r') as f:
        n_ep = f.attrs['n_episodes']

    max_ep = train_cfg.get('_max_episodes', n_ep)
    all_ep = list(range(min(n_ep, max_ep)))
    n_val  = max(1, int(len(all_ep) * train_cfg['val_split']))
    n_train = len(all_ep) - n_val
    rng = np.random.default_rng(42)
    rng.shuffle(all_ep)
    train_ep = all_ep[:n_train]
    val_ep   = all_ep[n_train:]

    T_obs = vis_cfg['T_obs']
    T_pred = act_cfg['T_pred']

    train_ds = FlowDatasetV4(h5_path, train_ep, T_obs=T_obs, T_pred=T_pred)
    val_ds   = FlowDatasetV4(h5_path, val_ep,   T_obs=T_obs, T_pred=T_pred)
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

    print(f"\n{'='*60}")
    print(f"Flow Matching Policy v4.0 — Supervised Pre-training")
    print(f"Epochs:     {num_epochs}   Batch: {train_cfg['batch_size']}   LR: {train_cfg['learning_rate']}")
    print(f"Train eps:  {n_train}  Val eps: {n_val}")
    print(f"Save dir:   {save_dir}")
    print(f"{'='*60}\n")

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

        for images, imu, actions in train_loader:
            # uint8 → float32 on GPU in one transfer
            images  = images.to(device=device, dtype=torch.float32, non_blocking=True)
            imu     = imu.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            # GPU augmentation then normalize [0,255] → [0,1]
            images = gpu_augment(images) / 255.0

            optimizer.zero_grad()
            with autocast('cuda'):
                loss = model.compute_loss(images, imu, actions)
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
            for images, imu, actions in val_loader:
                images  = images.to(device=device, dtype=torch.float32, non_blocking=True) / 255.0
                imu     = imu.to(device, non_blocking=True)
                actions = actions.to(device, non_blocking=True)
                with autocast('cuda'):
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
    args = parser.parse_args()
    train(args)

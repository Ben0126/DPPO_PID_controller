"""
P1 baseline: BC-vision-only (naive lower bound).

Same VisionEncoder (6 -> 256) as the flow policies, but **no IMU, no flow**: a small
MLP head regresses the T_pred x action_dim sequence with MSE. This isolates what a
plain feed-forward vision->action map achieves under the frozen protocol, i.e. the
value added by IMU fusion + flow modelling on top of pure visual BC.

The model exposes ``predict_action(images, imu=None, n_steps=None, task_cond=None)``
returning (B, action_dim, T_pred) — the SAME contract the flow policies use — so it
scores through ``scripts.evaluate_baselines_frozen`` (frozen rollout) unchanged.

Usage:
  dppo/Scripts/python.exe -m scripts.train_bc_vision_only \
      --recovery-h5 data/expert_demos_v4_recovery.h5 \
      --recovery-episodes 500 --hover-episodes 500 \
      --tag bc_vision_only_s0 --seed 0
"""
import os
import sys
import time
import argparse
import numpy as np
import yaml
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.vision_encoder import VisionEncoder
from models.ppo_expert import RunningMeanStd
from scripts.train_flow_v5 import FlowDatasetV5, gpu_augment, set_seed


class BCVisionOnly(nn.Module):
    """VisionEncoder (no IMU) + MLP head -> action sequence. Plain regression BC."""

    def __init__(self, vision_feature_dim=256, T_obs=2, T_pred=8,
                 action_dim=4, hidden_dim=256):
        super().__init__()
        self.T_pred = T_pred
        self.action_dim = action_dim
        self.vision_encoder = VisionEncoder(in_channels=T_obs * 3,
                                            feature_dim=vision_feature_dim)
        self.head = nn.Sequential(
            nn.Linear(vision_feature_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, action_dim * T_pred),
        )

    def forward(self, images):
        feat = self.vision_encoder(images)                 # (B, F)
        out = self.head(feat)                              # (B, action_dim*T_pred)
        return out.view(-1, self.action_dim, self.T_pred)  # (B, action_dim, T_pred)

    @torch.no_grad()
    def predict_action(self, images, imu=None, n_steps=None, task_cond=None):
        """Flow-policy-compatible inference contract (imu / n_steps ignored)."""
        return self.forward(images).clamp(-1.0, 1.0)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location='cpu'):
        self.load_state_dict(torch.load(path, map_location=map_location))


def _build_sources(args, train_cfg):
    """Hover (+ optional recovery) episode split — mirrors train_flow_v5."""
    h5_path = train_cfg['dataset_path']
    with h5py.File(h5_path, 'r') as f:
        n_ep = f.attrs['n_episodes']
    max_ep = args.hover_episodes if args.hover_episodes > 0 else n_ep
    all_ep = list(range(min(n_ep, max_ep)))
    n_val = max(1, int(len(all_ep) * train_cfg['val_split']))
    rng = np.random.default_rng(42); rng.shuffle(all_ep)
    train_sources = [(h5_path, all_ep[:len(all_ep) - n_val])]
    val_sources = [(h5_path, all_ep[len(all_ep) - n_val:])]
    if args.recovery_h5:
        with h5py.File(args.recovery_h5, 'r') as f:
            n_rec = f.attrs['n_episodes']
        max_rec = args.recovery_episodes if args.recovery_episodes > 0 else n_rec
        rec_ep = list(range(min(n_rec, max_rec)))
        n_rec_val = max(1, int(len(rec_ep) * train_cfg['val_split']))
        rng2 = np.random.default_rng(99); rng2.shuffle(rec_ep)
        train_sources.append((args.recovery_h5, rec_ep[:len(rec_ep) - n_rec_val]))
        val_sources.append((args.recovery_h5, rec_ep[len(rec_ep) - n_rec_val:]))
    return train_sources, val_sources


def train(args):
    if args.seed is not None:
        set_seed(args.seed)
        print(f"[seed] {args.seed}")

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    vis_cfg, act_cfg, train_cfg = cfg['vision'], cfg['action'], cfg['training']
    if args.quick:
        train_cfg['num_epochs'] = 5
        args.hover_episodes = min(args.hover_episodes, 20) if args.hover_episodes else 20

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T_obs, T_pred = vis_cfg['T_obs'], act_cfg['T_pred']

    # obs_rms is required by FlowDatasetV5 (for the state field we ignore here)
    obs_rms = RunningMeanStd(shape=(cfg['state_predictor']['state_dim'],))
    rd = np.load(cfg['teacher_obs_rms']['path'])
    obs_rms.load_state_dict({'mean': rd['mean'], 'var': rd['var'], 'count': float(rd['count'])})

    train_sources, val_sources = _build_sources(args, train_cfg)
    train_ds = FlowDatasetV5(train_sources, obs_rms, T_obs=T_obs, T_pred=T_pred)
    val_ds = FlowDatasetV5(val_sources, obs_rms, T_obs=T_obs, T_pred=T_pred)
    print(f"Train {len(train_ds):,} | Val {len(val_ds):,}")

    nw = train_cfg['num_workers']
    train_loader = DataLoader(train_ds, batch_size=train_cfg['batch_size'], shuffle=True,
                              num_workers=nw, pin_memory=(nw > 0),
                              persistent_workers=(nw > 0), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg['batch_size'], shuffle=False,
                            num_workers=nw, pin_memory=(nw > 0), persistent_workers=(nw > 0))

    model = BCVisionOnly(vision_feature_dim=vis_cfg['feature_dim'], T_obs=T_obs,
                         T_pred=T_pred, action_dim=act_cfg['action_dim']).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'],
                                  weight_decay=train_cfg['weight_decay'])
    num_epochs = train_cfg['num_epochs']
    warmup = len(train_loader) * train_cfg['warmup_epochs']
    total = len(train_loader) * num_epochs

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    run_name = args.tag if args.tag else datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join('./checkpoints/bc_vision_only/', run_name)
    log_dir = os.path.join('./logs/bc_vision_only/', run_name)
    os.makedirs(save_dir, exist_ok=True); os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    scaler = GradScaler('cuda')
    best_val = float('inf')
    for epoch in range(1, num_epochs + 1):
        model.train(); ep_loss = 0.0; t0 = time.time()
        for images, _imu, actions, _tilt, _state, _task in train_loader:
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            images = gpu_augment(images) / 255.0
            optimizer.zero_grad()
            with autocast('cuda'):
                pred = model(images)
                loss = F.mse_loss(pred, actions)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])
            scaler.step(optimizer); scaler.update(); scheduler.step()
            ep_loss += loss.item()
        ep_loss /= len(train_loader)

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for images, _imu, actions, _tilt, _state, _task in val_loader:
                images = images.to(device, dtype=torch.float32, non_blocking=True) / 255.0
                actions = actions.to(device, non_blocking=True)
                with autocast('cuda'):
                    val_loss += F.mse_loss(model(images), actions).item()
        val_loss /= len(val_loader)

        writer.add_scalar('train/mse', ep_loss, epoch)
        writer.add_scalar('val/mse', val_loss, epoch)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{num_epochs} | train_mse={ep_loss:.5f} "
                  f"val_mse={val_loss:.5f} | {time.time()-t0:.1f}s")
        if val_loss < best_val:
            best_val = val_loss
            model.save(os.path.join(save_dir, 'best_model.pt'))

    model.save(os.path.join(save_dir, 'final_model.pt'))
    writer.close()
    print(f"\nDone. best val_mse={best_val:.6f}  saved to {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='P1 baseline: BC-vision-only')
    parser.add_argument('--config', default='configs/flow_policy_v5.yaml')
    parser.add_argument('--recovery-h5', default=None)
    parser.add_argument('--recovery-episodes', type=int, default=0)
    parser.add_argument('--hover-episodes', type=int, default=0)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--tag', type=str, default=None)
    args = parser.parse_args()
    train(args)

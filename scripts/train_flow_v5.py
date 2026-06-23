"""
Phase 3a v5.0: Flow Matching Policy Supervised Pre-training (cross-attn + state aux).

Differences from train_flow_v4.py:
  * Uses FlowMatchingPolicyV5 (IMU->Vision cross-attention, state predictor head)
  * Dataset additionally returns normalised 15D state per sample
  * compute_loss is called with (states_gt, lambda_state) so the vision encoder
    is grounded on physics from the very start of training
  * --transfer-from-h4 partial-loads the H4 BC checkpoint (conv / fc / imu /
    flow_net / tilt_head); cross_attn and state_predictor stay random

Usage:
    dppo/Scripts/python.exe -m scripts.train_flow_v5 \
        --config configs/flow_policy_v5.yaml \
        --transfer-from-h4 checkpoints/flow_policy_v4/20260514_175219/best_model.pt \
        --hover-only --hover-episodes 1000
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
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.flow_policy_v5 import FlowMatchingPolicyV5
from models.ppo_expert import RunningMeanStd


# ---------------------------------------------------------------------------
# Dataset — same as v4 but additionally returns normalised 15D state.
# ---------------------------------------------------------------------------

class FlowDatasetV5(Dataset):
    """
    Sliding-window dataset with normalised state for auxiliary supervision.

    Each sample:
      images:  (T_obs*3, H, W) uint8
      imu:     (6,) float32
      actions: (action_dim, T_pred) float32
      tilt:    () float32
      state:   (15,) float32 — normalised by PPO expert obs_rms
      task:    (2,) float32 — [is_hover, is_recovery]
    """

    def __init__(self, sources: list, obs_rms: RunningMeanStd,
                 T_obs: int = 2, T_pred: int = 8,
                 hover_only: bool = False,
                 range_cue_mode: str = 'none', cue_scale: float = 3.0):
        self.T_obs  = T_obs
        self.T_pred = T_pred
        # Phase 3b range-cue positive control. The cue is the metric position
        # error the 64x64 FPV cannot encode past ~2 m (docs/experiment_report_
        # image_distance_info.md). It is FOLDED into the existing task-cond slot,
        # so the model concats it into global_cond unchanged (task_dim widens).
        #   'none'   -> cue_dim 0 (identical to the D0E1 frontier recipe)
        #   'scalar' -> cue_dim 1: ||pos_err_body|| / cue_scale
        #   'pos3d'  -> cue_dim 3: pos_err_body / cue_scale  (range + direction)
        # Clean cue is stored here; sensor noise (noised arms) is added at use.
        assert range_cue_mode in ('none', 'scalar', 'pos3d')
        self.range_cue_mode = range_cue_mode
        self.cue_scale = cue_scale
        self.cue_dim = {'none': 0, 'scalar': 1, 'pos3d': 3}[range_cue_mode]

        img_buf, imu_buf, act_buf, tilt_buf, state_buf, task_buf = [], [], [], [], [], []
        skipped_count = 0

        for h5_path, episode_indices in sources:
            print(f"  Loading {len(episode_indices)} episodes from {h5_path} ...")
            with h5py.File(h5_path, 'r') as f:
                for ep_idx in episode_indices:
                    key = f'episode_{ep_idx}'
                    if key not in f:
                        continue
                    ep_attrs = f[key].attrs
                    ep_type  = ep_attrs.get('episode_type', 'hover')
                    is_rec = ('init_tilt_deg' in ep_attrs or ep_type == 'recovery')
                    if hover_only and is_rec:
                        skipped_count += 1
                        continue
                    task_label = np.array([0.0, 1.0], dtype=np.float32) if is_rec else np.array([1.0, 0.0], dtype=np.float32)

                    imgs       = f[key]['images'][:]
                    acts       = f[key]['actions'][:]
                    imus       = f[key]['imu_data'][:]
                    states_ep  = f[key]['states'][:]
                    T = acts.shape[0]
                    for start in range(T_obs - 1, T - T_pred):
                        frames = imgs[start - T_obs + 1 : start + 1]
                        img_buf.append(np.concatenate(frames, axis=0))
                        imu_buf.append(imus[start])
                        act_buf.append(acts[start + 1 : start + 1 + T_pred].T)
                        s = states_ep[start]
                        r2z = float(s[3]*s[7] - s[4]*s[6])
                        tilt_buf.append(np.float32(np.arccos(np.clip(r2z, -1.0, 1.0))))
                        # Normalise full 15D state with PPO obs_rms
                        state_buf.append(obs_rms.normalize(s).astype(np.float32))
                        # Range cue from RAW (metres) body-frame pos error s[0:3].
                        if self.cue_dim == 0:
                            task_buf.append(task_label)
                        else:
                            if self.range_cue_mode == 'scalar':
                                cue = np.array([np.linalg.norm(s[0:3])], dtype=np.float32)
                            else:  # pos3d
                                cue = s[0:3].astype(np.float32)
                            cue = cue / self.cue_scale
                            task_buf.append(np.concatenate([task_label, cue]))

        self._img_arr   = np.stack(img_buf)
        self._imu_arr   = np.stack(imu_buf)
        self._act_arr   = np.stack(act_buf)
        self._tilt_arr  = np.array(tilt_buf, dtype=np.float32)
        self._state_arr = np.stack(state_buf)
        self._task_arr  = np.stack(task_buf)

        N = len(self._img_arr)
        if skipped_count > 0:
            print(f"  [hover_only] skipped {skipped_count} non-hover episodes")
        print(f"  Total: {N:,} samples  "
              f"img={self._img_arr.nbytes/1e9:.2f}GB  "
              f"imu+act+state={(self._imu_arr.nbytes + self._act_arr.nbytes + self._state_arr.nbytes + self._task_arr.nbytes)/1e6:.1f}MB")

    def __len__(self):
        return len(self._img_arr)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self._img_arr[idx]),
            torch.from_numpy(self._imu_arr[idx]),
            torch.from_numpy(self._act_arr[idx]),
            torch.tensor(self._tilt_arr[idx]),
            torch.from_numpy(self._state_arr[idx]),
            torch.from_numpy(self._task_arr[idx]),
        )


# ---------------------------------------------------------------------------
# GPU augmentation
# ---------------------------------------------------------------------------

def gpu_augment(images: torch.Tensor) -> torch.Tensor:
    B = images.shape[0]
    bright = torch.empty(B, 1, 1, 1, device=images.device).uniform_(0.7, 1.3)
    images = images * bright
    contrast = torch.empty(B, 1, 1, 1, device=images.device).uniform_(0.8, 1.2)
    mean = images.mean(dim=[2, 3], keepdim=True)
    images = (images - mean) * contrast + mean
    return images.clamp(0, 255)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Seed python / numpy / torch for reproducible ablation runs (P2)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    if args.seed is not None:
        set_seed(args.seed)
        print(f"[seed] python/numpy/torch seeded with {args.seed}")

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
    xattn_cfg  = cfg.get('cross_attn', {})
    sp_cfg     = cfg.get('state_predictor', {})
    rms_cfg    = cfg['teacher_obs_rms']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load PPO obs_rms for state normalisation
    # ------------------------------------------------------------------
    obs_rms = RunningMeanStd(shape=(sp_cfg.get('state_dim', 15),))
    rms_data = np.load(rms_cfg['path'])
    obs_rms.load_state_dict({
        'mean':  rms_data['mean'],
        'var':   rms_data['var'],
        'count': float(rms_data['count']),
    })
    print(f"Loaded obs_rms from {rms_cfg['path']}  "
          f"(mean range [{obs_rms.mean.min():.3f}, {obs_rms.mean.max():.3f}])")

    # ------------------------------------------------------------------
    # Dataset split
    # ------------------------------------------------------------------
    h5_path = train_cfg['dataset_path']
    with h5py.File(h5_path, 'r') as f:
        n_ep = f.attrs['n_episodes']

    max_ep = args.hover_episodes if args.hover_episodes > 0 \
        else train_cfg.get('_max_episodes', n_ep)
    all_ep = list(range(min(n_ep, max_ep)))
    n_val   = max(1, int(len(all_ep) * train_cfg['val_split']))
    n_train = len(all_ep) - n_val
    rng = np.random.default_rng(42)
    rng.shuffle(all_ep)
    hover_train_ep = all_ep[:n_train]
    hover_val_ep   = all_ep[n_train:]

    T_obs  = vis_cfg['T_obs']
    T_pred = act_cfg['T_pred']

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
        print(f"Recovery mix-in: {n_rec_train} train + {n_rec_val} val "
              f"episodes from {args.recovery_h5}")

    train_ds = FlowDatasetV5(train_sources, obs_rms,
                             T_obs=T_obs, T_pred=T_pred,
                             hover_only=args.hover_only,
                             range_cue_mode=args.range_cue, cue_scale=args.cue_scale)
    val_ds   = FlowDatasetV5(val_sources, obs_rms,
                             T_obs=T_obs, T_pred=T_pred,
                             hover_only=args.hover_only,
                             range_cue_mode=args.range_cue, cue_scale=args.cue_scale)
    print(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")
    cue_dim = train_ds.cue_dim
    if cue_dim:
        print(f"[range-cue] mode={args.range_cue} cue_dim={cue_dim} scale={args.cue_scale} "
              f"noise(m)={args.cue_noise}  -> task_dim={2 + cue_dim}")

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
    # Model (with optional H4 weight transfer)
    # ------------------------------------------------------------------
    model = FlowMatchingPolicyV5(
        vision_feature_dim     = vis_cfg['feature_dim'],
        imu_feature_dim        = imu_cfg['feature_dim'],
        time_embed_dim         = unet_cfg['time_embed_dim'],
        down_dims              = tuple(unet_cfg['down_dims']),
        T_obs                  = T_obs,
        T_pred                 = T_pred,
        action_dim             = act_cfg['action_dim'],
        n_inference_steps      = flow_cfg['n_inference_steps'],
        t_embed_scale          = flow_cfg['t_embed_scale'],
        cross_attn_heads       = xattn_cfg.get('n_heads', 8),
        state_predictor_hidden = sp_cfg.get('hidden_dim', 256),
        state_dim              = sp_cfg.get('state_dim', 15),
        task_dim               = 2 + cue_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    if args.transfer_from_h4:
        model.transfer_from_h4(args.transfer_from_h4)
    elif args.from_scratch:
        print("[--from-scratch] random initialisation, no H4 transfer")

    if args.pretrained:
        model.load(args.pretrained)
        print(f"[Stage D] Loaded pretrained weights: {args.pretrained}")

    # ------------------------------------------------------------------
    # Optional: freeze vision_encoder (Stage D — protect OOB-pretrained features)
    # ------------------------------------------------------------------
    if args.freeze_vision:
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
        frozen_n = sum(p.numel() for p in model.vision_encoder.parameters())
        trainable_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Stage D] vision_encoder FROZEN ({frozen_n:,} params)")
        print(f"[Stage D] Trainable: {trainable_n:,} params (flow_net + cross_attn + state_predictor + imu_encoder)")

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
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
    # Use --tag for deterministic run dirs (P2 ablation manifest mapping); else timestamp.
    run_name = args.tag if args.tag else datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir  = os.path.join(log_cfg['tensorboard_log'], run_name)
    save_dir = os.path.join(log_cfg['save_path'], run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    recovery_tag  = f" + recovery({args.recovery_h5})" if args.recovery_h5 else ""
    freeze_tag    = " [FREEZE vision_encoder]" if args.freeze_vision else ""
    print(f"\n{'='*60}")
    print(f"Flow Matching Policy v5.0 — Supervised Pre-training{recovery_tag}{freeze_tag}")
    print(f"Epochs:     {num_epochs}   Batch: {train_cfg['batch_size']}   "
          f"LR: {train_cfg['learning_rate']}")
    print(f"Train samples: {len(train_ds):,}  Val samples: {len(val_ds):,}")
    print(f"Lambda state: {train_cfg.get('lambda_state', 0.1)}   "
          f"Lambda tilt: {train_cfg.get('lambda_tilt', 0.1)}")
    print(f"Save dir:   {save_dir}")
    print(f"{'='*60}\n")

    lambda_state = train_cfg.get('lambda_state', sp_cfg.get('lambda_state', 0.1))
    lambda_tilt  = train_cfg.get('lambda_tilt', 0.1)
    best_val_loss = float('inf')
    global_step   = 0
    scaler = GradScaler('cuda')

    for epoch in range(1, num_epochs + 1):
        # ------------- train -------------
        model.train()
        ep_flow, ep_state, ep_tilt, ep_disp, ep_total = 0.0, 0.0, 0.0, 0.0, 0.0
        t_start = time.time()

        for images, imu, actions, tilt_gt, state_gt, task_cond in train_loader:
            images   = images.to(device=device, dtype=torch.float32, non_blocking=True)
            imu      = imu.to(device, non_blocking=True)
            actions  = actions.to(device, non_blocking=True)
            tilt_gt  = tilt_gt.to(device, non_blocking=True)
            state_gt = state_gt.to(device, non_blocking=True)
            task_cond = task_cond.to(device, non_blocking=True)
            # Noised range-cue arms: add sensor noise to the cue columns (trailing
            # cue_dim) each minibatch -> resampled augmentation. sigma is in metres,
            # cue is stored as metres/cue_scale, so divide sigma by cue_scale.
            if cue_dim and args.cue_noise > 0:
                noise = torch.randn(task_cond.shape[0], cue_dim, device=device) \
                    * (args.cue_noise / args.cue_scale)
                task_cond[:, 2:] = task_cond[:, 2:] + noise

            images = gpu_augment(images) / 255.0

            optimizer.zero_grad()
            with autocast('cuda'):
                total_loss, comp = model.compute_loss(
                    images, imu, actions,
                    states_gt=state_gt, lambda_state=lambda_state,
                    tilt_gt=tilt_gt,    lambda_tilt=lambda_tilt,
                    return_components=True,
                    task_cond=task_cond,
                    lambda_dispersive=args.lambda_disp,
                    dispersive_target=args.dispersive_target,
                    dispersive_tau=args.dispersive_tau,
                )
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ep_total += total_loss.item()
            ep_flow  += comp['flow_loss'].item()
            ep_state += comp.get('state_loss', torch.tensor(0.0)).item()
            ep_tilt  += comp.get('tilt_loss',  torch.tensor(0.0)).item()
            ep_disp  += comp.get('loss_dispersive', torch.tensor(0.0)).item()
            global_step += 1

        nb = len(train_loader)
        avg_total = ep_total / nb
        avg_flow  = ep_flow  / nb
        avg_state = ep_state / nb
        avg_tilt  = ep_tilt  / nb
        avg_disp  = ep_disp  / nb
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t_start

        # ------------- val (pure flow loss for comparability) -------------
        model.eval()
        val_flow = 0.0
        val_state = 0.0
        with torch.no_grad():
            for images, imu, actions, _, state_gt, task_cond in val_loader:
                images   = images.to(device=device, dtype=torch.float32,
                                     non_blocking=True) / 255.0
                imu      = imu.to(device, non_blocking=True)
                actions  = actions.to(device, non_blocking=True)
                state_gt = state_gt.to(device, non_blocking=True)
                task_cond = task_cond.to(device, non_blocking=True)
                with autocast('cuda'):
                    _, comp = model.compute_loss(
                        images, imu, actions,
                        states_gt=state_gt, lambda_state=lambda_state,
                        return_components=True,
                        task_cond=task_cond,
                        lambda_dispersive=args.lambda_disp,
                        dispersive_target=args.dispersive_target,
                        dispersive_tau=args.dispersive_tau,
                    )
                val_flow  += comp['flow_loss'].item()
                val_state += comp['state_loss'].item()
        val_flow  /= len(val_loader)
        val_state /= len(val_loader)

        # ------------- logging -------------
        writer.add_scalar('train/total_loss', avg_total, epoch)
        writer.add_scalar('train/flow_loss',  avg_flow,  epoch)
        writer.add_scalar('train/state_loss', avg_state, epoch)
        writer.add_scalar('train/tilt_loss',  avg_tilt,  epoch)
        writer.add_scalar('train/loss_dispersive', avg_disp, epoch)
        writer.add_scalar('val/flow_loss',    val_flow,  epoch)
        writer.add_scalar('val/state_loss',   val_state, epoch)
        writer.add_scalar('train/lr',         lr_now,    epoch)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{num_epochs} | "
                  f"flow={avg_flow:.5f} state={avg_state:.5f} tilt={avg_tilt:.5f} disp={avg_disp:.5f} "
                  f"| val_flow={val_flow:.5f} val_state={val_state:.5f} "
                  f"| lr={lr_now:.2e} | {elapsed:.1f}s")

        # ------------- checkpoint -------------
        if val_flow < best_val_loss:
            best_val_loss = val_flow
            model.save(os.path.join(save_dir, 'best_model.pt'))
        if epoch % log_cfg['save_freq'] == 0:
            model.save(os.path.join(save_dir, f'epoch_{epoch}.pt'))

    model.save(os.path.join(save_dir, 'final_model.pt'))
    writer.close()

    print(f"\nTraining complete!")
    print(f"  Best val flow loss: {best_val_loss:.6f}")
    print(f"  Saved to:           {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Phase 3a v5.0: Flow Matching BC with cross-attn + state aux')
    parser.add_argument('--config', type=str, default='configs/flow_policy_v5.yaml')
    parser.add_argument('--quick', action='store_true', help='5-epoch smoke test')
    parser.add_argument('--recovery-h5', type=str, default=None)
    parser.add_argument('--recovery-episodes', type=int, default=0)
    parser.add_argument('--hover-episodes', type=int, default=0)
    parser.add_argument('--hover-only', action='store_true')
    parser.add_argument('--transfer-from-h4', type=str, default=None,
                        help='Partial-load weights from a v4 (H4) checkpoint')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Skip H4 transfer; use random initialisation '
                             '(for ablation comparison)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Full checkpoint to load (e.g. OOB pretrain) before training')
    parser.add_argument('--freeze-vision', action='store_true',
                        help='Freeze vision_encoder weights; train only flow_net + cross_attn '
                             '(Stage D: re-align action head to OOB-pretrained features)')
    parser.add_argument('--range-cue', type=str, default='none',
                        choices=['none', 'scalar', 'pos3d'],
                        help='Phase 3b positive control: fold metric pos-error into '
                             'task-cond. none=D0E1 frontier; scalar=||pos_err||; pos3d=pos_err')
    parser.add_argument('--cue-noise', type=float, default=0.0,
                        help='sensor noise std (metres) added to the range cue (noised arms)')
    parser.add_argument('--cue-scale', type=float, default=3.0,
                        help='divide the metric cue by this so it is ~O(1)')
    parser.add_argument('--lambda-disp', type=float, default=0.05,
                        help='Dispersive loss weight (legacy default 0.05; faithful re-run '
                             'uses 0.5 per [13]/[14]; set 0.0 to disable — Dispersive OFF arm)')
    parser.add_argument('--dispersive-target', choices=['vis_pooled', 'flow_mid'],
                        default='vis_pooled',
                        help="Where to apply dispersive loss. 'vis_pooled' = legacy off-path "
                             "placement (pre-2026-06-22); 'flow_mid' = faithful to [13]/[14], "
                             "InfoNCE-L2 on the flow_net mid-block intermediate representation.")
    parser.add_argument('--dispersive-tau', type=float, default=0.5,
                        help='InfoNCE temperature for the flow_mid faithful dispersive loss (default 0.5)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed python/numpy/torch for reproducible P2 ablation runs')
    parser.add_argument('--tag', type=str, default=None,
                        help='Run name for log/checkpoint dirs (e.g. p2_D1E1_s0); '
                             'replaces the default timestamp so the sweep manifest can map runs')
    args = parser.parse_args()
    train(args)

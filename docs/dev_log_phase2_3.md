# DPPO PID Controller — Development Log Index (Phase 2–3c)

> Continuation of [dev_log.md](dev_log.md) (Phase 1 documented there)
> Phase 2 start: 2026-04-01 | Phase 3c v3.2→v3.3 renamed: 2026-04-11
> Predecessor: PPO Expert Run 6 (`checkpoints/ppo_expert/20260401_103107/`)

---

## Log Files

| 檔案 | 涵蓋範圍 | 大小 |
|------|----------|------|
| [dev_log_phase2.md](dev_log_phase2.md) | Phase 2：Expert demo 收集、HDF5 結構、pre-collection bug fixes | ~70 行 |
| [dev_log_phase3a.md](dev_log_phase3a.md) | Phase 3a：監督預訓練、Bug Audit × 2、Domain Randomization、Re-run 1 & 2 | ~870 行 |
| [dev_log_phase3b.md](dev_log_phase3b.md) | Phase 3b：RHC baseline eval、DPPO Runs 1–3、Key Lessons、Results Summary | ~280 行 |
| [dev_log_phase3c_v31.md](dev_log_phase3c_v31.md) | Phase 3c v3.1：IMU Late Fusion + FCN Depth 架構、DPPO v3.1 Runs 1–2 post-mortem | ~400 行 |
| [dev_log_phase3c_v33.md](dev_log_phase3c_v33.md) | Phase 3c v3.2/v3.3：DPPO Run 4、物理 IMU 實作、supervised eval、DPPO Run 1（v3.2→v3.3 重命名） | ~2170 行 |

---

## Results Snapshot

| Run | Model | RMSE | Crashes | 備註 |
|-----|-------|------|---------|------|
| Phase 1 | PPO Expert Run 6 | **0.069m** | **0/50** | 黃金標準 |
| Phase 3a | Supervised DP (original) | 0.286m | 50/50 | Covariate shift — 預期 |
| Phase 3b Run 1 | DPPO (β=1.0) | 0.378m | 50/50 | Collapse @ update ~100 |
| Phase 3b Run 2 | DPPO (β=0.1, u11) | **0.168m** | 50/50 | 歷史最佳 RMSE |
| Phase 3b Run 3 | DPPO (β=0.15) | 0.488m | 50/50 | β 稍高即退化 |
| Phase 3c v3.1 Run 1 | DPPO v3.1 | 0.518m | 50/50 | Value net lag |
| Phase 3c v3.1 Run 2 | DPPO v3.1 | 0.466m | 50/50 | finite-diff IMU 不穩定 |
| Phase 3b Run 4 | DPPO (原始架構改良) | TBD | TBD | β=0.05, warm-up 50 |
| Phase 3c v3.2 Run 1 | DPPO v3.2 | — | — | 已終止 u25，IMU 未歸一化（v3.2，棄用） |

---

## Key Checkpoints

| Artifact | Path |
|----------|------|
| PPO Expert Run 6 | `checkpoints/ppo_expert/20260401_103107/` |
| Supervised DP (original) | `checkpoints/diffusion_policy/20260402_032701/best_model.pt` |
| Supervised DP Re-run 2 | `checkpoints/diffusion_policy/20260405_044808/best_model.pt` |
| DPPO Run 2 best (歷史最佳) | `checkpoints/diffusion_policy/dppo_20260404_044552/best_dppo_model.pt` |
| v3.1 supervised | `checkpoints/diffusion_policy/v31_20260406_185128/best_model.pt` |
| v3.2 supervised（未歸一化，棄用） | `checkpoints/diffusion_policy/v32_20260410_120042/best_model.pt` |
| v3.2 DPPO Run 1 (已終止 u25) | `checkpoints/diffusion_policy/dppo_v32_20260411_114141/` |

---

## Active Run

**`dppo_v32_20260411_114141`** — Phase 3c v3.2 DPPO Run 1（已終止，棄用）
- Started: 2026-04-11 11:41
- Log: `logs/train_dppo_v32_20260411_114133.log` (空，未寫入)
- 詳細配置與監控方式見 [dev_log_phase3c_v33.md §17](dev_log_phase3c_v33.md)

---

## Known Failure Modes (跨版本通用)

| 問題 | 症狀 | 解法 |
|------|------|------|
| Covariate shift | 監督模型 100% crash | D²PPO closed-loop 訓練 (必須) |
| Policy collapse | per-step reward 正→負 | β 減小 + LR 降低 |
| Value net lag | VLoss > 5 至 update ~150 | value warmup (凍結 policy 50 updates) |
| IMU 未歸一化（v3.2） | v3.2 supervised RMSE 1.985m | **已修復 → v3.3** gyro/2.0, sf 中心化後/5.0 |
| 推論速度 14Hz | DDIM 10步 = 74ms, 無法達 50Hz | Phase 3d OneDP 單步蒸餾 |

---

## §18 — Phase 3b Run 5：P5 假設驗證（2026-04-11）

**目標：** 驗證 P5 假設（DR-aug 特徵模糊是 Run 4 RMSE 0.409m 的根因）。
**策略：** 路線 B — 原始 pretrained（無 DR-aug）+ Run 4 改良配置，無需等待 v3.3 pipeline。

### Hyperparameter Diff（vs Run 4）

| 參數 | Run 4 | Run 5 | 理由 |
|------|-------|-------|------|
| pretrained | `20260405_044808`（DR-aug Re-run 2） | `20260402_032701`（原始，無 DR-aug） | 驗證 P5：特徵更尖銳 → DPPO 梯度更有方向性 |
| advantage_beta | 0.05 | 0.05（不變） | — |
| value_warmup_updates | 50 | 50（不變） | — |
| vloss_best_threshold | 500 | 500（不變） | — |
| script | train_dppo.py | train_dppo.py（不變） | 無 IMU，原始架構 |

### 預期行為

- VLoss 應在 u50 前降至 < 500（Run 4 u50 時 VLoss = 17，歷史最佳）
- Reward 應維持 0.47–0.56 穩定範圍
- Best ckpt RMSE 目標：**< 0.168m**（超越 Run 2）
- 若成立：原始 pretrained 的尖銳特徵空間 + Run 4 配置 = 最優組合
- 若不成立（RMSE ≈ 0.4m+）：DR-aug 不是主因，v3.2 IMU 歸一化修復是主要突破口

### 判斷點

u155 附近的 best ckpt 評估結果：
- **成功：** RMSE < 0.168m → P5 假設成立
- **中性：** RMSE 0.168–0.268m → 部分改善，繼續 v3.3 路線
- **失敗：** RMSE > 0.268m → P5 假設否定，優先 v3.3 IMU 歸一化（已修復）

**Status:** Starting — 2026-04-11

---
<!-- auto-log 2026-04-11 17:41:39 bash -->
### [Auto-Log] 2026-04-11 17:41:39 — DPPO Training — Started

**Command:** `cd c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller && source dppo/Scripts/activate && nohup python -m scripts.train_dppo \
    --pretrained checkpoints/diffusion_policy/20260402_032701/best_model.pt \
    --total-updates 500 \
    > logs/train_dppo_run5_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 17:42:00 bash -->
### [Auto-Log] 2026-04-11 17:42:00 — DPPO Training — Started

**Command:** `ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/ | grep "train_dppo_run5" | sort -r | head -2`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 17:42:12 bash -->
### [Auto-Log] 2026-04-11 17:42:12 — DPPO Training — Started

**Command:** `sleep 10 && tail -20 c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_run5_20260411_174138.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 19:12:19 edit -->
### [Auto-Log] 2026-04-11 19:12:19 — Script Fix

**File:** `scripts\auto_log.py`

**Before:**
```python
LOG_FILES = {
    "phase3c_v32": REPO / "docs" / "dev_log_phase3c_v33.md",
    "phase3c_v31": REPO / "docs" / "dev_log_phase3c_v31.md",
```

**After:**
```python
LOG_FILES = {
    "phase3c_v33": REPO / "docs" / "dev_log_phase3c_v33.md",
    "phase3c_v31": REPO / "docs" / "dev_log_phase3c_v31.md",
```

---
<!-- auto-log 2026-04-11 19:12:23 edit -->
### [Auto-Log] 2026-04-11 19:12:23 — Script Fix

**File:** `scripts\auto_log.py`

**Before:**
```python
ROUTING_RULES = [
    (r"v33",                          "phase3c_v33"), # reserved for future use
    (r"v32",                          "phase3c_v32"),
    (r"v31",                          "phase3c_v31"),
```

**After:**
```python
ROUTING_RULES = [
    (r"v33",                          "phase3c_v33"),
    (r"v31",                          "phase3c_v31"),
```

---
<!-- auto-log 2026-04-11 19:19:44 edit -->
### [Auto-Log] 2026-04-11 19:19:44 — Script Fix

**File:** `scripts\auto_log.py`

**Before:**
```python
LOG_FILES = {
    "phase3c_v33": REPO / "docs" / "dev_log_phase3c_v33.md",
    "phase3c_v31": REPO / "docs" / "dev_log_phase3c_v31.md",
```

**After:**
```python
LOG_FILES = {
    "phase3c_v33": REPO / "docs" / "dev_log_phase3c_v33.md",
    "phase3c_v32": REPO / "docs" / "dev_log_phase3c_v32.md",
    "phase3c_v31": REPO / "docs" / "dev_log_phase3c_v31.md",
```

---
<!-- auto-log 2026-04-11 19:20:38 edit -->
### [Auto-Log] 2026-04-11 19:20:38 — Script Fix

**File:** `scripts\auto_log.py`

**Before:**
```python
ROUTING_RULES = [
    (r"v33",                          "phase3c_v33"),
    (r"v31",                          "phase3c_v31"),
```

**After:**
```python
ROUTING_RULES = [
    (r"v33",                          "phase3c_v33"),
    (r"v32",                          "phase3c_v32"),
    (r"v31",                          "phase3c_v31"),
```

---
<!-- auto-log 2026-04-19 12:59:11 write -->
### [Auto-Log] 2026-04-19 12:59:11 — New File: Config / HP Change

**File:** `configs\quadrotor_v4.yaml`

**Content:**
```yaml
# Vision-DPPO v4.0 Configuration
# CTBR action space + INDI inner-loop rate controller

# Quadrotor Physical Parameters (unchanged from v3.3)
quadrotor:
  mass: 0.5
  arm_length: 0.17
  inertia: [0.0023, 0.0023, 0.004]
  gravity: 9.81
  motor_max_thrust: 4.0           # N per motor (total 16N max collective)
  motor_time_constant: 0.02       # s
  drag_coeff: 0.01
  torque_coeff: 0.016

# Time Step Configuration (unchanged)
timing:
  dt_inner: 0.005                 # 200 Hz physics integration
  dt_outer: 0.02                  # 50 Hz outer policy decision
  n_inner_steps: 4

# CTBR Action Space
# Action: [F_c_norm, wx_norm, wy_norm, wz_norm] in [-1, 1]
# F_c_norm -> collective thrust [0, 16N]
# w_norm   -> body rates [-omega_max, omega_max] per axis
ctbr:
  F_c_max: 16.0                  
... [truncated — 2348 chars total]
```

---
<!-- auto-log 2026-04-19 13:28:51 write -->
### [Auto-Log] 2026-04-19 13:28:51 — New File: Script Fix

**File:** `scripts\validate_srt_hypothesis.py`

**Content:**
```python
"""
SRT Root-Cause Validation Experiment
=====================================
Two tests in one script:

Test A -- INDI Angular Stability  (--test indi_hover)
    Runs QuadrotorEnvV4 with hardcoded hover CTBR (F_c=hover, omega=0),
    external disturbances DISABLED to isolate INDI.
    Verifies that tilt stays < 5 deg and angular rates < 0.5 rad/s.
    Note: translational drift is expected (no outer position loop).
    Phase 0 gate: required before any PPO training.

Test B -- BC-Vision-MLP RHC  (--test bc_mlp)
    Trains a single-step BC policy (VisionEncoder + MLP -> SRT action) on
    existing expert_demos_v33.h5 for N epochs.  Evaluates the trained policy
    in RHC closed-loop on QuadrotorEnv (v3.3 SRT environment).

    Hypothesis check:
      crash_rate ~= 50/50 -> covariate shift i
... [truncated — 16737 chars total]
```

---
<!-- auto-log 2026-04-19 18:08:56 edit -->
### [Auto-Log] 2026-04-19 18:08:56 — Model Fix

**File:** `models\flow_policy_v4.py`

**Before:**
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
```

**After:**
```python
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---
<!-- auto-log 2026-04-19 18:09:02 edit -->
### [Auto-Log] 2026-04-19 18:09:02 — Model Fix

**File:** `models\flow_policy_v4.py`

**Before:**
```python
n_steps: int | None = None,
```

**After:**
```python
n_steps: Optional[int] = None,
```

---
<!-- auto-log 2026-04-19 18:32:41 edit -->
### [Auto-Log] 2026-04-19 18:32:41 — Script Fix

**File:** `scripts\train_flow_v4.py`

**Before:**
```python
if args.quick:
        cfg['training']['num_epochs'] = 5
        cfg['training']['batch_size'] = 64
        print("[Quick Test Mode]")
```

**After:**
```python
if args.quick:
        cfg['training']['num_epochs'] = 5
        cfg['training']['batch_size'] = 64
        cfg['training']['num_workers'] = 0
        cfg['training']['_max_episodes'] = 20
        print("[Quick Test Mode] 5 epochs, 20 episodes, batch=64")
```

---
<!-- auto-log 2026-04-19 18:32:46 edit -->
### [Auto-Log] 2026-04-19 18:32:46 — Script Fix

**File:** `scripts\train_flow_v4.py`

**Before:**
```python
all_ep = list(range(n_ep))
    n_val  = max(1, int(n_ep * train_cfg['val_split']))
    n_train = n_ep - n_val
```

**After:**
```python
max_ep = train_cfg.get('_max_episodes', n_ep)
    all_ep = list(range(min(n_ep, max_ep)))
    n_val  = max(1, int(len(all_ep) * train_cfg['val_split']))
    n_train = len(all_ep) - n_val
```

---
<!-- auto-log 2026-04-19 18:33:10 edit -->
### [Auto-Log] 2026-04-19 18:33:10 — Script Fix

**File:** `scripts\train_flow_v4.py`

**Before:**
```python
train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=train_cfg['num_workers'],
        pin_memory=True,
        persistent_workers=True,
    )
```

**After:**
```python
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
```

---
<!-- auto-log 2026-04-19 18:51:42 edit -->
### [Auto-Log] 2026-04-19 18:51:42 — Config / HP Change

**File:** `configs\flow_policy_v4.yaml`

**Before:**
```yaml
num_workers: 4
```

**After:**
```yaml
num_workers: 0                  # data already in RAM; workers cause 5.8GB spawn-copy overhead on Windows
```

---
<!-- auto-log 2026-04-19 18:52:04 edit -->
### [Auto-Log] 2026-04-19 18:52:04 — Script Fix

**File:** `scripts\train_flow_v4.py`

**Before:**
```python
print(f"  Loading {len(episode_indices)} episodes from {h5_path} ...")
        with h5py.File(h5_path, 'r') as f:
            for ep_idx in episode_indices:
                key = f'episode_{ep_idx}'
                if key not in f:
                    continue
                imgs = f[key]['images'][:]      # (T, 3, H, W) uint8
                acts = f[key]['actions'][:]     # (T, 4) float32
                imus = f[key]['imu_data'][:]    # (T, 6) float32
                T = acts.shape[0]
                self._images[ep_idx]  = imgs
                self._actions[ep_idx] = acts
                self._imu[ep_idx]     = imus
                for start in range(T_obs - 1, T - T_pred):
                    self.index.append((ep_idx, start))

        print(f"  {len(self.index):,} sliding-window sam
... [truncated — 1539 chars total]
```

**After:**
```python
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
... [truncated — 2142 chars total]
```

---
<!-- auto-log 2026-04-20 03:35:42 edit -->
### [Auto-Log] 2026-04-20 03:35:42 — Script Fix

**File:** `scripts\train_flow_v4.py`

**Before:**
```python
import os
import sys
import time
import argparse
import numpy as np
import yaml
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
```

**After:**
```python
import os
import sys
import time
import argparse
import numpy as np
import yaml
import h5py
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
```

---
<!-- auto-log 2026-04-20 03:44:35 edit -->
### [Auto-Log] 2026-04-20 03:44:35 — Script Fix

**File:** `scripts\train_flow_v4.py`

**Before:**
```python
from torch.cuda.amp import autocast, GradScaler
```

**After:**
```python
from torch.amp import autocast, GradScaler
```

---
<!-- auto-log 2026-04-20 03:44:39 edit -->
### [Auto-Log] 2026-04-20 03:44:39 — Script Fix

**File:** `scripts\train_flow_v4.py`

**Before:**
```python
scaler = GradScaler()
```

**After:**
```python
scaler = GradScaler('cuda')
```

---
<!-- auto-log 2026-04-20 03:44:40 edit -->
### [Auto-Log] 2026-04-20 03:44:40 — Script Fix

**File:** `scripts\train_flow_v4.py`

**Before:**
```python
with autocast():
```

**After:**
```python
with autocast('cuda'):
```

---
<!-- auto-log 2026-04-21 13:18:38 edit -->
### [Auto-Log] 2026-04-21 13:18:38 — Model Fix

**File:** `models\flow_policy_v4.py`

**Before:**
```python
weights = torch.exp(beta * advantages).clamp(max=20.0).detach()  # (B,)

        mse = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=[1, 2])  # (B,)
        return (weights * mse).mean()
```

**After:**
```python
# Only learn from positive-advantage steps (avoid reinforcing bad actions)
        pos_mask = (advantages > 0).float().detach()  # (B,)
        weights  = (torch.exp(beta * advantages).clamp(max=20.0) * pos_mask).detach()

        if weights.sum() < 1e-8:
            return F.mse_loss(v_pred, v_target) * 0.0  # no positive samples; skip update

        mse = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=[1, 2])  # (B,)
        return (weights * mse).sum() / (weights.sum() + 1e-8)
```

---
<!-- auto-log 2026-04-21 13:18:55 edit -->
### [Auto-Log] 2026-04-21 13:18:55 — Script Fix

**File:** `scripts\train_reinflow_v4.py`

**Before:**
```python
with torch.no_grad():
            global_cond = policy._encode(img_tensor, imu_tensor)  # (1, 288)
            value       = value_net(global_cond).item()
            action_seq  = policy.predict_action(img_tensor, imu_tensor)
            # (1, action_dim, T_pred)
```

**After:**
```python
with torch.no_grad():
            global_cond = policy._encode(img_tensor, imu_tensor)  # (1, 288)
            value       = value_net(global_cond).item()
            # Sample noise explicitly so we can store x1 for stable updates
            x1 = torch.randn(1, policy.action_dim, policy.T_pred, device=device)
            action_seq  = policy.predict_action(img_tensor, imu_tensor,
                                                _fixed_x1=x1)
            # (1, action_dim, T_pred)
```

---
<!-- auto-log 2026-04-21 13:19:07 edit -->
### [Auto-Log] 2026-04-21 13:19:07 — Script Fix

**File:** `scripts\train_reinflow_v4.py`

**Before:**
```python
for a_idx in range(min(T_action, actions_to_exec.shape[0])):
            action = actions_to_exec[a_idx]

            rollout['image_stacks'].append(img_stack.copy())
            rollout['action_seqs'].append(action_seq_np.copy())  # (action_dim, T_pred)
            rollout['imu_data'].append(imu_vec.copy())
```

**After:**
```python
noise_np = x1.squeeze(0).cpu().numpy()  # (action_dim, T_pred)

        for a_idx in range(min(T_action, actions_to_exec.shape[0])):
            action = actions_to_exec[a_idx]

            rollout['image_stacks'].append(img_stack.copy())
            rollout['action_seqs'].append(action_seq_np.copy())  # (action_dim, T_pred)
            rollout['noise_seqs'].append(noise_np.copy())        # (action_dim, T_pred)
            rollout['imu_data'].append(imu_vec.copy())
```

---
<!-- auto-log 2026-04-21 13:19:13 edit -->
### [Auto-Log] 2026-04-21 13:19:13 — Model Fix

**File:** `models\flow_policy_v4.py`

**Before:**
```python
@torch.no_grad()
    def predict_action(
        self,
        images: torch.Tensor,
        imu: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
```

**After:**
```python
@torch.no_grad()
    def predict_action(
        self,
        images: torch.Tensor,
        imu: torch.Tensor,
        n_steps: Optional[int] = None,
        _fixed_x1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
```

---
<!-- auto-log 2026-04-21 13:19:18 edit -->
### [Auto-Log] 2026-04-21 13:19:18 — Model Fix

**File:** `models\flow_policy_v4.py`

**Before:**
```python
global_cond = self._encode(images, imu)

        x = torch.randn(B, self.action_dim, self.T_pred, device=device)
```

**After:**
```python
global_cond = self._encode(images, imu)

        x = _fixed_x1 if _fixed_x1 is not None else \
            torch.randn(B, self.action_dim, self.T_pred, device=device)
```

---
<!-- auto-log 2026-04-21 13:19:40 edit -->
### [Auto-Log] 2026-04-21 13:19:40 — Script Fix

**File:** `scripts\train_reinflow_v4.py`

**Before:**
```python
# ---- Policy update (skip during warmup) ----
        policy_loss_t = torch.tensor(0.0)
        if not in_warmup:
            policy.train()
            for _ in range(n_epochs):
                idx = torch.randperm(N)
                for start in range(0, N, mini_batch):
                    mb = idx[start:start + mini_batch]
                    imgs_gpu = img_cpu[mb].to(device=device, dtype=torch.float32) / 255.0
                    act_gpu  = act_cpu[mb].to(device)
                    imu_gpu  = imu_cpu[mb].to(device)
                    adv_gpu  = adv_t[mb].to(device)

                    pl = policy.compute_weighted_loss(
                        imgs_gpu, imu_gpu, act_gpu, adv_gpu, beta)
                    policy_opt.zero_grad()
                    pl.backward()
                    t
... [truncated — 947 chars total]
```

**After:**
```python
# ---- Policy update (skip during warmup) ----
        policy_loss_t = torch.tensor(0.0)
        frac_pos = (advantages > 0).mean()
        if not in_warmup:
            policy.train()
            for _ in range(n_epochs):
                idx = torch.randperm(N)
                for start in range(0, N, mini_batch):
                    mb = idx[start:start + mini_batch]
                    imgs_gpu  = img_cpu[mb].to(device=device, dtype=torch.float32) / 255.0
                    act_gpu   = act_cpu[mb].to(device)
                    noise_gpu = noise_cpu[mb].to(device)
                    imu_gpu   = imu_cpu[mb].to(device)
                    adv_gpu   = adv_t[mb].to(device)

                    pl = policy.compute_weighted_loss(
                        imgs_gpu, imu_gpu, act_gpu, adv_gpu, 
... [truncated — 1152 chars total]
```

---
<!-- auto-log 2026-04-21 13:20:06 edit -->
### [Auto-Log] 2026-04-21 13:20:06 — Script Fix

**File:** `scripts\train_reinflow_v4.py`

**Before:**
```python
writer.add_scalar('reinflow/mean_reward',  mean_reward,         update)
        writer.add_scalar('reinflow/policy_loss',  policy_loss_t.item(), update)
        writer.add_scalar('reinflow/value_loss',   value_loss_t.item(),  update)
        writer.add_scalar('reinflow/warmup',       int(in_warmup),       update)
```

**After:**
```python
writer.add_scalar('reinflow/mean_reward',  mean_reward,          update)
        writer.add_scalar('reinflow/policy_loss',  policy_loss_t.item(), update)
        writer.add_scalar('reinflow/value_loss',   value_loss_t.item(),  update)
        writer.add_scalar('reinflow/frac_pos_adv', float(frac_pos),      update)
        writer.add_scalar('reinflow/warmup',       int(in_warmup),       update)
```

---
<!-- auto-log 2026-04-22 03:02:41 edit -->
### [Auto-Log] 2026-04-22 03:02:41 — Model Fix

**File:** `models\flow_policy_v4.py`

**Before:**
```python
# Only learn from positive-advantage steps (avoid reinforcing bad actions)
        pos_mask = (advantages > 0).float().detach()  # (B,)
        weights  = (torch.exp(beta * advantages).clamp(max=20.0) * pos_mask).detach()

        if weights.sum() < 1e-8:
            return F.mse_loss(v_pred, v_target) * 0.0  # no positive samples; skip update

        mse = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=[1, 2])  # (B,)
        return (weights * mse).sum() / (weights.sum() + 1e-8)
```

**After:**
```python
weights = torch.exp(beta * advantages).clamp(max=20.0).detach()  # (B,)

        mse = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=[1, 2])  # (B,)
        return (weights * mse).mean()
```

---
<!-- auto-log 2026-04-22 03:02:46 edit -->
### [Auto-Log] 2026-04-22 03:02:46 — Script Fix

**File:** `scripts\train_reinflow_v4.py`

**Before:**
```python
pl = policy.compute_weighted_loss(
                        imgs_gpu, imu_gpu, act_gpu, adv_gpu, beta,
                        fixed_x1=noise_gpu)
```

**After:**
```python
pl = policy.compute_weighted_loss(
                        imgs_gpu, imu_gpu, act_gpu, adv_gpu, beta)
```

---
<!-- auto-log 2026-04-22 07:54:46 edit -->
### [Auto-Log] 2026-04-22 07:54:46 — Script Fix

**File:** `scripts\train_reinflow_v4.py`

**Before:**
```python
value_warmup         = rl_cfg['value_warmup_updates']
    vloss_thresh         = rl_cfg['vloss_best_threshold']
    total_updates        = rl_cfg['total_updates']

    print(f"\n{'='*60}")
    print(f"ReinFlow v4.0 (CTBR + INDI + Flow Matching RL)")
    print(f"Total updates: {total_updates} | Rollout steps: {n_rollout}")
    print(f"beta={beta} | LR_policy={rl_cfg['learning_rate']:.1e} | LR_value={rl_cfg['value_lr']:.1e}")
    print(f"Value warmup: {value_warmup} updates | VLoss threshold: {vloss_thresh}")
    print(f"Save: {save_dir}")
    print(f"{'='*60}\n")

    best_reward  = -float('inf')
    value_loss_t = torch.tensor(float('inf'))

    for update in range(total_updates):
        in_warmup = (update < value_warmup)
```

**After:**
```python
value_warmup         = rl_cfg['value_warmup_updates']
    vloss_thresh         = rl_cfg['vloss_best_threshold']
    vloss_gate           = rl_cfg.get('vloss_gate', 10.0)
    total_updates        = rl_cfg['total_updates']

    print(f"\n{'='*60}")
    print(f"ReinFlow v4.0 (CTBR + INDI + Flow Matching RL)")
    print(f"Total updates: {total_updates} | Rollout steps: {n_rollout}")
    print(f"beta={beta} | LR_policy={rl_cfg['learning_rate']:.1e} | LR_value={rl_cfg['value_lr']:.1e}")
    print(f"Value warmup: {value_warmup} updates (+ VLoss gate <{vloss_gate}) | Save threshold: {vloss_thresh}")
    print(f"Save: {save_dir}")
    print(f"{'='*60}\n")

    best_reward  = -float('inf')
    value_loss_t = torch.tensor(float('inf'))

    for update in range(total_updates):
        in_warmup = (upd
... [truncated — 857 chars total]
```

---
<!-- auto-log 2026-04-23 02:21:39 edit -->
### [Auto-Log] 2026-04-23 02:21:39 — Script Fix

**File:** `scripts\train_reinflow_v4.py`

**Before:**
```python
return advantages, returns


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------
```

**After:**
```python
return advantages, returns


# ---------------------------------------------------------------------------
# Demo dataset loader (for BC regularization)
# ---------------------------------------------------------------------------

def load_demo_subset(h5_path: str, n_episodes: int,
                     T_obs: int, T_pred: int):
    """Load n_episodes from expert demo h5 into RAM for BC loss."""
    img_buf, imu_buf, act_buf = [], [], []
    with h5py.File(h5_path, 'r') as f:
        keys = sorted([k for k in f.keys() if k.startswith('episode_')])[:n_episodes]
        for key in keys:
            imgs = f[key]['images'][:]      # (T, 3, H, W) uint8
            acts = f[key]['actions'][:]     # (T, 4) float32
            imus = f[key]['imu_data'][:]    # (T, 6) float32
            T = acts.
... [truncated — 1673 chars total]
```

---
<!-- auto-log 2026-04-23 02:22:15 edit -->
### [Auto-Log] 2026-04-23 02:22:15 — Script Fix

**File:** `scripts\train_reinflow_v4.py`

**Before:**
```python
policy_opt = torch.optim.AdamW(policy.parameters(), lr=rl_cfg['learning_rate'])
    value_opt  = torch.optim.Adam(value_net.parameters(), lr=rl_cfg['value_lr'])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag   = f"reinflow_v4_{timestamp}"
    log_dir   = os.path.join(log_cfg['tensorboard_log'], run_tag)
    save_dir  = os.path.join(log_cfg['save_path'],       run_tag)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    beta                 = rl_cfg['advantage_beta']
    gamma                = rl_cfg['gamma']
    gae_lambda           = rl_cfg['gae_lambda']
    n_rollout            = rl_cfg['n_rollout_steps']
    n_epochs             = rl_cfg['n_epochs']
    mini_batch           = rl_cfg['mini_batch'
... [truncated — 1710 chars total]
```

**After:**
```python
policy_opt = torch.optim.AdamW(policy.parameters(), lr=rl_cfg['learning_rate'])
    value_opt  = torch.optim.Adam(value_net.parameters(), lr=rl_cfg['value_lr'])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag   = f"reinflow_v4_{timestamp}"
    log_dir   = os.path.join(log_cfg['tensorboard_log'], run_tag)
    save_dir  = os.path.join(log_cfg['save_path'],       run_tag)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    beta                 = rl_cfg['advantage_beta']
    gamma                = rl_cfg['gamma']
    gae_lambda           = rl_cfg['gae_lambda']
    n_rollout            = rl_cfg['n_rollout_steps']
    n_epochs             = rl_cfg['n_epochs']
    mini_batch           = rl_cfg['mini_batch'
... [truncated — 2829 chars total]
```

---
<!-- auto-log 2026-04-23 02:22:31 edit -->
### [Auto-Log] 2026-04-23 02:22:31 — Script Fix

**File:** `scripts\train_reinflow_v4.py`

**Before:**
```python
for _ in range(n_epochs):
                idx = torch.randperm(N)
                for start in range(0, N, mini_batch):
                    mb = idx[start:start + mini_batch]
                    imgs_gpu  = img_cpu[mb].to(device=device, dtype=torch.float32) / 255.0
                    act_gpu   = act_cpu[mb].to(device)
                    noise_gpu = noise_cpu[mb].to(device)
                    imu_gpu   = imu_cpu[mb].to(device)
                    adv_gpu   = adv_t[mb].to(device)

                    pl = policy.compute_weighted_loss(
                        imgs_gpu, imu_gpu, act_gpu, adv_gpu, beta)
                    if pl.requires_grad:
                        policy_opt.zero_grad()
                        pl.backward()
                        torch.nn.utils.clip_grad_norm_(policy.par
... [truncated — 911 chars total]
```

**After:**
```python
demo_perm = torch.randperm(N_demo) if N_demo > 0 else None
            demo_ptr  = 0
            for _ in range(n_epochs):
                idx = torch.randperm(N)
                for start in range(0, N, mini_batch):
                    mb = idx[start:start + mini_batch]
                    imgs_gpu = img_cpu[mb].to(device=device, dtype=torch.float32) / 255.0
                    act_gpu  = act_cpu[mb].to(device)
                    imu_gpu  = imu_cpu[mb].to(device)
                    adv_gpu  = adv_t[mb].to(device)

                    rl_loss = policy.compute_weighted_loss(
                        imgs_gpu, imu_gpu, act_gpu, adv_gpu, beta)

                    # BC regularization: anchor policy to expert demos
                    if lambda_bc > 0 and N_demo > 0:
                        i
... [truncated — 1762 chars total]
```

---
<!-- auto-log 2026-04-23 02:22:41 edit -->
### [Auto-Log] 2026-04-23 02:22:41 — Script Fix

**File:** `scripts\train_reinflow_v4.py`

**Before:**
```python
# ---- Logging ----
        mean_reward = np.mean(rollout['rewards'])
        writer.add_scalar('reinflow/mean_reward',  mean_reward,          update)
        writer.add_scalar('reinflow/policy_loss',  policy_loss_t.item(), update)
        writer.add_scalar('reinflow/value_loss',   value_loss_t.item(),  update)
        writer.add_scalar('reinflow/frac_pos_adv', float(frac_pos),      update)
        writer.add_scalar('reinflow/warmup',       int(in_warmup),       update)

        warmup_tag = " [WARMUP]" if in_warmup else ""
        print(f"Update {update+1:>4}/{total_updates}{warmup_tag} | "
              f"Reward: {mean_reward:.4f} | "
              f"PLoss: {policy_loss_t.item():.6f} | "
              f"VLoss: {value_loss_t.item():.6f}")
```

**After:**
```python
# ---- Logging ----
        mean_reward = np.mean(rollout['rewards'])
        writer.add_scalar('reinflow/mean_reward',  mean_reward,          update)
        writer.add_scalar('reinflow/policy_loss',  policy_loss_t.item(), update)
        writer.add_scalar('reinflow/value_loss',   value_loss_t.item(),  update)
        writer.add_scalar('reinflow/frac_pos_adv', float(frac_pos),      update)
        writer.add_scalar('reinflow/warmup',       int(in_warmup),       update)

        warmup_tag = " [WARMUP]" if in_warmup else ""
        print(f"Update {update+1:>4}/{total_updates}{warmup_tag} | "
              f"Reward: {mean_reward:.4f} | "
              f"RLLoss: {policy_loss_t.item():.6f} | "
              f"VLoss: {value_loss_t.item():.6f}")
```

---
<!-- auto-log 2026-04-23 14:53:16 edit -->
### [Auto-Log] 2026-04-23 14:53:16 — Script Fix

**File:** `scripts\train_reinflow_v4.py`

**Before:**
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReinFlow v4.0 RL Fine-tuning")
    parser.add_argument('--flow-config',      type=str, default='configs/flow_policy_v4.yaml')
    parser.add_argument('--rl-config',        type=str, default='configs/reinflow_v4.yaml')
    parser.add_argument('--quadrotor-config', type=str, default='configs/quadrotor_v4.yaml')
    parser.add_argument('--pretrained',       type=str, default=None,
                        help='Path to pretrained FlowMatchingPolicyV4 checkpoint')
    parser.add_argument('--pretrained-value', type=str, default=None)
    args = parser.parse_args()
    train(args)
```

**After:**
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReinFlow v4.0 RL Fine-tuning")
    parser.add_argument('--flow-config',          type=str, default='configs/flow_policy_v4.yaml')
    parser.add_argument('--rl-config',            type=str, default='configs/reinflow_v4.yaml')
    parser.add_argument('--quadrotor-config',     type=str, default='configs/quadrotor_v4.yaml',
                        help='Env config for evaluation (standard)')
    parser.add_argument('--rl-quadrotor-config',  type=str, default=None,
                        help='Env config for RL rollout (wider disturbances). '
                             'Defaults to --quadrotor-config if not set.')
    parser.add_argument('--pretrained',           type=str, default=None,
                        help
... [truncated — 975 chars total]
```

---
<!-- auto-log 2026-04-23 14:53:33 edit -->
### [Auto-Log] 2026-04-23 14:53:33 — Config / HP Change

**File:** `configs\reinflow_v4.yaml`

**Before:**
```yaml
# ReinFlow v4.0 Configuration
# Advantage-weighted Flow Matching RL fine-tuning (CTBR + INDI)
# Run 7: BC regularization (lambda_bc=0.1) + one-way VLoss gate + beta=0.1

rl:
  learning_rate: 5.0e-7
  value_lr: 1.0e-3
  value_hidden_dim: 256
  n_rollout_steps: 4096
  n_epochs: 1
  mini_batch: 256
  gamma: 0.99
  gae_lambda: 0.95
  advantage_beta: 0.1            # middle ground (0.05 too weak, 0.15 too fast decline)
  grad_clip: 1.0
  value_warmup_updates: 200
  vloss_gate: 10.0               # one-way: once passed, never re-enter warmup
  vloss_best_threshold: 100.0
  save_freq_ckpt: 10
  total_updates: 600
  lambda_bc: 0.1                 # BC regularization weight
  demo_path: data/expert_demos_v4.h5
  demo_episodes: 100             # ~50k samples (~600MB RAM)
```

**After:**
```yaml
# ReinFlow v4.0 Configuration
# Advantage-weighted Flow Matching RL fine-tuning (CTBR + INDI)
# Run 8: OOD training distribution (quadrotor_v4_rl.yaml) + BC reg + one-way gate

rl:
  learning_rate: 5.0e-7
  value_lr: 1.0e-3
  value_hidden_dim: 256
  n_rollout_steps: 4096
  n_epochs: 1
  mini_batch: 256
  gamma: 0.99
  gae_lambda: 0.95
  advantage_beta: 0.1
  grad_clip: 1.0
  value_warmup_updates: 200
  vloss_gate: 10.0               # one-way gate (from Run 5)
  vloss_best_threshold: 100.0
  save_freq_ckpt: 10
  total_updates: 600
  lambda_bc: 0.1                 # BC regularization (from Run 7)
  demo_path: data/expert_demos_v4.h5
  demo_episodes: 100
```

---
<!-- auto-log 2026-04-26 07:04:31 edit -->
### [Auto-Log] 2026-04-26 07:04:31 — Script Fix

**File:** `scripts\train_reinflow_v4.py`

**Before:**
```python
beta                 = rl_cfg['advantage_beta']
    gamma                = rl_cfg['gamma']
    gae_lambda           = rl_cfg['gae_lambda']
    n_rollout            = rl_cfg['n_rollout_steps']
    n_epochs             = rl_cfg['n_epochs']
    mini_batch           = rl_cfg['mini_batch']
    grad_clip            = rl_cfg['grad_clip']
    value_warmup         = rl_cfg['value_warmup_updates']
    vloss_thresh         = rl_cfg['vloss_best_threshold']
    vloss_gate           = rl_cfg.get('vloss_gate', 10.0)
    lambda_bc            = rl_cfg.get('lambda_bc', 0.0)
    total_updates        = rl_cfg['total_updates']
```

**After:**
```python
beta                 = rl_cfg['advantage_beta']
    gamma                = rl_cfg['gamma']
    gae_lambda           = rl_cfg['gae_lambda']
    n_rollout            = rl_cfg['n_rollout_steps']
    n_epochs             = rl_cfg['n_epochs']
    mini_batch           = rl_cfg['mini_batch']
    grad_clip            = rl_cfg['grad_clip']
    value_warmup         = rl_cfg['value_warmup_updates']
    vloss_thresh         = rl_cfg['vloss_best_threshold']
    vloss_gate           = rl_cfg.get('vloss_gate', 10.0)
    lambda_bc            = rl_cfg.get('lambda_bc', 0.0)
    total_updates        = rl_cfg['total_updates']

    # Curriculum config
    with open(args.rl_config, 'r', encoding='utf-8') as f:
        full_cfg = yaml.safe_load(f)
    cur_cfg = full_cfg.get('curriculum', {})
    curriculum_enabl
... [truncated — 1206 chars total]
```

---
<!-- auto-log 2026-04-26 07:05:05 edit -->
### [Auto-Log] 2026-04-26 07:05:05 — Script Fix

**File:** `scripts\train_reinflow_v4.py`

**Before:**
```python
# ---- Logging ----
        mean_reward = np.mean(rollout['rewards'])
        writer.add_scalar('reinflow/mean_reward',  mean_reward,          update)
        writer.add_scalar('reinflow/policy_loss',  policy_loss_t.item(), update)
        writer.add_scalar('reinflow/value_loss',   value_loss_t.item(),  update)
        writer.add_scalar('reinflow/frac_pos_adv', float(frac_pos),      update)
        writer.add_scalar('reinflow/warmup',       int(in_warmup),       update)
```

**After:**
```python
# ---- Logging ----
        mean_reward = np.mean(rollout['rewards'])
        writer.add_scalar('reinflow/mean_reward',  mean_reward,          update)
        writer.add_scalar('reinflow/policy_loss',  policy_loss_t.item(), update)
        writer.add_scalar('reinflow/value_loss',   value_loss_t.item(),  update)
        writer.add_scalar('reinflow/frac_pos_adv', float(frac_pos),      update)
        writer.add_scalar('reinflow/warmup',       int(in_warmup),       update)
        if curriculum_enabled:
            writer.add_scalar('curriculum/pos_range', base_env.initial_pos_range, update)
            writer.add_scalar('curriculum/vel_range', base_env.initial_vel_range, update)
```

---
<!-- auto-log 2026-04-26 07:05:57 edit -->
### [Auto-Log] 2026-04-26 07:05:57 — Script Fix

**File:** `scripts\train_reinflow_v4.py`

**Before:**
```python
warmup_tag = " [WARMUP]" if in_warmup else ""
        print(f"Update {update+1:>4}/{total_updates}{warmup_tag} | "
              f"Reward: {mean_reward:.4f} | "
              f"RLLoss: {policy_loss_t.item():.6f} | "
              f"VLoss: {value_loss_t.item():.6f}")
```

**After:**
```python
if curriculum_enabled and vloss_gate_passed:
            if updates_since_gate <= cur_n_hover:
                cur_stage_tag = f" [CUR S1 hover pos={base_env.initial_pos_range:.2f}m]"
            elif updates_since_gate <= cur_n_hover + cur_n_ramp:
                cur_stage_tag = f" [CUR S2 ramp pos={base_env.initial_pos_range:.2f}m]"
            else:
                cur_stage_tag = f" [CUR S3 OOD pos={base_env.initial_pos_range:.2f}m]"
        else:
            cur_stage_tag = ""
        warmup_tag = " [WARMUP]" if in_warmup else ""
        print(f"Update {update+1:>4}/{total_updates}{warmup_tag}{cur_stage_tag} | "
              f"Reward: {mean_reward:.4f} | "
              f"RLLoss: {policy_loss_t.item():.6f} | "
              f"VLoss: {value_loss_t.item():.6f}")
```

---
<!-- auto-log 2026-04-28 01:27:50 edit -->
### [Auto-Log] 2026-04-28 01:27:50 — Script Fix

**File:** `scripts\train_reinflow_v4.py`

**Before:**
```python
if args.pretrained:
        policy.load(args.pretrained)
        print(f"Loaded pretrained: {args.pretrained}")

    # Value network
    global_cond_dim = vis_cfg['feature_dim'] + imu_cfg['feature_dim']   # 288
    value_net = ValueNetworkV4(
        global_cond_dim=global_cond_dim,
        hidden_dim=rl_cfg['value_hidden_dim'],
    ).to(device)
```

**After:**
```python
if args.pretrained:
        policy.load(args.pretrained)
        print(f"Loaded pretrained policy: {args.pretrained}")

    # Value network
    global_cond_dim = vis_cfg['feature_dim'] + imu_cfg['feature_dim']   # 288
    value_net = ValueNetworkV4(
        global_cond_dim=global_cond_dim,
        hidden_dim=rl_cfg['value_hidden_dim'],
    ).to(device)

    if args.pretrained_value:
        value_net.load_state_dict(torch.load(args.pretrained_value, map_location=device))
        print(f"Loaded pretrained value net: {args.pretrained_value}")
```

---
<!-- auto-log 2026-04-29 05:15:06 edit -->
### [Auto-Log] 2026-04-29 05:15:06 — Env Fix

**File:** `envs\quadrotor_env_v4.py`

**Before:**
```python
def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        init_pos = self.np_random.uniform(
            -self.initial_pos_range, self.initial_pos_range, size=3
        )
        init_pos[2] = -abs(init_pos[2]) - 1.0

        init_vel = self.np_random.uniform(
            -self.initial_vel_range, self.initial_vel_range, size=3
        )
```

**After:**
```python
def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Anchored curriculum: with prob `hover_anchor_prob`, force a near-hover
        # init (pos<=0.1m, vel<=0.05) to prevent catastrophic forgetting of hover.
        # Set externally via env.hover_anchor_prob = 0.2 from training script.
        anchor_prob = getattr(self, 'hover_anchor_prob', 0.0)
        if anchor_prob > 0.0 and self.np_random.random() < anchor_prob:
            pos_range_now = 0.1
            vel_range_now = 0.05
        else:
            pos_range_now = self.initial_pos_range
            vel_range_now = self.initial_vel_range

        init_pos = self.np_random.uniform(
            -pos_range_now, pos_range_now, siz
... [truncated — 963 chars total]
```

---
<!-- auto-log 2026-04-29 05:15:42 edit -->
### [Auto-Log] 2026-04-29 05:15:42 — Script Fix

**File:** `scripts\train_reinflow_v4.py`

**Before:**
```python
# Curriculum config
    with open(args.rl_config, 'r', encoding='utf-8') as f:
        full_cfg = yaml.safe_load(f)
    cur_cfg = full_cfg.get('curriculum', {})
    curriculum_enabled = cur_cfg.get('enabled', False)
    cur_n_hover  = cur_cfg.get('n_hover_updates', 50)
    cur_n_ramp   = cur_cfg.get('n_ramp_updates', 200)
    cur_pos_start = cur_cfg.get('pos_start', base_env.initial_pos_range)
    cur_vel_start = cur_cfg.get('vel_start', base_env.initial_vel_range)
    cur_pos_end  = cur_cfg.get('pos_end',   cur_pos_start)
    cur_vel_end  = cur_cfg.get('vel_end',   cur_vel_start)
```

**After:**
```python
# Curriculum config
    with open(args.rl_config, 'r', encoding='utf-8') as f:
        full_cfg = yaml.safe_load(f)
    cur_cfg = full_cfg.get('curriculum', {})
    curriculum_enabled = cur_cfg.get('enabled', False)
    cur_n_hover  = cur_cfg.get('n_hover_updates', 50)
    cur_n_ramp   = cur_cfg.get('n_ramp_updates', 200)
    cur_pos_start = cur_cfg.get('pos_start', base_env.initial_pos_range)
    cur_vel_start = cur_cfg.get('vel_start', base_env.initial_vel_range)
    cur_pos_end  = cur_cfg.get('pos_end',   cur_pos_start)
    cur_vel_end  = cur_cfg.get('vel_end',   cur_vel_start)
    cur_anchor_prob = cur_cfg.get('hover_anchor_prob', 0.0)

    # Softened crash penalty during RL rollout (Run 12: prevents shock gradient
    # destroying hover behaviour when drone crashes during OOD ramp)
  
... [truncated — 1107 chars total]
```

---
<!-- auto-log 2026-04-29 05:16:02 edit -->
### [Auto-Log] 2026-04-29 05:16:02 — Script Fix

**File:** `scripts\train_reinflow_v4.py`

**Before:**
```python
# ---- Curriculum: update env params ----
        if curriculum_enabled and vloss_gate_passed:
            if updates_since_gate <= cur_n_hover:
                # Stage 1: hover stabilisation
                cur_pos = cur_pos_start
                cur_vel = cur_vel_start
            elif updates_since_gate <= cur_n_hover + cur_n_ramp:
                # Stage 2: linear ramp
                t = (updates_since_gate - cur_n_hover) / cur_n_ramp
                cur_pos = cur_pos_start + t * (cur_pos_end - cur_pos_start)
                cur_vel = cur_vel_start + t * (cur_vel_end - cur_vel_start)
            else:
                # Stage 3: full OOD
                cur_pos = cur_pos_end
                cur_vel = cur_vel_end
            base_env.initial_pos_range = cur_pos
            base_env.init
... [truncated — 823 chars total]
```

**After:**
```python
# ---- Curriculum: update env params ----
        if curriculum_enabled and vloss_gate_passed:
            if updates_since_gate <= cur_n_hover:
                # Stage 1: hover stabilisation (no anchor needed yet)
                cur_pos = cur_pos_start
                cur_vel = cur_vel_start
                base_env.hover_anchor_prob = 0.0
            elif updates_since_gate <= cur_n_hover + cur_n_ramp:
                # Stage 2: linear ramp + anchor (prevents catastrophic forgetting)
                t = (updates_since_gate - cur_n_hover) / cur_n_ramp
                cur_pos = cur_pos_start + t * (cur_pos_end - cur_pos_start)
                cur_vel = cur_vel_start + t * (cur_vel_end - cur_vel_start)
                base_env.hover_anchor_prob = cur_anchor_prob
            else:
         
... [truncated — 1070 chars total]
```

---
<!-- auto-log 2026-04-29 05:16:09 edit -->
### [Auto-Log] 2026-04-29 05:16:09 — Script Fix

**File:** `scripts\train_reinflow_v4.py`

**Before:**
```python
if curriculum_enabled:
            writer.add_scalar('curriculum/pos_range', base_env.initial_pos_range, update)
            writer.add_scalar('curriculum/vel_range', base_env.initial_vel_range, update)
```

**After:**
```python
if curriculum_enabled:
            writer.add_scalar('curriculum/pos_range', base_env.initial_pos_range, update)
            writer.add_scalar('curriculum/vel_range', base_env.initial_vel_range, update)
            writer.add_scalar('curriculum/anchor_prob', getattr(base_env, 'hover_anchor_prob', 0.0), update)
```

---
<!-- auto-log 2026-04-29 15:51:36 edit -->
### [Auto-Log] 2026-04-29 15:51:36 — Env Fix

**File:** `envs\quadrotor_env_v4.py`

**Before:**
```python
self.w_action     = r['w_action']
        self.alive_bonus  = r['alive_bonus']
        self.crash_penalty = r['crash_penalty']
```

**After:**
```python
self.w_action     = r['w_action']
        self.alive_bonus  = r['alive_bonus']
        self.crash_penalty = r['crash_penalty']
        self.w_brake      = r.get('w_brake', 0.0)
        self.sigma_brake  = r.get('sigma_brake', 0.3)
```

---
<!-- auto-log 2026-04-29 15:51:49 edit -->
### [Auto-Log] 2026-04-29 15:51:49 — Env Fix

**File:** `envs\quadrotor_env_v4.py`

**Before:**
```python
# Action penalty: deviation of CTBR from hover point
        F_hover_norm = (self.hover_thrust * 4 / self.F_c_max) * 2 - 1   # ~-0.387
        action_dev   = np.array([
            action[0] - F_hover_norm,   # F_c deviation
            action[1],                  # omega_x (target 0)
            action[2],                  # omega_y (target 0)
            action[3],                  # omega_z (target 0)
        ])
        action_pen = self.w_action * np.sum(action_dev**2)

        return pos_rew + z_rew + vel_rew + ang_rew - action_pen + self.alive_bonus
```

**After:**
```python
# Action penalty: deviation of CTBR from hover point
        F_hover_norm = (self.hover_thrust * 4 / self.F_c_max) * 2 - 1   # ~-0.387
        action_dev   = np.array([
            action[0] - F_hover_norm,   # F_c deviation
            action[1],                  # omega_x (target 0)
            action[2],                  # omega_y (target 0)
            action[3],                  # omega_z (target 0)
        ])
        action_pen = self.w_action * np.sum(action_dev**2)

        # Continuous velocity brake penalty: closer + faster -> larger penalty.
        # Forces "soft landing" approach instead of suicide-rush. Z-axis excluded
        # to avoid interfering with anisotropic z_rew design.
        dist_xy   = np.sqrt(pos_error[0]**2 + pos_error[1]**2)
        vel_xy_sq = vel[0]**2 + ve
... [truncated — 986 chars total]
```

---
<!-- auto-log 2026-04-29 15:52:21 edit -->
### [Auto-Log] 2026-04-29 15:52:21 — Script Fix

**File:** `scripts\train_reinflow_v4.py`

**Before:**
```python
adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
```

**After:**
```python
adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        adv_t = torch.clamp(adv_t, -3.0, 3.0)
```

---
<!-- auto-log 2026-04-29 15:53:50 write -->
### [Auto-Log] 2026-04-29 15:53:50 — New File: Script Fix

**File:** `scripts\merge_expert_demos.py`

**Content:**
```python
"""
Merge multiple expert_demos*.h5 files into a single mixed dataset.

Used by Run 13 to combine hover-only demos (expert_demos_v4.h5) with
approach demos (expert_demos_v4_approach.h5) so BC loss anchors the policy
to both regimes simultaneously.

Episode groups are renumbered consecutively in the output file. Per-episode
attributes (e.g. initial_pos_range) are preserved.

Usage:
    python -m scripts.merge_expert_demos \
        --inputs data/expert_demos_v4.h5 data/expert_demos_v4_approach.h5 \
        --output data/expert_demos_v4_mixed.h5
"""

import os
import sys
import argparse
import numpy as np
import h5py


def copy_episode(src_grp: h5py.Group, dst_grp: h5py.Group):
    for key in src_grp.keys():
        ds = src_grp[key]
        kwargs = {}
        if ds.compression is not None:
... [truncated — 3772 chars total]
```

# DPPO PID Controller — Phase 3c v3.3: Physics-based IMU + Normalization

> Part of the dev log series. See [index](dev_log_phase2_3.md) for all phases.
> Covers: v3.3 architecture, Phase 3a supervised pre-training, DPPO Run 1 & Run 2 results.

---

## Table of Contents

1. [v3.3 Architecture & Motivation](#1-v33-architecture--motivation)
2. [Phase 3a Supervised Pre-training](#2-phase-3a-supervised-pre-training)
3. [DPPO Run 1 — warmup=50](#3-dppo-run-1--warmup50)
4. [DPPO Run 2 — warmup=100](#4-dppo-run-2--warmup100)
5. [Phase 3c Summary & Root Cause](#5-phase-3c-summary--root-cause)

---

## 1. v3.3 Architecture & Motivation

**Problem v3.2 solved:** finite-diff IMU (`v_body` derivative) amplified Coriolis noise 20× during RL rollouts (ax std ratio expert/perturbed = 23×). Replaced with `get_specific_force_body()` = `R^T @ (F_world − mg) / m`.

**Problem v3.2 introduced:** `specific_force` at hover ≈ −9.81 m/s² (gravity not subtracted in body frame at rest). IMU input never near zero → supervised RMSE 1.985m → DPPO aborted u25.

**v3.3 fix:** Per-axis normalisation applied at collection time (`collect_data.py --v33`):
```
gyro       / 2.0          →  σ ≈ 1
sf_x, sf_y / 5.0          →  centred near 0
(sf_z + 9.81) / 5.0       →  zero-centred at hover
```
Result: ax std ratio 23× → **1.4×** (within <2× target). Supervised RMSE recovered to 0.286m baseline level.

**Architecture:** same as v3.1 — VisionDPPOv31, IMUEncoder MLP(6→64→32), 288D global_cond, FCN DepthDecoder (training-only auxiliary loss).

---

## 2. Phase 3a Supervised Pre-training

| Item | Value |
|------|-------|
| Dataset | `data/expert_demos_v33.h5` (4.0 GB, 1000 ep, collected 2026-04-11) |
| Checkpoint | `checkpoints/diffusion_policy/v33_20260412_052333/best_model.pt` |
| Best loss | −1.4435 @ epoch 488 |
| Epochs | 500 |
| Duration | ~14h (RTX 3090) |

---

## 3. DPPO Run 1 — warmup=50

**Run ID:** `dppo_v33_20260413_033647`
**Period:** 2026-04-13 03:36 → 2026-04-13 17:16 (~13.6h)
**Config:** β=0.05, lr=5e-6, warmup=50, n_rollout=4096, vloss_threshold=500

### Training Curve

| Update | Reward/step | VLoss | Note |
|--------|-------------|-------|------|
| 1 | 0.433 | 5,372,091 | start |
| 50 | 0.419 | ~3,300 | warmup ends — VLoss not yet converged |
| 85 | — | 191 | first dip < 200 → checkpoint trigger |
| ~200 | 0.43 | ~400 | best checkpoint region |
| 407 | 0.429 | 81 | stable |
| 500 | 0.307 | 18 | final (slight late decline) |

### Checkpoints

| File | Saved | Notes |
|------|-------|-------|
| `best_dppo_v33_model.pt` | 09:05 | ~u200, VLoss first stably < 500 |
| `final_dppo_v33_model.pt` | 17:16 | last epoch |
| `deploy_model.pt` | 17:16 | FCN decoder stripped |

### RHC Evaluation

| Metric | Value |
|--------|-------|
| **Position RMSE** | **0.1039 m** ← best across all architectures |
| Crashes | 50 / 50 |
| Mean reward | 24.51 ± 9.58 |
| Inference time | 70.2 ms (median 69.0 ms) |
| PPO Expert RMSE | 0.0693 m (reference) |

### Assessment

VLoss at u50 was still ~3300 (far from converged), so policy updates began with noisy advantage estimates. Despite this, Run 1 achieved the best RMSE to date (0.1039m vs 0.168m baseline). Reward held stable throughout training with only a mild late decline.

---

## 4. DPPO Run 2 — warmup=100

**Run ID:** `dppo_v33_20260414_023817`
**Period:** 2026-04-14 02:38 → 2026-04-14 15:57 (~13.3h)
**Config:** β=0.05, lr=5e-6, **warmup=100**, n_rollout=4096, vloss_threshold=500
**Motivation:** VLoss was ~3300 at u50 in Run 1; extend warmup so value net converges before policy learning starts.

### Training Curve

| Update | Reward/step | VLoss | Note |
|--------|-------------|-------|------|
| 1 | 0.433 | 4,388,943 | start |
| 100 | ~0.42 | ~1,300 | warmup ends — VLoss still high |
| 101 | 0.461 | 3,621 | policy starts; VLoss spikes from new rollout distribution |
| 176 | 0.515 | 953 | reward rising |
| **225** | **0.7077** | 488 | **peak reward — highest in entire project** |
| 251 | 0.560 | 124 | declining |
| 275 | 0.319 | 774 | collapse begins |
| 301 | 0.270 | 87 | |
| 500 | 0.348 | 37 | final (partial recovery) |

### Checkpoints

| File | Saved | Notes |
|------|-------|-------|
| `best_dppo_v33_model.pt` | 08:30 | ~u225, peak reward region |
| `final_dppo_v33_model.pt` | 15:57 | last epoch |
| `deploy_model.pt` | 15:57 | FCN decoder stripped |

### RHC Evaluation

| Metric | Value |
|--------|-------|
| Position RMSE | 0.1335 m ← worse than Run 1 |
| Crashes | 50 / 50 |
| Mean reward | **30.23 ± 14.95** ← higher than Run 1 |
| Inference time | 74.1 ms (median 71.2 ms) |
| PPO Expert RMSE | 0.0693 m (reference) |

### Assessment

Extended warmup enabled much higher peak reward (0.71 vs ~0.43), but RMSE worsened (0.1335m vs 0.1039m). The policy learned to fly *longer* (avg ~42 steps vs ~37 in Run 1) but with *worse positional accuracy* — long-surviving episodes tended to drift further from the target. This reveals a **reward/RMSE misalignment**: reward accumulates with alive_bonus and velocity smoothness regardless of positional drift.

Reward collapse at u275 is consistent with over-optimization: advantage estimates drove the policy to high-reward-but-imprecise flight modes that eventually destabilized.

---

## 5. Phase 3c Summary & Root Cause

### Results Comparison

| Run | RMSE | Crashes | Mean Reward | Peak Reward | warmup |
|-----|------|---------|-------------|-------------|--------|
| 3b Run 2 (baseline) | 0.168m | 50/50 | — | — | — |
| **v3.3 Run 1** | **0.1039m** ← best | 50/50 | 24.51 | ~0.49 | 50 |
| v3.3 Run 2 | 0.1335m | 50/50 | 30.23 | **0.7077** | 100 |
| PPO Expert | 0.069m | 0/50 | 538.8 | — | — |

### Root Cause: Inference Latency Bottleneck

Every run hit the same 50/50 crash ceiling despite improving RMSE and reward. The fundamental constraint:

```
DDIM 10-step inference:  74 ms
Control loop period:     20 ms  (50 Hz)
Lag ratio:               3.7×

At 1 m/s, drone moves ~74 mm between consecutive observations.
Policy was trained with instantaneous inference — never saw this lag.
Covariate shift from latency is structurally guaranteed.
```

Hyperparameter tuning (β, LR, warmup) can shift the reward/RMSE tradeoff but cannot close a 3.7× latency gap. The crash ceiling will persist until inference is faster than the control period.

### Next Step: Phase 3d OneDP Distillation

- **Goal:** single-step inference < 16ms → 62Hz+ (closes the latency gap)
- **Teacher model:** `checkpoints/diffusion_policy/dppo_v33_20260413_033647/best_dppo_v33_model.pt` (RMSE 0.1039m)
- **Expected impact:** crash rate should drop significantly once the drone receives fresh observations at control frequency

---
<!-- auto-log 2026-04-12 04:18:56 bash -->
### [Auto-Log] 2026-04-12 04:18:56 — DPPO Training — Started

**Command:** `cd /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller

# Collect progress
echo "=== COLLECT V33 ==="
CLOG=$(ls -t logs/collect_v33_*.log 2>/dev/null | head -1)
tail -1 "$CLOG" 2>/dev/null | tr -d '\r'

# Chain status
echo "=== CHAIN ==="
CHALOG=$(ls -t logs/chain_v33_*.log 2>/dev/null | head -1)
tail -2 "$CHALOG" 2>/dev/null | tr -d '\r'

# DPPO Run 5
echo "=== DPPO RUN 5 ==="
DLOG=$(ls -t logs/train_dppo_run5_*.log 2>/dev/null | head -1)
tail -3 "$DLOG" 2>/dev/null | tr -d '\r'`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-12 04:19:07 bash -->
### [Auto-Log] 2026-04-12 04:19:07 — DPPO Training — Started

**Command:** `cd /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller

# Segfault details
echo "=== DIFFUSION V33 LOG (last 20) ==="
DIFFLOG=$(ls -t logs/train_diffusion_v33_*.log 2>/dev/null | head -1)
echo "Log: $DIFFLOG"
cat "$DIFFLOG" 2>/dev/null | tr -d '\r' | tail -20

# DPPO Run 5 full recent
echo ""
echo "=== DPPO RUN 5 RECENT ==="
DLOG=$(ls -t logs/train_dppo_run5_*.log 2>/dev/null | head -1)
echo "Log: $DLOG"
cat "$DLOG" | tr -d '\r' | grep "Update" | tail -10`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-12 04:20:43 bash -->
### [Auto-Log] 2026-04-12 04:20:43 — Diffusion v3.3 Training — Started

**Command:** `cd /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller

# Segfault - what happened
DIFFLOG=$(ls -t logs/train_diffusion_v33_*.log 2>/dev/null | head -1)
wc -l "$DIFFLOG"
cat "$DIFFLOG" | tr -d '\r'

# Check chain log fully
CHALOG=$(ls -t logs/chain_v33_*.log 2>/dev/null | head -1)
tail -10 "$CHALOG" | tr -d '\r'`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-12 04:21:15 bash -->
### [Auto-Log] 2026-04-12 04:21:15 — Diffusion v3.3 Training — Started

**Command:** `cd /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller

# Check first diffusion v33 log (from first chain attempt)
DIFFLOG1="logs/train_diffusion_v33_20260411_192314.log"
echo "=== First diffusion v33 log ==="
wc -l "$DIFFLOG1"
cat "$DIFFLOG1" | tr -d '\r' | head -30`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-12 04:27:48 bash -->
### [Auto-Log] 2026-04-12 04:27:48 — Diffusion v3.3 Training — Started

**Command:** `cd /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller
grep -n "resume\|checkpoint\|load" scripts/train_diffusion_v33.py | head -15`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-12 05:23:40 bash -->
### [Auto-Log] 2026-04-12 05:23:40 — Diffusion v3.3 Training — Started

**Command:** `cd /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller
cat logs/train_diffusion_v33_20260412_052326.log | tr -d '\r' | head -20`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-12 05:23:43 bash -->
### [Auto-Log] 2026-04-12 05:23:43 — Diffusion v3.3 Training — Started

**Command:** `ls -lh logs/train_diffusion_v33_20260412_052326.log && wc -l logs/train_diffusion_v33_20260412_052326.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-12 05:25:51 bash -->
### [Auto-Log] 2026-04-12 05:25:51 — Diffusion v3.3 Training — Started

**Command:** `cd /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller
# Wait and recheck
ls -lt logs/train_diffusion_v33_*.log
# Check if python is running
tasklist 2>/dev/null | grep python || true
# Check new TB dir
ls -lt logs/diffusion_policy/ | head -3`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-12 05:25:58 bash -->
### [Auto-Log] 2026-04-12 05:25:58 — Diffusion v3.3 Training — Started

**Command:** `ls -lh logs/diffusion_policy/v33_20260412_052333/ && wc -l logs/train_diffusion_v33_20260412_052326.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-13 03:36:39 bash -->
### [Auto-Log] 2026-04-13 03:36:39 — DPPO v3.3 Training — Started

**Command:** `source dppo/Scripts/activate && BEST_CKPT="checkpoints/diffusion_policy/v33_20260412_052333/best_model.pt" && DPPO_LOG="logs/train_dppo_v33_$(date +%Y%m%d_%H%M%S).log" && echo "Launching Step 5..." && echo "pretrained: $BEST_CKPT" && echo "log: $DPPO_LOG" && nohup python -u -m scripts.train_dppo_v33 --pretrained "$BEST_CKPT" > "$DPPO_LOG" 2>&1 & echo "PID: $!"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-13 03:36:49 bash -->
### [Auto-Log] 2026-04-13 03:36:49 — DPPO v3.3 Training — Started

**Command:** `ls logs/train_dppo_v33_*.log 2>/dev/null && head -5 logs/train_dppo_v33_*.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-13 03:36:53 bash -->
### [Auto-Log] 2026-04-13 03:36:53 — DPPO v3.3 Training — Started

**Command:** `cat logs/train_dppo_v33_20260413_033639.log | tr -d '\r'`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-13 04:09:23 bash -->
### [Auto-Log] 2026-04-13 04:09:23 — DPPO v3.3 Training — Started

**Command:** `cd "C:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && tail -80 logs/train_dppo_v33_20260413_033639.log 2>&1`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-13 04:09:28 bash -->
### [Auto-Log] 2026-04-13 04:09:28 — DPPO v3.3 Training — Started

**Command:** `cd "C:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && wc -l logs/train_dppo_v33_20260413_033639.log && grep "^Update" logs/train_dppo_v33_20260413_033639.log | tail -20`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-13 04:10:05 bash -->
### [Auto-Log] 2026-04-13 04:10:05 — DPPO v3.3 Training — Started

**Command:** `cd "C:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && stat logs/train_dppo_v33_20260413_033639.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-13 14:42:04 bash -->
### [Auto-Log] 2026-04-13 14:42:04 — DPPO v3.3 Training — Started

**Command:** `cd "C:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && grep "^Update" logs/train_dppo_v33_20260413_033639.log | tail -10 && echo "---" && wc -l logs/train_dppo_v33_20260413_033639.log && stat -c "%y" logs/train_dppo_v33_20260413_033639.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-13 14:42:16 bash -->
### [Auto-Log] 2026-04-13 14:42:16 — DPPO v3.3 Training — Started

**Command:** `cd "C:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && ls -la checkpoints/diffusion_policy/dppo_v33_20260413_033647/ && echo "---" && grep "^Update" logs/train_dppo_v33_20260413_033639.log | awk -F'[| ]+' '{print $2, $6}' | awk 'NR%50==1 || NR>395' | head -20`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-14 02:32:21 bash -->
### [Auto-Log] 2026-04-14 02:32:21 — DPPO v3.3 Training — Started

**Command:** `cd "C:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && grep "^Update" logs/train_dppo_v33_20260413_033639.log | tail -10 && echo "---" && wc -l logs/train_dppo_v33_20260413_033639.log && stat -c "%y" logs/train_dppo_v33_20260413_033639.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-14 02:33:11 bash -->
### [Auto-Log] 2026-04-14 02:33:11 — RHC v3.3 Evaluation

**Command:** `cd "C:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && source dppo/Scripts/activate && python -m scripts.evaluate_rhc_v33 \
    --diffusion-model checkpoints/diffusion_policy/dppo_v33_20260413_033647/best_dppo_v33_model.pt \
    --ppo-model  checkpoints/ppo_expert/20260401_103107/best_model.pt \
    --ppo-norm   checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz \
    2>&1 | tee logs/eval_rhc_v33_20260414.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-14 02:35:11 bash -->
### [Auto-Log] 2026-04-14 02:35:11 — RHC v3.3 Evaluation

**Command:** `cd "C:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && grep -n "RMSE\|crash\|Crash\|mean\|median\|survived" scripts/evaluate_rhc_v33.py | head -30`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-14 02:38:13 bash -->
### [Auto-Log] 2026-04-14 02:38:13 — DPPO v3.3 Training — Started

**Command:** `cd "C:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && source dppo/Scripts/activate && python -m scripts.train_dppo_v33 \
    --pretrained checkpoints/diffusion_policy/v33_20260412_052333/best_model.pt \
    2>&1 | tee logs/train_dppo_v33_run2_20260414.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-14 02:39:21 bash -->
### [Auto-Log] 2026-04-14 02:39:21 — DPPO v3.3 Training — Started

**Command:** `cd "C:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && cat logs/train_dppo_v33_run2_20260414.log 2>&1 | head -30`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-14 02:39:40 bash -->
### [Auto-Log] 2026-04-14 02:39:40 — DPPO v3.3 Training — Started

**Command:** `cd "C:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && wc -c logs/train_dppo_v33_run2_20260414.log && ls checkpoints/diffusion_policy/ | grep dppo_v33 | sort -r | head -3`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-14 14:07:14 bash -->
### [Auto-Log] 2026-04-14 14:07:14 — DPPO v3.3 Training — Started

**Command:** `cd "C:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && wc -c logs/train_dppo_v33_run2_20260414.log && tasklist | grep python`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-14 14:07:20 bash -->
### [Auto-Log] 2026-04-14 14:07:20 — DPPO v3.3 Training — Started

**Command:** `cd "C:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && grep "^Update" logs/train_dppo_v33_run2_20260414.log | tail -10`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-14 14:07:30 bash -->
### [Auto-Log] 2026-04-14 14:07:30 — DPPO v3.3 Training — Started

**Command:** `cd "C:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && grep "^Update" logs/train_dppo_v33_run2_20260414.log | awk 'NR%50==1 || NR<=5' | head -20`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-14 14:07:50 bash -->
### [Auto-Log] 2026-04-14 14:07:50 — DPPO v3.3 Training — Started

**Command:** `cd "C:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && ls -la checkpoints/diffusion_policy/dppo_v33_20260414_023817/ && echo "---" && grep "^Update" logs/train_dppo_v33_run2_20260414.log | awk -F'[| ]+' '{print $2, $6, $NF}' | awk '{if ($2+0 > 0.5) print}' | head -20`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-14 14:08:06 bash -->
### [Auto-Log] 2026-04-14 14:08:06 — DPPO v3.3 Training — Started

**Command:** `cd "C:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && grep "^Update" logs/train_dppo_v33_run2_20260414.log | grep -v "WARMUP" | sort -t'=' -k2 -rn | head -10`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-14 14:08:16 bash -->
### [Auto-Log] 2026-04-14 14:08:16 — DPPO v3.3 Training — Started

**Command:** `cd "C:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && python -c "
import re
with open('logs/train_dppo_v33_run2_20260414.log') as f:
    lines = f.readlines()

updates = []
for l in lines:
    m = re.match(r'Update\s+(\d+)/500.*Reward:\s*([\d.]+).*VLoss:\s*([\d.]+)', l)
    if m:
        u, r, v = int(m.group(1)), float(m.group(2)), float(m.group(3))
        updates.append((u, r, v))

if updates:
    print(f'Total updates logged: {len(updates)}')
    print(f'Latest update: {updates[-1][0]}')
    best = max(updates, key=lambda x: x[1])
    print(f'Best reward: u{best[0]} reward={best[1]:.4f} VLoss={best[2]:.2f}')
    print()
    print('Reward progression (every 25 updates):')
    for u, r, v in updates[::25]:
        print(f'  u{u:3d}: reward={r:.4f}  VLoss={v:.1f}')
    print(f'  u{updates[-1][0]:3d}: reward={updates[-1][1]:.4f}  VLoss={updates[-1][2]:.1f}')
"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-14 15:57:18 bash -->
### [Auto-Log] 2026-04-14 15:57:18 — DPPO v3.3 Training — Started

**Command:** `cd "C:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && python -c "
import re
with open('logs/train_dppo_v33_run2_20260414.log') as f:
    lines = f.readlines()
updates = []
for l in lines:
    m = re.match(r'Update\s+(\d+)/500.*Reward:\s*([\d.]+).*VLoss:\s*([\d.]+)', l)
    if m:
        updates.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
best = max(updates, key=lambda x: x[1])
print(f'Total: {len(updates)} updates')
print(f'Best: u{best[0]} reward={best[1]:.4f} VLoss={best[2]:.2f}')
print(f'Final: u{updates[-1][0]} reward={updates[-1][1]:.4f} VLoss={updates[-1][2]:.2f}')
" && ls -la checkpoints/diffusion_policy/dppo_v33_20260414_023817/`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-15 03:43:08 bash -->
### [Auto-Log] 2026-04-15 03:43:08 — RHC v3.3 Evaluation

**Command:** `cd "C:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && source dppo/Scripts/activate && python -m scripts.evaluate_rhc_v33 \
    --diffusion-model checkpoints/diffusion_policy/dppo_v33_20260414_023817/best_dppo_v33_model.pt \
    --ppo-model  checkpoints/ppo_expert/20260401_103107/best_model.pt \
    --ppo-norm   checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz \
    2>&1 | tee logs/eval_rhc_v33_run2_20260414.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-15 08:48:57 bash -->
### [Auto-Log] 2026-04-15 08:48:57 — RHC v3.3 Evaluation

**Command:** `python -c "
print('Inference analysis:')
print('  10-step DDIM total: 74ms')
print('  Per UNet call: ~7.4ms')
print('  Encoder: ~6ms (included in 74ms)')
print('  Estimated OneDP: 7.4 + overhead = 8-10ms total')
print('  Net with encoder: ~13-14ms')
print('  Target <16ms: ACHIEVABLE')

print()
print('predict_action already accepts ddim_steps kwarg (line 449 of v31)')
print('Just needs sample_onestep branch for ddim_steps==1')
print('evaluate_rhc_v33 does NOT currently pass ddim_steps to predict_action')
print('Need: --ddim-steps CLI arg, passed through evaluate_v33 to predict_action')
" 2>&1`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-15 09:00:45 edit -->
### [Auto-Log] 2026-04-15 09:00:45 — Script Fix

**File:** `scripts\evaluate_rhc_v33.py`

**Before:**
```python
def evaluate_v33(env: QuadrotorVisualEnv, base_env: QuadrotorEnv,
                 policy: VisionDPPOv31,
                 n_episodes: int, T_obs: int, T_action: int,
                 device: torch.device) -> Dict:
```

**After:**
```python
def evaluate_v33(env: QuadrotorVisualEnv, base_env: QuadrotorEnv,
                 policy: VisionDPPOv31,
                 n_episodes: int, T_obs: int, T_action: int,
                 device: torch.device, ddim_steps: int = 10) -> Dict:
```

---
<!-- auto-log 2026-04-15 09:00:51 edit -->
### [Auto-Log] 2026-04-15 09:00:51 — Script Fix

**File:** `scripts\evaluate_rhc_v33.py`

**Before:**
```python
t_start = time.perf_counter()
            with torch.no_grad():
                action_seq = policy.predict_action(img_tensor, imu_tensor)
            results['inference_times'].append((time.perf_counter() - t_start) * 1000)
```

**After:**
```python
t_start = time.perf_counter()
            with torch.no_grad():
                action_seq = policy.predict_action(img_tensor, imu_tensor, ddim_steps=ddim_steps)
            results['inference_times'].append((time.perf_counter() - t_start) * 1000)
```

---
<!-- auto-log 2026-04-15 09:00:56 edit -->
### [Auto-Log] 2026-04-15 09:00:56 — Script Fix

**File:** `scripts\evaluate_rhc_v33.py`

**Before:**
```python
diff_results = evaluate_v33(
        visual_env, base_env, policy, args.n_episodes,
        T_obs=vision_cfg['T_obs'],
        T_action=action_cfg['T_action'],
        device=device,
    )
```

**After:**
```python
diff_results = evaluate_v33(
        visual_env, base_env, policy, args.n_episodes,
        T_obs=vision_cfg['T_obs'],
        T_action=action_cfg['T_action'],
        device=device,
        ddim_steps=args.ddim_steps,
    )
```

---
<!-- auto-log 2026-04-15 09:01:02 edit -->
### [Auto-Log] 2026-04-15 09:01:02 — Script Fix

**File:** `scripts\evaluate_rhc_v33.py`

**Before:**
```python
parser.add_argument('--n-episodes',       type=int, default=50)
    parser.add_argument('--output-dir',       type=str,
                        default='evaluation_results/rhc_v33/')
```

**After:**
```python
parser.add_argument('--n-episodes',       type=int, default=50)
    parser.add_argument('--ddim-steps',       type=int, default=10,
                        help='DDIM inference steps (1=OneDP single-step, default=10)')
    parser.add_argument('--output-dir',       type=str,
                        default='evaluation_results/rhc_v33/')
```

---
<!-- auto-log 2026-04-15 09:02:05 write -->
### [Auto-Log] 2026-04-15 09:02:05 — New File: Script Fix

**File:** `scripts\train_onedp_v33.py`

**Content:**
```python
"""
Phase 3d: OneDP Single-Step Distillation (v3.3)

Teacher  : VisionDPPOv31 (DPPO fine-tuned, frozen, 10-step DDIM)
Student  : VisionDPPOv31 (1-step inference, trainable)

Distillation loss:
    L = MSE(x0_student, x0_teacher)
      + lambda_dispersive * L_dispersive
      + lambda_depth * L_depth

Root cause of Phase 3c bottleneck: DDIM 10-step = 74ms >> 20ms control
period (50Hz). 1-step distillation targets ~13ms inference.

Usage:
    python -m scripts.train_onedp_v33 \\
        --teacher checkpoints/diffusion_policy/dppo_v33_20260413_033647/best_dppo_v33_model.pt \\
        --config  configs/diffusion_policy.yaml \\
        2>&1 | tee logs/train_onedp_v33_$(date +%Y%m%d_%H%M%S).log
"""

import os
import sys
import argparse
import yaml
import logging
import numpy as np
import torch
f
... [truncated — 13994 chars total]
```

---
<!-- auto-log 2026-04-16 02:35:12 bash -->
### [Auto-Log] 2026-04-16 02:35:12 — RHC v3.3 Evaluation

**Command:** `powershell -Command "
\$proc = Start-Process \`
    -FilePath 'C:\Users\User\Desktop\DPPO_PID_controller\DPPO_PID_controller\dppo\Scripts\python.exe' \`
    -ArgumentList '-m', 'scripts.evaluate_rhc_v33', '--diffusion-model', 'checkpoints/diffusion_policy/onedp_v33_20260415_134933/best_onedp_model.pt', '--ppo-model', 'checkpoints/ppo_expert/20260401_103107/best_model.pt', '--ppo-norm', 'checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz', '--ddim-steps', '1' \`
    -WorkingDirectory 'C:\Users\User\Desktop\DPPO_PID_controller\DPPO_PID_controller' \`
    -RedirectStandardOutput 'logs\eval_rhc_onedp_v33.log' \`
    -RedirectStandardError 'logs\eval_rhc_onedp_v33_err.log' \`
    -NoNewWindow -PassThru
Write-Host \"PID: \$(\$proc.Id)\"
"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-16 02:48:20 bash -->
### [Auto-Log] 2026-04-16 02:48:20 — RHC v3.3 Evaluation

**Command:** `powershell -Command "
\$logdir = 'C:\Users\User\Desktop\DPPO_PID_controller\DPPO_PID_controller\logs'
\$proc = Start-Process \`
    -FilePath 'C:\Users\User\Desktop\DPPO_PID_controller\DPPO_PID_controller\dppo\Scripts\python.exe' \`
    -ArgumentList '-m', 'scripts.evaluate_rhc_v33', '--diffusion-model', 'checkpoints/diffusion_policy/onedp_v33_20260415_134933/best_onedp_model.pt', '--ppo-model', 'checkpoints/ppo_expert/20260401_103107/best_model.pt', '--ppo-norm', 'checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz', '--ddim-steps', '1' \`
    -WorkingDirectory 'C:\Users\User\Desktop\DPPO_PID_controller\DPPO_PID_controller' \`
    -RedirectStandardOutput \"\$logdir\eval_rhc_onedp_v33_run2.log\" \`
    -RedirectStandardError \"\$logdir\eval_rhc_onedp_v33_run2_err.log\" \`
    -NoNewWindow -PassThru
Write-Host \"PID: \$(\$proc.Id)\"
"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-19 12:43:39 bash -->
### [Auto-Log] 2026-04-19 12:43:39 — DPPO v3.3 Training — Started

**Command:** `find . -name "*.log" -type f | grep -E "(eval_rhc|train_dppo_v33)" | sort -r | head -10`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-19 12:43:43 bash -->
### [Auto-Log] 2026-04-19 12:43:43 — DPPO v3.3 Training — Started

**Command:** `grep -E "Update.*Reward|Peak|Final" /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v33_20260413_033639.log | tail -20`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-19 12:44:05 bash -->
### [Auto-Log] 2026-04-19 12:44:05 — DPPO v3.3 Training — Started

**Command:** `grep "Update.*225\|Best.*reward\|Final" /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v33_run2_20260414.log | head -15`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-29 15:15:28 bash -->
### [Auto-Log] 2026-04-29 15:15:28 — DPPO v3.3 Training — Started

**Command:** `grep -n "compute_gae\|advantages_t = " /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/scripts/train_dppo_v33.py`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-29 15:15:39 bash -->
### [Auto-Log] 2026-04-29 15:15:39 — DPPO v3.3 Training — Started

**Command:** `grep -n "hover_anchor_prob\|pos_end\|initial_pos_range" /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/scripts/train_dppo_v33.py`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-29 15:19:12 bash -->
### [Auto-Log] 2026-04-29 15:19:12 — Expert Data Collection

**Command:** `cd "/c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && grep -A 10 -B 5 "v33\|v3.3" scripts/collect_data.py`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-29 15:19:19 bash -->
### [Auto-Log] 2026-04-29 15:19:19 — Diffusion v3.3 Training — Started

**Command:** `cd "/c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && grep -B 5 -A 10 "collect_data\|expert_demos_v33\|initial_pos_range" scripts/train_diffusion_v33.py | head -80`

**Output:**
```
(empty)
```

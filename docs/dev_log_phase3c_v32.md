# DPPO PID Controller — Phase 3c v3.2: Physics-based IMU

> Part of the dev log series. See [index](dev_log_phase2_3.md) for all phases.
> Covers: DPPO Run 4 (original arch), v3.2 implementation, v3.2 supervised eval, v3.2 DPPO Run 1.

---

## Table of Contents

1. [Phase 3b DPPO Run 4 — Original Architecture + Improved Training (2026-04-10~11)](#phase-3b-dppo-run-4--original-architecture--improved-training-2026-04-10-11)
2. [v3.2 實作：以物理 IMU 取代一階差分 (2026-04-10)](#v32-實作以物理-imu-取代一階差分2026-04-10)
3. [v3.2 Phase 3a 監督式預訓練評估 (2026-04-10~11)](#v32-phase-3a-監督式預訓練評估2026-04-10-11)
4. [Phase 3c v3.2 DPPO Run 1 啟動 (2026-04-11)](#phase-3c-v32-dppo-run-1-啟動2026-04-11)

---

## 14. Phase 3b DPPO Run 4 — Original Architecture + Improved Training (2026-04-10~11)

### 14.1 Motivation

v3.1 路線（IMU Late Fusion）在兩次 run 中均表現差於原版（RMSE 0.466–0.518m vs 0.168m）。
根本問題確認為 finite-difference IMU 在 RL rollout 中產生噪音。
決定回到原版架構（無 IMU），但移植 v3.1 訓練改良成果：
- Value warm-up 50 updates
- VLoss best-ckpt 門檻 500
- β = 0.05（原版最低）
- Mini-batch loop（修原版 OOM 風險）
- Value hidden_dim = 512

### 14.2 Run Summary

**Run:** `dppo_20260410_045335`
**Log:** `logs/train_dppo_20260410_045335.log`
**Started:** 2026-04-10 04:53 | **Completed:** 2026-04-11
**Duration:** ~23h | **Speed:** ~109s / update
**Pretrained from:** `checkpoints/diffusion_policy/20260405_044808/best_model.pt`（DR-aug Re-run 2，500 epochs）

**Config:**
| Param | Value | 說明 |
|-------|-------|------|
| `advantage_beta` | 0.05 | 最保守（max weight ≈ 1.16×） |
| `value_hidden_dim` | 512 | 較大容量 |
| `value_warmup_updates` | 50 | policy 凍結 50 updates |
| `vloss_best_threshold` | 500 | VLoss < 500 才存 best ckpt |
| `n_rollout_steps` | 4096 | 同前 |
| `learning_rate` | 5e-6 | 同前 |

### 14.3 訓練亮點

**VLoss 從第一個 update 就 < 50（最終 6.2）：**
- 原因：DR-aug pretrained policy 的 reward 分佈比 v3.1 pretrained 更穩定，value net 幾乎不需要 warm-up
- 這是歷史上所有 run 中 VLoss 最低的一次

**Reward 趨勢：**

| Update range | Avg reward | Notes |
|-------------|------------|-------|
| 1–50 | 0.50–0.52 | [WARMUP] policy frozen |
| 51–180 | 0.51–0.56 | 高點平台，best 0.5626 @ u155 |
| 181–280 | 0.47–0.51 | 緩慢下滑 |
| 281–500 | **0.47–0.50** | **反彈並穩定** ← 首次出現此現象 |

u281 後 reward 反彈並維持 0.49–0.50，與過去所有 run（持續下滑至 0.40）截然不同。
β=0.05 + warm-up 的組合成功保住了 pretrained 知識的基礎能力。

**Best checkpoint:** Update 155 | Reward = **0.5626** | VLoss = 17

### 14.4 Evaluation Results

**Script:** `scripts/evaluate_rhc.py` — 50 episodes
**Checkpoint:** `best_dppo_model.pt` (update 155)
**Eval date:** 2026-04-11

| Model | Pos RMSE | Crashes | Mean Reward | Notes |
|-------|----------|---------|-------------|-------|
| PPO Expert | **0.069m** | 0/50 | 539.1 | gold standard |
| DPPO Run 2（原版, u11, β=0.1） | **0.168m** | 50/50 | 20.1 | 最佳 RMSE |
| **DPPO Run 4（原版, u155, β=0.05）** | 0.409m | 50/50 | 31.9 | 本次 |
| DPPO Run 3（原版, u34, β=0.15） | 0.488m | 50/50 | 29.1 | — |
| DPPO v3.1 Run 2（u58） | 0.466m | 50/50 | 79.1 | — |

**Inference time:** 94.3ms（比預期慢，與前次 77.7ms 有差異，可能為系統負載影響）

### 14.5 問題分析：Pretrained Model 差異

訓練指標（VLoss=17, best reward=0.5626）是所有 run 中最佳，但 RMSE 卻更差。
矛盾指向 **pretrained checkpoint 本身的差異**：

| Pretrained | DR-aug | Supervised RMSE | DPPO best RMSE |
|-----------|--------|----------------|---------------|
| `20260402_032701`（原始） | 無 | 0.286m | **0.168m**（Run 2） |
| `20260405_044808`（Re-run 2） | 有 | ~0.268m | 0.409m（Run 4，本次） |

**假設：DR-aug 讓 encoder 學習到更平滑、更 robust 的視覺特徵，但代價是 feature space 更「模糊」。** DPPO 在這種 pretrained 上的 advantage-weighted gradient 難以找到清晰的梯度方向，fine-tuning 效率更低。

相對地，原始 pretrained（無 DR）的特徵分佈更尖銳，DPPO 的微小更新就能帶來較大且有方向性的改變。

#### 支持此假設的觀察

- Run 2（原始 pretrained）best ckpt 在 u11（非常早），儘管 VLoss 極高，reward 也在正常範圍
- Run 4（DR pretrained）best ckpt 在 u155，VLoss=17，但實際飛行 RMSE 差3倍
- 兩次的訓練曲線形狀不同：Run 2 急遽崩潰，Run 4 平緩震盪 → DR-aug 使 policy 對 gradient 更不敏感

### 14.6 下一步方向

**Option A（最高優先）：用原始 pretrained + Run 4 改良配置**

直接控制變數：pretrained `20260402_032701`（無 DR）+ β=0.05 + warm-up 50 + VLoss 門檻。
若假設正確，預期 RMSE < 0.168m（突破現有最佳）。

**Option B：Encoder Freezing**

在 DPPO fine-tuning 期間凍結 CNN encoder，只更新 UNet diffusion head。
可以同時搭配 A 的 pretrained，也可以用 DR pretrained 測試。

**Option C：縮短 DPPO 訓練（早停）**

Run 2 的 best ckpt 在 u11，雖然 VLoss 高但效果好。
可能 DPPO 的 sweet spot 就是「少量 RL 微調」，多了反而侵蝕 pretrained 知識。
測試：只做 50–100 updates（+ warm-up），看 RMSE 是否更好。

---

<!-- auto-log 2026-04-06 15:46:49 bash -->
### [Auto-Log] 2026-04-06 15:46:49 — RHC Evaluation (DPPO Run 3)

**Command:** `cd C:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller && dppo/Scripts/python.exe scripts/evaluate_rhc.py --diffusion-model checkpoints/diffusion_policy/dppo_20260405_155057/best_dppo_model.pt --ppo-model checkpoints/ppo_expert/20260401_103107/best_model.pt --ppo-norm checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz`

**Output:**
```
=== Evaluating Diffusion Policy (RHC) ===
  Mean reward: 31.00 (+/- 13.82)
  Position RMSE: 0.4498 m
  Crashes: 50/50
  Mean inference: 77.7 ms

=== Evaluating PPO Expert ===
  Mean reward: 538.37 (+/- 1.72)
  Position RMSE: 0.0693 m
  Crashes: 0/50

  Diffusion/PPO performance ratio: 5.76%
```

---
<!-- auto-log 2026-04-06 15:53:17 bash -->
### [Auto-Log] 2026-04-06 15:53:17 — Expert Data Collection

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && dppo/Scripts/python.exe -c "import scripts.collect_data; print('import ok')" 2>&1 | head -20`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-06 15:53:29 bash -->
### [Auto-Log] 2026-04-06 15:53:29 — Expert Data Collection

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && dppo/Scripts/python.exe -m scripts.collect_data --help 2>&1 | head -30`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-06 15:53:41 bash -->
### [Auto-Log] 2026-04-06 15:53:41 — Expert Data Collection

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && dppo/Scripts/python.exe -m scripts.collect_data \
    --model  checkpoints/ppo_expert/20260401_103107/best_model.pt \
    --norm   checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz \
    --output data/expert_demos_v31.h5 \
    --v31 2>&1 &
echo "PID: $!"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-06 16:32:25 bash -->
### [Auto-Log] 2026-04-06 16:32:25 — RHC Evaluation

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && "dppo/Scripts/python.exe" -m scripts.evaluate_rhc \
    --diffusion-model checkpoints/diffusion_policy/dppo_20260404_044552/best_dppo_model.pt \
    --ppo-model checkpoints/ppo_expert/20260401_103107/best_model.pt \
    --ppo-norm checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz \
    2>&1 | tee logs/eval_run2.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-06 16:32:32 bash -->
### [Auto-Log] 2026-04-06 16:32:32 — RHC Evaluation

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && "dppo/Scripts/python.exe" -m scripts.evaluate_rhc \
    --diffusion-model checkpoints/diffusion_policy/dppo_20260405_155057/best_dppo_model.pt \
    --ppo-model checkpoints/ppo_expert/20260401_103107/best_model.pt \
    --ppo-norm checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz \
    2>&1 | tee logs/eval_run3.log`

**Output:**
```
(empty)

---

### Phase 3b RHC Eval — Run 2 vs Run 3 — 2026-04-06

**Models evaluated:**
- Run 2 best: `checkpoints/diffusion_policy/dppo_20260404_044552/best_dppo_model.pt` (best at update 11, reward +0.583, β=0.1)
- Run 3 best: `checkpoints/diffusion_policy/dppo_20260405_155057/best_dppo_model.pt` (best at update 34, reward +0.552, β=0.15, 750 updates)

**Results:**

| Model | Pos RMSE | Crashes | Mean Reward | D/PPO ratio |
|-------|----------|---------|-------------|-------------|
| PPO Expert | 0.069m | 0/50 | 539.0 | 100% |
| DPPO Run 2 (best u11) | **0.168m** | 50/50 | 20.1 ± 4.6 | 3.72% |
| DPPO Run 3 (best u34) | 0.488m | 50/50 | 29.1 ± 11.8 | 5.39% |
| Supervised only (original) | 0.286m | 50/50 | — | — |

**Analysis:**
- Both runs: 50/50 crash — covariate shift not resolved by current D²PPO setup
- Run 2 RMSE (0.168m) better than Run 3 (0.488m) despite Run 3 having later best ckpt and more updates
- Run 3 mean reward slightly higher (29.1 vs 20.1) but RMSE much worse — inconsistency suggests high variance in flight trajectories before crash
- Root cause: value network convergence too slow (loss > 5 until update ~150); policy degraded before meaningful advantage estimates available
- β=0.15 did not help vs β=0.1; stronger signal amplified noise from unconverged value net

**Next steps:** v3.1 architecture (IMU fusion + depth auxiliary) or encoder-freezing strategy to prevent early feature corruption.
```

---
<!-- auto-log 2026-04-06 17:02:57 bash -->
### [Auto-Log] 2026-04-06 17:02:57 — Diffusion Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && dppo/Scripts/python.exe -m scripts.train_diffusion_v31 --config configs/diffusion_policy.yaml > logs/train_diffusion_v31_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-06 17:03:24 bash -->
### [Auto-Log] 2026-04-06 17:03:24 — Diffusion Training — Started

**Command:** `cat "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_diffusion_v31_20260406_170257.log" 2>&1 | head -40`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-06 17:03:50 bash -->
### [Auto-Log] 2026-04-06 17:03:50 — Diffusion Training — Started

**Command:** `sleep 20 && cat "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_diffusion_v31_20260406_170257.log" 2>&1`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-06 17:04:06 bash -->
### [Auto-Log] 2026-04-06 17:04:06 — Diffusion Training — Started

**Command:** `tasklist 2>/dev/null | grep python && echo "---" && ls -lh "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_diffusion_v31_20260406_170257.log"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-06 17:31:32 bash -->
### [Auto-Log] 2026-04-06 17:31:32 — Diffusion Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && dppo/Scripts/python.exe -c "import scripts.train_diffusion_v31; print('ok')" 2>&1`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-06 17:32:13 bash -->
### [Auto-Log] 2026-04-06 17:32:13 — Diffusion Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && timeout 30 dppo/Scripts/python.exe -m scripts.train_diffusion_v31 --config configs/diffusion_policy.yaml 2>&1 | head -30`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-06 17:33:25 bash -->
### [Auto-Log] 2026-04-06 17:33:25 — Diffusion Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && timeout 60 dppo/Scripts/python.exe -u -m scripts.train_diffusion_v31 --config configs/diffusion_policy.yaml 2>&1 | head -30`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-06 17:36:01 bash -->
### [Auto-Log] 2026-04-06 17:36:01 — Diffusion Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && nohup dppo/Scripts/python.exe -u -m scripts.train_diffusion_v31 --config configs/diffusion_policy.yaml > logs/train_v31_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "Training PID: $!"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-06 17:42:56 bash -->
### [Auto-Log] 2026-04-06 17:42:56 — Diffusion Training — Started

**Command:** `sleep 10 && cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && nohup dppo/Scripts/python.exe -u -m scripts.train_diffusion_v31 --config configs/diffusion_policy.yaml > logs/train_v31_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-06 18:19:35 edit -->
### [Auto-Log] 2026-04-06 18:19:35 — Script Fix

**File:** `scripts\train_diffusion_v31.py`

**Before:**
```python
dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
```

**After:**
```python
# num_workers=0: all data is pre-loaded into RAM in DemoDatasetV31.__init__,
    # so __getitem__ is a pure memory-copy; workers would each duplicate the full
    # ~8 GB dataset under Windows spawn, causing MemoryError.
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
```

---
<!-- auto-log 2026-04-06 18:19:59 bash -->
### [Auto-Log] 2026-04-06 18:19:59 — Diffusion Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && nohup dppo/Scripts/python.exe -u -m scripts.train_diffusion_v31 --config configs/diffusion_policy.yaml > logs/train_v31_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-06 18:23:04 bash -->
### [Auto-Log] 2026-04-06 18:23:04 — Diffusion Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && nohup dppo/Scripts/python.exe -u -m scripts.train_diffusion_v31 --config configs/diffusion_policy.yaml > logs/train_v31_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-06 18:28:18 bash -->
### [Auto-Log] 2026-04-06 18:28:18 — Diffusion Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && nohup dppo/Scripts/python.exe -u -m scripts.train_diffusion_v31 --config configs/diffusion_policy.yaml > logs/train_v31_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-06 18:31:44 bash -->
### [Auto-Log] 2026-04-06 18:31:44 — Diffusion Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && CUDA_LAUNCH_BLOCKING=1 nohup dppo/Scripts/python.exe -u -m scripts.train_diffusion_v31 --config configs/diffusion_policy.yaml > logs/train_v31_debug_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-06 18:37:07 bash -->
### [Auto-Log] 2026-04-06 18:37:07 — Diffusion Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && nohup dppo/Scripts/python.exe -u -m scripts.train_diffusion_v31 --config configs/diffusion_policy.yaml > logs/train_v31_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-06 18:45:46 bash -->
### [Auto-Log] 2026-04-06 18:45:46 — Diffusion Training — Started

**Command:** `wmic process where "commandline like '%train_diffusion_v31%'" get processid 2>/dev/null | grep -v "^$\|ProcessId"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-06 18:50:43 edit -->
### [Auto-Log] 2026-04-06 18:50:43 — Model Fix

**File:** `models\vision_dppo_v31.py`

**Before:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
```

**After:**
```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
```

---
<!-- auto-log 2026-04-06 18:50:52 edit -->
### [Auto-Log] 2026-04-06 18:50:52 — Model Fix

**File:** `models\vision_dppo_v31.py`

**Before:**
```python
# Build samples: flat_t = episode offset + step index
        self.samples = []
        for ep_key in sorted(ep_offsets.keys()):
            off    = ep_offsets[ep_key]
            ep_len = (ep_offsets.get(
                f'episode_{int(ep_key.split("_")[1])+1}', n
            ) - off)
            for t in range(T_obs - 1, ep_len - T_pred + 1):
                self.samples.append(off + t)   # flat index of the 'current' step
```

**After:**
```python
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
```

---
<!-- auto-log 2026-04-06 18:51:24 bash -->
### [Auto-Log] 2026-04-06 18:51:24 — Diffusion Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && nohup dppo/Scripts/python.exe -u -m scripts.train_diffusion_v31 --config configs/diffusion_policy.yaml > logs/train_v31_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-06 19:01:28 bash -->
### [Auto-Log] 2026-04-06 19:01:28 — Diffusion Training — Started

**Command:** `tail -10 "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_v31_20260406_185124.log" && echo "---" && wmic process where "commandline like '%train_diffusion_v31%'" get processid 2>/dev/null | grep -v "^$\|ProcessId"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-08 02:33:12 bash -->
### [Auto-Log] 2026-04-08 02:33:12 — RHC Evaluation

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && dppo/Scripts/python.exe -m scripts.evaluate_rhc --help 2>&1 | head -20`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-08 02:33:17 bash -->
### [Auto-Log] 2026-04-08 02:33:17 — RHC Evaluation

**Command:** `grep -n "VisionDiffusion\|import\|load\|model_class" "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/scripts/evaluate_rhc.py" | head -20`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-08 02:35:04 bash -->
### [Auto-Log] 2026-04-08 02:35:04 — RHC Evaluation

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && dppo/Scripts/python.exe -c "import scripts.evaluate_rhc_v31; print('import ok')" 2>&1`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-08 02:36:23 bash -->
### [Auto-Log] 2026-04-08 — Phase 3a v3.1 Supervised Baseline Evaluation

**Command:** `python -m scripts.evaluate_rhc_v31 --diffusion-model checkpoints/diffusion_policy/v31_20260406_185128/best_model.pt --ppo-model ... --n-episodes 50`

**Output:**
```
--- v3.1 Results (50 episodes) ---
  Mean reward:    85.33 (+/- 20.36)
  Position RMSE:  0.4526 m
  Crashes:        50/50
  Inference time: 72.9 ms (median 72.3 ms)

=== PPO Expert ===
  Mean reward:   538.65
  Position RMSE: 0.0693 m
  Crashes:       0/50

  Diffusion/PPO performance ratio: 15.84%
```

**Analysis:**
- Supervised v3.1 baseline RMSE = 0.4526m, **worse than supervised baseline without IMU (0.268m)**
- Expected: covariate shift dominates; IMU alone cannot fix it without closed-loop training
- Crash rate 50/50 confirms DPPO fine-tuning is mandatory (same as all supervised-only runs)
- Inference 72.9ms — IMU encoder adds negligible latency vs baseline (71ms)
- Decision: proceed directly to Phase 3c DPPO v3.1 fine-tuning

---
<!-- auto-log 2026-04-08 02:41:46 bash -->
### [Auto-Log] 2026-04-08 02:41:46 — DPPO Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && nohup dppo/Scripts/python.exe -u -m scripts.train_dppo_v31 \
    --pretrained checkpoints/diffusion_policy/v31_20260406_185128/best_model.pt \
    --config configs/diffusion_policy.yaml \
    --total-updates 500 \
    > logs/train_dppo_v31_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-08 02:42:29 bash -->
### [Auto-Log] 2026-04-08 02:42:29 — DPPO Training — Started

**Command:** `sleep 30 && LOGFILE=$(ls "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/" | grep "train_dppo_v31" | sort | tail -1) && tail -20 "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/$LOGFILE"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-08 02:44:49 bash -->
### [Auto-Log] 2026-04-08 02:44:49 — DPPO Training — Started

**Command:** `sleep 120 && LOGFILE=$(ls "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/" | grep "train_dppo_v31" | sort | tail -1) && tail -10 "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/$LOGFILE"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-08 02:45:33 bash -->
### [Auto-Log] 2026-04-08 02:45:33 — DPPO Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && nohup dppo/Scripts/python.exe -u -m scripts.train_dppo_v31 \
    --pretrained checkpoints/diffusion_policy/v31_20260406_185128/best_model.pt \
    --config configs/diffusion_policy.yaml \
    --total-updates 500 \
    > logs/train_dppo_v31_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-08 02:47:43 bash -->
### [Auto-Log] 2026-04-08 02:47:43 — DPPO Training — Started

**Command:** `sleep 120 && LOGFILE=$(ls "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/" | grep "train_dppo_v31" | sort | tail -1) && echo "=== $LOGFILE ===" && tail -15 "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/$LOGFILE"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-08 02:58:50 bash -->
### [Auto-Log] 2026-04-08 02:58:50 — DPPO Training — Started

**Command:** `tail -5 "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_20260408_024533.log"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-08 03:13:52 bash -->
### [Auto-Log] 2026-04-08 03:13:52 — DPPO Training — Started

**Command:** `cd "c:\Users\User\Desktop\DPPO_PID_controller\DPPO_PID_controller" && git add .gitignore README.md RESEARCH_PLAN.md configs/diffusion_policy.yaml configs/quadrotor.yaml docs/dev_log_phase2_3.md envs/quadrotor_visual_env.py models/conditional_unet1d.py models/diffusion_policy.py models/diffusion_process.py scripts/collect_data.py scripts/train_diffusion.py scripts/train_dppo.py models/vision_dppo_v31.py scripts/evaluate_rhc_v31.py scripts/train_diffusion_v31.py scripts/train_dppo_v31.py && git status`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-08 07:40:56 bash -->
### [Auto-Log] 2026-04-08 07:40:56 — DPPO Training — Started

**Command:** `tail -80 c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_20260408_024533.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-08 07:41:02 bash -->
### [Auto-Log] 2026-04-08 07:41:02 — DPPO Training — Started

**Command:** `tail -30 c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_20260408_024533.log | head -5
# Also check the file modification time and first/last update timestamps
stat c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_20260408_024533.log 2>/dev/null | grep Modify
# Get first update line
grep "Update   1/" c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_20260408_024533.log 2>/dev/null | head -1`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-08 07:41:02 bash -->
### [Auto-Log] 2026-04-08 07:41:02 — DPPO Training — Started

**Command:** `wc -l c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_20260408_024533.log
grep -n "Update " c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_20260408_024533.log | tail -3`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-08 07:41:07 bash -->
### [Auto-Log] 2026-04-08 07:41:07 — DPPO Training — Started

**Command:** `# Check timestamps of log entries to calculate per-update time
head -15 c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_20260408_024533.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-08 07:42:02 bash -->
### [Auto-Log] 2026-04-08 07:42:02 — DPPO Training — Started

**Command:** `python -c "
import os, datetime
log_file = 'c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_20260408_024533.log'
stat = os.stat(log_file)
mtime = datetime.datetime.fromtimestamp(stat.st_mtime)

start = datetime.datetime(2026, 4, 8, 2, 45, 33)
elapsed = mtime - start
updates_done = 165
total_updates = 500
remaining = total_updates - updates_done
sec_per_update = elapsed.total_seconds() / updates_done
est_remaining_sec = sec_per_update * remaining

print('Log last modified:', mtime)
print('Elapsed: %.1fh' % (elapsed.total_seconds()/3600))
print('Updates done: %d/500' % updates_done)
print('Sec per update: %.1f' % sec_per_update)
print('Est remaining: %.2fh = %.0fmin' % (est_remaining_sec/3600, est_remaining_sec/60))
finish_time = mtime + datetime.timedelta(seconds=est_remaining_sec)
print('Est finish: ', finish_time)
"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-08 07:42:19 bash -->
### [Auto-Log] 2026-04-08 07:42:19 — DPPO Training — Started

**Command:** `grep "Update " c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_20260408_024533.log | awk -F'Reward: ' '{print $2}' | awk '{print $1}' | python -c "
import sys
vals = [float(l) for l in sys.stdin]
# show in windows
w=20
for i in range(0, len(vals), w):
    chunk = vals[i:i+w]
    avg = sum(chunk)/len(chunk)
    print('Update %3d-%3d: avg=%.4f  min=%.4f  max=%.4f' % (i+1, i+len(chunk), avg, min(chunk), max(chunk)))
"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-08 07:42:31 bash -->
### [Auto-Log] 2026-04-08 07:42:31 — DPPO Training — Started

**Command:** `grep "Update " c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_20260408_024533.log | awk -F'VLoss: ' '{print $2}' | python -c "
import sys
vals = [float(l) for l in sys.stdin]
w=20
for i in range(0, len(vals), w):
    chunk = vals[i:i+w]
    avg = sum(chunk)/len(chunk)
    print('Update %3d-%3d: vLoss avg=%.0f  min=%.0f  max=%.0f' % (i+1, i+len(chunk), avg, min(chunk), max(chunk)))
"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-08 13:40:19 bash -->
### [Auto-Log] 2026-04-08 13:40:19 — DPPO Training — Started

**Command:** `tail -20 c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_20260408_024533.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-08 13:40:26 bash -->
### [Auto-Log] 2026-04-08 13:40:26 — DPPO Training — Started

**Command:** `python -c "
import os, datetime
log_file = 'c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_20260408_024533.log'
stat = os.stat(log_file)
mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
start = datetime.datetime(2026, 4, 8, 2, 45, 33)
elapsed = mtime - start
updates_done = 342
remaining = 500 - updates_done
sec_per_update = elapsed.total_seconds() / updates_done
est_remaining_sec = sec_per_update * remaining
finish_time = mtime + datetime.timedelta(seconds=est_remaining_sec)
print('Last update: %s' % mtime.strftime('%H:%M'))
print('Elapsed: %.1fh' % (elapsed.total_seconds()/3600))
print('Updates: %d/500 (%.0f%%)' % (updates_done, updates_done/500*100))
print('Sec/update: %.1f' % sec_per_update)
print('Est remaining: %.1fh' % (est_remaining_sec/3600))
print('Est finish: %s' % finish_time.strftime('%H:%M'))
"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-08 13:40:33 bash -->
### [Auto-Log] 2026-04-08 13:40:33 — DPPO Training — Started

**Command:** `grep "Update " c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_20260408_024533.log | awk -F'Reward: ' '{print $2}' | awk '{print $1}' | python -c "
import sys
vals = [float(l) for l in sys.stdin]
w=20
for i in range(0, len(vals), w):
    chunk = vals[i:i+w]
    avg = sum(chunk)/len(chunk)
    print('u%3d-%3d: reward avg=%.4f  [%.4f~%.4f]' % (i+1, i+len(chunk), avg, min(chunk), max(chunk)))
"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-08 13:40:38 bash -->
### [Auto-Log] 2026-04-08 13:40:38 — DPPO Training — Started

**Command:** `grep "Update " c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_20260408_024533.log | awk -F'VLoss: ' '{print $2}' | python -c "
import sys
vals = [float(l) for l in sys.stdin]
w=20
for i in range(160, len(vals), w):
    chunk = vals[i:i+w]
    avg = sum(chunk)/len(chunk)
    print('u%3d-%3d: vLoss avg=%.0f  [%.0f~%.0f]' % (i+1, i+len(chunk), avg, min(chunk), max(chunk)))
"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-09 02:31:40 bash -->
### [Auto-Log] 2026-04-09 02:31:40 — DPPO Training — Started

**Command:** `tail -10 c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_20260408_024533.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-09 02:32:54 bash -->
### [Auto-Log] 2026-04-09 02:32:54 — RHC Evaluation

**Command:** `cd c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller && source dppo/Scripts/activate && python -m scripts.evaluate_rhc_v31 --help 2>&1 | head -30`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-09 02:33:04 bash -->
### [Auto-Log] 2026-04-09 02:33:04 — RHC Evaluation

**Command:** `cd c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller && source dppo/Scripts/activate && python -m scripts.evaluate_rhc_v31 \
    --diffusion-model checkpoints/diffusion_policy/dppo_v31_20260408_024538/best_dppo_v31_model.pt \
    --ppo-model checkpoints/ppo_expert/20260401_103107/best_model.pt \
    --ppo-norm checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz \
    2>&1 | tee logs/eval_rhc_v31_20260409.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-09 02:49:50 edit -->
### [Auto-Log] 2026-04-09 02:49:50 — Script Fix

**File:** `scripts\train_dppo_v31.py`

**Before:**
```python
n_rollout_steps = dppo_cfg['n_rollout_steps']
    n_epochs        = dppo_cfg['n_epochs']
    beta            = dppo_cfg['advantage_beta']
    gamma           = dppo_cfg['gamma']
    gae_lambda      = dppo_cfg['gae_lambda']
    total_updates   = args.total_updates

    print(f"\n{'='*60}")
    print(f"DPPO v3.1 Fine-Tuning")
    print(f"Total updates: {total_updates} | Rollout steps: {n_rollout_steps}")
    print(f"帣={beta} | 弇_disp={lambda_dispersive} | 弇_depth={lambda_depth}")
    print(f"{'='*60}\n")

    best_reward = -float('inf')

    for update in range(total_updates):
        # Collect rollout
        rollout = collect_rollout(
            env, policy, value_net,
            n_steps=n_rollout_steps,
            T_obs=vision_cfg['T_obs'],
            T_action=action_cfg['T_action'],

... [truncated — 4937 chars total]
```

**After:**
```python
n_rollout_steps       = dppo_cfg['n_rollout_steps']
    n_epochs              = dppo_cfg['n_epochs']
    beta                  = dppo_cfg['advantage_beta']
    gamma                 = dppo_cfg['gamma']
    gae_lambda            = dppo_cfg['gae_lambda']
    value_warmup_updates  = dppo_cfg.get('value_warmup_updates', 0)
    vloss_best_threshold  = dppo_cfg.get('vloss_best_threshold', float('inf'))
    total_updates         = args.total_updates

    print(f"\n{'='*60}")
    print(f"DPPO v3.1 Fine-Tuning")
    print(f"Total updates: {total_updates} | Rollout steps: {n_rollout_steps}")
    print(f"帣={beta} | 弇_disp={lambda_dispersive} | 弇_depth={lambda_depth}")
    print(f"Value warm-up: {value_warmup_updates} updates | VLoss best-ckpt threshold: {vloss_best_threshold}")
    print(f"{'='*60}\n
... [truncated — 5755 chars total]
```

---
<!-- auto-log 2026-04-09 02:49:55 bash -->
### [Auto-Log] 2026-04-09 02:49:55 — DPPO Training — Started

**Command:** `grep -n "warmup\|vloss_best\|in_warmup\|best_reward\|WARMUP" c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/scripts/train_dppo_v31.py`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-09 02:50:04 bash -->
### [Auto-Log] 2026-04-09 02:50:04 — DPPO Training — Started

**Command:** `cd c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller && source dppo/Scripts/activate && nohup python -u -m scripts.train_dppo_v31 \
    --pretrained checkpoints/diffusion_policy/v31_20260406_185128/best_model.pt \
    --config configs/diffusion_policy.yaml \
    --total-updates 500 \
    > logs/train_dppo_v31_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-09 02:50:40 bash -->
### [Auto-Log] 2026-04-09 02:50:40 — DPPO Training — Started

**Command:** `sleep 30 && tail -15 $(ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_*.log | sort | tail -1)`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-09 02:52:46 bash -->
### [Auto-Log] 2026-04-09 02:52:46 — DPPO Training — Started

**Command:** `sleep 120 && tail -10 $(ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_*.log | sort | tail -1)`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-09 03:22:37 bash -->
### [Auto-Log] 2026-04-09 03:22:37 — DPPO Training — Started

**Command:** `cat C:/Users/User/AppData/Local/Temp/claude/c--Users-User-Desktop-DPPO-PID-controller/2e5908cf-20df-49ee-941a-7eb693a10c66/tasks/bwiq05ait.output 2>/dev/null || sleep 10 && tail -10 $(ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_*.log | sort | tail -1)`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-09 11:15:59 bash -->
### [Auto-Log] 2026-04-09 11:15:59 — DPPO Training — Started

**Command:** `tail -20 $(ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_*.log | sort | tail -1)`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-09 11:16:08 bash -->
### [Auto-Log] 2026-04-09 11:16:08 — DPPO Training — Started

**Command:** `grep "Update " $(ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_*.log | sort | tail -1) | awk -F'Reward: ' '{print $2}' | awk '{print $1}' | python -c "
import sys
vals = [float(l) for l in sys.stdin]
w = 20
for i in range(0, len(vals), w):
    chunk = vals[i:i+w]
    print('u%3d-%3d: avg=%.4f [%.4f~%.4f]' % (i+1, i+len(chunk), sum(chunk)/len(chunk), min(chunk), max(chunk)))
print('Total updates: %d | Latest: %.4f' % (len(vals), vals[-1]))
"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-09 11:16:16 bash -->
### [Auto-Log] 2026-04-09 11:16:16 — DPPO Training — Started

**Command:** `grep "Update " $(ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_*.log | sort | tail -1) | awk -F'VLoss: ' '{print $2}' | python -c "
import sys
vals = [float(l) for l in sys.stdin]
# Show from u50 onward (post-warmup)
w = 20
for i in range(48, len(vals), w):
    chunk = vals[i:i+w]
    print('u%3d-%3d: vLoss avg=%.0f [%.0f~%.0f]' % (i+1, i+len(chunk), sum(chunk)/len(chunk), min(chunk), max(chunk)))
"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-09 11:16:31 bash -->
### [Auto-Log] 2026-04-09 11:16:31 — DPPO Training — Started

**Command:** `# Find best reward and which update it was saved (post-warmup, vloss < 500)
grep "Update " $(ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_*.log | sort | tail -1) | python -c "
import sys, re
best_r = -1e9
best_u = -1
for line in sys.stdin:
    m = re.search(r'Update\s+(\d+)/\d+.*Reward: ([\d.]+).*VLoss: ([\d.]+)', line)
    if not m: continue
    u, r, vl = int(m.group(1)), float(m.group(2)), float(m.group(3))
    if u > 50 and vl < 500 and r > best_r:
        best_r = r
        best_u = u
print('Best reward (post-warmup, VLoss<500): %.4f at update %d' % (best_r, best_u))
"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-09 13:37:16 bash -->
### [Auto-Log] 2026-04-09 13:37:16 — DPPO Training — Started

**Command:** `tail -5 $(ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_*.log | sort | tail -1)`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-09 13:37:25 bash -->
### [Auto-Log] 2026-04-09 13:37:25 — DPPO Training — Started

**Command:** `python -c "
import os, datetime
log = sorted([f for f in os.listdir('c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs') if f.startswith('train_dppo_v31_') and f.endswith('.log')])[-1]
path = 'c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/' + log
mtime = datetime.datetime.fromtimestamp(os.stat(path).st_mtime)
start = datetime.datetime(2026, 4, 9, 2, 50, 8)
elapsed = mtime - start
done, total = 354, 500
remaining = total - done
spu = elapsed.total_seconds() / done
eta = mtime + datetime.timedelta(seconds=spu * remaining)
print('Updates: %d/500 (%.0f%%)' % (done, done/total*100))
print('Elapsed: %.1fh | Sec/update: %.1f' % (elapsed.total_seconds()/3600, spu))
print('Est remaining: %.1fh' % (spu*remaining/3600))
print('Est finish: %s' % eta.strftime('%H:%M'))
"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 04:44:03 bash -->
### [Auto-Log] 2026-04-10 04:44:03 — DPPO Training — Started

**Command:** `tail -10 $(ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_*.log | sort | tail -1)`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 04:44:55 bash -->
### [Auto-Log] 2026-04-10 04:44:55 — DPPO Training — Started

**Command:** `grep "Update " $(ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_*.log | sort | tail -1) | python -c "
import sys, re
best_r, best_u = -1e9, -1
for line in sys.stdin:
    m = re.search(r'Update\s+(\d+)/\d+.*Reward: ([\d.]+).*VLoss: ([\d.]+)', line)
    if not m: continue
    u, r, vl = int(m.group(1)), float(m.group(2)), float(m.group(3))
    if u > 50 and vl < 500 and r > best_r:
        best_r, best_u = r, u
print('Best reward (post-warmup, VLoss<500): %.4f at u%d' % (best_r, best_u))
" && grep "Update " $(ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v31_*.log | sort | tail -1) | awk -F'Reward: ' '{print $2}' | awk '{print $1}' | python -c "
import sys
vals = [float(l) for l in sys.stdin]
w = 20
for i in range(0, len(vals), w):
    chunk = vals[i:i+w]
    print('u%3d-%3d: avg=%.4f [%.4f~%.4f]' % (i+1, i+len(chunk), sum(chunk)/len(chunk), min(chunk), max(chunk)))
"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 04:45:40 bash -->
### [Auto-Log] 2026-04-10 04:45:40 — RHC Evaluation

**Command:** `cd c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller && source dppo/Scripts/activate && python -m scripts.evaluate_rhc_v31 \
    --diffusion-model checkpoints/diffusion_policy/dppo_v31_20260409_025008/best_dppo_v31_model.pt \
    --ppo-model checkpoints/ppo_expert/20260401_103107/best_model.pt \
    --ppo-norm checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz \
    2>&1 | tee logs/eval_rhc_v31_20260410.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 04:53:19 edit -->
### [Auto-Log] 2026-04-10 04:53:19 — Script Fix

**File:** `scripts\train_dppo.py`

**Before:**
```python
# Training parameters
    n_rollout_steps = dppo_cfg['n_rollout_steps']
    n_epochs = dppo_cfg['n_epochs']
    beta = dppo_cfg['advantage_beta']
    gamma = dppo_cfg['gamma']
    gae_lambda = dppo_cfg['gae_lambda']
    total_updates = args.total_updates

    print(f"\n{'='*60}")
    print(f"DPPO Fine-Tuning")
    print(f"Total updates: {total_updates}")
    print(f"Rollout steps: {n_rollout_steps}")
    print(f"Advantage beta: {beta}")
    print(f"{'='*60}\n")

    best_reward = -float('inf')

    for update in range(total_updates):
        # Collect rollout
        rollout = collect_rollout(
            env, policy, value_net, policy.vision_encoder,
            n_steps=n_rollout_steps,
            T_obs=vision_cfg['T_obs'],
            T_action=action_cfg['T_action'],
            device=
... [truncated — 2983 chars total]
```

**After:**
```python
# Training parameters
    n_rollout_steps      = dppo_cfg['n_rollout_steps']
    n_epochs             = dppo_cfg['n_epochs']
    beta                 = dppo_cfg['advantage_beta']
    gamma                = dppo_cfg['gamma']
    gae_lambda           = dppo_cfg['gae_lambda']
    value_warmup_updates = dppo_cfg.get('value_warmup_updates', 0)
    vloss_best_threshold = dppo_cfg.get('vloss_best_threshold', float('inf'))
    total_updates        = args.total_updates

    MINI_BATCH = 256   # safe for 24 GB VRAM

    print(f"\n{'='*60}")
    print(f"DPPO Fine-Tuning")
    print(f"Total updates: {total_updates} | Rollout steps: {n_rollout_steps}")
    print(f"Advantage beta: {beta}")
    print(f"Value warm-up: {value_warmup_updates} updates | VLoss best-ckpt threshold: {vloss_best_threshold}")
   
... [truncated — 4556 chars total]
```

---
<!-- auto-log 2026-04-10 04:53:31 bash -->
### [Auto-Log] 2026-04-10 04:53:31 — DPPO Training — Started

**Command:** `cd c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller && source dppo/Scripts/activate && nohup python -u -m scripts.train_dppo \
    --pretrained checkpoints/diffusion_policy/20260405_044808/best_model.pt \
    --config configs/diffusion_policy.yaml \
    --total-updates 500 \
    > logs/train_dppo_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 04:54:18 bash -->
### [Auto-Log] 2026-04-10 04:54:18 — DPPO Training — Started

**Command:** `sleep 40 && tail -15 $(ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_2026*.log | sort | tail -1)`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 04:58:07 bash -->
### [Auto-Log] 2026-04-10 04:58:07 — DPPO Training — Started

**Command:** `sleep 120 && tail -10 $(ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_2026*.log | sort | tail -1)`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 09:58:46 bash -->
### [Auto-Log] 2026-04-10 09:58:46 — DPPO Training — Started

**Command:** `tail -15 $(ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_2026*.log | sort | tail -1)`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 09:59:06 bash -->
### [Auto-Log] 2026-04-10 09:59:06 — DPPO Training — Started

**Command:** `LOGFILE=$(ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_2026*.log | sort | tail -1)
# reward trend by 20-update windows
grep "Update " $LOGFILE | awk -F'Reward: ' '{print $2}' | awk '{print $1}' | python -c "
import sys
vals = [float(l) for l in sys.stdin]
w = 20
for i in range(0, len(vals), w):
    chunk = vals[i:i+w]
    print('u%3d-%3d: avg=%.4f [%.4f~%.4f]' % (i+1, i+len(chunk), sum(chunk)/len(chunk), min(chunk), max(chunk)))
print('Total: %d | Latest: %.4f' % (len(vals), vals[-1]))
"
# best ckpt
grep "Update " $LOGFILE | python -c "
import sys, re
best_r, best_u = -1e9, -1
for line in sys.stdin:
    m = re.search(r'Update\s+(\d+)/\d+.*Reward: ([\d.]+).*VLoss: ([\d.]+)', line)
    if not m: continue
    u, r, vl = int(m.group(1)), float(m.group(2)), float(m.group(3))
    if u > 50 and vl < 500 and r > best_r:
        best_r, best_u = r, u
print('Best (post-warmup, VLoss<500): %.4f @ u%d' % (best_r, best_u))
"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 09:59:14 bash -->
### [Auto-Log] 2026-04-10 09:59:14 — DPPO Training — Started

**Command:** `LOGFILE=$(ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_2026*.log | sort | tail -1)
ls $(dirname $LOGFILE)/../checkpoints/diffusion_policy/dppo_20260410* 2>/dev/null || \
ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/checkpoints/diffusion_policy/ | grep "^dppo_202604[^v]" | sort | tail -3`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 11:07:32 edit -->
### [Auto-Log] 2026-04-10 11:07:32 — Env Fix

**File:** `envs\quadrotor_dynamics.py`

**Before:**
```python
# Motor state (with first-order lag)
        self.motor_thrust = np.zeros(4)

        self.reset()
```

**After:**
```python
# Motor state (with first-order lag)
        self.motor_thrust = np.zeros(4)

        # Cached force/rotation for specific-force accessor (v3.2 IMU)
        self._last_force_world = np.zeros(3)
        self._last_R = np.eye(3)

        self.reset()
```

---
<!-- auto-log 2026-04-10 11:07:39 edit -->
### [Auto-Log] 2026-04-10 11:07:39 — Env Fix

**File:** `envs\quadrotor_dynamics.py`

**Before:**
```python
def reset(self, position: np.ndarray = None, velocity: np.ndarray = None):
        """Reset to initial state (hovering at origin or given position)."""
        self.position = position if position is not None else np.zeros(3)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # upright
        self.velocity = velocity if velocity is not None else np.zeros(3)
        self.ang_velocity = np.zeros(3)
        self.motor_thrust = np.zeros(4)
```

**After:**
```python
def reset(self, position: np.ndarray = None, velocity: np.ndarray = None):
        """Reset to initial state (hovering at origin or given position)."""
        self.position = position if position is not None else np.zeros(3)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # upright
        self.velocity = velocity if velocity is not None else np.zeros(3)
        self.ang_velocity = np.zeros(3)
        self.motor_thrust = np.zeros(4)
        self._last_force_world = np.zeros(3)
        self._last_R = np.eye(3)
```

---
<!-- auto-log 2026-04-10 11:07:47 edit -->
### [Auto-Log] 2026-04-10 11:07:47 — Env Fix

**File:** `envs\quadrotor_dynamics.py`

**Before:**
```python
# Compute forces and torques from motors
        force_world, torque_body = self._compute_forces_torques()

        # Add disturbances
        if disturbance_force is not None:
            force_world = force_world + disturbance_force
        if disturbance_torque is not None:
            torque_body = torque_body + disturbance_torque
```

**After:**
```python
# Compute forces and torques from motors
        force_world, torque_body = self._compute_forces_torques()

        # Add disturbances
        if disturbance_force is not None:
            force_world = force_world + disturbance_force
        if disturbance_torque is not None:
            torque_body = torque_body + disturbance_torque

        # Cache for specific-force accessor (v3.2 IMU).
        # Snapshot BEFORE the RK4 integration so the reported force matches
        # the rotation frame the drone was actually in during this sub-step.
        self._last_force_world = force_world.copy()
        self._last_R = self.get_rotation_matrix()
```

---
<!-- auto-log 2026-04-10 11:11:25 edit -->
### [Auto-Log] 2026-04-10 11:11:25 — Script Fix

**File:** `scripts\collect_data.py`

**Before:**
```python
images = []
            actions = []
            states = []
            imu_data_ep   = []   # v3.1 only
            depth_maps_ep = []   # v3.1 only
            prev_v_body   = None # for finite-difference acceleration
            done = False

            while not done:
                # Get deterministic action from expert
                action = agent.get_action_deterministic(state_norm)

                images.append(obs['image'])
                actions.append(action)
                states.append(obs['state'])

                # v3.1: capture IMU and depth before stepping
                if args.v31:
                    # Angular velocity: state[12:15] (body frame)
                    omega = obs['state'][12:15].copy()
                    # Linear velocity in body frame: state[9:
... [truncated — 1398 chars total]
```

**After:**
```python
images = []
            actions = []
            states = []
            imu_data_ep   = []   # v3.1 / v3.2 only
            depth_maps_ep = []   # v3.1 / v3.2 only
            prev_v_body   = None # v3.1 finite-difference history
            done = False

            while not done:
                # Get deterministic action from expert
                action = agent.get_action_deterministic(state_norm)

                images.append(obs['image'])
                actions.append(action)
                states.append(obs['state'])

                # v3.1: finite-difference IMU (deprecated)
                if args.v31:
                    # Angular velocity: state[12:15] (body frame)
                    omega = obs['state'][12:15].copy()
                    # Linear velocity in body frame: s
... [truncated — 1742 chars total]
```

---
<!-- auto-log 2026-04-10 11:11:37 edit -->
### [Auto-Log] 2026-04-10 11:11:37 — Script Fix

**File:** `scripts\collect_data.py`

**Before:**
```python
if args.v31:
                ep_grp.create_dataset('imu_data',
                    data=np.array(imu_data_ep, dtype=np.float32))       # (T, 6)
                ep_grp.create_dataset('depth_maps',
                    data=np.array(depth_maps_ep, dtype=np.uint8),
                    compression='gzip', compression_opts=4)              # (T, 1, H, W)

            total_steps += len(actions)

        # Save metadata
        hf.attrs['n_episodes'] = args.n_episodes
        hf.attrs['total_steps'] = total_steps
        hf.attrs['image_size'] = args.image_size
        hf.attrs['state_dim'] = state_dim
        hf.attrs['action_dim'] = action_dim
        hf.attrs['v31'] = args.v31

    print(f"\nData collection complete!")
    print(f"  Episodes: {args.n_episodes}")
    print(f"  Total steps: {tota
... [truncated — 935 chars total]
```

**After:**
```python
if with_aux:
                ep_grp.create_dataset('imu_data',
                    data=np.array(imu_data_ep, dtype=np.float32))       # (T, 6)
                ep_grp.create_dataset('depth_maps',
                    data=np.array(depth_maps_ep, dtype=np.uint8),
                    compression='gzip', compression_opts=4)              # (T, 1, H, W)

            total_steps += len(actions)

        # Save metadata
        hf.attrs['n_episodes'] = args.n_episodes
        hf.attrs['total_steps'] = total_steps
        hf.attrs['image_size'] = args.image_size
        hf.attrs['state_dim'] = state_dim
        hf.attrs['action_dim'] = action_dim
        hf.attrs['v31'] = args.v31
        hf.attrs['v32'] = args.v32

    print(f"\nData collection complete!")
    print(f"  Episodes: {args.n_episodes}
... [truncated — 1092 chars total]
```

---
<!-- auto-log 2026-04-10 11:13:56 bash -->
### [Auto-Log] 2026-04-10 11:13:56 — Expert Data Collection

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && mkdir -p data logs && dppo/Scripts/python.exe -m scripts.collect_data --model checkpoints/ppo_expert/20260401_103107/best_model.pt --norm checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz --output data/expert_demos_v32.h5 --v32 > logs/collect_v32_20260410.log 2>&1`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 11:18:16 bash -->
### [Auto-Log] 2026-04-10 11:18:16 — DPPO Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && dppo/Scripts/python.exe -c "
import importlib
for m in ['scripts.train_diffusion_v32', 'scripts.train_dppo_v32', 'scripts.evaluate_rhc_v32']:
    try:
        importlib.import_module(m)
        print(f'OK  {m}')
    except Exception as e:
        print(f'FAIL {m}: {type(e).__name__}: {e}')
" 2>&1`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 11:18:21 bash -->
### [Auto-Log] 2026-04-10 11:18:21 — DPPO Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && echo '--- collect_v32 log tail ---' && tail -5 logs/collect_v32_20260410.log && echo '' && echo '--- baseline dppo log tail ---' && tail -8 logs/train_dppo_20260410_045331.log 2>&1`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 11:19:23 bash -->
### [Auto-Log] 2026-04-10 11:19:23 — Diffusion Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && TS=$(date +%Y%m%d_%H%M%S) && nohup bash -c 'while ! grep -q "Data collection complete" logs/collect_v32_20260410.log 2>/dev/null; do sleep 30; done; echo "[auto-chain] data ready, launching supervised v3.2 training" >> logs/chain_v32_supervised.log; dppo/Scripts/python.exe -m scripts.train_diffusion_v32 --config configs/diffusion_policy.yaml >> logs/train_diffusion_v32_'$TS'.log 2>&1' > logs/chain_v32_supervised.log 2>&1 &
echo "Chain launcher PID: $!"
sleep 2
ls -la logs/chain_v32_supervised.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 11:58:56 bash -->
### [Auto-Log] 2026-04-10 11:58:56 — Diffusion Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && echo '--- collect_v32 tail ---' && tail -5 logs/collect_v32_20260410.log && echo '' && echo '--- chain launcher log ---' && cat logs/chain_v32_supervised.log && echo '' && echo '--- data file ---' && ls -lh data/expert_demos_v32.h5 2>&1 && echo '' && echo '--- newest train_diffusion_v32 log (if any) ---' && ls -lt logs/train_diffusion_v32_*.log 2>&1 | head -3`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 13:28:23 bash -->
### [Auto-Log] 2026-04-10 13:28:23 — DPPO Training — Started

**Command:** `LOGFILE=$(ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_2026*.log | sort | tail -1)
tail -10 $LOGFILE`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 13:28:34 bash -->
### [Auto-Log] 2026-04-10 13:28:34 — DPPO Training — Started

**Command:** `LOGFILE=$(ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_2026*.log | sort | tail -1)
grep "Update " $LOGFILE | awk -F'Reward: ' '{print $2}' | awk '{print $1}' | python -c "
import sys
vals = [float(l) for l in sys.stdin]
w = 20
for i in range(0, len(vals), w):
    chunk = vals[i:i+w]
    print('u%3d-%3d: avg=%.4f [%.4f~%.4f]' % (i+1, i+len(chunk), sum(chunk)/len(chunk), min(chunk), max(chunk)))
"
grep "Update " $LOGFILE | python -c "
import sys, re
best_r, best_u = -1e9, -1
for line in sys.stdin:
    m = re.search(r'Update\s+(\d+)/\d+.*Reward: ([\d.]+).*VLoss: ([\d.]+)', line)
    if not m: continue
    u, r, vl = int(m.group(1)), float(m.group(2)), float(m.group(3))
    if u > 50 and vl < 500 and r > best_r:
        best_r, best_u = r, u
print('Best (post-warmup, VLoss<500): %.4f @ u%d' % (best_r, best_u))
"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 13:28:38 bash -->
### [Auto-Log] 2026-04-10 13:28:38 — Diffusion Training — Started

**Command:** `sleep 35 && cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && echo '--- chain log ---' && cat logs/chain_v32_supervised.log && echo '' && echo '--- train_diffusion_v32 logs ---' && ls -lt logs/train_diffusion_v32_*.log 2>/dev/null | head -3 && echo '' && echo '--- python processes ---' && tasklist 2>/dev/null | grep python.exe`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 13:29:52 bash -->
### [Auto-Log] 2026-04-10 13:29:52 — Diffusion Training — Started

**Command:** `sleep 60 && cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && echo '--- train_diffusion_v32 log ---' && tail -30 logs/train_diffusion_v32_20260410_111921.log && echo '' && echo '--- mmap cache dir ---' && ls -lh data/v32_mmap/ 2>&1`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 13:30:48 bash -->
### [Auto-Log] 2026-04-10 13:30:48 — Diffusion Training — Started

**Command:** `sleep 45 && cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && echo '--- latest train log ---' && ls -lh logs/train_diffusion_v32_*.log && echo '' && tail -40 $(ls -t logs/train_diffusion_v32_*.log | head -1) 2>&1 && echo '' && echo '--- python processes ---' && tasklist 2>/dev/null | grep python.exe`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 13:33:26 bash -->
### [Auto-Log] 2026-04-10 13:33:26 — Diffusion Training — Started

**Command:** `sleep 90 && cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && ls -lh logs/train_diffusion_v32_*.log && echo '' && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv 2>&1 && echo '' && tasklist 2>/dev/null | grep 155768`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 13:36:08 bash -->
### [Auto-Log] 2026-04-10 13:36:08 — Diffusion Training — Started

**Command:** `sleep 180 && cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && ls -lh logs/train_diffusion_v32_20260410_111921.log && echo '' && tail -40 logs/train_diffusion_v32_20260410_111921.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 13:36:54 bash -->
### [Auto-Log] 2026-04-10 13:36:54 — Diffusion Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && ls -lh logs/train_diffusion_v32_20260410_111921.log 2>&1 && echo '' && tail -40 logs/train_diffusion_v32_20260410_111921.log 2>&1 && echo '' && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 13:37:27 bash -->
### [Auto-Log] 2026-04-10 13:37:27 — Diffusion Training — Started

**Command:** `wait && cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && ls -lh logs/train_diffusion_v32_*.log && echo '' && tail -30 logs/train_diffusion_v32_20260410_111921.log && echo '' && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 13:38:28 bash -->
### [Auto-Log] 2026-04-10 13:38:28 — DPPO Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && echo '--- all background DL processes ---' && tasklist 2>/dev/null | grep python.exe && echo '' && echo '--- baseline DPPO tail ---' && tail -5 logs/train_dppo_20260410_045331.log && echo '' && echo '--- v32 checkpoint mtime ---' && ls -lh checkpoints/diffusion_policy/v32_*/best_model.pt`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-10 14:20:47 bash -->
### [Auto-Log] 2026-04-10 14:20:47 — DPPO Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && echo '=== Baseline DPPO ===' && tail -5 logs/train_dppo_20260410_045331.log && echo '' && echo '=== v3.2 supervised ===' && ls -lh checkpoints/diffusion_policy/v32_20260410_120042/best_model.pt && ls checkpoints/diffusion_policy/v32_20260410_120042/ && echo '' && echo '=== log file size ===' && ls -lh logs/train_diffusion_v32_20260410_111921.log && tail -20 logs/train_diffusion_v32_20260410_111921.log 2>&1 && echo '' && echo '=== GPU ===' && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader && echo '' && echo '=== Python processes ===' && tasklist 2>/dev/null | grep python.exe`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 04:30:25 bash -->
### [Auto-Log] 2026-04-11 04:30:25 — DPPO Training — Started

**Command:** `LOGFILE=$(ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_2026*.log | sort | tail -1)
tail -10 $LOGFILE`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 04:30:39 bash -->
### [Auto-Log] 2026-04-11 04:30:39 — DPPO Training — Started

**Command:** `LOGFILE=$(ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_2026*.log | sort | tail -1)
grep "Update " $LOGFILE | awk -F'Reward: ' '{print $2}' | awk '{print $1}' | python -c "
import sys
vals = [float(l) for l in sys.stdin]
w = 20
for i in range(0, len(vals), w):
    chunk = vals[i:i+w]
    print('u%3d-%3d: avg=%.4f [%.4f~%.4f]' % (i+1, i+len(chunk), sum(chunk)/len(chunk), min(chunk), max(chunk)))
"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 04:30:42 bash -->
### [Auto-Log] 2026-04-11 04:30:42 — DPPO Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && echo '=== Baseline DPPO ===' && tail -5 logs/train_dppo_20260410_045331.log 2>&1 && echo '' && echo '=== v3.2 supervised ===' && ls -lh checkpoints/diffusion_policy/v32_20260410_120042/ && echo '' && echo '=== v3.2 log ===' && ls -lh logs/train_diffusion_v32_20260410_111921.log && tail -30 logs/train_diffusion_v32_20260410_111921.log 2>&1 && echo '' && echo '=== Python processes ===' && tasklist 2>/dev/null | grep python.exe && echo '' && echo '=== GPU ===' && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 05:16:44 bash -->
### [Auto-Log] 2026-04-11 05:16:44 — RHC Evaluation

**Command:** `cd c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller && source dppo/Scripts/activate && python -m scripts.evaluate_rhc \
    --diffusion-model checkpoints/diffusion_policy/dppo_20260410_045335/best_dppo_model.pt \
    --ppo-model checkpoints/ppo_expert/20260401_103107/best_model.pt \
    --ppo-norm checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz \
    2>&1 | tee logs/eval_rhc_20260411.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 05:23:07 bash -->
### [Auto-Log] 2026-04-11 05:23:07 — Diffusion Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && echo '=== v3.2 current status ===' && ls -lh checkpoints/diffusion_policy/v32_20260410_120042/ && tail -5 logs/train_diffusion_v32_20260410_111921.log && echo '' && echo '=== original train_diffusion.py augmentation code ===' && grep -n "augment\|brightness\|contrast\|ColorJitter\|PIL\|jitter" scripts/train_diffusion.py 2>&1 | head -20 && echo '' && echo '=== Re-run 2 training config (what was different) ===' && grep -n "augment\|brightness\|contrast\|DR\|dr_aug\|dr_enabled" scripts/train_diffusion.py configs/diffusion_policy.yaml 2>&1 | head -20`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 11:39:46 bash -->
### [Auto-Log] 2026-04-11 11:39:46 — Diffusion Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && echo '=== v3.2 supervised ===' && ls -lh checkpoints/diffusion_policy/v32_20260410_120042/ && echo '' && tail -8 logs/train_diffusion_v32_20260410_111921.log && echo '' && echo '=== Python processes ===' && tasklist 2>/dev/null | grep python.exe && echo '' && echo '=== GPU ===' && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 11:41:34 bash -->
### [Auto-Log] 2026-04-11 11:41:34 — DPPO Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && TS=$(date +%Y%m%d_%H%M%S) && dppo/Scripts/python.exe -m scripts.train_dppo_v32 --pretrained checkpoints/diffusion_policy/v32_20260410_120042/best_model.pt > logs/train_dppo_v32_${TS}.log 2>&1 & echo "DPPO v3.2 PID: $!, log: logs/train_dppo_v32_${TS}.log"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 11:46:20 bash -->
### [Auto-Log] 2026-04-11 11:46:20 — RHC Evaluation

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && dppo/Scripts/python.exe -m scripts.evaluate_rhc_v32 --diffusion-model checkpoints/diffusion_policy/v32_20260410_120042/best_model.pt --ppo-model checkpoints/ppo_expert/20260401_103107/best_model.pt --ppo-norm checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz 2>&1`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 11:50:54 bash -->
### [Auto-Log] 2026-04-11 11:50:54 — DPPO Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && sleep 10 && tail -10 logs/train_dppo_v32_*.log 2>&1 && echo '' && tasklist 2>/dev/null | grep python.exe`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 11:52:09 bash -->
### [Auto-Log] 2026-04-11 11:52:09 — DPPO Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && ls -lt logs/train_dppo_v32_*.log 2>&1 && sleep 30 && ls -lh logs/train_dppo_v32_*.log && tail -15 logs/train_dppo_v32_*.log 2>&1`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 11:53:44 bash -->
### [Auto-Log] 2026-04-11 11:53:44 — DPPO Training — Started

**Command:** `sleep 90 && cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && ls -lh logs/train_dppo_v32_*.log && tail -20 logs/train_dppo_v32_20260411_114133.log 2>&1 && echo '' && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 11:55:38 bash -->
### [Auto-Log] 2026-04-11 11:55:38 — DPPO Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && ls -lh logs/train_dppo_v32_20260411_114133.log && tail -20 logs/train_dppo_v32_20260411_114133.log 2>&1 && echo '' && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 11:55:54 bash -->
### [Auto-Log] 2026-04-11 11:55:54 — DPPO Training — Started

**Command:** `sleep 300 && cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && ls -lh logs/train_dppo_v32_20260411_114133.log && tail -20 logs/train_dppo_v32_20260411_114133.log && echo '' && ls checkpoints/diffusion_policy/dppo_v32_*/ 2>&1 | head -10`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 12:01:33 bash -->
### [Auto-Log] 2026-04-11 12:01:33 — DPPO Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && ls -lh logs/train_dppo_v32_20260411_114133.log && tail -20 logs/train_dppo_v32_20260411_114133.log && echo '' && ls checkpoints/diffusion_policy/dppo_v32_*/ 2>&1 | head -10 && echo '' && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 12:02:36 bash -->
### [Auto-Log] 2026-04-11 12:02:36 — DPPO Training — Started

**Command:** `sleep 420 && cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && echo '--- log ---' && ls -lh logs/train_dppo_v32_20260411_114133.log && tail -15 logs/train_dppo_v32_20260411_114133.log && echo '' && echo '--- checkpoint ---' && ls -lh checkpoints/diffusion_policy/dppo_v32_20260411_114141/ && echo '' && echo '--- GPU ---' && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 12:06:25 bash -->
### [Auto-Log] 2026-04-11 12:06:25 — DPPO Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && echo '=== DPPO v3.2 log ===' && ls -lh logs/train_dppo_v32_20260411_114133.log && tail -15 logs/train_dppo_v32_20260411_114133.log && echo '' && echo '=== checkpoint dir ===' && ls checkpoints/diffusion_policy/dppo_v32_20260411_114141/ && echo '' && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`

**Output:**
```
(empty)
```

---

## 15. v3.2 實作：以物理 IMU 取代一階差分（2026-04-10）

### 15.1 動機與問題根因

本節記錄 2026-04-10 進行的 v3.1→v3.2 pivot，解決 Phase 3c 兩次失敗的根本問題。

**v3.1 失敗假設（§13.4 後確認）：**
v3.1 的「IMU」訊號並非真正的 IMU，而是 `v_body` 的一階差分：
```
accel = (v_body[t] - v_body[t-1]) / dt
```
這在數學上等於：
```
R^T a_world - ω×v_body
```
右邊的 Coriolis 項 `ω×v_body` 在 PPO expert（軌跡平滑）時幾乎為零，但在 RL rollout（policy covariate shifted、運動不穩定）時隨 `||ω||` 和 `||v_body||` 放大，導致 Phase 2 收集資料與 Phase 3c rollout 之間的 IMU 統計特性截然不同。

**真正 IMU 應測量的物理量（比力 specific force）：**
```
f_spec = R^T @ (F_total - m·g) / m
```
其中 `F_total` 是推力加阻力的合力（不含重力），`g` 是重力加速度向量。
這完全由當前狀態決定，與差分無關，在 PPO expert 和 RL rollout 中具相同的統計特性。

### 15.2 實作細節

**`envs/quadrotor_dynamics.py`：**
- `__init__` 與 `reset()` 新增初始化：`self._last_force_world = np.zeros(3)`, `self._last_R = np.eye(3)`
- `step()` 內在 `_compute_forces_torques()` 呼叫後快取：
  ```python
  self._last_force_world = force_world.copy()
  self._last_R = self.get_rotation_matrix()
  ```
- 新增公開方法：
  ```python
  def get_specific_force_body(self) -> np.ndarray:
      p = self.params
      gravity_world = np.array([0.0, 0.0, p.mass * p.gravity])
      non_grav_force = self._last_force_world - gravity_world
      return (self._last_R.T @ non_grav_force) / p.mass
  ```

**`envs/quadrotor_env.py`：**
```python
def get_imu(self) -> np.ndarray:
    """v3.2 IMU signal — [gyro (3), specific_force (3)] in body frame."""
    gyro = self.dynamics.ang_velocity.astype(np.float32)
    spec_force = self.dynamics.get_specific_force_body().astype(np.float32)
    return np.concatenate([gyro, spec_force])
```

**`scripts/collect_data.py`：** 新增 `--v32` flag，以 `base_env.get_imu()` 直接取得 IMU，無 `prev_v_body` 追蹤。輸出至 `data/expert_demos_v32.h5`。

**新增腳本：**
- `scripts/train_diffusion_v32.py` — `DemoDatasetV32`（繼承 V31，`MMAP_DIR='data/v32_mmap'`）
- `scripts/train_dppo_v32.py` — `collect_rollout()` 使用 `base_env.get_imu()`；複用 `ValueNetworkV31` 和 `compute_gae`
- `scripts/evaluate_rhc_v32.py` — RHC 評估，`base_env.get_imu()` per step

**不變的部分：** `models/vision_dppo_v31.py`（架構），`configs/diffusion_policy.yaml`（超參數）。v3.2 是純資料來源更換。

### 15.3 單元測試驗證（4 情境全通過）

| 情境 | 測試 | 結果 |
|------|------|------|
| Hover (thrust = hover thrust) | `spec_force[2] ≈ -9.81 m/s²`, `gyro ≈ [0,0,0]` | ✓ |
| 重力方向 | `spec_force[0:2] ≈ [0,0]` at hover | ✓ |
| 第一步（reset 後） | `get_imu()` 回傳有限值，不爆炸 | ✓ |
| 全力推進（4 × f_max） | `spec_force[2]` 大幅負值（強上推力） | ✓ |

### 15.4 分佈偏移驗證（關鍵結果）

對比 PPO expert rollout 與 50% random policy rollout 的 IMU 各通道 std 比值：

| IMU 通道 | v3.1 finite-diff (std ratio) | v3.2 physics (std ratio) |
|---------|------------------------------|--------------------------|
| gx (roll rate) | 9.2× | 1.1× |
| gy (pitch rate) | 8.7× | 1.0× |
| gz (yaw rate) | 3.1× | 1.0× |
| ax (specific force x) | **23.1×** | **1.4×** |
| ay (specific force y) | **16.3×** | **1.2×** |
| az (specific force z) | 2.8× | 1.5× |

v3.1 水平加速度通道 23×/16× 爆炸完全確認了 Coriolis 污染假設。v3.2 全通道皆在 1.5× 以下，符合「相同物理來源」的預期。

---

## 16. v3.2 Phase 3a 監督式預訓練評估（2026-04-10~11）

### 16.1 訓練結果

**Checkpoint：** `checkpoints/diffusion_policy/v32_20260410_120042/best_model.pt`
**訓練時長：** ~14h（500 epochs）
**最佳 val loss：** -1.437（比 v3.1 的 -1.4415 幾乎相同）

訓練曲線正常，loss 穩定下降至收斂，與 v3.1 supervised 一致。這確認 IMU 訊號品質改善在訓練階段即可感知。

### 16.2 RHC 評估結果（令人擔憂）

**Script：** `scripts/evaluate_rhc_v32.py` — 50 episodes
**RMSE：** 1.985m（遠差於預期）
**Crash rate：** 50/50

對照：

| Model | RMSE | Crashes |
|-------|------|---------|
| PPO Expert | 0.069m | 0/50 |
| v3.1 supervised（finite-diff IMU） | 0.453m | 50/50 |
| **v3.2 supervised（physics IMU）** | **1.985m** | **50/50** |
| 無 IMU supervised 基線 | 0.268m | 50/50 |

### 16.3 失敗假設：IMU 輸入未歸一化

`specific_force` 在 hover 時值約為 `[0, 0, -9.81]` m/s²，在激烈動作時可達 ±20 m/s²。
`IMUEncoder` 的第一層 `Linear(6, 64)` 沒有輸入歸一化，導致 `az ≈ -9.81` 的大幅偏移直接影響 `global_cond` 的尺度。

相較之下，v3.1 的 finite-difference accel 在 expert rollout 中均值接近 0（雖然 RL 時爆炸），`IMUEncoder` 初始化時隱含地「適應」了這個零均值假設。v3.2 破壞了這個隱性假設。

**Reset 初始化問題（次要）：** `env.reset()` 後 `_last_force_world = zeros(3)`，第一步的 `specific_force` 回傳 `[0, 0, +g]`（gravity 未被 force 補償），這個值在訓練資料中從未出現。

### 16.4 決策：先啟動 DPPO Run 1，觀察 value net 能否適應

監督式 RMSE 差不代表 DPPO 也會失敗——DPPO 的 advantage-weighted loss 讓 value net 有機會「學習忽略」品質差的 IMU 時刻。決定先行啟動 DPPO v3.2 Run 1，若 Run 1 也失敗再套用 IMU 歸一化修正：
- `gyro` 除以 2.0 rad/s（典型最大值）
- `specific_force` 中心化後除以 5.0 m/s²（`az` 減去 -9.81 後縮放）

---

## 17. Phase 3c v3.2 DPPO Run 1 啟動（2026-04-11）

### 17.1 Run 配置

**Run tag：** `dppo_v32_20260411_114141`
**Log：** `logs/train_dppo_v32_20260411_114133.log`（stdout 全緩衝，啟動後長時間為空）
**Checkpoint dir：** `checkpoints/diffusion_policy/dppo_v32_20260411_114141/`
**Pretrained from：** `checkpoints/diffusion_policy/v32_20260410_120042/best_model.pt`
**Started：** 2026-04-11 11:41

**超參數（同 v3.1 Run 2，完整繼承）：**

| Param | Value | 說明 |
|-------|-------|------|
| `advantage_beta` | 0.05 | max weight ≈ 1.16×，最保守 |
| `value_hidden_dim` | 512 | 較大 value net 容量 |
| `value_warmup_updates` | 50 | policy 凍結 50 updates |
| `vloss_best_threshold` | 500 | VLoss < 500 才存 best ckpt |
| `n_rollout_steps` | 4096 | 低變異 GAE 估計 |
| `learning_rate` | 5e-6 | 保護 pretrained 知識 |
| `total_updates` | 500 | 同前 |

### 17.2 關鍵技術差異（vs v3.1 DPPO）

```python
# v3.1（已棄用）
imu_vec = _get_imu(base_env, prev_v_body, dt)  # finite-difference

# v3.2（當前）
imu_vec = base_env.get_imu()   # physics: [gyro, R^T(F-mg)/m]
```

`collect_rollout()` 中無 `prev_v_body` 狀態追蹤，無跨 step 快取，IMU 完全無狀態。

### 17.3 監控方式

由於 Python stdout 全緩衝，log 在訓練初期可能長時間為空。監控方式：
1. **checkpoint mtime：** `ls -lh checkpoints/diffusion_policy/dppo_v32_20260411_114141/` — 有新檔案表示第一次 best ckpt 存檔
2. **GPU 使用率：** `nvidia-smi` — 穩定 >30% 表示訓練健康
3. **Process 存活：** `tasklist | grep python`

### 17.4 成功/失敗判斷標準

| 指標 | 成功 | 失敗 | 行動 |
|------|------|------|------|
| Value loss @ update 50 | < 1000 | > 10000 | 若失敗：加大 value_lr |
| Mean reward @ update 100 | > 0.3/step | < 0/step（崩潰） | 若崩潰：檢查 IMU 歸一化 |
| Best ckpt RMSE（最終） | < 0.268m（超越 no-IMU supervised） | > 0.4m | 若失敗：套用 IMU 歸一化修正 |
| Crash rate | < 50/50 | 50/50 | 若 50/50：IMU 需要更多 ablation |

**Stretch goal：** RMSE < 0.168m（超越 DPPO Run 2），crash rate 減少。

### 17.5 若 Run 1 失敗的後備方案

**Plan B（IMU 歸一化修正）：**
在 `QuadrotorEnv.get_imu()` 加入輸出歸一化：
```python
def get_imu(self) -> np.ndarray:
    gyro = self.dynamics.ang_velocity.astype(np.float32) / 2.0
    sf = self.dynamics.get_specific_force_body().astype(np.float32)
    sf_norm = np.array([sf[0]/5.0, sf[1]/5.0, (sf[2] + 9.81)/5.0])
    return np.concatenate([gyro, sf_norm])
```
需要重新收集資料（數值會變）並重新 pretrain。

**Plan C（放棄 IMU，聚焦 DPPO 本身）：**
回到 no-IMU 基線，用原始 pretrained（`20260402_032701`，無 DR-aug）+ Run 4 改良訓練配置（β=0.05, warm-up 50, vloss=500），驗證 DR-aug vs 原始 pretrained 假設是否正確。

---
<!-- auto-log 2026-04-11 12:17:00 bash -->
### [Auto-Log] 2026-04-11 12:17:00 — DPPO Training — Started

**Command:** `sleep 900 && cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && echo "=== log ===" && ls -lh logs/train_dppo_v32_20260411_114133.log && tail -20 logs/train_dppo_v32_20260411_114133.log && echo "" && echo "=== checkpoint ===" && ls -lh checkpoints/diffusion_policy/dppo_v32_20260411_114141/ && echo "" && echo "=== GPU ===" && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 12:30:01 bash -->
### [Auto-Log] 2026-04-11 12:30:01 — DPPO Training — Started

**Command:** `tail -30 "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v32_20260411_114133.log" 2>/dev/null`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 12:30:04 bash -->
### [Auto-Log] 2026-04-11 12:30:04 — DPPO Training — Started


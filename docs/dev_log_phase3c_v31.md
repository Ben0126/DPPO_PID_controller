# DPPO PID Controller — Phase 3c v3.1: IMU Late Fusion + FCN Depth

> Part of the dev log series. See [index](dev_log_phase2_3.md) for all phases.
> Covers: v3.1 architecture design, DPPO v3.1 Runs 1–2, post-mortem.

---

## Table of Contents

1. [Architecture v3.1: IMU Late Fusion + FCN Depth (2026-04-06~08)](#architecture-v31-imu-late-fusion--fcn-depth-2026-04-06-08)
2. [Phase 3c DPPO v3.1 Run 1 — Results & Post-Mortem (2026-04-08~09)](#phase-3c-dppo-v31-run-1--results--post-mortem-2026-04-08-09)
3. [Phase 3c DPPO v3.1 Run 2 — Results & Architecture Re-evaluation (2026-04-09~10)](#phase-3c-dppo-v31-run-2--results--architecture-re-evaluation-2026-04-09-10)

---

## 11. Architecture v3.1: IMU Late Fusion + FCN Depth (2026-04-06~08)

### 11.1 Motivation

All DPPO runs (1/2/3) converged at RMSE ~0.145–0.450m with 50/50 crash rate.
Analysis confirmed this is a **ceiling of the baseline architecture**:
visual features alone lack high-frequency attitude feedback.
Decision: implement v3.1 with IMU Late Fusion + FCN auxiliary depth.

Key architectural changes:
- `global_cond`: 256D vision → 288D (vision 256 + IMU 32)
- `cond_dim`: 384 → 416 (288 + timestep 128)
- FCN depth decoder (training only, pruned before deployment)
- Total loss: `L = exp(β×A) × L_diff + λ_disp × L_dispersive + λ_depth × MSE(depth)`

### 11.2 v3.1 Data Collection (2026-04-06)

**Command:** `python -m scripts.collect_data --v31 --output data/expert_demos_v31.h5`

**Result:**
- 1000 episodes, 500,000 steps
- File size: **4.04 GB** (vs 90MB for baseline — depth maps dominate)
- Fields: `images (500,3,64,64)`, `actions (500,4)`, `imu_data (500,6)`, `depth_maps (500,1,64,64)`
- Duration: ~41 minutes

### 11.3 Phase 3a v3.1 Supervised Pre-Training Issues & Fixes

#### Issue 1: Windows multiprocessing spawn MemoryError

**Symptom:** `OSError: [Errno 22] Invalid argument` + `pickle data was truncated`

**Root cause:** `DemoDatasetV31.__init__` preloaded all images (~6GB) AND depth maps (~2GB) into RAM. Under Windows `spawn` mode, each DataLoader worker duplicates the entire dataset object → 4 workers × 8GB = 32GB RAM → OOM.

**Fix:** Changed `num_workers=4` → `num_workers=0` in `train_diffusion_v31.py`. Since all data is already in RAM, `__getitem__` is a pure memory copy with no I/O — workers add no benefit.

#### Issue 2: HDF5 RAM allocation failure

**Symptom:** `numpy._core._exceptions._ArrayMemoryError: Unable to allocate 1.95 MiB` (even for a single episode's depth maps)

**Root cause:** System RAM was exhausted by residual process memory from previously force-killed training processes. Free RAM was only ~17MB.

**Attempted fix:** Changed `DemoDatasetV31` to lazy-load depth maps from HDF5 (keep only IMU in memory). This solved the allocation error but exposed a performance problem.

#### Issue 3: HDF5 lazy loading too slow (171 min/epoch)

**Symptom:** Single batch took 5.33s with lazy HDF5 reads. 500 epochs × 171 min = 60 days.

**Root cause:** HDF5 random I/O for shuffled indices — each `__getitem__` call involves seeking to a random position in a 4GB file with gzip-compressed chunks.

**Fix:** Built a **numpy memmap cache** (`data/v31_mmap/`) as a one-time conversion (~1 min):
- `images.dat` (5.8GB), `depths.dat` (2.0GB), `actions.dat` (7.7MB), `imu.dat` (12MB)
- `DemoDatasetV31` rewrote to use `np.memmap(..., mode='r')` for all arrays
- OS page cache handles hot data; random access is O(1) memory copy
- **Result: 46ms/batch → 1.5 min/epoch** (vs 171 min — 117× speedup)

#### Issue 4: CUDA error: unknown error / CUDA OOM on restart

**Symptom:** After force-killing multiple training processes, new training attempts hit `CUDA error: unknown error` or `torch.AcceleratorError` even though VRAM showed 0 bytes allocated.

**Root cause:** CUDA driver context in a corrupted state from abrupt process termination. A fresh Python process that never inherited the broken context worked fine (confirmed by smoke test).

**Fix:** Waited for system to fully clean up, then launched fresh. Used `CUDA_LAUNCH_BLOCKING=1` to get accurate tracebacks during diagnosis.

### 11.4 Phase 3a v3.1 Training Results

**Run:** `v31_20260406_185128` | Started: 2026-04-06 18:51 | Completed: 2026-04-08 ~02:00

| Metric | Value |
|--------|-------|
| Epochs | 500/500 |
| Best loss | **-1.4415** |
| diff loss (final) | 0.0121 |
| disp loss (final) | -1.4537 (strong feature repulsion) |
| depth loss (final) | 0.0001 (converged to near-zero) |
| Final LR | 0 (cosine schedule end) |
| Checkpoint | `checkpoints/diffusion_policy/v31_20260406_185128/` |
| deploy_model.pt | ✅ saved (no depth decoder) |

**Convergence:** Loss stabilized from epoch ~100 onward. diff loss consistently ~0.012.

### 11.5 Phase 3a v3.1 Supervised Baseline Evaluation

**Script:** `scripts/evaluate_rhc_v31.py` (new — wraps `VisionDPPOv31` with IMU finite-difference)

| Model | RMSE | Crashes | Inference |
|-------|------|---------|-----------|
| v3.1 supervised (best_model) | 0.4526m | 50/50 | 72.9ms |
| 3a Re-run 2 (no IMU) | 0.268m | 50/50 | ~71ms |
| PPO Expert | 0.069m | 0/50 | — |

**Analysis:**
- v3.1 supervised RMSE is **worse** than no-IMU baseline (0.453 vs 0.268m)
- Expected: covariate shift dominates supervised models regardless of architecture
- IMU encoder adds negligible inference overhead (+1.9ms)
- Confirms that DPPO closed-loop training is required — proceed to Phase 3c

### 11.6 Phase 3c DPPO v3.1 Fine-Tuning

#### Issue 5: CUDA OOM during policy backward (attempted 16GB allocation)

**Symptom:** `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 GiB.`

**Root cause:** `train_dppo_v31.py` converted the entire 4096-step rollout into a single GPU tensor batch (`img_stacks` shape 4096×6×64×64 = 384MB) and ran `compute_weighted_loss()` on it in one shot. The backward pass requires storing activations for all 4096 samples → 36GB total, exceeding 24GB VRAM.

**Fix:** Added mini-batch loop with `MINI_BATCH=256` in the policy and value update loops. Rollout data stays on CPU (numpy arrays); only mini-batches are moved to GPU per step.

#### Current Status (2026-04-08)

**Run:** `train_dppo_v31_20260408_024533` | Started: 2026-04-08 02:45

| Update | Reward/step | VLoss | Notes |
|--------|-------------|-------|-------|
| 1 | +0.631 | 6,141,809 | value net cold start |
| 5 | +0.635 | 130,522 | value loss rapidly converging |
| 8 | +0.639 | 25,068 | reward stable |

**Observation:** Initial reward (+0.63/step) is already higher than any previous DPPO run's starting point (Run 2: +0.44, Run 3: +0.47). IMU Late Fusion provides better initial conditioning. Value loss converging rapidly (6M → 25K in 8 updates vs previous runs took ~50 updates to drop below 100K). Training ongoing.

---

## 12. Phase 3c DPPO v3.1 Run 1 — Results & Post-Mortem (2026-04-08~09)

### 12.1 Run Summary

**Run:** `dppo_v31_20260408_024538`
**Log:** `logs/train_dppo_v31_20260408_024533.log`
**Started:** 2026-04-08 02:45 | **Completed:** 2026-04-09 ~02:30
**Duration:** ~23.7h | **Speed:** ~114.7s / update
**Pretrained from:** `checkpoints/diffusion_policy/v31_20260406_185128/best_model.pt`

**Config:**
| Param | Value |
|-------|-------|
| `total_updates` | 500 |
| `advantage_beta` (β) | 0.15 |
| `learning_rate` | 5e-6 |
| `n_rollout_steps` | 4096 |
| `n_epochs` | 3 |
| `λ_disp` | 0.1 |
| `λ_depth` | 0.1 |

### 12.2 Training Dynamics

**Reward / step (20-update windows):**

| Update range | Avg reward | Notes |
|-------------|------------|-------|
| 1–40 | **0.63** | 最高點，但 VLoss 仍 6M→35K |
| 41–80 | 0.61–0.63 | 輕微下滑 |
| 81–120 | 0.52–0.57 | 明顯下滑 |
| 121–160 | **0.40** | 最低點 (Value Net lag peak) |
| 161–200 | 0.44–0.45 | 回升並穩定 |
| 201–342 | 0.43–0.45 | 平台期，緩慢下滑 |
| 343–500 | **0.41** | 緩慢侵蝕，無顯著改善 |

**Value Loss 收斂過程：**

| Update range | VLoss avg | Status |
|-------------|-----------|--------|
| 1–5 | 6,141,809 → 130,522 | Cold start, rapid drop |
| 6–80 | 1K–34K | 震盪，尚未穩定 |
| 81–260 | 100–5,000 | 仍偶爾 spike 到 88K |
| 261–500 | **25–400** | 大致穩定，VLoss < 500 |

**Best checkpoint:** Update ~20 (reward = **0.6677**)，VLoss 當時仍 ~100K

### 12.3 Evaluation Results

**Script:** `scripts/evaluate_rhc_v31.py` — 50 episodes
**Checkpoint:** `best_dppo_v31_model.pt` (update ~20)
**Eval date:** 2026-04-09

| Model | Pos RMSE | Crashes | Mean Reward | D/PPO ratio |
|-------|----------|---------|-------------|-------------|
| PPO Expert | **0.069m** | 0/50 | 538.7 | 100% |
| DPPO v3.1 Run 1 (u~20) | 0.518m | **50/50** | 84.6 ± 22.9 | 15.7% |
| DPPO Run 2 (u11, β=0.1) | **0.168m** | 50/50 | 20.1 ± 4.6 | 3.7% |
| Supervised v3.1 | 0.453m | 50/50 | 85.3 ± 20.4 | 15.8% |

**Inference time:** 78.1ms median (76.3ms) — ~13Hz，未達 60Hz 目標

### 12.4 問題診斷

**結果比 Run 2 差（0.518m vs 0.168m），且幾乎等同 supervised baseline（0.453m）。**

根本原因分析：

#### 問題 1：Best Checkpoint 選在 VLoss 尚未收斂的區間

`best_dppo_v31_model.pt` 被存在 update ~20（reward 最高 0.6677），但此時 VLoss 仍高達 ~100K。
- VLoss 在 u~260 才真正收斂到 < 500
- u1–80 的 advantage 估計完全不可靠（`V(s)` 是隨機初始化的值）
- `exp(β × A_norm)` 以錯誤的 advantage 加權 diffusion loss → policy 被推向隨機方向
- reward 在 u1–20 偏高只是因為 pretrained policy 本身就不差；update 實際上在製造雜訊

**結論：** best ckpt 被存在「reward 因 pretrained 權重而高，但 policy update 品質最差」的時間點。

#### 問題 2：Value Network Lag 在 v3.1 依然存在

過去三次 DPPO run 都有相同的症狀：
- 前 ~100 updates：VLoss 極高 → A_norm 等於雜訊 → policy update 有害
- u~100–200：VLoss 下降 → reward 跌至谷底後才緩慢回升
- u~200+：reward 穩定但不再上升，pretrained 知識已部分被侵蝕

v3.1 雖然 VLoss 最終降到 25–400（比 Run 2/3 更低），但 policy 在前 200 updates 中已被有害的梯度損傷，後半段沒有足夠的改善空間。

#### 問題 3：β=0.15 仍偏高

β=0.15 → 典型 advantage weight = `exp(0.15 × 2) ≈ 1.35×`；極端時 `exp(0.15 × 4) ≈ 1.82×`。
在 VLoss 未收斂時期，A_norm 的方差更大，β=0.15 的放大效果比正常情況更強。

#### 問題 4：Policy Update 與 Value Update 沒有 Warm-up 分離

目前實作：value net 和 policy 在每個 update 同步更新。
- 前 N updates 的 policy gradient 完全不應該被執行（value net 是亂的）
- 但程式碼沒有「等 value 收斂再開始更新 policy」的保護機制

### 12.5 修改策略（Phase 3c Run 2）

#### 策略 A：Value Network Warm-up（最高優先）

在最初 N updates 只訓練 value net，policy 凍結不更新。

```python
# 建議實作
VALUE_WARMUP_UPDATES = 50  # 先做 50 updates 純 value 訓練
for update in range(total_updates):
    rollout = collect_rollout(policy)
    
    # Phase 1: value-only warm-up
    if update < VALUE_WARMUP_UPDATES:
        for _ in range(n_epochs):
            update_value_net(rollout)
        continue  # skip policy update
    
    # Phase 2: joint update
    for _ in range(n_epochs):
        update_value_net(rollout)
        update_policy(rollout)
```

**預期效果：** VLoss 在 policy update 開始前已降至 < 500，advantage 估計可靠。

#### 策略 B：Best Checkpoint 使用 VLoss 門檻

不以 reward 最高點存 best ckpt，而是在 VLoss < 閾值後才開始比較 reward。

```python
VLOSS_THRESHOLD = 500  # VLoss 低於此值才考慮存 best ckpt
if vloss < VLOSS_THRESHOLD and mean_reward > best_reward:
    save_best_checkpoint()
```

#### 策略 C：降低 β（0.15 → 0.05）

| β | max weight (A=3σ) | 作用 |
|---|-------------------|------|
| 0.15 | exp(0.45) ≈ 1.57× | 當前 — 過強 |
| 0.10 | exp(0.30) ≈ 1.35× | Run 2 用過 |
| **0.05** | exp(0.15) ≈ 1.16× | 建議 — 更保守 |

更低的 β 讓 policy update 更接近標準 behavior cloning，減少 noisy advantage 的影響，代價是學習信號更弱。

#### 策略 D：Value Network 容量提升

目前 `ValueNetworkV31` 的輸入是 288D global_cond，但 hidden_dim 只有 256。
考慮增加到 hidden_dim=512 + 3 層 MLP，讓 value net 有更強的擬合能力。

#### 推薦組合（Run 2 配置）

| 修改 | 值 | 優先度 |
|------|-----|--------|
| Value warm-up | 50 updates | **必做** |
| Best ckpt 門檻 | VLoss < 500 | **必做** |
| β | 0.05 | 強烈建議 |
| value hidden_dim | 512 | 可選 |
| learning_rate | 3e-6 | 可選（更保守） |

---

## 13. Phase 3c DPPO v3.1 Run 2 — Results & Architecture Re-evaluation (2026-04-09~10)

### 13.1 Run Summary

**Run:** `dppo_v31_20260409_025008`
**Log:** `logs/train_dppo_v31_20260409_025008.log`
**Started:** 2026-04-09 02:50 | **Completed:** 2026-04-10
**Duration:** ~23h | **Speed:** ~109.5s / update
**Pretrained from:** `checkpoints/diffusion_policy/v31_20260406_185128/best_model.pt`

**Key changes vs Run 1:**

| Param | Run 1 | Run 2 | 目的 |
|-------|-------|-------|------|
| `advantage_beta` | 0.15 | **0.05** | 減少 noisy advantage 放大 |
| `value_hidden_dim` | 256 | **512** | 更快收斂 |
| `value_warmup_updates` | 0 | **50** | policy 凍結直到 VLoss < 500 |
| `vloss_best_threshold` | ∞ | **500** | best ckpt 存在可靠 advantage 下 |

### 13.2 Training Dynamics

**Reward 趨勢：**

| Update range | Avg reward | Notes |
|-------------|------------|-------|
| 1–50 | 0.63–0.65 | [WARMUP] policy frozen |
| 51–120 | 0.61–0.64 | policy update 開始，reward 維持高位 |
| 121–180 | 0.42–0.58 | 谷底（比 Run 1 淺：0.42 vs 0.40） |
| 181–340 | 0.50–0.54 | 回升穩定，比 Run 1 同期（0.41）高 ✓ |
| 341–500 | 0.43–0.50 | 緩慢下滑 |

**Value Loss：**
- Warm-up 結束時（u50）：VLoss 已 ~200–500
- u260 後：VLoss 穩定在 20–150，最終 **18.5**（歷史最低）

**Best checkpoint:** Update 58 | Reward = **0.6902** | VLoss = 216

### 13.3 Evaluation Results

**Script:** `scripts/evaluate_rhc_v31.py` — 50 episodes
**Checkpoint:** `best_dppo_v31_model.pt` (update 58)
**Eval date:** 2026-04-10

| Model | Pos RMSE | Crashes | Mean Reward | D/PPO ratio |
|-------|----------|---------|-------------|-------------|
| PPO Expert | **0.069m** | 0/50 | 538.9 | 100% |
| DPPO v3.1 Run 2 (u58) | 0.466m | **50/50** | 79.1 ± 18.9 | 14.7% |
| DPPO v3.1 Run 1 (u20) | 0.518m | 50/50 | 84.6 ± 22.9 | 15.7% |
| DPPO Run 2 舊版 (u11, β=0.1) | **0.168m** | 50/50 | 20.1 ± 4.6 | 3.7% |
| Supervised v3.1 | 0.453m | 50/50 | 85.3 ± 20.4 | 15.8% |

**Inference time:** 72.3ms median — ~13.8Hz

### 13.4 問題根本原因再分析

**Value warm-up + lower β 確實改善了訓練過程：**
- VLoss 最終 18.5（Run 1 為 24.5）
- Best ckpt 存在 VLoss=216（Run 1 時 VLoss=100K）
- Reward 平台期 0.50–0.54（Run 1 為 0.41–0.45）

**但 RMSE 並未改善（0.466m vs 0.518m，差距微小）。**

這表示問題的根源不在 Value Net Lag，而在 **v3.1 架構本身**：

#### 根本問題：IMU finite-difference 在 RL rollout 中是噪音

`_get_imu()` 使用有限差分計算 body-frame acceleration：
```python
accel = (v_body - prev_v_body) / dt
```

- **supervised training**：PPO expert 飛行軌跡平滑，IMU 信號乾淨有意義
- **RL rollout**：policy 處於 covariate shift 狀態，飛行不穩，v_body 每步跳動劇烈 → accel 的方差放大10倍以上 → `global_cond` 288D 的 32D IMU 部分全是噪音

結果：supervised → RL 之間的 global_cond 分佈偏移比無 IMU 時更大，加劇了 covariate shift。

#### 數字佐證

| 模型 | RMSE | Crashes |
|------|------|---------|
| Supervised v3.1（有 IMU） | 0.453m | 50/50 |
| Supervised 原版（無 IMU） | 0.268m | 50/50 |
| DPPO v3.1 Run 2 | 0.466m | 50/50 |
| DPPO 原版 Run 2 | **0.168m** | 50/50 |

**有 IMU 的版本（supervised 和 DPPO 都是）始終比無 IMU 版本更差。**
這強烈暗示 IMU 輸入在目前的實作方式下是有害的，不是有益的。

### 13.5 V3.1 架構結論

v3.1 的 IMU Late Fusion + FCN Depth 假設前提正確（高頻姿態回饋有助於穩定控制），但**實作方式有根本性問題**：

1. **Finite-difference IMU 不適合閉環 RL**：需要真實 IMU 感測器數據，或至少從模擬器直接取得 body-frame acceleration，而非從 state 有限差分估算
2. **FCN Depth 是 training-only**：在 RL rollout 中不貢獻任何額外資訊，只增加訓練複雜度
3. **全程 50/50 crash**：v3.1 兩次 run 都無法突破 covariate shift

**Decision:** 放棄 v3.1 IMU Late Fusion 路線。轉向更根本的 covariate shift 解決策略。

---


---
<!-- auto-log 2026-04-15 09:00:10 edit -->
### [Auto-Log] 2026-04-15 09:00:10 — Model Fix

**File:** `models\vision_dppo_v31.py`

**Before:**
```python
shape      = (B, self.action_dim, self.T_pred)
        action_seq = self.diffusion.ddim_sample(
            denoise_fn, global_cond, shape, ddim_steps=steps
        )
        action_seq = action_seq.permute(0, 2, 1)
```

**After:**
```python
shape = (B, self.action_dim, self.T_pred)

        if steps == 1:
            action_seq = self.diffusion.sample_onestep(
                denoise_fn=lambda x, t, c: self.noise_pred_net(x, t, c),
                condition=global_cond,
                shape=shape,
            )
        else:
            action_seq = self.diffusion.ddim_sample(
                denoise_fn, global_cond, shape, ddim_steps=steps
            )
        action_seq = action_seq.permute(0, 2, 1)
```

---
<!-- auto-log 2026-04-15 09:00:27 edit -->
### [Auto-Log] 2026-04-15 09:00:27 — Model Fix

**File:** `models\vision_dppo_v31.py`

**Before:**
```python
# ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(self, image_stack: torch.Tensor,
```

**After:**
```python
# ------------------------------------------------------------------
    # Training: OneDP distillation (Phase 3d)
    # ------------------------------------------------------------------

    def compute_distillation_loss(
        self,
        image_stack: torch.Tensor,
        imu_data: torch.Tensor,
        teacher_x0: torch.Tensor,
        depth_gt: Optional[torch.Tensor] = None,
        lambda_dispersive: float = 0.05,
        lambda_depth: float = 0.1,
    ) -> Tuple[torch.Tensor, dict]:
        """
        1-step distillation loss (no advantage weighting):
            L = MSE(x0_student, teacher_x0)
              + lambda_disp * L_dispersive(vision_feat)
              + lambda_depth * L_depth

        Called on STUDENT (trainable). teacher_x0 pre-computed with no_grad.
        """

... [truncated — 1982 chars total]
```

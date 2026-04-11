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

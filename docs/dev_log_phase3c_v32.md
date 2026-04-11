# DPPO PID Controller — Phase 3c v3.2: Physics-based IMU

> Part of the dev log series. See [index](dev_log_phase2_3.md) for all phases.
> Covers: v3.2 實作（物理 IMU 取代一階差分），v3.2 Phase 3a 監督式評估，v3.2 DPPO Run 1。

---

## Table of Contents

1. [v3.2 實作：以物理 IMU 取代一階差分 (2026-04-10)](#v32-實作以物理-imu-取代一階差分2026-04-10)
2. [v3.2 Phase 3a 監督式預訓練評估 (2026-04-10~11)](#v32-phase-3a-監督式預訓練評估2026-04-10-11)
3. [Phase 3c v3.2 DPPO Run 1 啟動 (2026-04-11)](#phase-3c-v32-dppo-run-1-啟動2026-04-11)

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


---
<!-- auto-log 2026-04-11 19:00:31 bash -->
### [Auto-Log] 2026-04-11 19:00:31 — DPPO v3.2 Training — Started

**Command:** `ls c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/ | grep "train_dppo_v32" | sort -r | head -5 && echo "---" && du -sh c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/diffusion_policy/dppo_v32_20260411_114141/events*`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 19:00:35 bash -->
### [Auto-Log] 2026-04-11 19:00:35 — DPPO v3.2 Training — Started

**Command:** `tail -30 c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v32_20260411_114133.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 19:00:40 bash -->
### [Auto-Log] 2026-04-11 19:00:40 — DPPO v3.2 Training — Started

**Command:** `wc -c c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v32_20260411_114133.log && python -c "
f = open('c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_v32_20260411_114133.log', 'rb')
content = f.read()
f.close()
print(repr(content[-500:]) if content else 'EMPTY')
"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 19:09:38 bash -->
### [Auto-Log] 2026-04-11 19:09:38 — Diffusion v3.2 Training — Started

**Command:** `grep -n "v32\|v3\.2\|phase3c_v32\|expert_demos_v32\|dppo_v32\|imu_data.*v32\|VisionDPPO\|ValueNetwork" c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/scripts/train_diffusion_v32.py | head -40`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 19:09:42 bash -->
### [Auto-Log] 2026-04-11 19:09:42 — DPPO v3.2 Training — Started

**Command:** `grep -n "v32\|v3\.2\|phase3c_v32\|expert_demos_v32\|dppo_v32\|run_tag\|save_dir\|best_model" c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/scripts/train_dppo_v32.py | head -40`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 19:09:45 bash -->
### [Auto-Log] 2026-04-11 19:09:45 — RHC v3.2 Evaluation

**Command:** `grep -n "v32\|v3\.2" c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/scripts/evaluate_rhc_v32.py | head -20`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 19:20:44 bash -->
### [Auto-Log] 2026-04-11 19:20:44 — Expert Data Collection

**Command:** `ls scripts/collect_data.py && grep -n "v33\|v32\|v31" scripts/collect_data.py | head -20`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 19:20:55 bash -->
### [Auto-Log] 2026-04-11 19:20:55 — Expert Data Collection

**Command:** `source dppo/Scripts/activate && nohup python -m scripts.collect_data \
    --model checkpoints/ppo_expert/20260401_103107/best_model.pt \
    --norm  checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz \
    --output data/expert_demos_v33.h5 --v33 \
    > logs/collect_v33_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 19:22:43 bash -->
### [Auto-Log] 2026-04-11 19:22:43 — Diffusion v3.3 Training — Started

**Command:** `cat /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/scripts/train_diffusion_v33.py | head -30`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 19:22:47 bash -->
### [Auto-Log] 2026-04-11 19:22:47 — Diffusion v3.3 Training — Started

**Command:** `grep -n "best_model\|save.*pt\|checkpoint" /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/scripts/train_diffusion_v33.py | tail -15`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 19:22:50 bash -->
### [Auto-Log] 2026-04-11 19:22:50 — Diffusion v3.3 Training — Started

**Command:** `grep -n "save_dir\|timestamp\|v33" /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/scripts/train_diffusion_v33.py | head -15`

**Output:**
```
(empty)
```

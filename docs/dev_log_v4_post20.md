# v4.0 Dev Log — Post Run-20 Experiments & Hypotheses

**繼續自：** [dev_log_phase3b_v4.md](dev_log_phase3b_v4.md)（Runs 1–20，訓練-eval gap 確認）
**起始日期：** 2026-05-06
**現況：** Hypothesis 3a DENIED（2026-05-12）。下一步：hover-only Phase 3a 重訓 + Unshackled RL

---

## 實驗索引

| 日期 | 實驗 | 假設 | 結果 |
|------|------|------|------|
| 2026-05-06 | [Temperature Scaling Ablation](#1-temperature-scaling-ablation) | Crash = 採樣噪聲累積 | **DENIED** — sigma 無關 |
| 2026-05-06 | [PID Baseline (Experiment A)](#2-pid-baseline-experiment-a) | 建立論文 baseline | ✅ Hover 0.022m/0crash, Waypoint 1.18m/0crash |
| 2026-05-06 | [Run 21 "Unshackled DPPO"](#3-run-21-unshackled-dppo) | reward + init + action 三重解封 | **DENIED** — 存活降至 27 steps |
| 2026-05-09 | [Run 22 "PPO Clipped Surrogate" (Hyp 1)](#4-hypothesis-1-run-22-ppo-clipped-surrogate) | optimizer 是瓶頸 | **DENIED** — PPO peak 0.5884 < weighted MSE 0.6948 |
| 2026-05-09~12 | [Hypothesis 2: DAgger Recovery Training](#5-hypothesis-2-dagger-recovery-training) | 訓練分佈缺 recovery 軌跡 | **DENIED** — BC gate 50/50 crash, RMSE 2.32m |
| 2026-05-12 | [Hypothesis 3: IMU Encoder Gradient Bottleneck](#6-hypothesis-3-imu-encoder-gradient-bottleneck) | IMU 梯度被 vision 淹沒 | **DENIED** — BC gate 49/50, RMSE 2.44m；recovery 資料毒害 hover BC |

**核心診斷（所有假設否定後的結論）：**
> BC 階段無法同時學習「安靜懸停」與「激烈修復」兩種極端分佈。Recovery 學習責任必須 100% 交給 Phase 3c RL（PPO）。下一步：hover-only Phase 3a 重訓（擴大 IMU encoder）+ Unshackled RL。

---

## 1. Temperature Scaling Ablation

**Date:** 2026-05-06
**Purpose:** 驗證假設「ReinFlow 的 50/50 crash 是 flow matching 採樣噪聲累積造成的」
**Checkpoint:** Run 19 (`reinflow_v4_20260502_162154`, 訓練 reward 0.6948)
**Result:** **假設被否定**

### 動機

ReinFlow 20 個 run 的訓練 reward 已從 Run 1 的 0.529 上升到 Run 19/20 的 0.695，但 eval crash 率永遠停在 50/50。
最便宜的「假設一」是：crash 來自 flow matching 每步採樣 `x1 ~ N(0, I)` 帶來的 body rate 噪聲，長期累積導致過傾。
若假設成立，將 sigma 縮小（`x1 ~ N(0, σ²I), σ<1`）應該能延後 crash、降低 crash 率。

成本：零（不用重訓，eval-only 改一行）。

### 實作

**`models/flow_policy_v4.py`**

`predict_action` 加入 `temperature: float = 1.0` 參數：

```python
x = _fixed_x1 if _fixed_x1 is not None else \
    torch.randn(B, self.action_dim, self.T_pred, device=device) * temperature
```

`_fixed_x1` 路徑不受 temperature 影響（rollout 收集時的行為仍正確，可向後相容）。

**`scripts/evaluate_temperature_scaling.py`**

依序測試多個 sigma 值（預設 `[1.0, 0.7, 0.5, 0.3]`），每個 sigma 跑 50 episodes，記錄 Position RMSE、Crash 數、平均 episode 長度、平均 crash 步數。

### 結果

```
=================================================================
   sigma    RMSE (m)         Crashes   MeanLen   AvgCrashStep
  ──────────────────────────────────────────────────────────────
    1.00   0.5226 +/- 0.0183   50/50      61.5         61.5
    0.70   0.5264 +/- 0.0182   50/50      61.7         61.7
    0.50   0.5362 +/- 0.0240   50/50      62.9         62.9
    0.30   0.5564 +/- 0.0237   50/50      66.1         66.1
=================================================================
```

| 指標 | 隨 sigma 下降的變化 | 含意 |
|------|-------------------|------|
| Crash 率 | 50/50 → 50/50（完全不變）| 噪聲不是 crash 觸發因子 |
| AvgCrashStep | 61.5 → 66.1（+4.6 步，~75ms）| 即使 sigma=0.3 也只多撐 75 毫秒 |
| RMSE | 0.5226 → 0.5564（**變差**）| 降噪反而傷害追蹤能力 |

### 結論

**假設否定：crash 不是採樣噪聲造成的，是結構性 distribution shift。**

1. Flow matching 1-step Euler：`x_0 = x_1 - v_θ(x_1, 1.0, cond)` — `cond` 信息量遠大於 `x_1` 的隨機性，縮小 `x_1` 振幅只會讓 action 集中到 `-v_θ(0, 1.0, cond)` 附近，不會改變 action 的「方向」
2. RMSE 隨 sigma 下降而變差，代表 policy 訓練時習慣了 sigma=1.0 的噪聲，eval 時改 sigma 反而 OOD

**真正的元兇：** Training rollout 自己就在 ~60 steps crash。整個訓練資料分佈是「初始狀態 → 60 steps → crash」，policy 從沒看過高傾角下如何主動修正、waypoint 接近時如何主動煞車、連續 100+ steps 的穩定飛行。這是 short-horizon RL + on-policy 的經典 coverage 問題，不是 noise 問題。

### 結果檔案

| 檔案 | 說明 |
|------|------|
| `evaluation_results/temperature_scaling/results_run19.json` | 4 sigma × 50 ep 完整結果 |
| `models/flow_policy_v4.py` | `predict_action` 新增 `temperature` 參數 |
| `scripts/evaluate_temperature_scaling.py` | 評估腳本 |

---

## 2. PID Baseline (Experiment A)

**Date:** 2026-05-06
**Purpose:** 建立傳統 PID position controller 的 baseline 數據，填補論文審稿人必問的空白

### 動機

論文目前有以下數據：

| 方法 | RMSE | Crash | 模式 |
|------|------|-------|------|
| PPO Expert (CTBR+INDI) | 0.065m | 0/50 | hover |
| ReinFlow Run 10 | 0.300m | 50/50 | waypoint 2.0m |
| BC (supervised only) | 0.522m | 50/50 | waypoint |

缺口：沒有傳統 PID 的數據。審稿人會直接問「PID 也 0 crash 嗎？如果是，你的 RL 貢獻是什麼？」

### 架構

4 層 Cascade PID，全部在 NED world frame：

```
pos_err (world NED)
  → Level 1: Position P  → vel_cmd = Kp_pos * pos_err  [clip ±vel_max]
  → Level 2: Velocity PI → accel_cmd = Kp_vel * vel_err + Ki_vel * ∫vel_err
  → Level 2b: Thrust decomposition
       thrust_vec = m * (g_vec - accel_cmd)       # from Newton's 2nd law
       F_total = ||thrust_vec||
       des_z   = thrust_vec / F_total             # desired body-Z in world
  → Level 3: Attitude P  → omega_cmd (SO3 error, body frame) [clip ±omega_max]
  → Level 4: Rate P      → torques = Kp_rate * omega_err     (body frame)
  → Motor inverse mixer  → 4D action ∈ [-1, 1]
```

**NED 物理推導：**
```
Newton: m*a = [0,0,m*g] - F_total * R[:,2]
→ F_total * R[:,2] = m*(g_vec - a_des) = thrust_vec
→ hover check: a_des=0 → thrust_vec=[0,0,m*g]=4.905N, R[:,2]=[0,0,1] ✓
```

**預設增益：**

| 參數 | 值 | 單位 | 說明 |
|------|----|------|------|
| Kp_pos | 1.5 | m/s / m | 1m pos_err → 1.5 m/s vel_cmd |
| Kp_vel | 3.0 | m/s² / m/s | 最大加速度 6 m/s²，最大傾角 ~31° |
| Ki_vel | 0.5 | m/s² / (m·s) | 消除穩態誤差 |
| vel_max | 2.0 | m/s | clip vel_cmd（避免過大傾角）|
| Kp_att | 8.0 | rad/s / rad | 攻角誤差 → 角速度指令 |
| omega_max | 2.0 | rad/s | clip omega_cmd |
| Kp_rate | 0.15 | Nm / (rad/s) | 速率誤差 → 扭矩 |

### 關鍵 Bug：SO3 姿態誤差旋轉順序寫反

**症狀：** 所有 episode 在 27–44 steps（< 1 秒）因 tilt > 60° 崩潰，甚至在 hover 模式也發生。

**根本原因：**

```python
# 錯誤（原始）：
R_err = R_des.T @ R      # = (R.T @ R_des).T = body-frame 誤差的轉置

# 正確：
R_err = R.T @ R_des      # = body-frame 誤差本身
```

Body-frame 誤差的轉置在 vee-map 後符號全部反號，導致每一個修正方向都反向 → 正反饋 → 不穩定。

**修復後的驗證：** hover 模式 500 steps 全通過，最大傾角僅 1.7°。

### 結果

**Hover 模式：**
```
Episodes:       50  (全部 500 steps)
Position RMSE:  0.0219 m  (+/- 0.0025 m)
Crashes:        0/50
Compute time:   ~177 us/step  (CPU only)
```

**Waypoint 模式（target_type=waypoint, range=2.0m）：**
```
Episodes:       50  (全部 500 steps)
Position RMSE:  1.177 m  (+/- 0.275 m)
Crashes:        0/50
```

**完整比較表：**

| 方法 | 評估模式 | RMSE | Crash | 備註 |
|------|---------|------|-------|------|
| **Cascade PID** | Hover | **0.022m** | **0/50** | 比 PPO Expert 好 3× |
| PPO Expert (CTBR+INDI) | Hover | 0.065m | 0/50 | 現有黃金標準（同模式） |
| **Cascade PID** | Waypoint 2.0m | 1.177m | **0/50** | 穩定但慢，無法即時追蹤 |
| ReinFlow Run 10 | Waypoint 2.0m | 0.300m | 50/50 | RL 最佳 eval，但全 crash |
| BC baseline | Waypoint 2.0m | 0.522m | 50/50 | 監督學習基線 |

### 結論與論文價值

**Waypoint 追蹤——PID 穩定但慢，RL 快但 crash：**
- PID waypoint：0/50 crash，但 RMSE 1.18m（waypoint 3 秒一換追不上）
- ReinFlow waypoint：50/50 crash，但 RMSE 0.30m（有辦法接近目標，但失去穩定性）

**核心論點：** 傳統控制器能維持穩定，但無法快速追蹤動態目標。學習方法能學到快速接近，但尚未解決 crash 問題。論文貢獻在「如何讓 RL 在快速追蹤時保持穩定」。

**PID 為何不 crash？**
1. 無隨機採樣噪聲
2. 顯式角速度反饋（Level 4 rate controller 直接控制 ω）
3. 顯式傾角限制（vel_max=2.0m/s 對應最大傾角 ~31°，遠低於 60° 終止線）

### 新增檔案

| 檔案 | 說明 |
|------|------|
| `controllers/pid_controller.py` | `CascadePIDController`（NED，SO3 誤差） |
| `controllers/__init__.py` | package |
| `scripts/evaluate_pid_baseline.py` | 50-episode 評估 |
| `evaluation_results/pid_baseline/results.json` | Hover 50-ep 結果 |
| `evaluation_results/pid_baseline/results_waypoint.json` | Waypoint 50-ep 結果 |

---

## 3. Run 21 "Unshackled DPPO"

**Date:** 2026-05-06
**Purpose:** 同時解開過去 20 個 run 互相掣肘的三個隱藏設定，測試 ReinFlow 在「正確 reward 地形 + 正確訓練分佈」下能否突破 50/50 crash 上限
**Pretrained from:** Phase 3a Flow Matching BC (`flow_policy_v4/20260420_034314/best_model.pt`)

### 動機

過去 20 個 run 都困在 RMSE 0.30–0.55m / 50 crash 的地板。Run 19/20 把訓練 reward 推到 0.6948 但 eval 完全不變。
溫度縮放實驗已排除「採樣噪聲是 crash 主因」這個假設。
診斷指向三個結構性問題，過去從未被同時處理：

| # | 隱藏問題 | 表現 |
|---|---------|------|
| 1 | Reward 缺乏煞車信號 | Policy 學到「衝向 target」但不會減速 → overshoot 後 crash |
| 2 | `w_action` 懲罰偏離 hover 的指令 | Policy 物理上不敢打高 body rate → 沒辦法執行煞車姿態 |
| 3 | `hover_anchor_prob` 把初始狀態變更簡單 | 訓練資料缺乏「需要恢復的危險狀態」→ 沒有 recovery 行為 |

### 改動（5 檔，最小化）

**`envs/quadrotor_dynamics.py`** — `reset()` 新增 `quaternion` 參數，允許從非單位姿態啟動。

**`envs/quadrotor_env_v4.py`** — 三向互斥 reset：

```python
anchor_prob = getattr(self, 'hover_anchor_prob',       0.0)
swift_prob  = getattr(self, 'swift_perturbation_prob', 0.0)
r = self.np_random.random()
if   r < anchor_prob:                # 安全 hover（防遺忘）
    pos_range_now=0.1, vel_range_now=0.05, init_tilt_deg=0
elif r < anchor_prob + swift_prob:   # Swift 擾動（學恢復）
    vel_range_now = swift_perturb_vel
    init_tilt_deg = U(0, swift_perturb_tilt_deg)
else:                                # 標準 curriculum
    ...
```

**`envs/quadrotor_env_v4.py`** — Radial brake formula：

```python
vel_radial = (vel · pos_error_xy) / (dist_xy + 1e-6)   # +ve 朝向 target
brake_pen  = w_brake * max(0, vel_radial)**2 * exp(-dist_xy / sigma_brake)
```

**`configs/quadrotor_v4.yaml`：**

| 參數 | 舊值 | 新值 | 理由 |
|------|------|------|------|
| `w_action` | 0.05 | **0.005** | 解開動作封印，允許激進操控 |
| `w_brake` | 0.0 | **0.1** | 開啟 radial brake |
| `sigma_brake` | 0.3 | **0.5** | 確保信號在現有 RMSE 操作距離仍可達 |

**`configs/reinflow_v4.yaml`：**

| 參數 | 舊值 | 新值 | 理由 |
|------|------|------|------|
| `hover_anchor_prob` | 0.2 | **0.1** | 釋出 10% 機率給 swift |
| `swift_perturbation_prob` | – | **0.2** | 強制 OOD 恢復訓練 |
| `swift_perturb_tilt_deg` | – | **30.0** | 初始最大傾角 |
| `swift_perturb_vel` | – | **1.0** | 初始最大速度 m/s |

機率分配：anchor 0.1 + swift 0.2 + normal 0.7 = 1.0。

新增 `self._current_max_tilt_deg`：swift episode 終止傾角放寬到 80°（給 30° 初始傾角留 50° 復原餘地）；eval 維持標準 60° — train/eval 在 termination 上對齊。

### 假設與成功判準

| 指標 | Run 19/20 | Run 21 預期 | 達成判準 |
|------|----------|------------|---------|
| Eval RMSE | 0.5226m | < 0.40m | RMSE 降至少 20% |
| Crashes | 50/50 | < 30/50 | 至少 40% episodes 存活到 500 steps |
| AvgCrashStep | 61.5 | > 100 | crash 步數翻倍 |

### 結果

**DENIED。** Run 21 的 swift episode 把 max_tilt_deg 放寬到 80°，訓練出「70° 也 OK」的 policy，但 eval 仍用 60° 終止 → policy 比以前更激進 → 平均存活只剩 27 steps（vs Run 19/20 的 62 steps）。

---

## 4. Hypothesis 1: Run 22 "PPO Clipped Surrogate"

**Date:** 2026-05-09
**Purpose:** 以真正的 PPO Clipped Surrogate 取代 ReinFlow 的 exp(beta*A) 加權 MSE loss，驗證 Hypothesis 1：21 個 run 困在 0.6948 training reward 上限的根本原因是優化器本身被閹割，而非 reward/init 設計問題
**Pretrained from:** Phase 3a Flow Matching BC (`flow_policy_v4/20260420_034314/best_model.pt`)
**Run timestamp:** `reinflow_v4_20260509_050911`

### 動機

`exp(beta * A_norm)` with beta=0.1：weights 動態範圍只有 exp(-0.3) ~ exp(0.3) ≈ 0.74x–1.35x。數學上等同於帶有微小 reward 偏差的 BC。Policy 實際上無法離 pretrained 太遠 → 21 個 run 都是在做 BC，RL 形同虛設。

### 關鍵數學：SDE Rollout 讓 Flow Matching 有 Likelihood

Flow Matching 1-step Euler 是確定性的，無法直接定義 likelihood ratio。解法：rollout 加 Gaussian 噪聲（SDE）：

```
x_1   ~ N(0, I)
mu_old = x_1 - v_theta_old(x_1, t=1, s)
eps    ~ N(0, I)
a      = mu_old + sigma * eps    # 實際送進 env 的 action
```

Likelihood ratio（正規化常數 sigma 固定，完全相消）：

```
log_ratio = -0.5/sigma^2 * [||a - mu_new||^2 - ||a - mu_old||^2]
```

### 改動

**`models/flow_policy_v4.py`** — 新增 `compute_clipped_loss`（保留 `compute_weighted_loss`，由 `loss_type` config 切換）

**`scripts/train_reinflow_v4.py`** — `collect_rollout` SDE 化；新增 TensorBoard scalars：`ppo/clip_fraction`, `ppo/approx_kl`, `ppo/mean_ratio`, `ppo/log_ratio_std`, `ppo/pct_action_oob`

**`configs/reinflow_v4.yaml` — Run 22 設定：**

| 參數 | 舊值（Run 21） | 新值 | 理由 |
|------|--------------|------|------|
| `loss_type` | _(新增)_ | `clipped` | PPO 替換 weighted MSE |
| `sde_noise_std` | _(新增)_ | `0.1` | 噪聲 = action 範圍的 10% |
| `clip_epsilon` | _(新增)_ | `0.2` | 標準 PPO |
| `learning_rate` | `1e-7` | `1e-5` | PPO trust region 是安全網 |
| `n_epochs` | `1` | `4` | PPO clipping 讓多 epoch 安全 |
| `total_updates` | `700` | `200` | 短期 sanity test |
| `lambda_bc` | `0.01` | `0.001` | PPO 已約束 policy |

### Sanity Check 結果

```
compute_clipped_loss OK
  clip_fraction=0.0000  (expect ~0 at first iter) ✓
  approx_kl=0.000000    (expect ~0 at first iter) ✓
  mean_ratio=1.000000   (expect ~1.0 at first iter) ✓
SANITY CHECK PASSED: mu_new == mu_old at first iter

Smoke test:
Update  2/5 | Reward: 0.7334 | RLLoss: -0.232907 | VLoss: 63.44
```

### 結果

**Hypothesis 1 DENIED。**

- PPO（Runs 22/22b/22c）training reward 峰值：**0.5884** < weighted MSE 0.6948
- `clip_fraction` 一直 > 0.70 — SDE noise σ 放大策略梯度，即使 σ=0.1 也有 50× 靈敏度
- 根本原因：50/50 crash 在 rollout 期間就毒化了 advantage 估計；不管什麼 optimizer 都無法從充滿 crash_penalty=-10 的 returns 中提取清楚信號。**瓶頸在訓練分佈，不在優化器。**

---

## 5. Hypothesis 2: DAgger Recovery Training

**Date:** 2026-05-09~12
**Context:** Hypothesis 1 否定後
**核心洞察：** 瓶頸在訓練分佈。Policy 從未見過「危險狀態的恢復軌跡」→ eval 遇到 OOD 就崩潰

### 理論

Swift 冠軍無人機論文的關鍵細節：
> 在每次環境重置時，代理會被初始化在賽道上，並在先前觀察到的狀態周圍加入有界的擾動 (bounded perturbation)

**DAgger Recovery 三步驟：**

| 步驟 | 動作 | 成功判準 |
|------|------|---------|
| Step 1 | PPO Expert 從危險狀態 (tilt 20-30°, vel 2 m/s) 收集 500 recovery 軌跡 | PPO success rate > 50% |
| Step 2 | 純 BC Gate：用 mixed dataset 重訓 Phase 3a，eval BC model | BC crash < 50/50 |
| Step 3 | 若 BC gate 通過，重啟 RL (PPO Clipped + brake penalty) | RMSE < 0.4m，crash < 30/50 |

### Step 1：Recovery 資料收集

腳本：`scripts/collect_data_v4_recovery.py`

- 所有 episode 都是 swift 模式（`swift_perturbation_prob=1.0`）
- 傾角：uniform[0°, 30°]（expected ≈ 15°）；速度：±2.0 m/s per axis；位置：±1.0m from target；終止傾角：80°
- 500 episodes：**90.2% PPO success rate**（成功判準通過）
- 結果：`data/expert_demos_v4_recovery.h5`（500 ep, 1.86GB）

### Step 2：BC Gate 測試（2026-05-11~12）

| 設定 | 值 |
|------|---|
| Hover | 500 episodes（475 train + 25 val）— 原 1000，因 RAM 限制降為 500 |
| Recovery | 500 episodes（475 train + 25 val）|
| 總計 | 446,700 train samples，10.98 GB |
| batch_size | 256（512 → CUDA OOM，已修正 config）|
| 訓練時間 | ~10.5 小時（500 epochs）|

訓練曲線：

| Epoch | Train loss | Val loss | 備注 |
|-------|-----------|---------|------|
| 1 | 0.5029 | 0.2001 | warmup 開始 |
| **65** | **0.0638** | **0.0671** | **best_model.pt** |
| 500 | 0.0247 | 0.1387 | 嚴重過擬合 |

BC Gate 評估結果：

```
Position RMSE: 2.3201 m   (原始 BC: 0.522m — 惡化 4.5×)
Crashes:       50/50
```

**BC Gate：FAILED。**

### 失敗原因

**RMSE 從 0.522m 惡化至 2.32m 的根因：** 50:50 hover:recovery 混合比例過激進。Recovery 軌跡帶有大幅非對稱推力（用於糾正 30° 傾角），model 從正常起點出發時試圖執行「恢復動作」→ 主動把自己搞崩。

根本瓶頸（確認）：**IMU encoder 梯度被 VisionEncoder 淹沒 46.8×** → 見 Hypothesis 3。

### 重要硬體問題（此次發現）

| 問題 | 原因 | 修正 |
|------|------|------|
| 1000+500 eps = 17.6 GB → OOM | 12 GB free RAM 不足 | `--hover-episodes 500` |
| batch_size=512 → CUDA OOM backward | Recovery mix 增加 activation 量 | batch_size: 256（已寫入 config）|
| nohup/pipe 在 Cygwin Bash 失效 | shell 生命週期問題 | 直接使用 `dppo/Scripts/python.exe` + `run_in_background=true` |

### 修改的檔案

- `scripts/collect_data_v4_recovery.py` — 新增（recovery 資料收集）
- `scripts/train_flow_v4.py` — FlowDatasetV4 支援多 h5 混合 + `--recovery-h5` arg

---

## 6. Hypothesis 3: IMU Encoder Gradient Bottleneck

**Date:** 2026-05-12
**Context:** Hypothesis 2 BC gate 失敗後啟動
**Root cause confirmed:** IMU encoder 梯度被 VisionEncoder 淹沒 46.8×，policy 無法感知姿態危機

### 診斷數據

```python
# FlowMatchingPolicyV4 梯度診斷（batch=4, random input）
VisionEncoder:     456,032 params  (4.1%)   grad_norm_sum = 4.3512
IMUEncoder:          2,528 params  (0.02%)  grad_norm_sum = 0.0929
ConditionalUNet: 10,571,784 params (95.8%)
Total:           11,030,344 params

Gradient ratio (VisionEncoder / IMUEncoder) = 46.8×
```

Policy 在 FPV 64×64 圖像中看不清楚 30° 傾角（只有幾個像素差），又忽視 IMU 中的角速度信號 → 無法感知危機 → 50/50 crash。

### 修復方案（Hypothesis 3a）

**策略：擴大 IMU encoder + 加入 auxiliary tilt supervision**

1. **IMU encoder 擴大**：6→64→32 改為 6→256→128（參數量 2.5k → 34k，13× 增加）
2. **Auxiliary tilt loss**：從 `imu_feat`（128D）預測 tilt angle，建立直接梯度路徑

代價：
- `global_cond_dim` 從 288D 變為 384D（256 vision + 128 IMU）
- `cond_dim`（UNet）從 416D 變為 512D（384 global + 128 time_embed）
- 需要重新訓練 Phase 3a（~14h）

**修改：**

```python
# models/flow_policy_v4.py
class FlowMatchingPolicyV4(nn.Module):
    def __init__(self, ..., imu_feature_dim=128, ...):  # 32→128
        self.imu_encoder = nn.Sequential(
            nn.Linear(6, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.tilt_head = nn.Linear(128, 1)  # imu_feat → tilt_angle_rad

def compute_loss(self, images, imu, actions, tilt_gt=None, lambda_tilt=0.1):
    ...
    imu_feat = self.imu_encoder(imu)
    tilt_loss = F.mse_loss(self.tilt_head(imu_feat).squeeze(-1), tilt_gt) if tilt_gt is not None else 0.0
    return flow_loss + lambda_tilt * tilt_loss
```

### 3a 執行結果（2026-05-12）

| 修改項目 | 數值 | 狀態 |
|---------|------|------|
| IMU encoder params | 2,528 → 34,688 (13×) | ✅ 完成 |
| Gradient ratio | 46.8× → **9.9×** (lambda_tilt=0.2) | ✅ < 10× 目標達成 |
| global_cond_dim | 288 → 384 | ✅ 正確 |

Phase 3a 訓練（hover + recovery 混合，500 ep × 2）：

| 指標 | 數值 | 目標 | 狀態 |
|------|------|------|------|
| 最佳 val loss | **0.0668 @ epoch 68** | < 0.080 | ✅ |
| Checkpoint | `flow_policy_v4/20260512_055304/best_model.pt` | — | — |

BC Gate 評估結果：

```
Position RMSE:  2.44m   (原始 BC: 0.522m — 惡化 4.7×)
Crashes:        49/50   (無顯著改善)
```

**結論：Hypothesis 3a DENIED。**

### 最終失敗原因（所有 Hypotheses 否定後的共識）

**問題根源是 recovery 資料毒害 hover BC，而非 IMU 感知能力：**

```
Hover demos:    微幅推力調整，懸停在 0.1m 附近
Recovery demos: 從 30° 傾角做出全推力激烈修正動作

50:50 混合訓練後：
  policy 在正常懸停狀態下也「幻覺」出需要激烈修正的動作
  → 主動自我失穩 → 比原始 BC 崩潰得更快
  → RMSE 2.44m >> 0.522m hover-only BC
```

**BC 階段無法同時學習「安靜懸停」與「激烈修復」兩種極端分佈。Recovery 學習責任必須 100% 交給 Phase 3c RL（PPO）。**

### IMU 架構升級為永久性改動

擴大後的 IMU encoder（34k params, grad ratio 9.9×）**保留**。只改變訓練資料策略：回歸 hover-only 訓練集。

### 替代方案（若後續仍失敗）

- **Hypothesis 3b：Cross-Attention Fusion** — IMU feature 作為 query，vision feature map 作為 key/value；代價：架構更改更大
- **Hypothesis 3c：更大 Image Size（64→96 或 128）** — 給 VisionEncoder 更高解析度；代價：需要重新收集資料，計算量 2-4×

---

## 下一步：Hover-Only BC + Unshackled RL

### Step 1：Hover-Only Phase 3a 重訓（~14h）

```bash
cd /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller
dppo/Scripts/python.exe -m scripts.train_flow_v4 \
    --hover-episodes 500
# 注意：Bash tool run_in_background=true，不要用 nohup
```

- 資料：`data/expert_demos_v4.h5` only（hover，500 ep，不混 recovery）
- 架構：現有擴大版 IMU encoder（imu_feature_dim=128，已 commit）
- 預期 val loss：< 0.065（有望優於原始 0.0630）

### Step 2：Unshackled RL（Phase 3c）

關鍵設定（Run 10 成功經驗 + 本輪診斷成果）：

| 參數 | 值 | 依據 |
|------|---|------|
| `sigma_pos` | 0.30 | 擴大獎勵盆地，drone 在 0.5m 外仍有梯度 |
| `lambda_bc` | 0.01 | Run 13-17 確認 0.1 會鎖死策略 |
| `w_action` | 0.005 | Run 21 經驗（解開動作封印）|
| Curriculum | 0.1m → 2.0m | Run 10 最佳 RMSE 0.3005m |
| Advantage hard clip | [-3, 3] | 保護 hover 權重免受 crash_penalty=-10 梯度爆炸 |

### 梯度診斷（Phase 3a 完成後驗證 ratio < 10×）

```bash
dppo/Scripts/python.exe -c "
import torch
from models.flow_policy_v4 import FlowMatchingPolicyV4
model = FlowMatchingPolicyV4(imu_feature_dim=128)
dummy = torch.randn(4, 6, 64, 64), torch.randn(4, 6), torch.randn(4, 4, 8)
loss = model.compute_loss(*dummy); loss.backward()
v = sum(p.grad.norm().item() for p in model.vision_encoder.parameters() if p.grad is not None)
i = sum(p.grad.norm().item() for p in model.imu_encoder.parameters() if p.grad is not None)
print(f'Ratio: {v/i:.1f}x  (target: < 10x)')
"
```

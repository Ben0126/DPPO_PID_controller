# Phase 3c 問題彙整與行動計畫
# Known Issues & Resolution Plan — DPPO Closed-Loop RL Fine-tuning

**日期：** 2026-04-11
**現況：** v3.2 DPPO Run 1 已手動終止。所有已知問題重新評估後，制定本計畫。
**目標：** RMSE < 0.168m（超越 Run 2 最佳），crash rate < 50/50。

---

## 問題總覽

| # | 問題 | 嚴重度 | 狀態 | 影響 |
|---|------|--------|------|------|
| P1 | Covariate shift（監督→RL） | 高 | 未解決 | 100% crash rate |
| P2 | Value network lag | 中 | 已修 | 早期優勢估計噪音大 |
| P3 | Policy collapse（β 過大） | 中 | 已修 | RL 更新摧毀 pretrained 知識 |
| P4 | Finite-difference IMU 分佈偏移 | 高 | 已修（v3.2 架構） | v3.1 全面失敗根因 |
| P5 | DR-aug pretrained 特徵模糊 | 中 | **待驗證** | Run 4 RMSE 0.409m vs Run 2 0.168m |
| P6 | IMU 輸入未歸一化 | 高 | **待修復** | v3.2 supervised RMSE 1.985m |
| P7 | Reset 第一步 IMU 異常 | 低 | **待修復** | 每集第一步 specific_force 異常值 |

---

## 問題詳解

### P1 — Covariate Shift（監督→RL）

**現象：** 監督式預訓練 policy 在所有評估中 crash rate 100%。DPPO fine-tuning 是必要條件，不可跳過。

**根因：** Policy 在訓練資料（expert 軌跡）之外的狀態分佈上行為退化。
監督訓練只覆蓋 expert 軌跡附近的狀態；RL rollout 時 policy 遭遇偏移狀態並雪球式崩潰。

**已知有效對策：**
- DPPO advantage-weighted fine-tuning（核心方法）
- 充足的 value net warm-up 使優勢估計收斂後再更新 policy

**剩餘風險：** 即使 DPPO 能減少 covariate shift，目前所有 run 仍 50/50 crash。
IMU fusion（v3.2）的動機之一就是提供額外動態信號幫助 policy 區分困難狀態。

---

### P2 — Value Network Lag（已修）

**現象（Run 1–3）：** VLoss > 1000 於前 100 updates，advantage 估計幾乎隨機，policy 在無意義梯度下崩潰。

**根因：** Value net 從隨機初始化開始，需要大量數據才能收斂，但 policy 更新與 value 更新同時進行。

**修復（v3.1 Run 2 起採用，Run 4 確認有效）：**
```yaml
# configs/diffusion_policy.yaml
dppo:
  value_warmup_updates: 50      # policy 凍結 50 updates，只更新 value net
  vloss_best_threshold: 500     # best ckpt 只在 VLoss < 500 時存檔
  value_hidden_dim: 512         # 較大 value net 容量
  value_lr: 3e-4
```
**驗證：** Run 4 VLoss 從 update 1 就 < 50（歷史最佳），u155 best ckpt 時 VLoss = 17。

---

### P3 — Policy Collapse（β 過大，已修）

**現象（Run 1：β=1.0）：** Update ~100 後 reward 急劇崩潰至負值，policy 遺忘 pretrained 知識。

**根因：** `exp(β × A_norm)` 是優勢加權因子。β=1.0 時，正優勢樣本權重可達 20×，梯度被少數高優勢樣本主導，相當於對 pretrained 參數做 large-step update。

**β 與最大加權的關係：**
| β | max weight（A_norm=3） |
|---|----------------------|
| 1.0 | 20.1× |
| 0.15 | 1.57× |
| 0.1 | 1.35× |
| **0.05** | **1.16×** |

**修復：** β = 0.05，加上 LR = 5e-6（保護 pretrained 知識）。Run 4 確認 β=0.05 可維持 500 updates 穩定，reward 0.47–0.56 範圍振盪但不崩潰。

---

### P4 — Finite-difference IMU 分佈偏移（已修，v3.2 架構）

**現象（v3.1 Run 1+2）：** RMSE 0.518/0.466m，遠差於無 IMU 監督基線（0.268m）。

**根因（數學）：**
```
(v_body[t] - v_body[t-1]) / dt  ≠  加速度計測量值

實際等於：
R^T a_world - ω×v_body   ← Coriolis 項在不穩定 rollout 時爆炸
```
PPO expert 收集時軌跡平滑，`ω×v_body` 接近零；RL rollout 時 policy 偏移，`ω×v_body` 隨角速度和速度放大。

**分佈偏移量化（expert vs 50%-random policy）：**
| 通道 | v3.1 std ratio | v3.2 std ratio |
|------|---------------|---------------|
| ax (specific force x) | **23.1×** | **1.4×** |
| ay (specific force y) | **16.3×** | **1.2×** |
| az | 2.8× | 1.5× |

**修復（v3.2）：** `QuadrotorDynamics.get_specific_force_body()` 回傳真實比力：
```python
def get_specific_force_body(self) -> np.ndarray:
    gravity_world = np.array([0.0, 0.0, self.params.mass * self.params.gravity])
    return (self._last_R.T @ (self._last_force_world - gravity_world)) / self.params.mass
```
`QuadrotorEnv.get_imu()` 為唯一呼叫點，回傳 `[gyro(3), specific_force(3)]`。

---

### P5 — DR-aug Pretrained 特徵模糊（待驗證）

**現象（Run 4）：** 訓練指標歷史最佳（VLoss=17, reward=0.5626），但 RMSE 0.409m，遠差於 Run 2（0.168m）。

**假設：**
| Pretrained | DR-aug | Supervised RMSE | DPPO best RMSE |
|-----------|--------|----------------|---------------|
| `20260402_032701`（原始） | 無 | 0.286m | **0.168m**（Run 2） |
| `20260405_044808`（Re-run 2） | 有（A+B） | 0.268m | 0.409m（Run 4） |

DR-aug 讓 encoder 學到更平滑、更 robust 的視覺特徵，但 feature space 更「模糊」。
DPPO 的 advantage-weighted gradient 在模糊 feature space 中難以找到清晰方向，fine-tuning 效率低。
原始 pretrained（無 DR-aug）的特徵更尖銳，微小 RL 更新即可產生有方向性的改變。

**驗證方式：**
用原始 pretrained（`20260402_032701`）+ Run 4 的改良 DPPO 配置，若 RMSE 明顯低於 0.409m，假設成立。

---

### P6 — IMU 輸入未歸一化（待修復，最高優先）

**現象（v3.2 supervised 評估）：** RMSE 1.985m（vs v3.1 supervised 0.453m、無 IMU 基線 0.268m）。

**根因：**
`specific_force` 在 hover 時值約為 `[0, 0, −9.81]` m/s²，激烈動作時可達 ±20 m/s²。
`IMUEncoder` 的 `Linear(6, 64)` 無輸入歸一化，`az ≈ −9.81` 的大幅偏移直接污染 `global_cond`（288D conditioning vector）的尺度。

v3.1 的 finite-difference accel 在 expert rollout 中均值接近 0（Coriolis 項小），`IMUEncoder` 初始化隱含地「適應」了零均值假設。v3.2 破壞了這個隱性假設。

**修復方案（需重新收集資料 + 重新 pretrain）：**

修改 `envs/quadrotor_env.py` 的 `get_imu()`：

```python
# 各通道的歸一化常數（根據物理意義決定）
_GYRO_SCALE    = 2.0    # rad/s，典型最大角速度
_SF_SCALE      = 5.0    # m/s²，除 gravity offset 後的典型比力幅度
_SF_Z_OFFSET   = -9.81  # m/s²，hover 時 az 的期望值

def get_imu(self) -> np.ndarray:
    """v3.2 normalized IMU signal — [gyro(3), specific_force_norm(3)]."""
    gyro  = self.dynamics.ang_velocity.astype(np.float32) / _GYRO_SCALE
    sf    = self.dynamics.get_specific_force_body().astype(np.float32)
    sf_n  = np.array([
        sf[0] / _SF_SCALE,
        sf[1] / _SF_SCALE,
        (sf[2] - _SF_Z_OFFSET) / _SF_SCALE,   # 中心化：hover 時接近 0
    ], dtype=np.float32)
    return np.concatenate([gyro, sf_n])
```

歸一化後各通道預期範圍：
| 通道 | hover 值 | 典型範圍 | 說明 |
|------|---------|---------|------|
| gyro_x/y/z | ~0 | [−1, +1] | 除以 2 rad/s |
| sf_x/y | ~0 | [−1, +1] | 除以 5 m/s² |
| sf_z（歸一後） | ~0 | [−1, +1] | (az − (−9.81)) / 5 |

---

### P7 — Reset 第一步 IMU 異常（待修復，低優先）

**現象：** `env.reset()` 後 `_last_force_world = zeros(3)`，第一步呼叫 `get_specific_force_body()` 時回傳 `[0, 0, +g]`（重力未被抵消），這個值在訓練資料中從未出現。

**根因：** `reset()` 初始化了 `_last_force_world = zeros(3)`，但在第一次 `step()` 呼叫之前快取尚未更新。

**修復方案：**
在 `QuadrotorDynamics.reset()` 中用 hover 推力估算初始 force：

```python
def reset(self, ...):
    ...
    # 用 hover 推力初始化快取，使第一步 specific_force ≈ [0, 0, -g]
    hover_force_per_motor = (self.params.mass * self.params.gravity) / 4.0
    total_hover_force     = self.params.mass * self.params.gravity
    # 初始姿態為水平，body Z 向上推力等於重力
    self._last_force_world = np.array([0.0, 0.0, -total_hover_force])
    self._last_R           = np.eye(3)
```

此修復可與 P6 一起在下次 pretrain 前套用。

---

## 行動計畫

### 策略決策

目前有兩條平行路線可選：

**路線 A：v3.2 + IMU 歸一化修復（延續 IMU fusion 路線）**
- 修復 P6（歸一化）和 P7（reset 初始化）
- 重新收集 v3.2 資料（~2h）+ 重新 supervised pretrain（~14h）
- 啟動 v3.2 DPPO Run 2
- 優點：若成功可同時解決 covariate shift 與 IMU 問題
- 風險：額外 ~16h 準備時間；IMU 仍可能不是 bottleneck

**路線 B：原始 pretrained + Run 4 配置（驗證 DR-aug 假設）**
- 直接用 `20260402_032701`（無 DR-aug）+ β=0.05, warm-up 50, VLoss threshold 500
- 無需額外資料收集或 pretrain，可立即啟動
- 優點：快速驗證 P5 假設；若成功（RMSE < 0.168m）貢獻不遜於 IMU fusion
- 風險：無 IMU，不能完整測試 v3.2 路線

**建議：先做路線 B（快速，無需等待），同步進行路線 A 的 IMU 歸一化修復。**

---

### 立即行動項目（按優先序）

#### Step 1 — 路線 B（可立即執行，~10h）

啟動 DPPO Run 5：原始 pretrained + Run 4 改良配置（無 IMU）：

```bash
cd DPPO_PID_controller
source dppo/Scripts/activate
nohup python -m scripts.train_dppo \
    --pretrained checkpoints/diffusion_policy/20260402_032701/best_model.pt \
    --total-updates 500 \
    > logs/train_dppo_run5_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

預期：若 P5 假設正確，RMSE 應 < 0.168m（突破 Run 2）。
判斷點：u155 附近的 best ckpt，評估後對比 Run 2/Run 4。

#### Step 2 — IMU 歸一化修復（在 Step 1 跑的同時進行）

修改以下兩個檔案（P6 + P7 同時修）：

1. `envs/quadrotor_env.py` — `get_imu()` 加入歸一化常數（見 P6 方案）
2. `envs/quadrotor_dynamics.py` — `reset()` 用 hover 推力初始化快取（見 P7 方案）

修改後必須重新執行 unit test 確認 hover specific_force 歸一化後 ≈ 0：
```bash
python -c "
from envs.quadrotor_env import QuadrotorEnv
import numpy as np
env = QuadrotorEnv()
env.reset()
# step with hover thrust
hover_action = np.full(4, -0.387)
env.step(hover_action)
imu = env.get_imu()
print('gyro:', imu[:3])          # 期望: ~[0,0,0]
print('sf_norm:', imu[3:])       # 期望: ~[0,0,0] (歸一化後)
"
```

#### Step 3 — v3.2 資料重收集與 pretrain（Step 2 完成後）

```bash
# 重新收集（使用歸一化的 get_imu()）
nohup python -m scripts.collect_data \
    --model checkpoints/ppo_expert/20260401_103107/best_model.pt \
    --norm  checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz \
    --output data/expert_demos_v32_norm.h5 --v32 \
    > logs/collect_v32_norm_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 完成後啟動 supervised pretrain
nohup python -m scripts.train_diffusion_v32 \
    --dataset data/expert_demos_v32_norm.h5 \
    > logs/train_diffusion_v32_norm_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### Step 4 — v3.2 DPPO Run 2（Step 3 完成後，Run 5 結果出爐後）

根據 Run 5 結果決定：
- 若 Run 5 RMSE < 0.168m → 原始 pretrained 假設成立 → v3.2 Run 2 應考慮用原始 pretrained 做 base（無 DR-aug）
- 若 Run 5 RMSE ≈ Run 4（0.4m+） → DR-aug 不是主因 → v3.2 IMU 歸一化是主要修復

---

## 成功判斷標準

### Run 5（路線 B 驗證）

| 指標 | 成功 | 中性 | 失敗 |
|------|------|------|------|
| Best ckpt RMSE | **< 0.168m** | 0.168–0.268m | > 0.268m |
| Crash rate | < 50/50 | 50/50 | — |
| VLoss @ u50 | < 100 | 100–500 | > 1000 |

### v3.2 Run 2（路線 A，歸一化修復後）

| 指標 | 成功 | 中性 | 失敗（IMU 假設否定） |
|------|------|------|------|
| Supervised RMSE（pretrain 後） | < 0.300m | 0.3–0.5m | > 0.5m → 歸一化仍不足 |
| DPPO best RMSE | **< 0.268m** | 0.268–0.4m | > 0.4m |
| IMU vs no-IMU 改進量 | > 10% | 0–10% | 負數 → 放棄 IMU 路線 |

---

## 如果全部路線都失敗的後備

若 Run 5 + v3.2 Run 2 均無法突破 0.268m 無 IMU 基線：

1. **Encoder Freezing：** DPPO 期間凍結 CNN encoder，只更新 diffusion UNet head。
   減少 RL 更新侵蝕視覺特徵的風險。

2. **DAgger-style 資料增強：** 在 RL rollout 的偏移狀態上混入少量 expert action，
   直接修復 covariate shift 而非用 IMU 補償。

3. **放棄 IMU fusion 路線：** 繼續純視覺 D²PPO，聚焦 oneDP 蒸餾（Phase 3d）
   達到推論速度目標，接受目前 0.168m RMSE 作為論文基線。

---

## 相關檔案

| 目的 | 路徑 |
|------|------|
| 歸一化修復（P6+P7） | `envs/quadrotor_env.py`, `envs/quadrotor_dynamics.py` |
| 原始 pretrained（路線 B） | `checkpoints/diffusion_policy/20260402_032701/best_model.pt` |
| v3.2 DPPO 腳本 | `scripts/train_dppo_v32.py` |
| 無 IMU DPPO 腳本 | `scripts/train_dppo.py` |
| 評估腳本（無 IMU） | `scripts/evaluate_rhc.py` |
| 評估腳本（v3.2） | `scripts/evaluate_rhc_v32.py` |
| 問題歷史 | `docs/dev_log_phase2_3.md §13–§17` |

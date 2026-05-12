# DPPO PID Controller — 完整架構圖（v4.0）

**最後更新：** 2026-05-12（Hypothesis 3a 完成後）

---

## 1. 整體研究流程（v4.0）

```
Phase 1               Phase 2               Phase 3a              Phase 3b/3c
─────────────         ─────────────         ─────────────         ─────────────
CTBR PPO Expert       Expert Demo           Flow Matching         ReinFlow
(State-based)   ──→   Collection      ──→   Pre-training    ──→   RL Fine-tuning
                      (FPV + IMU)           (hover demos)         (closed-loop)

QuadrotorEnv          QuadrotorEnvV4        FlowDatasetV4         QuadrotorEnvV4
  50Hz / 15D obs        1000 ep / 500k步      expert_demos_v4.h5    closed-loop RL
  6-DOF                 64×64 FPV + 6D IMU    500 epochs            PPO Clipped / ReinFlow
  RMSE 0.065m, 0 crash  3.9 GB h5             ~14h RTX 3090         best eval RMSE 0.3005m

Checkpoints:          Checkpoints:          Checkpoints:          Checkpoints:
ppo_expert_v4/        expert_demos_v4.h5    flow_policy_v4/       reinflow_v4/
20260419_142245/      (hover, 1000 ep)      20260420_034314/      reinflow_v4_*/
best_model.pt         expert_demos_v4_      best_model.pt         best_reinflow_model.pt
RMSE 0.065m           recovery.h5           val=0.0630
                      (recovery, 500 ep)    (hover-only)
```

---

## 2. 環境堆疊（v4.0）

```
┌─────────────────────────────────────────────────────────────────┐
│  QuadrotorEnvV4  (gymnasium Env, 50Hz outer loop)               │
│                                                                 │
│  obs = {                                                        │
│    "image": (T_obs×3, 64, 64) uint8  ← 2幀 stacked → (6,64,64) │
│    "imu":   (6,) float32             ← ω(3) + a_spec(3)        │
│  }                                                              │
│  action: (4,) CTBR thrusts ∈ [-1, 1]                           │
│                                                                 │
│  reward = +hover_bonus                                          │
│           − σ_pos^{-2} · ||pos_err||² / 2   (σ_pos=0.30)      │
│           − w_vel · ||vel||²                                    │
│           − w_action · ||action||²  (w_action=0.005)           │
│           − crash_penalty(−10) on tilt > 60° or OOB            │
│  healthy per-step: +0.3 ~ +0.6  |  collapse: < 0/step          │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Synthetic FPV Renderer  (每步呼叫)                       │   │
│  │                                                          │   │
│  │  ① Sky/Ground gradient  (pitch → horizon_y)             │   │
│  │  ② Horizon line         (roll → 傾斜角度)                │   │
│  │  ③ Target crosshair     (pinhole 投影, focal=0.3~0.5)   │   │
│  │  ④ Altitude bar         (左緣綠色條)                     │   │
│  │  ⑤ Center reticle       (白色十字中心)                   │   │
│  │                                                          │   │
│  │  Domain Randomization:                                   │   │
│  │    Per-episode: sky/ground 顏色偏移, 亮度(0.7~1.3),      │   │
│  │                 focal scale(0.30~0.50), crosshair 大小   │   │
│  │    Per-frame:   Gaussian pixel noise (σ=5)               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Physics IMU (6D, normalised)                            │   │
│  │                                                          │   │
│  │  ω(3)      = body angular rate  [rad/s] / 10.0           │   │
│  │  a_spec(3) = specific force     [m/s²]  / 30.0           │   │
│  │            = (thrust - gravity) rotated to body frame    │   │
│  │                                                          │   │
│  │  ← 直接查詢 QuadrotorDynamics，不用有限差分              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  QuadrotorDynamics  (200Hz inner loop, RK4)              │   │
│  │                                                          │   │
│  │  state: pos(3), vel(3), quat(4), ω(3) = 13D             │   │
│  │  RK4 積分 @ 200Hz → 外環 50Hz 執行 4 步內積分             │   │
│  │  重力 + 推力 + 阻力 + 陀螺力矩                            │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 模型架構 — FlowMatchingPolicyV4（11,358,313 params）

```
輸入 A: FPV image stack
  (B, T_obs×3, H, W) = (B, 6, 64, 64)  uint8 [0,255] / 255.0 → float [0,1]
  T_obs=2 frames 沿 channel 軸疊加 → 6ch

輸入 B: Physics IMU
  (B, 6)  = ω(3) + a_spec(3)，已歸一化（/10, /30）

                     ┌─────────────────────────────┐
  Image Stack   ──→  │   VisionEncoder (CNN)        │  ──→  (B, 256)  vis_feat
                     │   456,032 params             │
                     │                              │
                     │   Conv2d(6→32,  k=3,s=2,p=1) │  64×64 → 32×32
                     │   GroupNorm(8,32) + Mish      │
                     │   Conv2d(32→64, k=3,s=2,p=1) │  32×32 → 16×16
                     │   GroupNorm(8,64) + Mish      │
                     │   Conv2d(64→128,k=3,s=2,p=1) │  16×16 →  8×8
                     │   GroupNorm(8,128) + Mish     │
                     │   Conv2d(128→256,k=3,s=2,p=1)│   8×8 →  4×4
                     │   GroupNorm(8,256) + Mish     │
                     │   AdaptiveAvgPool2d(1)        │   4×4 →  1×1
                     │   Flatten → Linear(256, 256)  │
                     └─────────────────────────────-─┘

                     ┌─────────────────────────────┐
  IMU (6D)      ──→  │   IMUEncoder (MLP)           │  ──→  (B, 128)  imu_feat
                     │   34,688 params              │
                     │                              │
                     │   Linear(6, 256) + ReLU      │
                     │   Linear(256, 128) + ReLU    │
                     └──────────────────────┬───────┘
                                            │  [訓練時 only]
                                            │  TiltHead (129 params)
                                            ▼  Linear(128, 1) → tilt_pred (B,)
                                            │  L_tilt = MSE(tilt_pred, tilt_gt)
                                            │  (auxiliary supervision, λ_tilt=0.2)

cat([vis_feat(256), imu_feat(128)]) ──→ global_cond (B, 384)

         ┌──────────────────────────────────────────────┐
         │  global_cond ──→ [ReinFlow RL 訓練時]          │
         │                  ValueNetwork                  │
         │                  Linear(384→256) + ReLU        │
         │                  Linear(256→256) + ReLU        │
         │                  Linear(256→1)                 │
         │                  → V(s) scalar (GAE 計算用)    │
         └──────────────────────────────────────────────-─┘

         ┌──────────────────────────────────────────────────────────────┐
         │  ConditionalUnet1d  (flow velocity network v_θ)              │
         │  10,867,464 params                                           │
         │                                                              │
         │  Timestep embedding (training: t ∈ [0,1]; inference: t=1):  │
         │    SinusoidalPositionEmbeddings(128)                         │
         │    → Linear(128, 128) + Mish → t_emb (B, 128)               │
         │                                                              │
         │  cond = cat([global_cond(384), t_emb(128)]) = (B, 512)      │
         │                                                              │
         │  Input: x_t (B, action_dim=4, T_pred=8)                     │
         │         [training: interpolated; inference: x_1 ~ N(0,I)]   │
         │                                                              │
         │  ── Encoder ─────────────────────────────────────────────── │
         │                                                              │
         │  ResBlock(4→256, cond=512)   T:8                             │
         │  │  Conv1d(4→256,k=5) + GroupNorm(8,256) + Mish             │
         │  │  FiLM: h = h*(γ+1) + β   [γ,β ← Linear(512, 256×2)]    │
         │  │  Conv1d(256→256,k=5) + GroupNorm + Mish                  │
         │  │  Residual proj Conv1d(4→256,k=1)                         │
         │  └→ skip_1 (B, 256, 8)                                      │
         │  Downsample1d(256): Conv1d(256,256,3,s=2,p=1)  T:8→4        │
         │                                                              │
         │  ResBlock(256→512, cond=512)  T:4                            │
         │  └→ skip_2 (B, 512, 4)                                       │
         │  Downsample1d(512): Conv1d(512,512,3,s=2,p=1)  T:4→2        │
         │                                                              │
         │  ── Mid ─────────────────────────────────────────────────── │
         │                                                              │
         │  ResBlock(512→512, cond=512)  T:2                            │
         │                                                              │
         │  ── Decoder ─────────────────────────────────────────────── │
         │                                                              │
         │  Upsample1d(512): ConvTranspose1d(512,512,4,s=2,p=1) T:2→4  │
         │  cat([upsample, skip_2]) → (B, 1024, 4)                     │
         │  ResBlock(1024→256, cond=512)  T:4                           │
         │                                                              │
         │  Upsample1d(256): ConvTranspose1d(256,256,4,s=2,p=1) T:4→8  │
         │  cat([upsample, skip_1]) → (B, 512, 8)                      │
         │  ResBlock(512→4, cond=512)  T:8                              │
         │                                                              │
         │  Final Conv1d(4→4, k=1)                                      │
         └──────────────────────────────────────┬───────────────────────┘
                                                │
                              predicted velocity v̂_θ(x_t, t, cond)
                                       (B, 4, 8)
                                                │
                                                ▼
```

---

## 4. 推論流程 — Flow Matching 1-step Euler

```
隨機初始化
x_1 ~ N(0, I)  shape: (B, 4, T_pred=8)   [或 temperature scaling: N(0, σ²I)]
    │
    │  Flow: x_0 = x_1 - v_θ(x_1, t=1.0, global_cond)
    │  單步 Euler 積分  (t: 1.0 → 0.0)
    │
    ▼
action_seq: (B, 4, 8)   clamp[-1, 1]
    │
    │  shape 重排: (B, 4, 8) → (B, 8, 4)
    │  T_pred=8 步的 CTBR 指令序列
    │
    │  Inference latency:  ~8.2ms on RTX 3090 (~122Hz) ✓
    │  (舊 DDIM 10步: ~74ms @ ~14Hz)
    ▼
執行前 T_action=4 步，再重查 (RHC)
```

**訓練時 Flow Matching Loss：**
```
t ~ Uniform(0, 1)
x_0 = expert action,  x_1 ~ N(0, I)
x_t = (1-t)·x_0 + t·x_1             # 線性插值
u_t = x_1 - x_0                      # 目標 velocity (constant)

L_flow = MSE(v_θ(x_t, t, cond), u_t)
L_total = L_flow + λ_tilt · MSE(tilt_head(imu_feat), tilt_gt)   (λ_tilt=0.2)
```

---

## 5. RHC 閉環控制迴圈（推論時）

```
┌─────────────────────────────────────────────────────────────────────┐
│                      RHC Control Loop (50Hz env)                    │
│                                                                     │
│  環境重置                                                            │
│      │                                                              │
│      ▼                                                              │
│  image_buffer = [frame_0] × T_obs=2                                 │
│  imu = env.get_imu()                                                │
│      │                                                              │
│      └──────────────────────────────────────────────┐              │
│                                                      │ ← repeat     │
│  ┌── Observation ──────────────────────────────────┐│              │
│  │  img_stack = cat(image_buffer[-2:], axis=0)     ││              │
│  │  shape: (6, 64, 64)                             ││              │
│  │  imu: (6,) normalised physics IMU               ││              │
│  └─────────────────────┬───────────────────────────┘│              │
│                        │                             │              │
│  ┌── Inference (~8ms) ─▼───────────────────────────┐│              │
│  │  VisionEncoder → 256D                           ││              │
│  │  IMUEncoder    → 128D                           ││              │
│  │  global_cond   = cat([256, 128]) = 384D         ││              │
│  │  1-step Euler  → action_seq (8, 4)              ││              │
│  └─────────────────────┬───────────────────────────┘│              │
│                        │                             │              │
│  ┌── Execute T_action=4 ▼───────────────────────────┐│             │
│  │  step 0: env.step(action_seq[0]) → obs, reward  ││              │
│  │  step 1: env.step(action_seq[1]) → obs, reward  ││              │
│  │  step 2: env.step(action_seq[2]) → obs, reward  ││              │
│  │  step 3: env.step(action_seq[3]) → obs, reward  ││              │
│  │                                                  ││              │
│  │  image_buffer.append(new_frame) × 4              ││              │
│  │  imu = env.get_imu()            (last step)      ││              │
│  └─────────────────────┬───────────────────────────┘│              │
│                        │                             │              │
│                 done? ─┤ No ─────────────────────────┘              │
│                        │ Yes                                        │
│                        ▼                                            │
│                  episode end                                        │
│                                                                     │
│  latency: 8ms inference + 4×20ms steps = 88ms/cycle → 11Hz 重查    │
│  (目標 50Hz 達成於單步 = 8ms < 20ms ✓，RHC 週期為 4 步)             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. ReinFlow 訓練流程（Phase 3b）

```
┌────────────────────────────────────────────────────────────────────────┐
│  ReinFlow Training Loop                                                 │
│                                                                        │
│  ┌─ Rollout Collection (n_rollout_steps=4096) ──────────────────────┐  │
│  │                                                                  │  │
│  │  for step in range(4096):                                        │  │
│  │      global_cond = VisionEncoder(img) ‖ IMUEncoder(imu)  # nogr  │  │
│  │      V(s) = ValueNetwork(global_cond)          # GAE 計算用       │  │
│  │                                                                  │  │
│  │  [Option A: weighted MSE]                                        │  │
│  │      x_1 ~ N(0, I); action_seq = x_1 - v_θ(x_1, 1.0, cond)     │  │
│  │                                                                  │  │
│  │  [Option B: PPO Clipped (loss_type=clipped)]                     │  │
│  │      x_1 ~ N(0, I)                                               │  │
│  │      mu_old = x_1 - v_θ_old(x_1, 1.0, cond)   # rollout mu      │  │
│  │      action = mu_old + σ · ε,  ε ~ N(0,I)     # SDE noise σ=0.3  │  │
│  │      store: (img, imu, action, x_1, mu_old, reward, done, value)  │  │
│  │                                                                  │  │
│  │      Execute T_action=4 steps → rewards, new obs                 │  │
│  │                                                                  │  │
│  └──────────────────────────────────────────────┬───────────────────┘  │
│                                                 │                      │
│  ┌─ Advantage Estimation (GAE) ─────────────────▼───────────────────┐  │
│  │                                                                  │  │
│  │  δ_t = r_t + γ·V(s_{t+1})·(1−done_t) − V(s_t)                  │  │
│  │  A_t = Σ (γλ)^k · δ_{t+k}   [γ=0.99, λ=0.95]                   │  │
│  │  A_norm = (A − mean(A)) / (std(A) + ε)                          │  │
│  │                                                                  │  │
│  └──────────────────────────────────────────────┬───────────────────┘  │
│                                                 │                      │
│  ┌─ Policy Update (n_epochs=4) ─────────────────▼───────────────────┐  │
│  │                                                                  │  │
│  │  for epoch in range(4):                                          │  │
│  │    for batch in DataLoader(rollout, batch=256):                  │  │
│  │                                                                  │  │
│  │      ── [Option A] Weighted MSE Loss ──────────────────────────  │  │
│  │      cond = VisionEncoder(img) ‖ IMUEncoder(imu)   # with grad   │  │
│  │      t ~ Uniform(0,1);  x_t = (1-t)·action + t·x_1              │  │
│  │      v̂ = flow_net(x_t, t, cond)                                 │  │
│  │      per_sample = MSE(v̂, x_1 - action)  shape (B,)              │  │
│  │      weights = clamp(exp(β·A_norm), 0.1, 10.0)  β=0.1           │  │
│  │      L_policy = mean(weights × per_sample)                       │  │
│  │                                                                  │  │
│  │      ── [Option B] PPO Clipped Loss ───────────────────────────  │  │
│  │      mu_new = x_1 - flow_net(x_1, 1.0, cond)                    │  │
│  │      log_ratio = -0.5/σ² · [||a-mu_new||² - ||a-mu_old||²]      │  │
│  │      r = exp(log_ratio);  A_b = batch advantages                 │  │
│  │      L_clip = -mean(min(r·A, clip(r,1±ε)·A))  ε=0.2             │  │
│  │      L_policy = L_clip + λ_bc·L_bc            λ_bc=0.001        │  │
│  │                                                                  │  │
│  │      ── Value Loss ─────────────────────────────────────────     │  │
│  │      V_pred = ValueNetwork(cond.detach())                        │  │
│  │      L_value = MSE(V_pred, returns)                              │  │
│  │                                                                  │  │
│  │      ── Optimizers ─────────────────────────────────────────     │  │
│  │      policy_optim (AdamW, lr=1e-5):  L_policy.backward()        │  │
│  │      value_optim  (Adam,  lr=1e-3):  L_value.backward()         │  │
│  │                                                                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  VLoss Gate: policy update blocked until VLoss < 100  (value 先收斂)  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 資料流維度總覽

```
                               Shape                說明
                         ───────────────────────────────────────────────
輸入影像 (raw)          (2, 3, 64, 64)            T_obs幀 × RGB × H × W
影像 stack              (B, 6, 64, 64)            2幀疊合後送入 encoder
IMU 輸入                (B, 6)                    ω(3)/10 + a_spec(3)/30
VisionEncoder 輸出      (B, 256)                  視覺特徵
IMUEncoder 輸出         (B, 128)                  IMU 特徵（擴大後）
global_cond             (B, 384)                  cat([vision, imu])
timestep embedding      (B, 128)                  flow time t ∈ [0,1]
UNet conditioning       (B, 512)                  cat([global_cond, t_emb])
UNet 輸入 (x_t)         (B, 4, 8)                 action_dim × T_pred
UNet 輸出 (v̂_θ)        (B, 4, 8)                 predicted flow velocity
1-step Euler 動作       (B, 8, 4)                 T_pred × action_dim
執行動作 (單步)          (4,)                      4個馬達推力 ∈[-1,1]
Value 輸出              (B, 1)                    V(s) 純量

IMU grad ratio (v4.0)   VisionEncoder / IMUEncoder = 46.8×  (擴大前)
                                                  →  9.9×  (擴大後, λ_tilt=0.2)
```

---

## 8. 結果摘要

```
方法                    評估模式        RMSE     Crash    備註
────────────────────────────────────────────────────────────────────────
Cascade PID             Hover           0.022m   0/50    比 PPO Expert 好 3×
PPO Expert (CTBR+INDI)  Hover           0.065m   0/50    v4.0 黃金標準
Cascade PID             Waypoint 2.0m   1.177m   0/50    穩定但慢（每 3s 換目標）
BC (Flow Matching)      Waypoint 2.0m   0.522m   50/50   Covariate shift — 正常
ReinFlow Run 10         Waypoint 2.0m   0.300m   50/50   Best eval RMSE (v4.0)
ReinFlow Run 12         Waypoint 2.0m   0.298m   50/50   Hover quality ↑
ReinFlow Run 19/20      Waypoint 2.0m   0.523m   50/50   Training reward 0.695，eval 不變
DPPO v3.3 Run 1         Waypoint        0.104m   50/50   Best cross-arch RMSE (v3.x)

v4.0 training reward ceiling:  0.6948 @update200 (Runs 19-20)
v4.0 training-eval gap:        RL improves hover reward 0.529→0.695 但 eval crash 不變
Core blocker:                  Policy 從未見過危險狀態的 recovery 軌跡
```

---

## 9. 已知問題與根因

```
問題 1: Covariate Shift (100% crash rate on BC-only)
─────────────────────────────────────────────────────
  原因: 監督訓練 (Phase 3a) 使用靜態 expert 資料
        推論時累積誤差 → 影像/IMU 分布偏離訓練分布
  症狀: BC-only eval: RMSE 0.522m, 50/50 crash
  解法: ReinFlow closed-loop fine-tuning (Phase 3b)

問題 2: Training-Eval Gap (Runs 13-20)
──────────────────────────────────────
  原因: Policy 只學前 ~60 steps 的 reward 最大化
        訓練 rollout 全在 crash → short-horizon coverage
        eval 遇到 OOD 狀態（接近 target、傾角增大）→ 崩潰
  症狀: 訓練 reward 0.695，eval RMSE 0.52m（=BC 水準），crash 不變
  解法: Recovery 訓練責任給 Phase 3c RL（不是 BC 階段）

問題 3: Recovery Data 毒害 BC（Hypothesis 2/3 失敗原因）
─────────────────────────────────────────────────────────
  原因: Hover demos（微幅推力）+ Recovery demos（全推力激烈修正）
        50:50 混合 BC → policy 在正常起點幻覺「需要激烈修正」
        → 主動自我失穩 → RMSE 2.3m（>原始 BC 0.52m）
  結論: BC 無法同時學「安靜懸停」與「激烈修復」兩種極端分布
        Recovery 學習 100% 交給 RL 閉環訓練

問題 4: IMU Encoder 梯度瓶頸（已修復）
────────────────────────────────────────
  原因: 舊 IMUEncoder 6→64→32 (2,528 params) 梯度 = VisionEncoder 1/47
        Policy global_cond 被 vision 主導，IMU 訊號無效
  修復: 擴大到 6→256→128 (34,688 params) + auxiliary tilt loss
        梯度比 46.8× → 9.9×（< 10× 目標）
  狀態: 架構更新已 commit，需重訓 Phase 3a（hover-only）

問題 5: Value Net Lag
──────────────────────
  原因: VLoss > 10 直到 update ~20-30
        前期 advantage 估計幾乎是隨機雜訊
  解法: VLoss Gate（<100 才允許 policy update）+ value_lr=1e-3

問題 6: PPO 對 50/50 crash Rollout 的限制（已確認，Hypothesis 1 否定）
────────────────────────────────────────────────────────────────────────
  原因: 訓練 rollout 50/50 crash → returns 被 crash_penalty=-10 主導
        任何 optimizer（包含真正的 PPO Clip）都無法從毒化的 advantages 學習
        SDE noise σ=0.1 → sensitivity 50×；clip_fraction > 0.70
  確認: PPO peak reward 0.5884 < weighted MSE 0.6948
  結論: 瓶頸在訓練分布，不在優化器
```

---

## 10. 下一步：Hover-Only Phase 3a 重訓 + Unshackled RL

```
Step 1: Hover-Only Phase 3a 重訓（~14h）
  資料: data/expert_demos_v4.h5 only（hover，500 ep，不混 recovery）
  架構: 現有擴大版 IMU encoder（imu_feature_dim=128，已 commit）
  預期: val loss < 0.065（有望優於原始 0.0630）
  指令: dppo/Scripts/python.exe -m scripts.train_flow_v4 --hover-episodes 500

Step 2: Unshackled RL（Phase 3c）
  關鍵設定（基於 Run 10 成功 + 20+ run 診斷）：

  σ_pos = 0.30           擴大獎勵盆地（Run 18 修復）
  λ_bc  = 0.01           解除 BC 腳鐐（Run 13-17 確認 0.1 鎖死）
  w_action = 0.005       允許激進操控（Run 21 發現）
  curriculum 0.1m→2.0m   Run 10 最佳 RMSE 0.3005m
  Advantage hard clip [-3, 3]  保護免受 crash_penalty=-10 梯度爆炸
```

---

## 附錄：參數量對照

```
模組                     v3.3 (DDIM)      v4.0 current (Flow)
─────────────────────────────────────────────────────────────
VisionEncoder            456,032          456,032  (unchanged)
IMUEncoder               2,528            34,688   (13× 擴大)
TiltHead                 —                129      (training-only)
ConditionalUNet          ~10.47M          10,867,464
  (cond_dim)             (416D)           (512D)
Total                    ~10.93M          11,358,313
Inference method         DDIM 10步        1-step Euler
Inference latency        ~74ms / ~14Hz    ~8.2ms / ~122Hz
```

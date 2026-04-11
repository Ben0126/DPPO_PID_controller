# DPPO PID Controller — 完整架構圖

---

## 1. 整體研究流程

```
Phase 1               Phase 2               Phase 3a              Phase 3b              Phase 3d
─────────────         ─────────────         ─────────────         ─────────────         ──────────
PPO Expert            Expert Demo           Supervised            D²PPO                 OneDP
(State-based)   ──→   Collection      ──→   Pre-training    ──→   Fine-tuning     ──→   Distillation
                                                                                        (62Hz目標)

QuadrotorEnv          QuadrotorVisualEnv    DemoDataset           QuadrotorVisualEnv
  50Hz / 15D obs        1000 ep / 500k步      expert_demos.h5       closed-loop RL
  6-DOF 6DoF            64×64 RGB FPV         500 epochs            ~750 updates
  Run6: RMSE 0.069m     90MB h5 file          ~14h RTX 3090         β=0.1, LR=5e-6
```

---

## 2. 環境堆疊

```
┌─────────────────────────────────────────────────────────────────┐
│  QuadrotorVisualEnv  (gymnasium Wrapper)                        │
│                                                                 │
│  obs = { "image": (3,64,64) uint8,  "state": (15,) float32 }   │
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
│  │    Per-episode: sky/ground顏色偏移, 亮度(0.7~1.3),       │   │
│  │                 focal scale(0.30~0.50), crosshair大小    │   │
│  │    Per-frame:   Gaussian pixel noise (σ=5)              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                         │                                       │
│  ┌──────────────────────▼───────────────────────────────────┐   │
│  │  QuadrotorEnv  (50Hz outer loop)                         │   │
│  │                                                          │   │
│  │  obs (15D): pos(3) + vel(3) + quat(4) + ω(3) + err(2)   │   │
│  │  action (4D): motor thrusts ∈ [-1, 1]                    │   │
│  │                                                          │   │
│  │  reward = +hover_bonus − pos_error − vel_penalty         │   │
│  │           − crash_penalty(−10) − attitude_penalty        │   │
│  │  healthy per-step: +0.3 ~ +0.6                           │   │
│  │  collapse sign: < 0/step                                 │   │
│  │                                                          │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  QuadrotorDynamics  (200Hz inner loop, RK4)         │  │   │
│  │  │                                                    │  │   │
│  │  │  state: pos(3), vel(3), quat(4), ω(3) = 13D        │  │   │
│  │  │  RK4 积分 @ 200Hz → 外環 50Hz 執行 4 步內積分       │  │   │
│  │  │  重力 + 推力 + 阻力 + 陀螺力矩                      │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 模型架構 — VisionDiffusionPolicy（10,929,256 params）

```
輸入: FPV image stack
shape: (B, T_obs×C, H, W) = (B, 6, 64, 64)  uint8 [0,255]
  │
  │  T_obs=2 frames 沿 channel 軸疊加 → 6ch
  │
  ▼
┌────────────────────────────────────────────────┐
│  VisionEncoder (CNN)                           │
│                                                │
│  Conv2d(6→32,  k=3,s=2,p=1)  64×64 → 32×32   │
│  GroupNorm(8,32) + Mish                        │
│                                                │
│  Conv2d(32→64, k=3,s=2,p=1)  32×32 → 16×16   │
│  GroupNorm(8,64) + Mish                        │
│                                                │
│  Conv2d(64→128,k=3,s=2,p=1)  16×16 →  8×8    │
│  GroupNorm(8,128) + Mish                       │
│                                                │
│  Conv2d(128→256,k=3,s=2,p=1)  8×8 →  4×4     │
│  GroupNorm(8,256) + Mish                       │
│                                                │
│  AdaptiveAvgPool2d(1) → Flatten → (B, 256)    │
│  Linear(256 → 256)                             │
└───────────────────┬────────────────────────────┘
                    │ (B, 256) vision_feat
        ┌───────────┴──────────────────┐
        │                              │  [DPPO 訓練時]
        ▼                              ▼
        │                    ┌───────────────────┐
        │                    │  ValueNetwork     │
        │                    │  Linear(256→256)  │
        │                    │  ReLU             │
        │                    │  Linear(256→256)  │
        │                    │  ReLU             │
        │                    │  Linear(256→1)    │
        │                    └────────┬──────────┘
        │                             │ V(s) scalar
        │                             │ 用於 GAE 計算
        │
        ▼ (B, 256) condition
        │
        │  cat with timestep embedding
        │
┌───────┴──────────────────────────────────────────────────────────┐
│  ConditionalUnet1D                                               │
│                                                                  │
│  Timestep embedding:                                             │
│    SinusoidalPositionEmbeddings(128) → Linear(128) → Mish       │
│    → t_emb (B, 128)                                              │
│                                                                  │
│  cond = cat([vision_feat(256), t_emb(128)]) = (B, 384)          │
│                                                                  │
│  Input: noisy_action (B, action_dim=4, T_pred=8)                │
│                                                                  │
│  ── Encoder ──────────────────────────────────────────────────  │
│                                                                  │
│  ResBlock(4→256, cond=384)  T:8    ← FiLM: scale+shift from 384D│
│  │  Conv1d(4→256,k=5) + GroupNorm                               │
│  │  FiLM: h = h*(γ+1) + β   [γ,β ← MLP(cond)]                 │
│  │  Conv1d(256→256,k=5) + GroupNorm + Mish                      │
│  │  Residual proj Conv1d(4→256,k=1)                             │
│  └→ skip_1 (B,256,8)                                            │
│  Downsample1d(256) stride=2    T:8→4                             │
│                                                                  │
│  ResBlock(256→512, cond=384)  T:4                                │
│  └→ skip_2 (B,512,4)                                             │
│  Downsample1d(512) stride=2    T:4→2                             │
│                                                                  │
│  ── Mid ───────────────────────────────────────────────────────  │
│                                                                  │
│  ResBlock(512→512, cond=384)  T:2                                │
│                                                                  │
│  ── Decoder ──────────────────────────────────────────────────  │
│                                                                  │
│  Upsample1d(512) ConvTranspose1d stride=2  T:2→4                │
│  cat([upsample, skip_2]) → (B,1024,4)                            │
│  ResBlock(1024→256, cond=384)  T:4                               │
│                                                                  │
│  Upsample1d(256) ConvTranspose1d stride=2  T:4→8                │
│  cat([upsample, skip_1]) → (B,512,8)                             │
│  ResBlock(512→4, cond=384)  T:8                                  │
│                                                                  │
│  Final Conv1d(4→4, k=1)                                          │
└───────────────────────┬──────────────────────────────────────────┘
                        │ predicted noise ε̂ (B, 4, 8)
                        ▼
```

---

## 4. 推論流程 — DDIM 10步去噪

```
隨機初始化
x_T ~ N(0, I)  shape: (B, 4, 8)
    │
    │  cosine noise schedule (100訓練步 → 10推論步)
    │  step_size = 100 // 10 = 10
    │  timesteps = [90, 80, 70, 60, 50, 40, 30, 20, 10, 0]  (倒序)
    │
    ├─ iter 1: t=90 ──────────────────────────────────────────┐
    │   ε̂ = UNet(x_90, t=90, vision_feat)                     │
    │   x̂₀ = (x_90 − √(1−ᾱ₉₀)·ε̂) / √ᾱ₉₀  [clamp −5~5]    │
    │   x_80 = √ᾱ₈₀·x̂₀ + √(1−ᾱ₈₀)·ε̂  [η=0, 確定性]        │  ×10步
    ├─ iter 2: t=80 ──────────────────────────────────────────┤
    │   ...                                                   │
    └─ iter 10: t=0 ──────────────────────────────────────────┘
    │
    ▼
action_seq: (B, 4, 8)
    │ permute(0,2,1)
    ▼
action_seq: (B, 8, 4)  clamp[-1, 1]
    │
    │  T_pred=8 步的馬達推力序列
    │  ~74ms total on RTX 3090 (~14Hz)
    ▼
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
│      │                                                              │
│      └──────────────────────────────────────────────┐              │
│                                                      │ ← repeat     │
│  ┌── Observation ──────────────────────────────────┐│              │
│  │  img_stack = cat(image_buffer[-2:], axis=0)     ││              │
│  │  shape: (6, 64, 64)                             ││              │
│  └─────────────────────────┬───────────────────────┘│              │
│                            │                         │              │
│  ┌── Inference (~74ms) ────▼───────────────────────┐│              │
│  │  VisionEncoder → 256D                           ││              │
│  │  DDIM 10步去噪                                   ││              │
│  │  → action_seq: (8, 4)                           ││              │
│  └─────────────────────────┬───────────────────────┘│              │
│                            │                         │              │
│  ┌── Execute T_action=4 ───▼───────────────────────┐│              │
│  │  step 0: env.step(action_seq[0]) → obs, reward  ││              │
│  │  step 1: env.step(action_seq[1]) → obs, reward  ││              │
│  │  step 2: env.step(action_seq[2]) → obs, reward  ││              │
│  │  step 3: env.step(action_seq[3]) → obs, reward  ││              │
│  │                                                  ││              │
│  │  image_buffer.append(new_frame) × 4              ││              │
│  └─────────────────────────┬───────────────────────┘│              │
│                            │                         │              │
│                     done? ─┤ No ─────────────────────┘              │
│                            │ Yes                                    │
│                            ▼                                        │
│                      episode end                                    │
│                                                                     │
│  ⚠ 速度問題: 每次 inference 74ms，執行4步=4×20ms=80ms              │
│    → 有效重查頻率 ~12Hz，遠低於 50Hz 控制需求                        │
│    → 步驟 1~3 使用的是 t=0 時的舊影像預測，造成 covariate shift     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. D²PPO 訓練流程（Phase 3b）

```
┌────────────────────────────────────────────────────────────────────────┐
│  D²PPO Training Loop                                                   │
│                                                                        │
│  ┌─ Rollout Collection (n_steps=4096) ──────────────────────────────┐  │
│  │                                                                  │  │
│  │  for step in range(4096):                                        │  │
│  │      vis_feat = VisionEncoder(img_stack)      # no grad          │  │
│  │      V(s) = ValueNetwork(vis_feat)            # 用於 GAE         │  │
│  │      action_seq = DDIM(vis_feat)              # 10步, no grad     │  │
│  │      Execute T_action=4 steps → rewards                          │  │
│  │      Store: (img_stack, action_seq, reward, done, value)         │  │
│  │                                                                  │  │
│  └──────────────────────────────────────────────┬───────────────────┘  │
│                                                 │                      │
│  ┌─ Advantage Estimation (GAE) ─────────────────▼───────────────────┐  │
│  │                                                                  │  │
│  │  δ_t = r_t + γ·V(s_{t+1})·(1−done_t) − V(s_t)                  │  │
│  │  A_t = Σ (γλ)^k · δ_{t+k}   [γ=0.99, λ=0.95]                   │  │
│  │                                                                  │  │
│  │  A_norm = (A − mean(A)) / (std(A) + ε)                          │  │
│  │                                                                  │  │
│  └──────────────────────────────────────────────┬───────────────────┘  │
│                                                 │                      │
│  ┌─ Policy Update (n_epochs=3) ─────────────────▼───────────────────┐  │
│  │                                                                  │  │
│  │  for epoch in range(3):                                          │  │
│  │    for batch in DataLoader(rollout, batch=256):                  │  │
│  │                                                                  │  │
│  │      ── DPPO Loss ──────────────────────────────────────────     │  │
│  │      vis_feat = VisionEncoder(img_stack)   # with grad           │  │
│  │      ε̂ = UNet(q_sample(action_seq, t), t, vis_feat)             │  │
│  │      per_sample_loss = MSE(ε̂, ε)  shape (B,)                    │  │
│  │      weights = clamp(exp(β·A_norm), 0.1, 10.0)  β=0.1            │  │
│  │      L_policy = mean(weights × per_sample_loss)                  │  │
│  │                                                                  │  │
│  │      ── Value Loss ─────────────────────────────────────────     │  │
│  │      V_pred = ValueNetwork(vis_feat.detach())                    │  │
│  │      L_value = MSE(V_pred, returns)                              │  │
│  │                                                                  │  │
│  │      ── Optimizers ─────────────────────────────────────────     │  │
│  │      policy_optim (AdamW, lr=5e-6): L_policy.backward()         │  │
│  │      value_optim  (Adam,  lr=3e-4): L_value.backward()          │  │
│  │                                                                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 資料流維度總覽

```
                            Shape              說明
                         ──────────────────────────────────────────
輸入影像 (raw)          (2, 3, 64, 64)       T_obs幀 × RGB × H × W
影像 stack             (1, 6, 64, 64)        疊合後送入 encoder
VisionEncoder 輸出     (1, 256)              視覺特徵
timestep embedding     (1, 128)              DDIM 當前去噪步
UNet conditioning      (1, 384)              cat([視覺, 時步])
UNet 輸入 (noise)      (1, 4, 8)             action_dim × T_pred
UNet 輸出 (ε̂)         (1, 4, 8)             預測噪聲
DDIM 最終動作          (1, 8, 4)             T_pred × action_dim
執行動作 (單步)        (4,)                  4個馬達推力 ∈[-1,1]
Value 輸出             (1, 1)                V(s) 純量
```

---

## 8. 已知問題與根因

```
問題 1: Covariate Shift (100% crash rate)
─────────────────────────────────────────
  原因: 監督訓練 (Phase 3a) 使用靜態 expert 資料
        推論時累積誤差 → 影像分布偏離訓練分布
  症狀: 孤立 supervised model: RMSE 0.286m, 50/50 crash
  解法: D²PPO closed-loop fine-tuning (Phase 3b)

問題 2: Policy Collapse (D²PPO Run 1)
──────────────────────────────────────
  原因: β=1.0 → exp(1.0×A) 最大權重 ~20×, 梯度爆炸
        LR=3e-5 → 覆蓋預訓練權重
  症狀: update ~100 後 per-step reward 從正數 → 接近0 → 負數
  解法: β=0.1 (Run 2), LR=5e-6

問題 3: Value Net Lag
──────────────────────
  原因: Value loss > 5 直到 update ~150
        前 150 updates 的 advantage 估計幾乎是隨機雜訊
  症狀: Run 2 在 update 11 最佳 (RMSE 0.168m), 之後退化
  解法: 待 Phase 3c — ValueNetworkV31 + 更好初始化

問題 4: 推論速度 (14Hz vs 50Hz)
──────────────────────────────────
  原因: DDIM 10步 = ConditionalUnet1D × 10 forward pass = ~74ms
        馬達控制需要 50Hz = 20ms/step
  症狀: RHC 只能每 80ms 重查一次 (T_action=4 × 20ms)
  解法: Phase 3d OneDP 單步蒸餾 → 目標 62Hz (單次 forward)
```

---

## 9. 未來架構 v3.1（Phase 3c, 待實作）

```
FPV image stack (6×64×64)  ─→  VisionEncoder  ─→  256D vision_feat
6D IMU [ω,a]               ─→  IMUEncoder      ─→   32D imu_feat
                                  MLP(6→64→32)
                                                      │
                                         cat([256D, 32D])
                                                      │
                                              288D global_cond
                                            /               \
                              ConditionalUnet1D          [訓練時]
                              288D+128D→416D cond      DepthDecoder
                                      │                FCN(256→1×64×64)
                                      ▼                depth_pred
                                  action_seq
                                      │
                         L = exp(β×A) × L_diff
                           + λ_disp × L_dispersive   ← 防止特徵崩塌
                           + λ_depth × MSE(depth)    ← depth 輔助監督

ValueNetworkV31(global_cond_dim=288) → V(s)
```

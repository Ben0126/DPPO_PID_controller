# Vision-DPPO Research Plan: End-to-End Drone Control via Diffusion Policy
# 基於視覺與擴散策略的無人機端到端控制研究計劃

## Executive Summary / 執行摘要

This document outlines the research and development plan for **Vision-DPPO**, an end-to-end visuomotor drone controller that uses a **Diffusion Policy** to directly map FPV image sequences to 4D motor thrust commands. Unlike traditional cascaded PID approaches, Vision-DPPO learns a single policy that replaces the entire perception-planning-control stack.

本文件概述 **Vision-DPPO** 的研究與開發計劃，這是一個端到端視覺運動無人機控制器，使用**擴散策略**直接將 FPV 圖像序列映射到 4D 電機推力命令。與傳統級聯 PID 方法不同，Vision-DPPO 學習單一策略來替代整個感知-規劃-控制堆棧。

### Research Pipeline / 研究管線

```
Phase 1: 6-DOF Quadrotor Env + PPO Expert (State-based)
   ↓
Phase 2: FPV Rendering + Expert Data Collection (HDF5)
   ↓
Phase 3: Vision Diffusion Policy (Supervised + DPPO)
   ↓
Phase 4: Closed-Loop RHC Evaluation
   ↓
Phase 5: Hardware Deployment (NVIDIA Orin + PX4)
```

---

## Phase 1: Quadrotor Environment & State-Based PPO Expert

### 1.1 Quadrotor Dynamics (`envs/quadrotor_dynamics.py`)

**State vector (13D)**: position [x,y,z], quaternion [qw,qx,qy,qz], linear velocity [vx,vy,vz], angular velocity [ωx,ωy,ωz]

**Coordinate frame**: NED (North-East-Down), body frame X-forward/Y-right/Z-down

**Equations of motion**:

```
m · dv/dt = R(q) · [0, 0, F_total]^T - [0, 0, m·g]^T + F_drag
I · dω/dt = τ - ω × (I · ω)
dq/dt = 0.5 · q ⊗ [0, ω]
```

**Motor mixing (X-configuration)**:

```
F_total = Σ f_i
τ_x = L · (f1 - f2 - f3 + f4)       # Roll
τ_y = L · (f1 + f2 - f3 - f4)       # Pitch
τ_z = c_τ · (-f1 + f2 - f3 + f4)    # Yaw
```

**Integration**: RK4 at 200 Hz (dt = 0.005s)

**Motor dynamics**: First-order lag τ_motor = 0.02s

```
f_actual += (f_commanded - f_actual) · dt / τ_motor
```

**Physical parameters** (configurable via `configs/quadrotor.yaml`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Mass (m) | 0.5 kg | Total mass |
| Arm length (L) | 0.17 m | Motor to center distance |
| Inertia (I) | diag(2.5e-3, 2.5e-3, 4.5e-3) kg·m² | Moment of inertia |
| Max thrust per motor | 3.0 N | Motor saturation limit |
| Torque coefficient (c_τ) | 0.01 | Thrust-to-torque ratio |
| Drag coefficient | 0.1 | Linear drag |
| Motor time constant | 0.02 s | First-order lag |
| Gravity (g) | 9.81 m/s² | Standard gravity |

**Quaternion utilities**: `quaternion_multiply`, `quaternion_normalize`, `quaternion_to_rotation_matrix`, `quaternion_derivative`, `rotation_matrix_to_6d`, `get_tilt_angle`

### 1.2 Gymnasium Environment (`envs/quadrotor_env.py`)

**Observation space (15D)**:

| Index | Content | Description |
|-------|---------|-------------|
| 0-2 | R^T · (p_target - p) | Position error in body frame |
| 3-8 | 6D rotation representation | First 2 columns of rotation matrix R (avoids gimbal lock) |
| 9-11 | R^T · v | Linear velocity in body frame |
| 12-14 | ω | Angular velocity in body frame |

**Action space (4D)**: Normalized motor thrusts [-1, 1] → mapped to [0, f_max]

**Reward function** (Gaussian-based, bounded):

```
R = w_pos · exp(-||pos_err||² / σ_pos)     # Position tracking (w=0.6, σ=0.5)
  + w_vel · exp(-||vel||² / σ_vel)          # Velocity damping (w=0.2, σ=1.0)
  + w_ang · exp(-||ang_vel||² / σ_ang)      # Angular rate damping (w=0.1, σ=1.0)
  - w_action · ||action||²                  # Control effort penalty (w=0.05)
  + alive_bonus                             # Survival bonus (0.1 per step)
```

**Termination conditions**:
- Position out of bounds: ||p|| > 5.0 m
- Excessive tilt: tilt angle > 60°
- Ground contact: Z > 0 (NED frame, i.e., below ground)

**Timing**: Inner loop 200 Hz, outer loop 50 Hz (4 inner steps per RL decision)

### 1.3 PPO Expert (`models/ppo_expert.py`)

Adapted from custom PPO implementation with direct motor control (no PID).

**Architecture**:

| Component | Specification |
|-----------|--------------|
| State dim | 15 |
| Action dim | 4 |
| Hidden dim | 256 |
| Hidden layers | 2 (Actor and Critic) |
| Activation | ReLU |
| Distribution | TanhNormal (squashed Gaussian) |
| Output range | [-1, 1] (motor thrusts) |

**Training** (`scripts/train_ppo_expert.py`, `configs/ppo_expert.yaml`):
- Total timesteps: 5,000,000
- Learning rate: 3e-4
- GAE: γ=0.99, λ=0.95
- PPO clip range: 0.2
- Entropy coefficient: 0.01
- Batch size: 4096 steps per rollout
- Observation normalization: RunningMeanStd

**Verification target**: Stable hover (position error < 0.1m) and waypoint tracking within 100 episodes.

### 1.4 Configuration (`configs/quadrotor.yaml`, `configs/ppo_expert.yaml`)

YAML-based configuration with sections for:
- `quadrotor`: physics parameters, motor model
- `timing`: dt, rl_dt, inner_steps
- `reward`: Gaussian weights (σ and w for each term)
- `termination`: bounds, max_tilt, max_steps
- `disturbance`: wind model parameters
- `training`: PPO hyperparameters

---

## Phase 2: FPV Rendering & Expert Data Collection

### 2.1 Visual Environment (`envs/quadrotor_visual_env.py`)

Wraps `QuadrotorEnv` as a `gym.Wrapper`, renders synthetic FPV images (64×64 RGB).

**FPV rendering features**:
- **Sky/ground gradient**: Based on drone altitude
- **Horizon line**: Rotated according to drone roll/pitch attitude
- **Target crosshair**: Projected target position on image plane
- **Altitude indicator**: Vertical bar showing current altitude

**Observation space**: `Dict({"image": Box(0, 255, (3, 64, 64), uint8), "state": Box(-inf, inf, (15,))})`

### 2.2 Data Collection (`scripts/collect_data.py`)

**Process**:
1. Load trained PPO expert + RunningMeanStd normalization
2. Roll out in `QuadrotorVisualEnv` for ~1000 episodes
3. Record per-step: FPV image (uint8), motor action (float32), state (float32)

**HDF5 dataset format**:

```
data/expert_demos.h5
  /episode_0/images:  (T, 3, 64, 64) uint8, gzip compressed
  /episode_0/actions: (T, 4) float32
  /episode_0/states:  (T, 15) float32
  /episode_1/...
  ...
```

### 2.3 Dataset for Diffusion Training (`models/diffusion_policy.py: DemoDataset`)

Sliding window over episodes:
- **Input**: Image stack of T_obs=2 consecutive frames → (6, 64, 64)
- **Target**: Future action sequence of T_pred=8 motor commands → (8, 4)

**Verification target**: >100K image-action pairs, visually inspect FPV renders for correctness.

---

## Phase 3: Vision Diffusion Policy (CORE RESEARCH)

### 3.1 Vision Encoder (`models/vision_encoder.py`)

Lightweight CNN that extracts visual features from FPV image stacks.

```
Input: (B, T_obs × C, H, W) = (B, 6, 64, 64)

Conv2d(6, 32, 3, stride=2, pad=1)  → GroupNorm(8) → Mish    # (B, 32, 32, 32)
Conv2d(32, 64, 3, stride=2, pad=1) → GroupNorm(8) → Mish    # (B, 64, 16, 16)
Conv2d(64, 128, 3, stride=2, pad=1) → GroupNorm(8) → Mish   # (B, 128, 8, 8)
Conv2d(128, 256, 3, stride=2, pad=1) → GroupNorm(8) → Mish  # (B, 256, 4, 4)
AdaptiveAvgPool2d(1)  → Flatten → Linear(256, feature_dim)

Output: (B, 256)  — fixed during diffusion denoising iterations
```

### 3.2 Conditional 1D U-Net (`models/conditional_unet1d.py`)

Operates on action sequences with shape `(B, action_dim=4, T_pred=8)`.

**Conditioning**: `cond_dim = feature_dim(256) + time_embed_dim(128) = 384`

**Architecture**:

```
                    Condition: visual_features + sinusoidal_time_embedding
                         ↓ (384D)
Input: (B, 4, 8) ──→ ConditionalResBlock1d(4→256, FiLM)
                      ↓ Downsample (8→4)
                      ConditionalResBlock1d(256→512, FiLM)
                      ↓ Downsample (4→2)
                      ConditionalResBlock1d(512→512, FiLM)  ← Mid
                      ↑ Upsample (2→4) + skip connection
                      ConditionalResBlock1d(512+512→256, FiLM)
                      ↑ Upsample (4→8) + skip connection
                      ConditionalResBlock1d(256+256→4, FiLM)
                      → Conv1d(4, 4, 1)  ← Final
Output: (B, 4, 8) — predicted noise ε_θ
```

**FiLM conditioning**: Each ResBlock applies scale and shift from the condition vector:
```
h = GroupNorm(Conv1d(x))
scale, shift = FiLM_MLP(condition)
h = h * (1 + scale) + shift
h = Mish(h)
```

**Sinusoidal position embeddings** for timestep encoding (reused from original DPPO model).

### 3.3 Diffusion Process (`models/diffusion_process.py`)

**Forward process (q_sample)**: Add noise according to cosine beta schedule.

```
q(a_t | a_0) = N(a_t; √ᾱ_t · a_0, (1-ᾱ_t) · I)
```

**Cosine beta schedule** (100 timesteps):

```
ᾱ_t = f(t) / f(0), where f(t) = cos((t/T + s)/(1+s) · π/2)²
β_t = 1 - ᾱ_t / ᾱ_{t-1}, clipped to [0, 0.999]
```

**Reverse process (DDPM p_sample)**:

```
p_θ(a_{t-1} | a_t) = N(a_{t-1}; μ_θ(a_t, t), σ_t² · I)
μ_θ = (1/√α_t) · (a_t - β_t/√(1-ᾱ_t) · ε_θ(a_t, t, cond))
```

**DDIM sampling** (10 steps for fast inference):

```
a_{t-1} = √ᾱ_{t-1} · predicted_a_0 + √(1-ᾱ_{t-1}-σ²) · ε_θ + σ · z
predicted_a_0 = (a_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t
```

### 3.4 Vision Diffusion Policy (`models/diffusion_policy.py`)

Glue module combining all components.

**`VisionDiffusionPolicy`**:
- `vision_encoder`: VisionEncoder (CNN)
- `noise_pred_net`: ConditionalUnet1d (1D U-Net)
- `diffusion`: DiffusionProcess (DDPM/DDIM)

**Key methods**:

```python
compute_loss(image_stack, action_seq)
    # Supervised training: MSE noise prediction loss
    # visual_features = vision_encoder(image_stack)  — frozen during denoising
    # t ~ Uniform(0, T)
    # ε ~ N(0, I)
    # noisy_action = q_sample(action_seq, t, ε)
    # ε_θ = noise_pred_net(noisy_action, t, visual_features)
    # loss = MSE(ε_θ, ε)

compute_weighted_loss(image_stack, action_seq, advantages, beta)
    # DPPO: advantage-weighted noise prediction loss
    # weights = exp(β · A_normalized)
    # loss = mean(weights · MSE(ε_θ, ε))

predict_action(image_stack, ddim_steps=10)
    # Inference: DDIM sampling conditioned on visual features
    # Returns: (B, T_pred, action_dim) clamped to [-1, 1]
```

### 3.5 Supervised Pre-training (`scripts/train_diffusion.py`)

1. Load HDF5 dataset → `DemoDataset` with `DataLoader`
2. Train with MSE noise prediction loss
3. AdamW optimizer, cosine LR schedule with warmup
4. Gradient clipping (max norm = 1.0)

**Hyperparameters** (`configs/diffusion_policy.yaml`):

| Parameter | Value |
|-----------|-------|
| Batch size | 256 |
| Learning rate | 1e-4 |
| Weight decay | 1e-6 |
| Num epochs | 500 |
| Warmup epochs | 10 |
| Checkpoint frequency | 50 epochs |

**Verification target**: Training MSE loss < 0.01; qualitative action sequence comparison vs expert.

### 3.6 DPPO Fine-Tuning (`scripts/train_dppo.py`)

Optional reinforcement learning phase to improve beyond imitation.

**Algorithm**:
1. Collect rollouts in `QuadrotorVisualEnv` using RHC (predict T_pred, execute T_action)
2. Estimate values using `ValueNetwork` (MLP on visual features)
3. Compute GAE advantages (γ=0.99, λ=0.95)
4. Update policy with advantage-weighted diffusion loss:

```
L_policy = E[ exp(β · A_normalized) · ||ε_θ(a_t, t, s) - ε||² ]
L_value  = MSE(V_φ(features), returns)
```

**DPPO hyperparameters**:

| Parameter | Value |
|-----------|-------|
| Rollout steps | 2048 |
| Advantage beta (β) | 1.0 |
| Policy LR | 1e-5 |
| Value LR | 1e-4 |
| Update epochs | 4 |
| Total updates | 500 |

---

## Phase 4: Closed-Loop RHC Evaluation

### 4.1 Receding Horizon Control (`scripts/evaluate_rhc.py`)

**RHC loop**:
1. Capture T_obs=2 FPV frames → image stack (6, 64, 64)
2. DDIM sample → action sequence (8, 4)
3. Execute first T_action=4 actions on quadrotor
4. Re-observe and repeat from step 1

**Effective decision frequency**: 50 Hz / 4 = 12.5 Hz

### 4.2 Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Position RMSE | √(mean(||p_target - p||²)) | < 0.15m |
| Crash rate | Fraction of episodes with early termination | < 10% |
| Mean episode reward | Average cumulative reward | > 80% of PPO expert |
| Inference latency | DDIM sampling time (ms) | < 50ms |
| Diffusion/PPO ratio | Reward ratio vs expert baseline | > 80% |

### 4.3 Comparison Methodology

Run N=50 episodes with identical random seeds for both:
- **Diffusion RHC**: Vision Diffusion Policy with DDIM
- **PPO Expert**: State-based PPO with direct motor control

Generate comparison plots:
1. Bar chart: mean reward, position RMSE, crash count
2. 3D trajectory plot (first episode, both controllers)
3. Inference time histogram (DDIM sampling distribution)

---

## Phase 5: Hardware Deployment (Planned)

### 5.1 Target Platform

| Component | Specification |
|-----------|--------------|
| Compute | NVIDIA Jetson Orin Nano (8GB) |
| Flight controller | Pixhawk 6C (PX4 firmware) |
| Camera | 120° FOV FPV camera (640×480 → resize to 64×64) |
| Communication | ROS 2 Humble + uXRCE-DDS (PX4 ↔ Orin) |
| Frame | 250mm class racing quadrotor |

### 5.2 Inference Optimization

- **TensorRT**: Convert PyTorch model to TensorRT FP16
- **ONNX**: Export vision encoder + U-Net as single ONNX graph
- **Target**: < 20ms end-to-end (capture → inference → motor command)

### 5.3 Deployment Pipeline

```
FPV Camera → ROS 2 Image Topic → Vision-DPPO Node (Orin)
    → Motor Command → uXRCE-DDS → PX4 → ESCs → Motors
```

**Safety mechanisms**:
- PX4 failsafe: return-to-launch if no commands for 500ms
- Geofence: PX4 position limits
- Attitude limit: PX4 maximum tilt angle protection
- Watchdog: Monitor inference latency, fallback to hover if > 100ms

---

## Reusable Components from Original DPPO-PID Project

| Component | Original Source | Reuse in Vision-DPPO |
|-----------|----------------|---------------------|
| RK4 integration pattern | `dppo_pid_env.py` | `envs/quadrotor_dynamics.py` |
| Gaussian reward pattern | `dppo_pid_env.py` | `envs/quadrotor_env.py` |
| Inner/outer loop timing | `dppo_pid_env.py` | `envs/quadrotor_env.py` |
| SinusoidalPositionEmbeddings | `dppo_model.py` | `models/diffusion_process.py` |
| Forward diffusion (q_sample) | `dppo_model.py` | `models/diffusion_process.py` |
| DPPO loss skeleton | `dppo_model.py` | `models/diffusion_policy.py` |
| Actor/Critic + TanhNormal | `ppo_native.py` | `models/ppo_expert.py` |
| GAE computation | `ppo_native.py` | `models/ppo_expert.py`, `scripts/train_dppo.py` |
| PPO clipped objective | `ppo_native.py` | `models/ppo_expert.py` |
| RunningMeanStd | `ppo_native.py` | `models/ppo_expert.py` |

---

## Implementation Checklist

### Phase 1: Environment & Expert ✅

- [x] Quaternion utilities (multiply, normalize, to_rotation_matrix, derivative)
- [x] 6-DOF quadrotor dynamics with motor mixing matrix
- [x] RK4 integration at 200 Hz
- [x] First-order motor lag dynamics
- [x] Gymnasium environment (15D obs, 4D action, Gaussian reward)
- [x] 6D continuous rotation representation (gimbal-lock free)
- [x] PPO expert with TanhNormal distribution
- [x] RunningMeanStd observation normalization
- [x] Training script with TensorBoard logging
- [x] Configuration files (quadrotor.yaml, ppo_expert.yaml)

### Phase 2: Data Collection ✅

- [x] Synthetic FPV rendering (horizon, target, altitude)
- [x] QuadrotorVisualEnv wrapper (Dict observation space)
- [x] Data collection script (HDF5 format, gzip compression)
- [x] DemoDataset with sliding window (T_obs=2, T_pred=8)

### Phase 3: Diffusion Policy ✅

- [x] Vision encoder (4-layer CNN, GroupNorm, Mish)
- [x] Conditional 1D U-Net (FiLM conditioning, skip connections)
- [x] Cosine beta schedule (100 timesteps)
- [x] DDPM forward and reverse process
- [x] DDIM accelerated sampling (10 steps)
- [x] VisionDiffusionPolicy glue module
- [x] Supervised training script (MSE noise loss)
- [x] DPPO fine-tuning script (advantage-weighted loss)
- [x] Configuration file (diffusion_policy.yaml)

### Phase 4: Evaluation ✅

- [x] RHC evaluation loop (predict 8, execute 4)
- [x] PPO expert baseline evaluation
- [x] Comparison plots (bar charts, 3D trajectory, inference histogram)
- [x] Metrics: RMSE, crash rate, inference latency

### Phase 5: Hardware Deployment 📋

- [ ] TensorRT model optimization
- [ ] ONNX export pipeline
- [ ] ROS 2 node for Vision-DPPO inference
- [ ] PX4 integration via uXRCE-DDS
- [ ] Real camera calibration and preprocessing
- [ ] Sim-to-real domain adaptation
- [ ] Safety watchdog and failsafe logic

---

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*.
2. Song, J., Meng, C., & Ermon, S. (2020). "Denoising Diffusion Implicit Models." *ICLR*.
3. Chi, C., et al. (2023). "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." *RSS*.
4. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*.
5. Kaufmann, E., et al. (2023). "Champion-level drone racing using deep reinforcement learning." *Nature*.
6. Beard, R. W. & McLain, T. W. (2012). *Small Unmanned Aircraft: Theory and Practice*. Princeton University Press.

---

**Version**: 2.0
**Date**: 2026-03-17
**Status**: Phases 1–4 implemented, Phase 5 planned

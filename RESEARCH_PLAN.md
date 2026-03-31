# Vision-DPPO Research Plan: End-to-End Drone Control via Diffusion Policy
# 基於視覺與擴散策略的無人機端到端控制研究計劃

**Version:** 3.0
**Date:** 2026-03-31
**Status:** Phase 1c — PPO Expert tuning (Run 4 in progress)
**Target Venues:** CoRL 2025 / ICRA 2026 / RSS 2026

---

## Executive Summary / 執行摘要

**Research Question:** Can a Diffusion Policy map FPV image sequences directly to 4D motor thrust commands, resolving representation collapse via Dispersive Loss (D²PPO), and achieve real-time control (>60Hz) via single-step distillation (OneDP)?

**Core Hypothesis:**
> Adding Dispersive Loss to diffusion policy training significantly mitigates feature collapse in high-speed visual drone control, improving policy stability under sim-to-real transfer — independently verifiable via ablation.

**Three fundamental problems addressed:**

| Problem | Root Cause | Solution |
|---------|-----------|---------|
| Control frequency too low (12.5Hz) | Iterative denoising incompatible with real-time latency | OneDP single-step distillation → 62Hz+ |
| Visual representation collapse | Standard diffusion loss optimizes reconstruction only | D²PPO Dispersive Loss, forced discriminative representations |
| Sim-to-Real gap | Synthetic rendering differs from real camera statistics | High-fidelity simulator + domain randomization |

**Research Pipeline:**

```
Phase 1: 6-DOF Env + PPO Expert   ← CURRENT (Run 4 in progress)
    ↓ Gate: position error < 0.1m, crash rate = 0
Phase 2: FPV Data Collection
    ↓
Phase 3a: CNN Baseline Diffusion Policy (supervised pre-training)
    ↓
Phase 3b: D²PPO Dispersive Loss (core contribution, ablation required)
    ↓
Phase 3c: DPPO Closed-Loop RL Fine-tuning
    ↓
Phase 3d: OneDP Single-Step Distillation
    ↓
Phase 4: Full Benchmark Evaluation (BC-LSTM, VTD3, Standard DP)
    ↓
Phase 5: Jetson Orin Nano + TensorRT Hardware Deployment
```

---

## Phase 1: Quadrotor Environment & State-Based PPO Expert

### 1.1 Quadrotor Dynamics (`envs/quadrotor_dynamics.py`)

**State vector (13D):** position [x,y,z], quaternion [qw,qx,qy,qz], linear velocity [vx,vy,vz], angular velocity [ωx,ωy,ωz]

**Coordinate frame:** NED (North-East-Down), body frame X-forward/Y-right/Z-down

**Equations of motion:**

```
m · dv/dt = R(q) · [0, 0, F_total]^T - [0, 0, m·g]^T + F_drag
I · dω/dt = τ - ω × (I · ω)
dq/dt = 0.5 · q ⊗ [0, ω]
```

**Motor mixing (X-configuration):**

```
F_total = Σ f_i
τ_x = L · (f1 - f2 - f3 + f4)       # Roll
τ_y = L · (f1 + f2 - f3 - f4)       # Pitch
τ_z = c_τ · (-f1 + f2 - f3 + f4)    # Yaw
```

**Integration:** RK4 at 200 Hz (dt = 0.005s)

**Motor dynamics:** First-order lag τ_motor = 0.02s

**Physical parameters** (configurable via `configs/quadrotor.yaml`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Mass (m) | 0.5 kg | Total mass |
| Arm length (L) | 0.17 m | Motor to center distance |
| Inertia (I) | diag(2.3e-3, 2.3e-3, 4.0e-3) kg·m² | Moment of inertia |
| Max thrust per motor | 4.0 N | Motor saturation limit |
| Torque coefficient (c_τ) | 0.01 | Thrust-to-torque ratio |
| Drag coefficient | 0.1 | Linear drag |
| Motor time constant | 0.02 s | First-order lag |

**Hover calculation (critical):**

```
F_hover/motor = mg/4 = 0.5 × 9.81 / 4 = 1.226 N
action_hover  = (1.226 / 4.0) × 2 - 1 = -0.387
```

Actor output bias initialized to **-0.39** (pre-tanh) to avoid cold-start problem.

### 1.2 Gymnasium Environment (`envs/quadrotor_env.py`)

**Observation space (15D):**

| Index | Content | Design Rationale |
|-------|---------|-----------------|
| 0-2 | R^T · (p_target - p) | Position error in body frame (rotation-invariant) |
| 3-8 | 6D rotation (first 2 cols of R) | Gimbal-lock free |
| 9-11 | R^T · v | Linear velocity in body frame |
| 12-14 | ω | Angular velocity |

**Action space (4D):** Normalized motor thrusts [-1, 1] → [0, f_max]

**Reward function (Gaussian-based):**

```
R = w_pos × exp(−||pos_err||² / σ_pos)
  + w_vel × exp(−||vel||²    / σ_vel)
  + w_ang × exp(−||ang_vel||² / σ_ang)
  − w_action × ||action||²
  + alive_bonus
```

**Parameter evolution across training runs:**

| Parameter | Run 1 | Run 2 | Run 3 | Run 4 | Rationale |
|-----------|-------|-------|-------|-------|-----------|
| sigma_pos | 0.5 | 0.15 | 0.15 | **0.10** | Steeper gradient at target |
| w_pos | 0.5 | 0.65 | 0.65 | 0.65 | Position dominates signal |
| w_action | 0.05 | 0.03 | 0.03 | **0.01** | Don't penalize corrective actions |
| alive_bonus | 0.1 | 0.05 | 0.05 | **0.0** | Remove free reward diluting position |

**Termination conditions:**
- Position out of bounds: ||p|| > 5.0 m
- Excessive tilt: tilt angle > 60°
- Ground contact: Z > 0 (NED frame)

### 1.3 PPO Expert (`models/ppo_expert.py`)

**Architecture:**

| Component | Specification |
|-----------|--------------|
| Network depth | 2 hidden layers × 256 units |
| Activation | Tanh |
| Output distribution | TanhNormal (squashed Gaussian) |
| Critic | Independent MLP, Tanh activation |
| Observation normalization | RunningMeanStd (online) |
| Actor bias init | nn.init.constant_(mean_layer.bias, -0.39) |

**Training hyperparameters (`configs/ppo_expert.yaml`):**

| Parameter | Value | Notes |
|-----------|-------|-------|
| total_timesteps | 3,000,000 | |
| learning_rate | 3e-4 → 0 (linear annealing) | |
| n_steps | 4096 | Rollout length |
| batch_size | 256 | |
| gamma | 0.99 | |
| gae_lambda | 0.95 | |
| clip_range | 0.2 → 0.05 (annealed) | |
| target_kl | 0.04 | Raised from 0.01 to unblock optimization |
| vf_coef | 1.5 | Increased for critic convergence |
| ent_coef | 0.001 | Low; entropy already high |

**Phase gate (must pass before Phase 2):**

| Metric | Current | Target |
|--------|---------|--------|
| Mean position error | ~0.104m (Run 3) | **< 0.10 m** |
| Episodes < 0.1m error | 0/50 (Run 3) | **> 40/50** |
| Z-axis error | 0.0876m (Run 3) | **< 0.05 m** |
| Crash rate | 0/50 | **= 0** |

---

## Phase 2: FPV Data Collection

### 2.1 Visual Environment (`envs/quadrotor_visual_env.py`)

Wraps `QuadrotorEnv`, renders synthetic FPV images (64×64 RGB):
- Sky/ground gradient (altitude-based)
- Horizon line (roll/pitch-based)
- Target crosshair (relative position projection)
- Altitude indicator

**Sim-to-Real gap note:** Current renderer lacks motion blur, lens distortion, sensor noise. This is the primary risk factor for Phase 5. Upgrade path: Flightmare + domain randomization.

### 2.2 HDF5 Dataset Format

```
data/expert_demos.h5
  /episode_0/images:  (T, 3, 64, 64) uint8, gzip compressed
  /episode_0/actions: (T, 4) float32
  /episode_0/states:  (T, 15) float32
  ...
```

Sliding window: T_obs=2 frames (stacked as 6 channels) → T_pred=8 action steps

**Data quality gates (check before collecting):**
1. Visually inspect 10 random episodes for FPV rendering sanity
2. Confirm no action discontinuities: |action[t] - action[t-1]| < 0.5
3. Confirm episode mean position error < 0.1m (expert quality ceiling)

---

## Phase 3: Vision Diffusion Policy (Core Research)

### 3a. CNN Baseline (Supervised Pre-training)

Establishes baseline numbers for all subsequent ablations.

**Vision Encoder (`models/vision_encoder.py`):**

```
Input: (B, 6, 64, 64)
Conv2d(6, 32, 3, stride=2)  → GroupNorm(8) → Mish   # (B, 32, 32, 32)
Conv2d(32, 64, 3, stride=2) → GroupNorm(8) → Mish   # (B, 64, 16, 16)
Conv2d(64, 128, 3, stride=2)→ GroupNorm(8) → Mish   # (B, 128, 8, 8)
Conv2d(128, 256, 3, stride=2)→GroupNorm(8) → Mish   # (B, 256, 4, 4)
AdaptiveAvgPool2d(1) → Flatten → Linear(256, 256)
Output: (B, 256)
```

**Conditional 1D U-Net (`models/conditional_unet1d.py`):**

```
Condition: visual_features(256) + sinusoidal_time_embed(128) = 384D
Input: noisy action sequence (B, 4, 8)

Encoder: ResBlock(4→256, FiLM) ↓ ResBlock(256→512, FiLM) ↓
Mid:     ResBlock(512→512, FiLM)
Decoder: ResBlock(512→256, FiLM) ↑ ResBlock(256→4, FiLM) ↑
Final:   Conv1d(4, 4, 1)
Output:  predicted noise ε_θ (B, 4, 8)
```

**Diffusion process:** Cosine beta schedule, 100 DDPM timesteps, 10-step DDIM inference

**Training hyperparameters (`configs/diffusion_policy.yaml`):**

| Parameter | Value |
|-----------|-------|
| batch_size | 256 |
| learning_rate | 1e-4 (AdamW, cosine warmup) |
| num_epochs | 500 |
| ddim_steps | 10 (inference) |
| T_obs | 2 frames |
| T_pred | 8 steps |

### 3b. D²PPO: Dispersive Loss (Core Contribution)

**Problem:** Standard diffusion loss only optimizes denoising reconstruction. Encoder tends to map visually similar states (e.g., slight yaw differences) to identical feature vectors, preventing the policy from distinguishing flight states that require different corrective actions.

**Dispersive Loss:**

```python
def dispersive_loss(visual_features, margin=1.0):
    """
    Forces all feature vectors in a batch to repel each other.
    Args: visual_features: (B, feature_dim)
    Returns: scalar loss
    """
    B = visual_features.shape[0]
    diff = visual_features.unsqueeze(1) - visual_features.unsqueeze(0)
    dist = torch.norm(diff, dim=-1)        # (B, B)
    mask = 1 - torch.eye(B, device=visual_features.device)
    loss = -torch.log(dist + 1e-6) * mask
    return loss.sum() / (B * (B - 1))

# Total loss
L_total = L_diffusion + lambda_dispersive * dispersive_loss(features)
# lambda tuning range: 0.01 ~ 0.5, start at 0.1
```

**Dual-layer MDP framework:**
- **Inner MDP:** Diffusion model's iterative denoising — each step involves Gaussian likelihood, making policy gradient mathematically well-defined
- **Outer MDP:** Drone's real interaction with the physical environment

**Ablation design (mandatory for publication):**

| Combination | Training | Purpose |
|-------------|---------|---------|
| No dispersive loss | Standard diffusion loss | Baseline |
| Dispersive (early layers) | First 2 encoder blocks | Effect localization |
| Dispersive (late layers) | Last 2 decoder blocks | Best position search |
| Dispersive (all layers) | All blocks | Over-regularization check |

Each combination: 3 seeds × average. Non-negotiable for conference submission.

### 3c. DPPO Closed-Loop RL Fine-tuning

Advantage-weighted diffusion loss:

```
L_policy = E[ exp(β × A_normalized) × ||ε_θ(a_t, t, s) - ε||² ]
```

| Parameter | Value |
|-----------|-------|
| advantage_beta | 1.0 |
| gamma | 0.99, gae_lambda 0.95 |
| policy_lr | 1e-5 (10× lower than pre-training) |
| value_lr | 1e-4 |
| n_rollout_steps | 2048 |

### 3d. OneDP Single-Step Distillation

Solves the latency problem: 10-step DDIM (~80ms, 12.5Hz) → 1-step (~16ms, 62Hz+).

**Process:**
1. Use D²PPO-finetuned model as teacher
2. Minimize `KL(q_teacher(a|s) || q_student(a|s))`
3. Self-consistency training to preserve multimodal coverage

**Order matters:** Must have a high-quality teacher (3b+3c complete) before distilling.

**Latency targets (Jetson Orin Nano 8GB):**

| Version | Theoretical | Jetson Target |
|---------|-------------|--------------|
| 10-step DDIM | ~80ms | possibly >200ms |
| 1-step OneDP | <16ms | <30ms |

---

## Phase 4: Closed-Loop RHC Evaluation

### 4.1 RHC Loop

```
1. Capture T_obs=2 FPV frames → image stack (6, 64, 64)
2. OneDP single-step inference → action sequence (8, 4)
3. Execute first T_action=4 actions
4. Re-observe → repeat
Effective decision rate: 50Hz / 4 = 12.5Hz → 50Hz+ after OneDP
```

### 4.2 Benchmark Matrix

**Fair comparison principle:** All vision methods use identical RGB input. PPO Expert serves as oracle upper bound only.

| Method | Input | Purpose | Expected |
|--------|-------|---------|---------|
| BC-LSTM | RGB | BC multimodal collapse baseline | Action jitter at bifurcations |
| VTD3 | RGB | Standard visual DRL comparison | DPPO more stable at multimodal scenarios |
| Standard DP (3a) | RGB | Ablation: dispersive loss contribution | Quantify D²PPO gain |
| D²PPO (3b) | RGB | Ablation: DPPO finetuning contribution | Quantify RL gain |
| D²PPO + OneDP (full) | RGB | **Primary method** | — |
| VIO + Geometric Control | RGB + IMU | Modular baseline | End-to-end wins on error accumulation |
| PPO Expert | Full state | Oracle upper bound | — |

### 4.3 Evaluation Metrics

| Metric | Target | Conference Requirement |
|--------|--------|----------------------|
| Position RMSE | < 0.15m | ICRA |
| Crash Rate | < 10% | CoRL |
| Diffusion/PPO Ratio | > 80% | CoRL |
| Inference Latency | < 20ms | CoRL/ICRA |
| Control Frequency | > 60Hz | ICRA |
| Zero-shot Sim-to-Real | Qualitative + quantitative | CoRL |

---

## Phase 5: Hardware Deployment

### 5.1 Target Platform

| Component | Specification |
|-----------|--------------|
| Compute | NVIDIA Jetson Orin Nano 8GB |
| Flight controller | Pixhawk 6C (PX4 firmware) |
| Camera | 120° FOV FPV camera (640×480 → resize 64×64) |
| Communication | ROS 2 Humble + uXRCE-DDS |
| Frame | 250mm class racing quadrotor |

### 5.2 Inference Optimization

```
PyTorch Model → ONNX Export → TensorRT FP16/INT8 → Jetson deployment
Target: end-to-end latency < 30ms (capture + inference + command output)
```

### 5.3 Safety Mechanisms

- PX4 failsafe: return-to-launch if no commands for 500ms
- Geofence: maximum flight radius limit
- Attitude limit: maximum tilt angle protection
- Watchdog: inference latency > 100ms → switch to hover

### 5.4 Real Flight Test Scenarios

| Test | Description | Metric |
|------|-------------|--------|
| Static hover | No disturbance, RMSE < 0.1m | Position RMSE |
| Fan disturbance | Simulate wind gust from 3m | Survival rate, recovery time |
| Visual occlusion | 30% camera blocked | Recovery rate |
| Outdoor → Indoor | Zero-shot scene transfer | Success rate |

---

## Architecture Upgrade Path (v2.0 → v3.0)

**Immediate (current phase):**
- Reward: sigma_pos 0.5 → 0.10 (progressively tightened per run)
- Actor initialization: hover bias -0.39
- PPO: target_kl 0.01 → 0.04, vf_coef 0.5 → 1.5, ent_coef 0.01 → 0.001

**Medium-term (Phase 3):**
- Loss: Standard diffusion → D²PPO (dispersive loss + dual-layer MDP)
- Inference: 10-step DDIM → OneDP single-step distillation
- Baseline comparison: add BC-LSTM, VTD3 for fair ablation

**Long-term (Phase 5):**
- Encoder: CNN → Pretrained ViT-Small + privileged state decoder head
- Simulator: Custom Gymnasium → Flightmare + domain randomization

---

## Implementation Checklist

### Phase 1: Environment & Expert

- [x] Quaternion utilities (multiply, normalize, to_rotation_matrix, derivative)
- [x] 6-DOF quadrotor dynamics with motor mixing matrix
- [x] RK4 integration at 200 Hz + first-order motor lag
- [x] Gymnasium environment (15D obs, 4D action, Gaussian reward)
- [x] 6D continuous rotation representation (gimbal-lock free)
- [x] PPO expert with TanhNormal distribution + hover bias init
- [x] RunningMeanStd observation normalization
- [x] LR/clip range annealing
- [🔄] PPO tuning to meet phase gate (Run 4 in progress)

### Phase 2: Data Collection

- [x] Synthetic FPV rendering (horizon, target, altitude)
- [x] QuadrotorVisualEnv wrapper (Dict observation space)
- [x] Data collection script (HDF5 format, gzip compression)
- [x] DemoDataset with sliding window (T_obs=2, T_pred=8)

### Phase 3: Diffusion Policy

- [x] Vision encoder (4-layer CNN, GroupNorm, Mish)
- [x] Conditional 1D U-Net (FiLM conditioning, skip connections)
- [x] Cosine beta schedule (100 timesteps)
- [x] DDPM forward and reverse process
- [x] DDIM accelerated sampling (10 steps)
- [x] VisionDiffusionPolicy glue module
- [x] Supervised training script (MSE noise loss)
- [x] DPPO fine-tuning script (advantage-weighted loss)
- [ ] D²PPO dispersive loss implementation + ablation
- [ ] OneDP single-step distillation

### Phase 4: Evaluation

- [x] RHC evaluation loop (predict 8, execute 4)
- [x] PPO expert baseline evaluation
- [x] Comparison plots (bar charts, 3D trajectory, inference histogram)
- [ ] BC-LSTM, VTD3, Standard DP baselines
- [ ] Full 3-seed ablation experiments

### Phase 5: Hardware Deployment

- [ ] TensorRT model optimization
- [ ] ONNX export pipeline
- [ ] ROS 2 node for Vision-DPPO inference
- [ ] PX4 integration via uXRCE-DDS
- [ ] Real camera calibration and preprocessing
- [ ] Safety watchdog and failsafe logic

---

## References

1. Chi, C. et al. (2023). "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." *RSS*.
2. Zou, G. et al. (2025). "D²PPO: Diffusion Policy Policy Optimization with Dispersive Loss." *arXiv:2508.02644*
3. Ze, Y. et al. (2024). "One-Step Diffusion Policy." *arXiv:2410.21257*
4. Ren, A. et al. (2024). "Diffusion Policy Policy Optimization." *OpenReview:mEpqHvbD2h*
5. Kaufmann, E. et al. (2023). "Champion-level drone racing using deep reinforcement learning." *Nature*
6. Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*
7. Ho, J. et al. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*
8. Song, J. et al. (2020). "Denoising Diffusion Implicit Models." *ICLR*

---

**Detailed diagnostic log → [docs/dev_log.md](docs/dev_log.md)**
**Conference submission guide → [docs/TOP_CONF_GUIDE.md](docs/TOP_CONF_GUIDE.md)**

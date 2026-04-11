# Vision-DPPO Research Plan: End-to-End Drone Control via Diffusion Policy
# 基於視覺與擴散策略的無人機端到端控制研究計劃

**Version:** 3.5
**Date:** 2026-04-11
**Status:** Phase 3c v3.2 DPPO Run 1 in progress (`dppo_v32_20260411_114141`); v3.1 finite-diff IMU abandoned; physics-based IMU covariate shift ax 23×→1.4×; baseline DPPO ceiling at RMSE 0.168m (Run 2)
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
Phase 1: 6-DOF Env + PPO Expert   ✓ DONE (Run 6: RMSE 0.069m, 0 crashes)
    ↓
Phase 2: FPV Data Collection       ✓ DONE (expert_demos_dr.h5: 1000 ep, 500k steps, DR A+B)
    ↓                              ✓ v3.1 re-collect DONE (expert_demos_v31.h5: 4.04GB, IMU+depth)
    ↓                              ✓ v3.2 re-collect DONE (expert_demos_v32.h5: 4.0GB, physics IMU)
Phase 3a: CNN Baseline             ✓ Re-run 2 complete (20260405_044808, best loss converged)
    ↓      v3.1 supervised         ✓ DONE (v31_20260406_185128, best loss -1.4415, 500 epochs)
    ↓      v3.2 supervised         ✓ DONE (v32_20260410_120042, best loss -1.437, 500 epochs)
Phase 3b: D²PPO Dispersive Loss    ✓ Run 2 (u11, 0.168m) + Run 3 (u34, 0.488m) — both 50/50 crashes
    ↓       Root cause: value net lag; policy collapses before advantage estimates converge
    ↓      Run 4 (u155, 0.409m) — improved training (VLoss=17) but DR-aug pretrained hurt RMSE
Phase 3c: DPPO v3.1 RL Fine-tuning   ✗ ABANDONED — 2 runs (RMSE 0.518/0.466m), finite-diff IMU
    ↓       Root cause: finite-diff accel = R^T a_world − ω×v_body; Coriolis term 20× noise in RL
    ↓      DPPO v3.2 RL Fine-tuning  🔄 IN PROGRESS (dppo_v32_20260411_114141)
    ↓       Physics IMU: covariate shift ax 23×→1.4×, ay 16×→1.2×
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

**Domain Randomization (Option A — Renderer-level):**

Applied at each `env.reset()` call so every episode has a distinct visual appearance.
Expert uses full state for action decisions, so DR does not affect action quality.

| Param | Range | Granularity | Effect |
|-------|-------|-------------|--------|
| Sky base color offset | ±40 per R/G/B | per-episode | Prevents CNN from encoding sky hue as attitude |
| Ground base color offset | ±40 per R/G/B | per-episode | Same for ground |
| Global brightness | ×[0.7, 1.3] | per-episode | Simulates lighting variation |
| Focal scale (projection) | [0.30, 0.50] | per-episode | ≈ ±20% FOV variation |
| Crosshair size delta | ±2 px | per-episode | Distance-estimation robustness |
| Horizon color | [150, 255] per ch | per-episode | Prevents color-based horizon detection |
| Gaussian pixel noise | σ = 5 (uint8) | per-frame | Prevents encoder over-fitting to clean edges |

**Sim-to-Real gap note:** Motion blur, lens distortion, and shadow simulation are deferred to Phase 5 (Flightmare + domain randomization). Current DR addresses the highest-ROI factors.

### 2.2 HDF5 Dataset Format

```
data/expert_demos.h5
  /episode_0/images:  (T, 3, 64, 64) uint8, gzip compressed
  /episode_0/actions: (T, 4) float32
  /episode_0/states:  (T, 15) float32
  ...
```

Sliding window: T_obs=2 frames (stacked as 6 channels) → T_pred=8 action steps

**DR-enabled collection:** Phase 2 must be re-run with `QuadrotorVisualEnv(dr_enabled=True)`
(the default) to ensure collected images span the full randomization distribution.
Existing `data/expert_demos.h5` (no DR) is kept as a deterministic baseline for ablation.

**Data quality gates (check before collecting):**
1. Visually inspect 10 random episodes for FPV rendering sanity — confirm color variation across episodes
2. Confirm no action discontinuities: |action[t] - action[t-1]| < 0.5
3. Confirm episode mean position error < 0.1m (expert quality ceiling)
4. Confirm mean pixel diff across consecutive episodes > 10 (DR sanity check)

### 2.3 Data Collection Upgrade (v3.1)

Required before Phase 3c (Architecture v3.1 with IMU Late Fusion & Auxiliary Depth).
Not needed for the current Phase 3b run.

**New HDF5 fields:**

```
data/expert_demos_v31.h5
  /episode_0/images:      (T, 3, 64, 64) uint8    — unchanged
  /episode_0/actions:     (T, 4) float32           — unchanged
  /episode_0/states:      (T, 15) float32          — unchanged
  /episode_0/imu_data:    (T, 6) float32           — NEW: [ωx, ωy, ωz, ax, ay, az]
  /episode_0/depth_maps:  (T, 1, 64, 64) uint8     — NEW (optional, FCN branch only)
```

**IMU alignment:** Control loop runs at 50 Hz; IMU data is aligned at the same rate.
- **ω (gyro):** `dynamics.ang_velocity` (body-frame angular velocity, 3D)
- **a (specific force):** `dynamics.get_specific_force_body()` → `R^T @ (force_world − gravity_world) / mass`
  — what a body-mounted accelerometer physically measures; does NOT include gravity in free-fall.
  **v3.2 change (2026-04-10):** replaced the v3.1 finite-difference `(v_body[t] - v_body[t-1])/dt`
  which mathematically equals `R^T a_world − ω×v_body`. The Coriolis-like term `ω×v_body` amplifies
  20× during unstable RL rollouts, causing catastrophic distribution shift between Phase 2 collection
  and Phase 3c rollout. Physics-based specific force drops this to 1.3×.

**Depth map rendering:** requires `QuadrotorVisualEnv` modification to output a per-pixel depth
channel alongside the RGB image. In the synthetic environment depth can be derived analytically
from the target position and camera projection parameters — no neural estimation needed.

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

**Option B — Dataset-level Augmentation (applied during Phase 3a training):**

Applied per-frame independently before T_obs stacking. Complements Option A by adding
photometric variation at training time without requiring a Phase 2 re-run.

```python
transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1)
```

Combined A+B strategy:
- **A** covers geometric DR (FOV, horizon position, crosshair size) and episode-level appearance
- **B** covers per-sample photometric variation with zero data collection cost
- Together they force the encoder to learn pose-invariant, appearance-invariant features

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

### 3b-ext. Architecture v3.1: IMU Late Fusion & Auxiliary Depth

**Prerequisite:** Phase 3b Run 2 must complete first. Implement before Phase 3c begins.

**Motivation:**
- Dispersive Loss addresses *feature collapse* (same-looking states mapped to same vector)
- IMU Late Fusion addresses *visual ambiguity* (states that look identical but have different dynamics)
- FCN Auxiliary Depth acts as a *multi-task regularizer* (forces the encoder to learn 3D spatial structure)
  and is fully pruned at deployment — zero inference overhead.

**Architecture (dual-branch, late fusion):**

```
FPV Stack (6×64×64)
  → VisionEncoder CNN → 256D vision feature
                                              ┐
6D IMU [ωx,ωy,ωz, ax,ay,az]                  │ Late Fusion
  → MLP(6→64→32) → 32D IMU feature           │
                                              ┘
  cat([256D vision, 32D IMU]) → 288D global_cond
  + sinusoidal_time_embed(128D)
  → 416D conditioning vector
  → ConditionalUNet1D → predicted noise ε_θ

[Training only]
  256D vision feature → FCN depth decoder → (1,64,64) depth map
```

**cond_dim change:**

```
Current: cond = cat([vision(256), time(128)]) = 384
v3.1:    global_cond = cat([vision(256), imu(32)]) = 288
         cond        = cat([global_cond(288), time(128)]) = 416
```

**IMU Encoder (lightweight MLP):**

```python
imu_encoder = nn.Sequential(
    nn.Linear(6, 64),
    nn.Mish(),
    nn.Linear(64, 32)
)
```

**FCN Depth Decoder (training only):**

```
(B, 256) → Unflatten(256, 1, 1) → (B, 256, 1, 1)
→ ConvTranspose2d(256, 128, k=4, s=1)  → (B, 128, 4, 4)
→ ConvTranspose2d(128, 64, k=4, s=2, p=1) → (B, 64, 8, 8)
→ ConvTranspose2d(64, 32, k=4, s=2, p=1)  → (B, 32, 16, 16)
→ ConvTranspose2d(32, 16, k=4, s=2, p=1)  → (B, 16, 32, 32)
→ ConvTranspose2d(16, 1, k=4, s=2, p=1)   → (B, 1, 64, 64)
→ Sigmoid()  [depth normalised to 0–1]
```

**Total loss (Phase 3c with v3.1 architecture):**

```
L_total = L_action_diffusion
        + λ_disp  × L_dispersive(visual_features)
        + λ_depth × MSE(depth_pred, depth_gt)

Tuning range:
  λ_disp  ∈ [0.01, 0.5]  (start at 0.1, same as Phase 3b)
  λ_depth ∈ [0.01, 0.5]  (start at 0.1; zero to disable FCN branch)
```

**Deployment pruning:** Before ONNX/TensorRT export, remove `depth_decoder` entirely.
Only encoder + IMU MLP + diffusion UNet are exported. OneDP single-step latency target
remains < 30 ms on Jetson Orin Nano.

**Files to modify for v3.1:**

| File | Change |
|------|--------|
| `models/diffusion_policy.py` | Add `imu_encoder`, update `forward()` signature to accept `imu_data` |
| `models/conditional_unet1d.py` | Change `feature_dim=288` (no internal changes needed) |
| `envs/quadrotor_visual_env.py` | Add depth channel rendering to `_render()` |
| `scripts/collect_data.py` | Store `imu_data` and `depth_maps` in HDF5 |
| `scripts/train_dppo.py` | Pass `imu_data` from state vector to policy; add `λ_depth` loss term |
| `configs/diffusion_policy.yaml` | Add `imu_feature_dim: 32`, `lambda_depth: 0.1` |

**Ablation for paper (mandatory):**

| Config | Purpose |
|--------|---------|
| No IMU, No FCN (baseline = 3b) | D²PPO contribution isolation |
| IMU only | IMU fusion contribution |
| FCN only | Depth auxiliary contribution |
| IMU + FCN (v3.1 full) | Combined contribution |

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
| D²PPO v3.1 no-IMU no-depth | RGB | Ablation: v3.1 IMU + depth contribution baseline | — |
| D²PPO v3.1 IMU-only | RGB + IMU | Ablation: IMU fusion contribution | — |
| D²PPO v3.1 depth-only | RGB | Ablation: depth auxiliary contribution | — |
| D²PPO v3.1 full + OneDP | RGB + IMU | **Primary method** | — |
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

**v3.1 deployment graph (pruned):**

```
[Deploy]  FPV camera (64×64) → VisionEncoder → 256D
          IMU /fmu/out/vehicle_imu → MLP(6→32) → 32D
          cat([256D, 32D]) → 288D global_cond
          → OneDP single-step UNet → 4D motor thrust

[Removed] FCN depth decoder — training-only, stripped before ONNX export
```

ROS 2 node must subscribe to `/fmu/out/vehicle_imu` and align IMU readings to the
50 Hz control tick before passing to the policy.

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

## Architecture Upgrade Path (v2.0 → v3.2)

**Done (Phase 1):**
- Reward: sigma_pos 0.5 → 0.10 (progressively tightened per run)
- Actor initialization: hover bias -0.39
- PPO: target_kl 0.01 → 0.04, vf_coef 0.5 → 1.5, ent_coef 0.01 → 0.001

**Done (Phase 3a/3b):**
- Loss: Standard diffusion → D²PPO (dispersive loss + dual-layer MDP)
- Option B GPU augmentation replacing PIL ColorJitter (9× speedup)
- Phase 3b conservative HP: β=0.1, lr=5e-6, n_rollout=4096

**v3.1 (Phase 3c — ✗ ABANDONED 2026-04-10):**
- Sensing: vision-only → IMU Late Fusion (6D IMU → 32D MLP → cat with 256D vision feature → 288D global_cond)
- IMU source: finite-difference `(v_body[t] - v_body[t-1])/dt` — FLAWED: includes Coriolis `ω×v_body` term
- Failed: 2 DPPO runs RMSE 0.518/0.466m vs 0.268m no-IMU supervised baseline
- Failure confirmed: ax std ratio expert/perturbed = 23×, ay = 16× (v3.2 drops to 1.4×/1.2×)
- Files retained for historical reproduction: `scripts/train_diffusion_v31.py`, `scripts/train_dppo_v31.py`

**v3.2 (Phase 3c — 🔄 DPPO Run 1 in progress 2026-04-11):**
- IMU source: `QuadrotorDynamics.get_specific_force_body()` → `R^T @ (F_world − mg) / m`
  Gyro: `dynamics.ang_velocity`; exposed via `QuadrotorEnv.get_imu()` single call point
- Same architecture as v3.1: VisionDPPOv31, 288D global_cond, IMUEncoder MLP(6→64→32)
- Same DPPO hyperparameters as v3.1 Run 2: β=0.05, value_hidden_dim=512, warm-up 50, vloss_threshold 500
- Distribution-shift improvement: ax 23×→1.4×, ay 16×→1.2× (validated before DPPO launch)
- New files: `scripts/train_diffusion_v32.py`, `scripts/train_dppo_v32.py`, `scripts/evaluate_rhc_v32.py`
- Data: `data/expert_demos_v32.h5` (4.0 GB, collected 2026-04-10)
- Supervised pretraining complete: best loss -1.437 (vs v3.1 -1.4415 — essentially identical)
- Known issue: supervised RMSE 1.985m (IMU normalization gap — specific_force ≈ −9.81 m/s² not centered)
  DPPO Run 1 monitoring whether value net can adapt; fix ready if needed

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
- [x] PPO converged — Run 6 (RMSE 0.069m, 0 crashes)

### Phase 2: Data Collection

- [x] Synthetic FPV rendering (horizon, target, altitude)
- [x] QuadrotorVisualEnv wrapper (Dict observation space)
- [x] Data collection script (HDF5 format, gzip compression)
- [x] DemoDataset with sliding window (T_obs=2, T_pred=8)
- [x] Option A: Renderer-level DR (per-episode color/brightness/focal/noise)
- [x] Phase 2 re-run with DR enabled → `data/expert_demos_dr.h5`
- [x] `_render_depth()` added to `QuadrotorVisualEnv` (v3.1 geometry-based ray casting)
- [x] `collect_data.py --v31` flag implemented (imu_data + depth_maps fields)
- [x] v3.1 data collection → `data/expert_demos_v31.h5` (4.04GB, 2026-04-06)
- [x] `collect_data.py --v32` flag — physics IMU via `env.unwrapped.get_imu()`, no finite-diff
- [x] v3.2 data collection → `data/expert_demos_v32.h5` (4.0GB, 2026-04-10)

### Phase 3: Diffusion Policy

- [x] Vision encoder (4-layer CNN, GroupNorm, Mish)
- [x] Conditional 1D U-Net (FiLM conditioning, skip connections)
- [x] Cosine beta schedule (100 timesteps)
- [x] DDPM forward and reverse process
- [x] DDIM accelerated sampling (10 steps)
- [x] VisionDiffusionPolicy glue module
- [x] Supervised training script (MSE noise loss)
- [x] DPPO fine-tuning script (advantage-weighted loss)
- [x] Option B: GPU tensor augmentation (brightness/contrast, replaces PIL — 9× faster)
- [x] Phase 3a re-training with DR data + Option B augmentation (Re-run 2 complete)
- [x] Phase 3b D²PPO Run 2 (`dppo_20260404_044552`) — best u11, RMSE 0.145m, 50/50 crashes
- [x] Phase 3b D²PPO Run 3 (`dppo_20260405_155057`) — best u34, RMSE 0.450m, 50/50 crashes (750 updates, ablation)
- [x] RHC eval: Run 2 vs Run 3 — both 50/50 crash; baseline architecture ceiling confirmed
- [x] Architecture v3.1: `models/vision_dppo_v31.py` — IMUEncoder, DepthDecoder, VisionDPPOv31
- [x] Architecture v3.1: `scripts/train_diffusion_v31.py` — supervised pre-training
- [x] Architecture v3.1: `scripts/train_dppo_v31.py` — DPPO fine-tuning (mini-batch fix)
- [x] Architecture v3.1: `scripts/evaluate_rhc_v31.py` — RHC evaluator with IMU input
- [x] Architecture v3.1: `configs/diffusion_policy.yaml` v31 block (imu_feature_dim, lambda_depth)
- [x] numpy memmap cache `data/v31_mmap/` — 46ms/batch (117× speedup over HDF5 lazy-read)
- [x] Phase 3a v3.1 supervised pre-training complete (best loss -1.4415, 2026-04-06~08)
- [x] `QuadrotorDynamics.get_specific_force_body()` — physics IMU, caches `_last_force_world`+`_last_R` in `step()`
- [x] `QuadrotorEnv.get_imu()` — single call point, returns `[gyro(3), specific_force(3)]`
- [x] Architecture v3.2: `scripts/train_diffusion_v32.py` — DemoDatasetV32 + v32_mmap cache
- [x] Architecture v3.2: `scripts/train_dppo_v32.py` — physics IMU rollout, imports from v31
- [x] Architecture v3.2: `scripts/evaluate_rhc_v32.py` — RHC evaluator, physics IMU
- [x] Phase 3a v3.2 supervised pre-training complete (best loss -1.437, 2026-04-10~11)
- [x] Distribution-shift validation: ax 23×→1.4×, ay 16×→1.2× (confirmed fix justified)
- [✗] Phase 3c DPPO v3.1 — 2 runs abandoned (finite-diff IMU covariate shift)
- [🔄] Phase 3c DPPO v3.2 Run 1 in progress (`dppo_v32_20260411_114141`)
- [ ] λ_depth ablation (0, 0.01, 0.1, 0.5) + IMU ablation (3 seeds each)
- [ ] ONNX export script with FCN decoder stripping (`save_deployable()` already implemented)
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

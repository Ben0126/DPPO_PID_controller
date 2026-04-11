# Vision-DPPO: End-to-End Drone Control via Diffusion Policy
# 基於視覺與擴散策略的無人機端到端直接控制

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


---

## Research Motivation / 研究動機

Traditional drone control relies on a modular stack: Camera → VIO → Planning → Cascaded PID. This design has three fundamental limitations:

1. **Error accumulation:** VIO estimation errors propagate to the controller; positioning and control problems are inseparable
2. **Misaligned objectives:** VIO optimizes "estimation accuracy," PID optimizes "control error" — they are not jointly aligned
3. **Multimodal action collapse:** Traditional controllers and standard PPO can only express unimodal action distributions, unable to handle scenarios where multiple valid flight strategies coexist

**Core research question:** Can a Diffusion Policy map FPV image sequences directly to 4D motor thrust commands, resolving the above limitations while achieving real-time control (>60Hz) on resource-constrained onboard computers?

---

## Method Overview / 方法概覽

```
Traditional:
  Camera → VIO → Position PID → Attitude PID → Rate PID → Motor Mixing → Motors
  (each module trained independently, errors accumulate at each stage)

Vision-DPPO (This Project):
  FPV Image Stack → ViT Encoder → D²PPO (1D U-Net) → [OneDP Distillation] → Motor Thrusts
  (end-to-end joint optimization, single-step inference at 62Hz+)
```

**Why Diffusion Policy instead of PPO?**
PPO's Gaussian output assumes a unimodal action distribution. In complex flight scenarios (e.g., left or right around an obstacle), the optimal strategy is multimodal — Gaussian PPO outputs the average of two modes (i.e., crashes into the obstacle). Diffusion Policy learns the complete action distribution via iterative denoising, natively supporting multimodality.

---

## System Architecture / 系統架構

### Baseline (Phase 3a/3b)

```
┌─────────────────────────────────────────────────────────┐
│  FPV Image Stack (T_obs=2 frames, 64×64 RGB each)       │
│  → stacked as (B, 6, 64, 64)                            │
│  DR: ±sky/ground color, ±brightness, ±focal, σ=5 noise  │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│  Vision Encoder (4-layer CNN)                           │
│  → 256D feature vector                                  │
└────────────────────┬────────────────────────────────────┘
                     │ (B, 256) + timestep(128) = cond(384)
                     ↓
┌─────────────────────────────────────────────────────────┐
│  D²PPO: Conditional 1D U-Net                            │
│  + Dispersive Loss (prevents representation collapse)    │
│  → Predicted noise ε_θ (B, 4, 8)                       │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│  Inference: 10-step DDIM → 12.5Hz                       │
│  [Target v3.1+] OneDP 1-step → 62Hz+                   │
└────────────────────┬────────────────────────────────────┘
                     │ (B, T_pred, 4) → execute first T_action steps
                     ↓
┌─────────────────────────────────────────────────────────┐
│  6-DOF Quadrotor (RK4 @ 200Hz, NED frame)               │
└─────────────────────────────────────────────────────────┘
```

### Architecture v3.1 / v3.2 (Phase 3c — IMU Late Fusion + FCN Auxiliary Depth)

```
┌──────────────────────────────┐  ┌──────────────────────────────┐
│  FPV Image Stack (B,6,64,64) │  │  6D IMU [ωx,ωy,ωz, ax,ay,az] │
│  DR: color/brightness/focal  │  │  (body frame, 50Hz aligned)  │
└──────────────┬───────────────┘  └──────────────┬───────────────┘
               ↓                                 ↓
┌──────────────────────────┐     ┌───────────────────────────────┐
│  Vision Encoder (CNN)    │     │  IMU Encoder MLP              │
│  → 256D vision_feat      │     │  Linear(6→64→32)              │
└──────────────┬───────────┘     │  → 32D imu_feat               │
               │                 └──────────────┬────────────────┘
               │    cat([256D, 32D])             │
               └─────────────────┬──────────────┘
                                 ↓ 288D global_cond
                   + timestep_embed(128D)
                                 ↓ 416D cond
               ┌─────────────────────────────────────────┐
               │  D²PPO: Conditional 1D U-Net            │
               │  + Dispersive Loss                      │
               │  → Predicted noise ε_θ (B, 4, 8)       │
               └─────────────────┬───────────────────────┘
                                 ↓
               ┌─────────────────────────────────────────┐
               │  OneDP Single-Step Distillation         │
               │  → 4D motor thrusts @ 62Hz+             │
               └─────────────────┬───────────────────────┘
                                 ↓
               ┌─────────────────────────────────────────┐
               │  6-DOF Quadrotor (RK4 @ 200Hz)          │
               └─────────────────────────────────────────┘

[Training only — stripped before deployment]
256D vision_feat → FCN Depth Decoder (5× ConvTranspose2d) → (1,64,64) depth_pred
L_total = exp(β×A)×L_diff + λ_disp×L_dispersive + λ_depth×MSE(depth_pred, depth_gt)
```

---

## Development Phases / 開發階段

```
Phase 1: PPO Expert + 6-DOF Environment
         [✓]  Done — Run 6 (RMSE 0.069m, 0 crashes)

Phase 2: FPV Data Collection
         [✓]  Done — expert_demos_dr.h5 (1000 ep, 500k steps, DR enabled)
         [✓]  v3.1 re-collection complete → expert_demos_v31.h5 (4.04GB, IMU+depth)
         [✓]  v3.2 re-collection complete → expert_demos_v32.h5 (4.0GB, physics IMU)
              Physics-based specific force replaces finite-difference (2026-04-10)

Phase 3: Vision Diffusion Policy
   3a    [✓]  Supervised pre-training Re-run 2 complete (DR-aug, 500 epochs)
              checkpoints/diffusion_policy/20260405_044808/best_model.pt
   3a-v31[✓]  v3.1 supervised pre-training complete (500 epochs, best loss -1.4415)
              checkpoints/diffusion_policy/v31_20260406_185128/best_model.pt
   3a-v32[✓]  v3.2 supervised pre-training complete (500 epochs, best loss -1.437)
              checkpoints/diffusion_policy/v32_20260410_120042/best_model.pt
   3b    [✓]  D²PPO Run 2 (dppo_20260404_044552) — best u11, RMSE 0.168m, 50/50 crashes
   3b    [✓]  D²PPO Run 3 (dppo_20260405_155057) — best u34, RMSE 0.488m, 50/50 crashes
   3b    [✓]  D²PPO Run 4 (dppo_20260410_045335) — best u155, RMSE 0.409m, 50/50 crashes
              DR-aug pretrained; value loss converged (VLoss=17), reward stable, but RMSE worse
   3c    [✗]  DPPO v3.1 Run 1+2 ABANDONED — RMSE 0.518/0.466m, finite-diff IMU covariate shift
   3c-v32[🔄] DPPO v3.2 Run 1 IN PROGRESS (dppo_v32_20260411_114141)
              Physics-based IMU (get_specific_force_body); covariate shift ax: 23×→1.4×
   3d    [ ]  OneDP single-step distillation (solve deployment latency)

Phase 4: Evaluation
         [ ]  Full benchmark (BC-LSTM, VTD3, Standard DP)
         [ ]  Closed-loop RHC evaluation

Phase 5: Hardware Deployment
         [ ]  Jetson Orin Nano + TensorRT (FCN decoder pruned before export)
         [ ]  Real flight testing (with wind disturbance)
```

**Current status:** Phase 3c v3.2 DPPO Run 1 in progress (`dppo_v32_20260411_114141`).
v3.1 IMU (finite-difference) abandoned 2026-04-10 after 2 failed runs (RMSE 0.466–0.518m). Root cause confirmed: finite-diff `v_body` amplifies Coriolis noise ≥20× during RL rollouts.
v3.2 replaces with physics-based `get_specific_force_body()`: covariate shift ax 23×→1.4×, ay 16×→1.2×.
Run 4 baseline (no-IMU + improved training) reached RMSE 0.409m — training quality best so far (VLoss=17, reward stable), but DR-aug pretrained produced blurrier feature space than original pretrained, explaining the RMSE regression vs Run 2 (0.168m).
See [docs/dev_log_phase2_3.md](docs/dev_log_phase2_3.md) for detailed training analysis.

---

## Quick Start / 快速開始

### Installation

```bash
git clone <repository-url>
cd DPPO_PID_controller

python -m venv dppo
source dppo/bin/activate          # Linux/macOS
# .\\dppo\\Scripts\\activate         # Windows

pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Training Pipeline

**Step 1: Train PPO Expert (must meet gate before proceeding)**

```bash
python -m scripts.train_ppo_expert
tensorboard --logdir ./logs/ppo_expert/
```

Gate criteria: mean position error < 0.1m, crash rate = 0, episodes < 0.1m > 40/50.

**Step 2: Collect Expert Data**

```bash
python -m scripts.collect_data \
    --ppo-model checkpoints/ppo_expert/.../best_model.pt \
    --ppo-norm checkpoints/ppo_expert/.../best_obs_rms.npz
```

**Step 3: Train Diffusion Policy**

```bash
python -m scripts.train_diffusion --config configs/diffusion_policy.yaml
```

**Step 4: DPPO Fine-tuning (with Dispersive Loss)**

```bash
python -m scripts.train_dppo \
    --pretrained checkpoints/diffusion_policy/.../best_model.pt
```

**Step 5: Closed-Loop Evaluation**

```bash
python -m scripts.evaluate_rhc \
    --diffusion-model checkpoints/diffusion_policy/.../best_model.pt \
    --ppo-model checkpoints/ppo_expert/.../best_model.pt \
    --ppo-norm checkpoints/ppo_expert/.../best_obs_rms.npz
```

---

## Project Structure / 專案結構

```
DPPO_PID_controller/
├── configs/
│   ├── quadrotor.yaml           # Physics parameters, reward function
│   ├── ppo_expert.yaml          # PPO hyperparameters
│   └── diffusion_policy.yaml    # Diffusion policy hyperparameters
│
├── envs/
│   ├── quadrotor_dynamics.py    # Pure 6-DOF physics (quaternion, RK4, motors)
│   ├── quadrotor_env.py         # Gymnasium wrapper (15D obs, 4D action)
│   └── quadrotor_visual_env.py  # FPV image rendering wrapper
│
├── models/
│   ├── ppo_expert.py            # Actor-Critic PPO (TanhNormal distribution)
│   ├── vision_encoder.py        # CNN (current) → ViT (target upgrade)
│   ├── conditional_unet1d.py    # 1D U-Net (FiLM conditioning)
│   ├── diffusion_process.py     # DDPM/DDIM forward + reverse process
│   ├── diffusion_policy.py      # Baseline policy (with D²PPO loss)
│   └── vision_dppo_v31.py       # v3.1 policy (IMU Late Fusion + FCN Depth)
│
├── scripts/
│   ├── train_ppo_expert.py      # Phase 1
│   ├── collect_data.py          # Phase 2 (--v31 / --v32 flags for IMU+depth)
│   ├── train_diffusion.py       # Phase 3a (baseline)
│   ├── train_diffusion_v31.py   # Phase 3a v3.1 (finite-diff IMU — historical)
│   ├── train_diffusion_v32.py   # Phase 3a v3.2 (physics IMU — current)
│   ├── train_dppo.py            # Phase 3b (baseline DPPO)
│   ├── train_dppo_v31.py        # Phase 3c v3.1 (abandoned — finite-diff IMU)
│   ├── train_dppo_v32.py        # Phase 3c v3.2 (current — physics IMU)
│   ├── evaluate_rhc.py          # Phase 4 (baseline)
│   ├── evaluate_rhc_v31.py      # Phase 4 v3.1 (historical)
│   └── evaluate_rhc_v32.py      # Phase 4 v3.2 (current)
│
├── docs/
│   ├── dev_log.md               # Phase 1 training log (PPO Expert)
│   ├── dev_log_phase2_3.md      # Dev log index (Phase 2–3c) ← start here
│   ├── dev_log_phase2.md        # Phase 2: expert data collection
│   ├── dev_log_phase3a.md       # Phase 3a: supervised pre-training + bug audits
│   ├── dev_log_phase3b.md       # Phase 3b: DPPO Runs 1–3 + lessons learned
│   ├── dev_log_phase3c_v31.md   # Phase 3c v3.1: IMU late fusion, Runs 1–2
│   ├── dev_log_phase3c_v32.md   # Phase 3c v3.2: physics IMU, current run
│   ├── architecture.md          # Full architecture diagrams (all versions)
│   ├── phase3c_action_plan.md   # Phase 3c decision log & next steps
│   └── TOP_CONF_GUIDE.md        # Conference submission guide (CoRL/ICRA/RSS)
│
└── utils/
    ├── training_metrics.py
    └── visualization.py
```

---

## MDP Definition / MDP 定義

**Observation Space (15D):**

| Dims | Content | Design Rationale |
|------|---------|-----------------|
| 0-2 | Body-frame position error `R^T(p_target - p)` | Rotation invariant |
| 3-8 | 6D rotation (first 2 columns of R) | Gimbal-lock free |
| 9-11 | Body-frame linear velocity `R^T v` | Rotation invariant |
| 12-14 | Body-frame angular velocity `ω` | Attitude control signal |

**Action Space (4D):** Normalized motor thrusts `[-1,1]` → `[0, f_max]`

**Reward Function (Gaussian-based):**

```
R = w_pos × exp(−||pos_err||² / σ_pos)
  + w_vel × exp(−||vel||²    / σ_vel)
  + w_ang × exp(−||ang_vel||² / σ_ang)
  − w_action × ||action||²
  + alive_bonus
```

Current tuning focus (Run 4): `sigma_pos=0.10`, `w_action=0.01`, `alive_bonus=0.0`

**Timing Configuration:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Physics Δt | 0.005s (200Hz) | RK4 integration |
| RL Decision Δt | 0.02s (50Hz) | Motor command update |
| RHC Replan Rate | 12.5Hz (current) → 62Hz+ (target) | Every T_action steps |

---

## Evaluation Metrics / 評估指標

| Metric | Current Best | Latest Run | Target | Conference |
|--------|-------------|------------|--------|-----------|
| Position RMSE | 0.069m (PPO) / **0.168m** (DPPO Run 2) | 0.409m (Run 4, DR-aug pretrained) | **<0.168m (Phase 3c goal)** | ICRA |
| Crash Rate | 0% (PPO) / 100% (all diffusion runs to date) | 50/50 (Run 4) | **<50% (Phase 3c) → <10% (Phase 4)** | CoRL |
| Inference Latency | ~94ms (10-step DDIM) | — | **<20ms (after OneDP)** | CoRL/ICRA |
| Control Frequency | 12.5Hz | — | **>60Hz (after OneDP)** | ICRA |
| IMU covariate shift (ax std ratio) | 23× (v3.1 finite-diff) | **1.4×** (v3.2 physics) | <2× | — |

---

## References / 參考文獻

1. Chi et al. (2023). "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." *RSS*
2. Zou et al. (2025). "D²PPO: Diffusion Policy Policy Optimization with Dispersive Loss." *arXiv:2508.02644*
3. Ze et al. (2024). "One-Step Diffusion Policy." *arXiv:2410.21257*
4. Ren et al. (2024). "Diffusion Policy Policy Optimization." *OpenReview:mEpqHvbD2h*
5. Kaufmann et al. (2023). "Champion-level drone racing using deep reinforcement learning." *Nature*
6. Schulman et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*

---

## License / 許可證

MIT License

---

**Dev log index (Phase 2–3c) → [docs/dev_log_phase2_3.md](docs/dev_log_phase2_3.md)**
**Phase 1 training log → [docs/dev_log.md](docs/dev_log.md)**
**Architecture diagrams → [docs/architecture.md](docs/architecture.md)**
**Conference submission guide → [docs/TOP_CONF_GUIDE.md](docs/TOP_CONF_GUIDE.md)**

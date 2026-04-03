# DPPO PID Controller - Phase 2–3b Development Log

> Continuation of `dev_log.md` (Phase 1 documented there)
> Phase 2 start: 2026-04-01
> Phase 3b ongoing: 2026-04-03
> Predecessor: PPO Expert Run 6 (`checkpoints/ppo_expert/20260401_103107/`)

---

## Table of Contents

1. [Phase 2: Expert Demonstration Collection](#1-phase-2-expert-demonstration-collection)
2. [Phase 3a: Supervised Diffusion Policy Training](#2-phase-3a-supervised-diffusion-policy-training)
3. [Phase 3 Evaluation: RHC Closed-Loop Baseline](#3-phase-3-evaluation-rhc-closed-loop-baseline)
4. [Phase 3b: D²PPO Fine-Tuning](#4-phase-3b-dppo-fine-tuning)
   - [Run 1: Policy Collapse](#41-run-1-policy-collapse)
   - [Run 2: Conservative Hyperparameters (ongoing)](#42-run-2-conservative-hyperparameters-ongoing)
5. [Key Lessons Learned](#5-key-lessons-learned)
6. [Appendix: Results Summary](#6-appendix-results-summary)

---

## 1. Phase 2: Expert Demonstration Collection

### Objective

Collect (image, action, state) trajectories from the Run 6 PPO expert to serve as supervised training data for the Vision Diffusion Policy. The expert flies in `QuadrotorVisualEnv`, which wraps the base physics environment with a synthetic FPV renderer (64×64 RGB).

### Expert Chosen: Run 6

| Metric | Run 6 |
|--------|-------|
| X error | **0.0096m** |
| Y error | 0.0095m |
| Z error | 0.0684m |
| 3D RMSE | 0.0693m |
| Under 0.1m | 50/50 |
| Crashes | 0/50 |

Run 6 was selected over Runs 7–11 because it had the best X/Y accuracy and zero crashes. The Z bias (~6.8cm) is a consistent systematic offset — learnable and correctable via D²PPO fine-tuning in Phase 3b.

### Pre-Collection Environment Fixes

Before collection, two bugs were identified and fixed that would have silently corrupted the training data:

**Bug 1: `initial_z_range` fallback error**
- `_load_config()` had `self.initial_z_range = e.get('initial_z_range', self.initial_pos_range)` — wrong fallback
- When `initial_z_range` key was removed from YAML, the drone started with a ±10cm Z offset from the target
- This would introduce spurious altitude correction trajectories into the "hover" demonstration data
- **Fix:** Changed fallback to `0.0` → target_z equals init_z exactly

**Bug 2: YAML key not removed**
- `initial_z_range` key was still present in `quadrotor.yaml` from Run 10/11 experiments
- **Fix:** Removed key from config; verified `mean |Z init offset| = 0.000000m` ✓

### Collection Run

```
Episodes:     1,000
Steps/episode: 500 (10s at 50Hz)
Total steps:  500,000
Output:       data/expert_demos.h5  (90MB)
Duration:     ~33 minutes
```

**HDF5 structure per episode:**
```
episode_N/
  images   (500, 3, 64, 64)  uint8   FPV frames
  actions  (500, 4)           float32 motor thrusts ∈ [-1, 1]
  states   (500, 15)          float32 body-frame state observation
```

**Metadata:**
```
n_episodes:  1000
total_steps: 500,000
image_size:  64
state_dim:   15
action_dim:  4
```

---

## 2. Phase 3a: Supervised Diffusion Policy Training

### Architecture

```
FPV image stack (T_obs=2 frames, 6×64×64)
    → VisionEncoder (CNN) → feature vector (256D)
    → ConditionalUnet1D + timestep embedding
    → predicted noise (action_dim=4, T_pred=8)
```

**DemoDataset sliding window:** 491,000 training samples from 1,000 episodes  
(each step t generates one sample: obs[t-T_obs+1:t+1] → action[t:t+T_pred])

### Model

| Component | Details |
|-----------|---------|
| VisionEncoder | CNN, in_channels=6 (2 RGB frames), out=256D |
| ConditionalUnet1D | down_dims=[256, 512], time_embed_dim=128 |
| Diffusion | 100 DDIM steps (train), 10 DDIM steps (inference) |
| Beta schedule | Cosine |
| **Total parameters** | **10,929,256** |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 500 |
| Batch size | 256 |
| Learning rate | 1e-4 (cosine warmup, 10 epochs) |
| Weight decay | 0.01 (AdamW) |
| Grad clip | 1.0 |
| Device | RTX 3090 (CUDA) |

### Results

| Epoch | Loss |
|-------|------|
| 1 (init) | ~1.497 |
| 100 | — |
| 500 | 0.01845 |
| **Best** | **0.01841** |

Loss dropped **98.8%** over 500 epochs (1.497 → 0.018). Training took ~14 hours.

**Checkpoint:** `checkpoints/diffusion_policy/20260402_032701/best_model.pt`

---

## 3. Phase 3 Evaluation: RHC Closed-Loop Baseline

### Evaluation Protocol

Receding Horizon Control (RHC):
1. Stack last T_obs=2 frames → image tensor
2. Run DDIM (10 steps) → predicted action sequence of length T_pred=8
3. Execute first T_action=4 steps in the environment
4. Re-observe, repeat

### Results (50 episodes)

| Metric | Diffusion RHC | PPO Expert (Run 6) |
|--------|--------------|-------------------|
| Mean reward | 21.97 ± 8.77 | 539.14 ± 2.24 |
| Position RMSE | 0.2856m | 0.0693m |
| Crashes | **50/50 (100%)** | 0/50 |
| Mean inference | 71.3ms | — |
| Performance ratio | **4.08%** | 100% |

### Diagnosis: Covariate Shift

The policy crashes every episode. Root cause is **covariate shift**:
- Training distribution: images from expert trajectories (near-perfect hover)
- Inference distribution: images from own (imperfect) trajectories
- Once the drone deviates slightly from hover, the image is out of the training distribution → wrong action → larger deviation → crash

This is the expected failure mode of supervised-only imitation learning for dynamical systems. The fix is D²PPO closed-loop fine-tuning (Phase 3b).

**Inference time** of 71.3ms also slightly exceeds the 50ms target (20Hz decision rate with DDIM 10 steps).

---

## 4. Phase 3b: D²PPO Fine-Tuning

### Objective

Run the diffusion policy in the actual environment, compute GAE advantages, and update the policy with advantage-weighted diffusion loss:

```
L = E[ exp(β × A_normalized) × ||ε_θ(a_t, τ, s) − ε||² ]
```

where β controls how aggressively high-advantage samples are upweighted relative to low-advantage samples.

### Value Network

Shares visual features with the frozen encoder output:
- `ValueNetwork(feature_dim=256, hidden_dim=256)` → scalar V(s)
- Trained separately with MSE on GAE returns
- Used only for advantage estimation, not for inference

---

### 4.1 Run 1: Policy Collapse

**Configuration:**
| Parameter | Value |
|-----------|-------|
| advantage_beta | 1.0 |
| learning_rate | 3e-5 |
| n_rollout_steps | 2048 |
| n_epochs | 5 |
| value_lr | 1e-3 |
| total_updates | 500 |

**Training Curve:**

| Update | Mean Reward/step | Status |
|--------|-----------------|--------|
| 10 | +0.531 | Improving |
| 50 | **+0.561** | **Peak (best checkpoint saved)** |
| 100 | +0.479 | Declining |
| 150 | +0.206 | Degrading |
| 200 | +0.009 | Near zero |
| 300–500 | **−0.43** | Collapsed |

**Evaluation of best checkpoint:**

| Metric | DPPO Run 1 (best ckpt) | Phase 3 Supervised | PPO Expert |
|--------|----------------------|-------------------|------------|
| Mean reward | 27.44 | 21.97 | 539.14 |
| Position RMSE | 0.3781m | 0.2856m | 0.0693m |
| Crashes | **50/50** | 50/50 | 0/50 |

**RMSE worsened** (0.378 vs 0.286) despite saving the best checkpoint. Even the "best" point was already exhibiting collapse behavior, just less severely.

**Root Cause Analysis:**

Two interacting failure modes:

1. **β = 1.0 too aggressive:** The weight `exp(1.0 × A)` amplifies noisy early advantages by up to `e^1 ≈ 2.7×`. This creates a positive feedback loop — large updates → policy degrades → worse advantages → unstable updates.

2. **LR = 3e-5 too high for fine-tuning a pretrained diffusion model:** The supervised pretraining converged the noise prediction network to a specific weight configuration. RL updates at 3e-5 are large enough to overwrite this in ~100 updates, before the value network has converged enough to provide reliable advantage signals.

The value loss curve confirms this: Value Loss at update 10 = 36.3 → value network is far from converged when the policy already starts degrading.

---

### 4.2 Run 2: Conservative Hyperparameters (ongoing)

**Changes applied to `configs/diffusion_policy.yaml`:**

| Parameter | Run 1 | Run 2 | Rationale |
|-----------|-------|-------|-----------|
| `advantage_beta` | 1.0 | **0.1** | Max weight = exp(0.1×3) ≈ 1.35× vs exp(1.0×3) = 20×; prevents early collapse |
| `learning_rate` | 3e-5 | **5e-6** | 6× more conservative; preserves pretrained weights longer |
| `n_rollout_steps` | 2048 | **4096** | More data per update → lower-variance advantage estimates |
| `n_epochs` | 5 | **3** | Fewer gradient steps per update → smaller per-update drift |
| `value_lr` | 1e-3 | **3e-4** | Let value network converge more gradually alongside policy |

**Expected behavior:**
- Value network should converge (loss < 1.0) within first 50–100 updates before policy updates are large enough to matter
- Policy reward should rise monotonically without the "peak then collapse" pattern
- With β=0.1, the advantage weighting is soft enough that poor early estimates don't destroy pretraining

**Status:** Running (task `bi9e4gphn`), started 2026-04-03. ~10–11 hours estimated.

**Checkpoint:** `checkpoints/diffusion_policy/dppo_<timestamp>/best_dppo_model.pt`

---

## 5. Key Lessons Learned

### L1: Covariate Shift is Unavoidable Without Closed-Loop Training

Supervised diffusion policy trained on expert-only data will always fail at inference time when the drone deviates from the expert distribution. Even with 491k training samples and loss=0.018, position RMSE = 0.286m and crash rate = 100%. D²PPO (or DAgger-style online data collection) is mandatory, not optional.

### L2: DPPO advantage_beta Must Start Very Small

`β = 1.0` is commonly cited in DPPO papers for NLP tasks where the policy is already near-optimal and advantages are low-variance. For a quadrotor starting from supervised pretraining with a crashing policy, early advantages are large and noisy. `β = 0.1` limits the weight ratio to ~1.35× at 3σ, making the RL signal a gentle nudge rather than a hard override.

### L3: Value Network Must Converge Before Policy Updates Matter

The value loss at Run 1 update 10 was 36.3 (effectively random predictions). All the "improvement" from updates 1–50 likely came from lucky policy gradient steps, not reliable advantage signals. The right approach: either use a separate warmup phase for the value network, or use very conservative policy LR so the value network catches up naturally.

### L4: Per-Step vs Per-Episode Reward Tracking

The DPPO training script logs mean reward per rollout step (not per episode). A healthy hover step reward is ~0.3–0.6 (from reward terms: pos_reward ≈ 0.55 when within 10cm, vel_reward ≈ 0.2, ang_reward ≈ 0.2). A per-step reward of −0.43 indicates the policy is crashing within seconds every episode and accumulating crash penalty (−10 spread over few steps = large negative per-step).

### L5: 90MB HDF5 at float32 is Efficient for 1000 Episodes

500k timesteps × (3×64×64 uint8 + 4 float32 + 15 float32) = ~615MB raw. With gzip compression level 4 on images, actual file = 90MB (85% reduction). Loading all into RAM (DemoDataset loads eagerly) takes ~2GB — acceptable for a 32GB+ workstation.

---

## 6. Appendix: Results Summary

### Phase 2–3 Progression

| Phase | Model | Pos RMSE | Crashes | Notes |
|-------|-------|----------|---------|-------|
| 2 | PPO Expert (Run 6) | 0.0693m | 0/50 | Expert used for data collection |
| 3a | Diffusion (supervised) | 0.2856m | 50/50 | Covariate shift |
| 3b Run 1 | DPPO best ckpt | 0.3781m | 50/50 | Policy collapse after update ~100 |
| 3b Run 2 | DPPO (ongoing) | — | — | Conservative β=0.1, lr=5e-6 |

### File Locations

| Artifact | Path |
|----------|------|
| Expert demos | `data/expert_demos.h5` |
| Diffusion supervised | `checkpoints/diffusion_policy/20260402_032701/best_model.pt` |
| DPPO Run 1 best | `checkpoints/diffusion_policy/dppo_20260403_040722/best_dppo_model.pt` |
| DPPO Run 2 (ongoing) | `checkpoints/diffusion_policy/dppo_<timestamp>/` |
| RHC eval (supervised) | `evaluation_results/rhc_phase3/` |
| RHC eval (DPPO R1) | `evaluation_results/rhc_dppo/` |

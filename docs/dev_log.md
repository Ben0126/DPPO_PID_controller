# DPPO PID Controller - Research & Development Log

> Project: Vision-based Diffusion Policy with PPO for Quadrotor Control
> Date: 2026-03-30 ~ 2026-03-31
> Pipeline: PPO Expert → Expert Data Collection → Diffusion Policy → DPPO Fine-tuning → RHC Evaluation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Phase 1: PPO Expert Training](#2-phase-1-ppo-expert-training)
   - [Run 1: Baseline Training](#21-run-1-baseline-training)
   - [Run 2: Reward Shaping & Architecture Fix](#22-run-2-reward-shaping--architecture-fix)
   - [Run 3: KL Bottleneck Fix](#23-run-3-kl-bottleneck-fix)
   - [Run 4: Breaking the Reward Equilibrium](#24-run-4-breaking-the-reward-equilibrium)
3. [Phase 2-3: Diffusion Policy (Early Attempt)](#3-phase-2-3-diffusion-policy-early-attempt)
4. [Environment Setup Issues](#4-environment-setup-issues)
5. [Key Lessons Learned](#5-key-lessons-learned)
6. [Appendix: Hyperparameter Reference](#6-appendix-hyperparameter-reference)

---

## 1. Project Overview

### Pipeline Architecture

```
Phase 1: Train PPO Expert (state → motor thrust)
    ↓
Phase 2: Collect Expert Demos (state + FPV images + actions → HDF5)
    ↓
Phase 3: Train Vision Diffusion Policy (images → action sequences)
    ↓
Phase 3b: DPPO Fine-tuning (optional RL refinement)
    ↓
Phase 4: RHC Evaluation (predict T_pred actions, execute T_action)
```

### Core Technical Details

- **Quadrotor**: 6-DOF, mass=0.5kg, 4 motors, max thrust 4.0N/motor
- **Observation**: 15D state (body-frame position error [3], 6D rotation [6], body velocity [3], angular velocity [3])
- **Action**: 4D motor thrusts, TanhNormal distribution maps [-1, 1] → [0, f_max] via `thrust = (action + 1) * 0.5 * 4.0`
- **Coordinate System**: NED (Z-positive = downward), altitude = -Z
- **Hover Thrust**: F_hover = mg/4 = 0.5 × 9.81 / 4 = 1.226N per motor → normalized action ≈ -0.387

---

## 2. Phase 1: PPO Expert Training

The PPO expert must achieve **stable hover with position error < 0.1m** before it can serve as a reliable data source for downstream imitation learning. Poor expert quality propagates through the entire pipeline.

### 2.1 Run 1: Baseline Training

**Configuration:**
| Parameter | Value |
|-----------|-------|
| total_timesteps | 3,000,000 |
| learning_rate | 3e-4 (linearly annealed) |
| n_steps | 4096 |
| batch_size | 256 |
| clip_range | 0.2 → 0.05 (annealed) |
| n_epochs | 10 |
| ent_coef | 0.01 |
| vf_coef | 0.5 |
| target_kl | 0.01 |
| sigma_pos | 0.5 |
| w_pos / w_vel / w_ang | 0.5 / 0.2 / 0.2 |
| initial_pos_range | 0.5 |

**Results:**
- Reward peaked early (~400K steps) then **degraded** in later training
- Mean position error: **0.264m** (far from 0.1m target)
- 0/50 episodes achieved < 0.1m error
- Z-axis error: ~0.27m (agent hovered below target)

**Problem Analysis:**

#### Problem 1: Late-Training Degradation
Training reward peaked early then declined. The policy was making destructive updates in later stages when it should have been refining.

**Root Cause:** No learning rate or clip range annealing. A fixed LR=3e-4 and clip=0.2 throughout training meant the policy kept making large updates even after converging, causing "policy churn" — the agent oscillated rather than stabilized.

**Fix:** Added linear annealing for both LR and clip range:
```python
frac = 1.0 - timestep / total_timesteps
lr_now = learning_rate * frac          # 3e-4 → 0
clip = clip_start * frac + clip_end * (1 - frac)  # 0.2 → 0.05
```

#### Problem 2: Loose Reward Shaping (sigma_pos = 0.5)
The Gaussian reward function `r_pos = exp(-error² / sigma²)` with sigma_pos=0.5 was too lenient:

| Position Error | Reward (sigma=0.5) | Reward (sigma=0.15) |
|---------------|--------------------|--------------------|
| 0.0m          | 1.000              | 1.000              |
| 0.1m          | 0.961              | 0.641              |
| 0.27m         | 0.726              | 0.036              |

With sigma=0.5, the reward difference between perfect hover (0m) and 0.27m off-target was only **0.274 × w_pos = 0.137**. The agent had no incentive to refine beyond ~0.3m because the marginal reward gain was negligible compared to the action penalty cost.

**Fix:** Tighten sigma_pos from 0.5 to **0.15**, creating a steep reward gradient that strongly penalizes any error > 0.1m.

#### Problem 3: No Gravity Compensation in Actor Initialization
The Actor network's output layer bias initialized near zero, meaning initial actions ≈ tanh(0) = 0, which maps to thrust = 2.0N per motor (50% throttle). But hover requires 1.226N (≈31% throttle, action ≈ -0.387). The agent started every episode at 2× hover thrust, causing immediate upward acceleration and chaotic early trajectories.

**Fix:** Initialize Actor mean_layer bias to -0.39 (pre-tanh value for hover thrust):
```python
# F_hover = mg/4 = 1.226N, action_hover = (1.226/4.0)*2 - 1 = -0.387
# tanh(x) ≈ x for small x, so pre-tanh bias ≈ -0.39
nn.init.constant_(self.mean_layer.bias, -0.39)
```

#### Problem 4: Initial Conditions Too Wide
With initial_pos_range=0.5m and initial_vel_range=0.2m/s, the agent faced a wide distribution of starting states. Before the policy learns basic hover, many episodes start in unrecoverable positions, generating low-reward data that dilutes the learning signal.

**Fix:** Narrow to initial_pos_range=**0.1m**, initial_vel_range=**0.05m/s**. Let the agent master near-hover first, then widen later for generalization.

---

### 2.2 Run 2: Reward Shaping & Architecture Fix

**Changes Applied:**
| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| sigma_pos | 0.5 | **0.15** | Tighter Gaussian demands precision |
| w_pos | 0.5 | **0.65** | Position is dominant training signal |
| w_action | 0.05 | **0.03** | Reduced to not discourage corrective actions |
| alive_bonus | 0.1 | **0.05** | Reduced to prevent "lazy hover" strategy |
| initial_pos_range | 0.5 | **0.1** | Narrow for stable initial learning |
| initial_vel_range | 0.2 | **0.05** | Narrow for stable initial learning |
| Actor bias | 0 | **-0.39** | Gravity compensation from step 0 |

**Results:**
- Best eval reward: **426.45** at step 1,400,832
- Mean position error: **0.0993m** (improved from 0.264m, 62% reduction)
- 25/50 episodes under 0.1m (was 0/50)
- Z-axis error: **0.0934m** (was ~0.27m)
- 0 crashes

**Assessment Against Success Criteria:**
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Mean position error | < 0.10m | 0.0993m | Borderline |
| Episodes < 0.1m error | > 40/50 | 25/50 | NOT MET |
| Crash rate | 0 | 0 | MET |
| Z-axis error | < 0.05m | 0.0934m | NOT MET |

Significant improvement, but not yet at the quality threshold. TensorBoard analysis revealed the underlying cause.

**Problem Analysis:**

#### Problem 5: KL Early Stopping Throttling Optimization (Critical)

TensorBoard curves revealed a severe optimization bottleneck:

**Evidence:**
- **KL early stop triggered on 729/733 updates (99.5%)**
- **Entropy rose from ~3.5 to ~7.1** throughout training (should decrease as policy tightens)
- **Value Loss stuck at 60-110**, never converging (critic underfitting)
- **Eval reward plateaued at ~420** from step 1.4M to 3M — zero improvement over 1.6M steps

**Root Cause:** `target_kl = 0.01` was far too strict. With 4096 rollout steps and batch_size=256, each epoch has 16 mini-batches, and 10 epochs = 160 total mini-batches per update. But with KL firing after just 1-2 mini-batches, both the Actor and Critic were severely undertrained:

```
target_kl = 0.01 too strict
    → 99.5% of updates truncated after 1-2 mini-batches
    → Policy can't tighten (entropy rises from 3.5 → 7.1)
    → Critic can't fit returns (value loss stays at 60-110)
    → GAE advantage estimates are noisy
    → Reward plateaus at ~420, no further improvement possible
```

The entropy rising is particularly diagnostic: a healthy PPO training should show entropy **decreasing** as the policy becomes more confident about correct actions. Rising entropy means the policy is becoming MORE uncertain — the KL constraint is actively preventing convergence.

**Fix (Run 3):**
| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| target_kl | 0.01 | **0.04** | Allow more gradient steps per update |
| ent_coef | 0.005 | **0.001** | Stop encouraging exploration — entropy already too high |
| vf_coef | 1.0 | **1.5** | Give critic more gradient weight to fix value loss |

---

### 2.3 Run 3: KL Bottleneck Fix

**Changes Applied:**
| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| target_kl | 0.01 | **0.04** | Allow more gradient steps per update |
| ent_coef | 0.005 | **0.001** | Entropy was rising to 7.1; stop encouraging exploration |
| vf_coef | 1.0 | **1.5** | Give critic more gradient weight to fix value loss |

**Results:**
- Best eval reward: **502.70** at step 503,808 (Run 2: 426.45, +18%)
- Mean position error: **0.1041m** ± 0.0015m
- 0/50 episodes under 0.1m
- Z-axis error: **0.0876m**
- 0 crashes, all 50 episodes run full 500 steps
- KL early stop rate: **48.0%** (Run 2: 99.5%) — bottleneck resolved

**KL Fix Validation:**
| Metric | Run 2 | Run 3 | Improvement |
|--------|-------|-------|-------------|
| KL stop rate | 99.5% | 48.0% | Bottleneck eliminated |
| Best eval reward | 426.45 | 502.70 | +18% |
| Reward std | high variance | 2.97 | Extremely stable |
| Entropy (end) | 7.1 | 6.2 | Slight improvement |

**Problem Analysis:**

#### Problem 6: Reward Function Equilibrium (Reward Shaping Limit)

The position error converged to **0.1041m ± 0.0015m** across all 50 episodes. The incredibly tight std (0.0015m) proves this is NOT noise — it is a **deterministic equilibrium of the reward function**.

**Root Cause:** The agent found the optimal trade-off point where:
```
Marginal gain from reducing position error (via position reward)
≈ Marginal cost of larger corrective actions (via action penalty)
  + Free reward from alive_bonus that dilutes position signal
```

At error ≈ 0.104m with sigma_pos=0.15:
- Position reward = exp(-0.104²/0.15²) × 0.65 = 0.404
- Moving to 0.00m would give 0.650 — a gain of only 0.246 per step
- But the corrective actions cost w_action × action² = 0.03 × action²
- Plus alive_bonus = 0.05 per step is "free" regardless of precision

The agent correctly maximizes total reward by stabilizing at 0.104m rather than fighting for 0.00m. **The optimization is working perfectly — the objective function is wrong.**

Z-axis error (0.0876m) accounts for 84% of total error, confirming the agent still under-compensates for gravity on the vertical axis.

**Fix (Run 4):** Break the equilibrium by making position reward steeper and removing opposing forces:

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| sigma_pos | 0.15 | **0.10** | Steeper gradient — 0.1m error now gets much less reward |
| w_action | 0.03 | **0.01** | Reduce the "cost" of corrective actions |
| alive_bonus | 0.05 | **0.0** | Remove free reward that dilutes position signal |

Reward comparison (w_pos=0.65 weighted):

| Error | sigma=0.15 | sigma=0.10 | Delta |
|-------|-----------|-----------|-------|
| 0.00m | 0.650 | 0.650 | — |
| 0.05m | 0.586 | 0.490 | -0.096 |
| 0.10m | 0.404 | 0.239 | **-0.165** (penalty doubles) |
| 0.15m | 0.195 | 0.066 | -0.129 |

With sigma=0.10, the reward at 0.1m is only 37% of perfect (vs 62% with sigma=0.15). Combined with w_action=0.01 and alive_bonus=0, the equilibrium should shift well below 0.1m.

---

### 2.4 Run 4: Breaking the Reward Equilibrium

**Changes Applied:**
| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| sigma_pos | 0.15 | **0.10** | Steeper Gaussian penalizes 0.1m error more heavily |
| w_action | 0.03 | **0.01** | Remove action penalty as obstacle to precision |
| alive_bonus | 0.05 | **0.0** | Remove free reward; agent must earn from position |

**Status:** Training in progress (3M steps).

**Expected Outcomes:**
- Reward equilibrium should shift below 0.05m position error
- Z-axis error should decrease significantly
- Episodes < 0.1m target: > 40/50

*Results will be updated after training completes.*

---

## 3. Phase 2-3: Diffusion Policy (Early Attempt)

An early attempt was made to proceed through the full pipeline before the PPO expert met quality standards.

### Phase 2: Data Collection
- Used best PPO model (0.264m error) to collect expert demonstrations
- Deterministic policy rollouts in QuadrotorVisualEnv (64x64 RGB FPV images)
- Saved to HDF5: images (uint8), actions (float32), states (float32)

### Phase 3: Diffusion Policy Training
- Vision encoder + 1D U-Net + DDIM sampling (10 inference steps)
- Trained 136/500 epochs, loss converged to ~0.042
- Training was stopped early to test downstream performance

### Phase 4: RHC Evaluation
- **Result: 50/50 episodes crashed**
- Predicted actions were systematically biased (less negative than expert), causing upward acceleration
- Root cause: expert demonstrations themselves were low quality (0.264m position error), so the diffusion policy learned to imitate imprecise behavior and amplified the error

**Lesson:** The quality of the PPO expert is the single most important factor in this pipeline. A diffusion policy trained on poor demonstrations will always fail, regardless of how well the diffusion training loss converges. **Fix the expert first.**

---

## 4. Environment Setup Issues

### Issue: PowerShell Execution Policy
**Symptom:** `venv\Scripts\Activate.ps1` failed with execution policy error.
**Fix:** Use Git Bash, or run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in PowerShell.

### Issue: PyTorch Not Detecting GPU
**Symptom:** `torch.cuda.is_available()` returned `False` despite NVIDIA GTX 1650 being present.
**Root Cause:** PyTorch installed with wrong CUDA version (cu128 for a card that needed cu124, or CPU-only build).
**Fix:** Reinstall with correct CUDA version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

---

## 5. Key Lessons Learned

### 5.1 Reward Shaping

- **Gaussian reward width (sigma) directly controls precision ceiling.** A loose sigma makes the reward landscape nearly flat near the target, removing the gradient signal needed for fine-grained control. Calculate the reward difference at your target error threshold before choosing sigma.
- **Weight allocation matters.** Position weight should dominate (w_pos ≥ 0.6) for hover tasks. Over-weighting action penalty discourages the corrective actions needed for precision.
- **alive_bonus can create perverse incentives.** A large alive_bonus rewards survival regardless of performance — the agent learns to "hover lazily" at a safe-but-imprecise position rather than risking crashes to achieve precision.

### 5.2 Reward Equilibrium

- **A well-optimized agent can converge to the wrong answer.** If the agent's position error has extremely low variance (e.g., std=0.0015m) but is consistently above target, this is a reward equilibrium — not a training failure. The agent has found the point where marginal position improvement equals marginal action cost.
- **Diagnose by checking error std, not just mean.** High std = noisy optimization (train longer or fix PPO params). Low std + wrong mean = reward function equilibrium (change reward weights).
- **Three levers to shift the equilibrium:** (1) tighten sigma_pos to steepen the reward gradient, (2) reduce w_action to lower the cost of corrective actions, (3) remove alive_bonus to eliminate free reward that dilutes the position signal.

### 5.3 PPO Optimization

- **KL early stopping is a double-edged sword.** Too strict (0.01) and it chokes the entire optimization — both policy AND value function. Monitor the KL trigger rate in TensorBoard. If > 80% of updates are truncated, the threshold is too strict.
- **Entropy should decrease, not increase.** Rising entropy during training is a red flag that the policy can't converge. Common causes: KL stopping too aggressively, or ent_coef too high.
- **Value loss quality matters for advantage estimation.** If the critic can't fit returns (VL > 50), GAE advantages will be noisy, and policy updates will be inefficient. Consider increasing vf_coef or giving the critic more update steps.
- **LR and clip range annealing are essential** for stable late-training behavior. Without annealing, the policy oscillates instead of refining in later stages.

### 5.4 Actor Initialization

- **Physics-informed initialization dramatically accelerates learning.** For hover tasks, initializing the output bias to the gravity-compensating thrust value means the agent starts with approximate hover from step 0, rather than spending thousands of steps learning to not crash.
- The math: `action_hover = (F_hover / F_max) * 2 - 1`, where `F_hover = mg / n_motors`.

### 5.5 Pipeline Design

- **Expert quality is the bottleneck.** No amount of diffusion policy training can compensate for a poor expert. Set clear quality gates (position error, crash rate, consistency) and don't proceed until they're met.
- **Narrow initial conditions for learning, widen for generalization.** Start with easy states to build a strong base policy, then gradually increase difficulty. Curriculum learning in disguise.

### 5.6 Diagnostic Workflow

When training plateaus, follow this checklist:
1. **Check eval reward curve** — is it still improving or flat?
2. **Check KL early stop rate** — if > 80%, target_kl is too strict
3. **Check entropy trend** — should decrease; rising = policy can't converge
4. **Check value loss trend** — should decrease; flat/rising = critic underfitting
5. **Run evaluation with per-axis error breakdown** — identify which axis is the bottleneck

---

## 6. Appendix: Hyperparameter Reference

### PPO Training (`configs/ppo_expert.yaml`)

| Parameter | Run 1 | Run 2 | Run 3 | Run 4 | Notes |
|-----------|-------|-------|-------|-------|-------|
| total_timesteps | 3M | 3M | 3M | 3M | |
| learning_rate | 3e-4 | 3e-4 | 3e-4 | 3e-4 | Linearly annealed to 0 |
| n_steps | 4096 | 4096 | 4096 | 4096 | Rollout length |
| batch_size | 256 | 256 | 256 | 256 | |
| clip_range | 0.2→0.05 | 0.2→0.05 | 0.2→0.05 | 0.2→0.05 | Annealed |
| n_epochs | 10 | 10 | 10 | 10 | |
| ent_coef | 0.01 | 0.005 | **0.001** | 0.001 | Reduced Run 1→3 |
| vf_coef | 0.5 | 1.0 | **1.5** | 1.5 | Increased Run 1→3 |
| target_kl | 0.01 | 0.01 | **0.04** | 0.04 | Critical fix in Run 3 |

### Reward Function (`configs/quadrotor.yaml`)

| Parameter | Run 1 | Run 2/3 | Run 4 | Notes |
|-----------|-------|---------|-------|-------|
| reward_type | gaussian | gaussian | gaussian | |
| sigma_pos | 0.5 | 0.15 | **0.10** | Tighter each iteration |
| sigma_vel | 1.0 | 1.0 | 1.0 | |
| sigma_ang | 1.0 | 1.0 | 1.0 | |
| w_pos | 0.5 | 0.65 | 0.65 | Position dominates |
| w_vel | 0.2 | 0.2 | 0.2 | |
| w_ang | 0.2 | 0.2 | 0.2 | |
| w_action | 0.05 | 0.03 | **0.01** | Reduced to not fight precision |
| alive_bonus | 0.1 | 0.05 | **0.0** | Removed: diluted position signal |
| crash_penalty | 10.0 | 10.0 | 10.0 | |

### Environment

| Parameter | Run 1 | Run 2/3/4 | Notes |
|-----------|-------|-----------|-------|
| initial_pos_range | 0.5 | **0.1** | Narrowed for learning |
| initial_vel_range | 0.2 | **0.05** | Narrowed for learning |
| max_episode_steps | 500 | 500 | 10s at 50Hz |

### Training Results Summary

| Metric | Run 1 | Run 2 | Run 3 | Run 4 |
|--------|-------|-------|-------|-------|
| Best eval reward | ~380 | 426.45 | 502.70 | TBD |
| Mean position error | 0.264m | 0.0993m | 0.1041m | TBD |
| Episodes < 0.1m (of 50) | 0 | 25 | 0 | TBD |
| Z-axis error | ~0.27m | 0.0934m | 0.0876m | TBD |
| Crash rate | 0% | 0% | 0% | TBD |
| KL stop rate | ~99% | 99.5% | 48.0% | TBD |
| Final entropy | ~7 | ~7.1 | 6.2 | TBD |
| Pos error std | high | high | 0.0015m | TBD |

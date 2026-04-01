# DPPO PID Controller - Research & Development Log

> Project: Vision-based Diffusion Policy with PPO for Quadrotor Control
> Date: 2026-03-30 ~ ongoing
> Pipeline: PPO Expert → Expert Data Collection → Diffusion Policy → D²PPO (Dispersive Loss) → OneDP Distillation → RHC Evaluation
>
> **Research Plan Version:** v3.0 (updated 2026-03-31)
> **Target Venues:** CoRL 2025 / ICRA 2026 / RSS 2026
> **Core Contributions:** D²PPO Dispersive Loss (representation collapse prevention) + OneDP single-step distillation (62Hz+ deployment)

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
6. [Research Direction v3.0 Updates](#6-research-direction-v30-updates)
7. [Appendix: Hyperparameter Reference](#7-appendix-hyperparameter-reference)

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

**Training Setup:**
- Vectorized: 16 parallel `AsyncVectorEnv` (multiprocessing), RTX 3090 GPU (`cuda:0`)
- Transitions per PPO update: 65,536 (4,096 × 16 envs)
- Total PPO updates: ~46 (vs 732 in single-env runs)
- Estimated training time: ~15–20 minutes (vs ~35–50 min for single-env CPU)

**Results:**
- Best eval reward: **494.39** at step 2,555,904
- Final eval reward: 493.20 (stable, no degradation at end)
- 0 crashes observed throughout training
- Rapid improvement phase: step 720K–983K (reward 29 → 410, nearly linear ascent)
- Plateau onset: ~step 1.3M onward (mean ~400–413, eval ~471–494)

**Reward Curve:**
| Step | Mean Reward (100) | Eval Reward |
|------|-------------------|-------------|
| 65K | -1.87 | 6.20 |
| 720K | 29.85 | 37.74 |
| 917K | 139.37 | 355.93 |
| 1,048K | 357.59 | 448.59 |
| 1,179K | 395.28 | 465.87 |
| 1,507K | 409.02 | **479.86** |
| 2,555K | 407.33 | **494.39** ← best |
| 3,014K | 406.92 | 493.20 |

**Cross-run reward normalization:** Run 4 removed `alive_bonus=0.05`. Over 500 steps this subtracts 25 points from maximum episode reward. Adjusted comparison:

| Run | Best Eval | Alive Bonus (500 steps) | Adjusted Reward |
|-----|-----------|------------------------|-----------------|
| Run 3 | 502.70 | +25.0 | **477.70** |
| Run 4 | 494.39 | +0.0 | **494.39** |

Run 4 outperforms Run 3 by **+16.7 points** on an equivalent basis. Furthermore, Run 4 uses sigma_pos=0.10 (vs 0.15 in Run 3), meaning the same reward level corresponds to tighter position tracking. Absolute reward numbers are not comparable across runs with structural reward changes.

**Problem Analysis:**

#### Problem 7: Value Loss Divergence (Critic Underfitting at Scale)
Value Loss increased monotonically from 22.78 → 81.37 throughout training, never converging. This mirrors Run 3 (stuck at 60–110) but with a lower starting point.

**Root Cause:** With 16 parallel environments, each PPO update trains on 65,536 transitions covering far more state diversity than single-env 4,096-step rollouts. The critic (256-dim, 2-layer MLP) does not have sufficient capacity to fit the return function across this expanded state distribution. The current `vf_coef=1.5` is insufficient to compensate.

**Impact on Training:** Noisy advantage estimates from the underfitting critic slow convergence and likely explain the plateau at ~494. The policy plateau does NOT indicate reward equilibrium (unlike Run 3's 0.0015m std) because the critic's poor return estimates mean GAE advantages are inaccurate.

**Proposed Fix (Run 5 if needed):**
| Parameter | Current | Proposed | Reason |
|-----------|---------|----------|--------|
| hidden_dim | 256 | **512** | Wider critic to handle larger state diversity |
| vf_coef | 1.5 | **2.0** | More gradient weight on value function |
| Or: separate critic LR | — | **3× actor LR** | Allow critic to train faster independently |

**Phase gate assessment:** Detailed evaluation with per-axis error breakdown required to confirm position error < 0.1m. Run an explicit eval script with 50 episodes before proceeding to Phase 2.

**Phase gate to proceed to Phase 2:**

| Metric | Target | Run 4 Status |
|--------|--------|--------------|
| Mean position error | < 0.10m | Needs eval — reward level suggests likely met |
| Episodes < 0.1m | > 40/50 | Needs eval |
| Z-axis error | < 0.05m | Needs eval |
| Crash rate | 0 | MET (0 crashes during training) |

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

### Issue: PyTorch Not Detecting GPU (RTX 3090 / CUDA 12.9)
**Symptom:** `torch.cuda.is_available()` returned `False` despite RTX 3090 + CUDA 12.9 being present. Training ran on CPU and printed `Device: cpu`.
**Root Cause:** `pip install torch` (without an `--index-url`) always installs the CPU-only wheel from PyPI (`torch 2.8.0+cpu`). PyTorch CUDA wheels are hosted on a separate index and are not installed by default.
**Diagnosis:** `python -c "import torch; print(torch.__version__)"` → `2.8.0+cpu`
**Fix:** Reinstall from PyTorch's CUDA wheel index:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
```
PyTorch cu128 is compatible with CUDA 12.8 and 12.9. Also updated `requirements.txt` to include `--extra-index-url` so future installs automatically pick the correct wheel.

### Issue: AsyncVectorEnv Requires Picklable Env Factories (Windows)
**Symptom:** `AsyncVectorEnv` fails with pickling errors when using lambda or inner-function env factories.
**Root Cause:** Windows uses the `spawn` multiprocessing start method (unlike Linux's `fork`). The `spawn` method requires all objects passed to worker processes — including env factory functions — to be picklable. Lambda functions and closures are not picklable.
**Fix:** Replace lambda/closure factories with a picklable callable class:
```python
class _EnvFactory:
    def __init__(self, config_path: str):
        self.config_path = os.path.abspath(config_path)
    def __call__(self) -> QuadrotorEnv:
        return QuadrotorEnv(config_path=self.config_path)
```
Also resolves the config path to absolute before passing to workers, since worker processes may have a different working directory.

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

### 5.7 Reward Comparison Across Runs

- **Never compare absolute rewards across runs with structural reward changes.** Removing `alive_bonus` subtracts `bonus × max_steps` from every episode's ceiling. Tightening `sigma_pos` changes what position accuracy is needed to achieve the same per-step reward. Before concluding a run is "worse", adjust for these structural differences.
- **Formula:** adjusted_reward = raw_reward − alive_bonus × max_episode_steps. Compare only adjusted rewards when alive_bonus changed between runs.

### 5.8 Critic Capacity and Large Batch Sizes

- **Vectorized environments dramatically increase state diversity per PPO update.** A single-env 4096-step rollout samples a narrow trajectory; 16-env × 4096 samples 65,536 transitions covering much broader state space. A critic that converges with a single env may diverge (value loss rising monotonically) with 16 envs.
- **Symptom:** Value loss starts lower but rises continuously rather than converging. Unlike KL-choked critic (Run 2–3 pattern), this is a capacity issue, not a training speed issue.
- **Fix options:** (1) widen critic hidden_dim, (2) raise vf_coef further, (3) give critic a separate higher learning rate, (4) reduce n_envs if not enough GPU memory for larger critic.

---

## 7. Appendix: Hyperparameter Reference

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
| Best eval reward (raw) | ~380 | 426.45 | 502.70 | **494.39** |
| Best eval reward (adjusted¹) | ~355 | ~401 | **477.70** | **494.39** |
| Mean position error | 0.264m | 0.0993m | 0.1041m | Needs eval |
| Episodes < 0.1m (of 50) | 0 | 25 | 0 | Needs eval |
| Z-axis error | ~0.27m | 0.0934m | 0.0876m | Needs eval |
| Crash rate | 0% | 0% | 0% | 0% |
| KL stop rate | ~99% | 99.5% | 48.0% | N/A² |
| Final entropy | ~7 | ~7.1 | 6.2 | N/A² |
| Value loss (final) | — | high | 60–110 | ~81 (rising) |
| Pos error std | high | high | 0.0015m | Needs eval |
| Envs (parallel) | 1 | 1 | 1 | **16** |
| Device | CPU | CPU | CPU | **RTX 3090** |
| Est. training time | ~40 min | ~40 min | ~40 min | **~15–20 min** |

¹ Adjusted = raw − alive_bonus × max_episode_steps. Run 1–3: alive_bonus=0.1/0.05 × 500 steps. Run 4: alive_bonus=0.  
² KL/entropy not logged in vectorized run; would require adding TensorBoard tracking.

---

## 6. Research Direction v3.0 Updates

*Updated 2026-03-31. This section records the strategic shift from the original v2.0 plan to the v3.0 research direction that this dev log now targets.*

### 6.1 Why the Research Scope Expanded

The original plan (v2.0) treated this project primarily as an implementation exercise: build the pipeline, evaluate, deploy. Version 3.0 repositions it as a targeted research contribution aimed at CoRL 2025 / ICRA 2026.

The shift was driven by identifying three concrete gaps in existing work that this project is uniquely positioned to address:

1. **Representation collapse in visual diffusion policies** — standard diffusion loss provides no incentive for discriminative visual features. D²PPO's Dispersive Loss directly addresses this.
2. **Control frequency bottleneck** — 10-step DDIM at 12.5Hz is insufficient for aggressive quadrotor dynamics. OneDP single-step distillation targets 62Hz+.
3. **Missing fair baselines** — prior work compared visual methods against state-based oracles. This plan adds BC-LSTM and VTD3 for fair comparison.

### 6.2 Core Research Contribution (D²PPO)

**Dispersive Loss** forces visual feature vectors within a mini-batch to repel each other:

```python
L_dispersive = -log(||h_i - h_j||) for all i≠j in batch
L_total = L_diffusion + λ × L_dispersive   # λ = 0.1, tune 0.01~0.5
```

This prevents the encoder from mapping visually distinct flight states to identical embeddings, which would prevent the U-Net from generating appropriate corrective actions.

**Ablation design (non-negotiable for publication):**
- Baseline: no dispersive loss
- Early layers only / late layers only / all layers
- 3 seeds per condition

### 6.3 Deployment Upgrade (OneDP)

After D²PPO fine-tuning, distill the multi-step teacher into a 1-step student:

```
Teacher: D²PPO model (10-step DDIM, ~80ms latency)
Student: 1-step model via KL minimization
Target: <16ms inference, 62Hz+ control
```

Order is strict: poor teacher → poor student.

### 6.4 Architecture Upgrade Path

| Component | Current (v2.0) | Target (v3.0) |
|-----------|----------------|----------------|
| Vision encoder | 4-layer CNN | Pretrained ViT-Small |
| Encoder training | End-to-end | Pre-trained + auxiliary decoder head |
| Diffusion loss | Standard MSE | D²PPO (MSE + Dispersive) |
| Inference | 10-step DDIM | 1-step OneDP |
| Simulator | Custom Gymnasium | Flightmare (long-term) |

**Immediate priority:** Complete Phase 1 (Run 4). Architecture upgrades don't matter if the expert fails to meet the quality gate.

### 6.5 Conference Strategy

| Venue | Acceptance | Primary Requirement | This Project's Focus |
|-------|-----------|--------------------|--------------------|
| CoRL 2025 | ~15% | Generative policy generalization, real robot verification | D²PPO theory, multimodal distribution necessity |
| ICRA 2026 | ~40% | System integration, hardware experiments, closed-loop robustness | Real flight data, Jetson latency, wind disturbance |
| RSS 2026 | ~12% | Algorithm depth, mathematical rigor | DPPO convergence proof, ablation, manifold analysis |

**Strategy:** If time is tight, complete the D²PPO ablation for theoretical contribution and submit to CoRL. Hardware deployment is the ICRA complete version.

### 6.6 What Has NOT Changed from v2.0

- Core environment and dynamics (6-DOF, NED, RK4 @ 200Hz)
- Observation space (15D), action space (4D)
- Gaussian reward structure
- PPO phase gate criteria (< 0.1m error prerequisite for Phase 2)
- HDF5 data format, T_obs=2, T_pred=8
- DDPM/DDIM diffusion process
- Phase 4 RHC loop structure

---

## 7. Appendix: Hyperparameter Reference

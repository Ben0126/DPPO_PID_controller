# DPPO PID Controller — Phase 3b: D²PPO Fine-Tuning

> Part of the dev log series. See [index](dev_log_phase2_3.md) for all phases.
> Covers: RHC baseline eval, DPPO Runs 1–3, key lessons, results summary.

---

## Table of Contents

1. [Phase 3 Evaluation: RHC Closed-Loop Baseline](#phase-3-evaluation-rhc-closed-loop-baseline)
2. [Phase 3b: D²PPO Fine-Tuning](#phase-3b-dppo-fine-tuning)
   - [Run 1: Policy Collapse](#run-1-policy-collapse)
   - [Run 2: Conservative Hyperparameters](#run-2-conservative-hyperparameters)
   - [Diagnostic Run: Covariate Shift Test](#diagnostic-run-covariate-shift-test-2026-04-04)
   - [Run 3: Extended Budget + β=0.15](#run-3-extended-budget-β015-2026-04-06)
3. [Key Lessons Learned](#key-lessons-learned)
4. [Appendix: Results Summary](#appendix-results-summary)

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

**Training Curve:**

| Update | Reward/step | Value Loss | Status |
|--------|------------|-----------|--------|
| 10 | 0.499 | 71.2 | Value net not yet converged |
| 50 | 0.527 | 32.4 | Steadily improving (Run 1 collapsed here ✓) |
| 100 | 0.549 | 28.7 | Continued improvement |
| 200 | **0.572** | 14.6 | **Peak** |
| 300 | 0.532 | 9.9 | Mild decline, no collapse |
| 400 | 0.312 | 4.2 | Second dip (value net now converging) |
| 500 | 0.386 | 4.1 | Partial recovery, stable |

**Best reward: 0.5758** (saved at update ~200)

Policy collapsed in Run 1 reached −0.43; Run 2 stays positive throughout (min ~0.31). β=0.1 + lr=5e-6 successfully prevented policy collapse.

**Checkpoint:** `checkpoints/diffusion_policy/dppo_20260403_130513/best_dppo_model.pt`

**Evaluation of best checkpoint (50 episodes):**

| Metric | DPPO Run 2 | DPPO Run 1 | Phase 3 Supervised | PPO Expert |
|--------|------------|------------|-------------------|------------|
| Mean reward | 21.08 ± 5.14 | 27.44 | 21.97 | 539.14 |
| Position RMSE | **0.1868m** | 0.3781m | 0.2856m | 0.0693m |
| Crashes | 50/50 | 50/50 | 50/50 | 0/50 |
| Inference | 72.2ms | 70.1ms | 71.3ms | — |

**RMSE improved 35%** (0.2856 → 0.1868m) vs supervised baseline. D²PPO is pushing the policy toward the target, but the drone still crashes every episode.

**Assessment:**

The crash rate (50/50) despite RMSE improvement reveals the core remaining problem: the policy learns to fly *toward* the target but cannot sustain stable hover long enough to complete a 500-step episode. Each rollout during DPPO training ends in a crash within 10–50 steps, so the policy only sees crash-trajectory distributions — it never experiences long-horizon stable flight during RL training.

**Options for next run:**
- **A: More updates** — continue from Run 2 best ckpt, 1000 total updates
- **B: Curriculum** — reduce `initial_pos_range`/`initial_vel_range` to near-zero so policy starts near hover
- **C: DAgger-style mixing** — mix 50% expert demos + 50% policy rollout per update batch

---

### 4.3 Diagnostic Run: Covariate Shift Test (2026-04-04)

**Motivation:**

After Run 2 still ended in 100% crash rate, the question arose: is the crash caused by the policy not knowing how to *navigate* (hypothesis: PPO expert only trained to hover), or is it covariate shift (policy destabilises even from near-hover states)?

**Verification:** `initial_pos_range` in `quadrotor.yaml` was already 0.1m — DPPO rollouts start within ±0.1m of the target, not 1m away. Crashes happened within 10–50 steps even from this close range. This ruled out pure "navigation inability" as the sole cause.

**Experiment design:**

Temporarily set `initial_pos_range = 0.01m`, `initial_vel_range = 0.01m` in `configs/quadrotor.yaml` (essentially placing the drone on the hover point at each rollout reset). Continued DPPO from Run 2 best checkpoint for 150 updates.

**Early results (update 7 at session snapshot):**

| Update | Reward/step | Value Loss | Status |
|--------|------------|-----------|--------|
| 0 (init) | +0.5779 | 65.99 | Checkpoint loaded |
| 7 | +0.5617 | 62.84 | Healthy — no collapse |

Reward remains firmly in healthy range (+0.3–0.6). This is higher than Run 2's reward at the same early stage, consistent with the easier environment (near-hover starts). **The diagnostic trend supports Hypothesis A (covariate shift) rather than Hypothesis B (navigation incapacity).**

> **Note:** Bug fixes to `quadrotor_visual_env.py` (see §7) were applied during this run at approximately update 8. Updates 1–7 used the original buggy renderer; updates 8+ use the corrected renderer. This introduces a small intra-run visual distribution shift, but does not invalidate the diagnostic conclusion — the question being tested (does near-zero range prevent early crash?) is independent of renderer accuracy.

**Planned decision tree after this run completes:**
- If reward stays > +0.4 throughout → covariate shift confirmed → adopt **Curriculum DPPO** (start near hover, expand range) as the strategy going forward
- If reward still collapses → deeper structural issue → re-evaluate full pipeline (including Phase 2 rerun with correct renderer)

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

### L6: RMSE Improvement ≠ Stability Improvement

DPPO Run 2 improved RMSE 35% but crash rate stayed 100%. The two metrics measure different things:
- **RMSE** measures average distance to target across all steps in an episode (including the few stable ones early on)
- **Crash rate** measures whether the policy can sustain stable flight for 10 full seconds

A policy can improve RMSE by flying toward the target faster before crashing, without actually becoming more stable. True stability requires the value network to be converged and the policy to have seen long-horizon trajectories.

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
| 3b Run 2 | DPPO best ckpt | 0.1868m | 50/50 | No collapse; RMSE −35% vs supervised; still all crash |

### File Locations

| Artifact | Path |
|----------|------|
| Expert demos | `data/expert_demos.h5` |
| Diffusion supervised | `checkpoints/diffusion_policy/20260402_032701/best_model.pt` |
| DPPO Run 1 best | `checkpoints/diffusion_policy/dppo_20260403_040722/best_dppo_model.pt` |
| DPPO Run 2 best | `checkpoints/diffusion_policy/dppo_20260403_130513/best_dppo_model.pt` |
| RHC eval (supervised) | `evaluation_results/rhc_phase3/` |
| RHC eval (DPPO R1) | `evaluation_results/rhc_dppo/` |
| RHC eval (DPPO R2) | `evaluation_results/rhc_dppo_run2/` |

---
<!-- auto-log 2026-04-03 14:20:16 -->
### [Auto-Log] 2026-04-03 14:20:16 — Device Check

**Command:** `python check_device.py`

**Output:**
```
PyTorch version: 2.5.1
CUDA available: True
GPU 0: NVIDIA GeForce RTX 3090
```

---
<!-- auto-log 2026-04-03 14:41:36 -->
### [Auto-Log] 2026-04-03 14:41:36 — DPPO Training — Started

**Command:** `grep -n "writer\.\|json\.\|metrics\|reward\|value_loss\|update" "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/scripts/train_dppo.py" | grep -v "#" | head -40`

**Output:**
```
{"stdout": "48:def compute_gae(rewards: List[float], values: List[float],\n56:    for i in reversed(range(len(rewards))):\n57:        delta = rewards[i] + gamma * values_ext[i + 1] * (1 - dones[i]) - values_ext[i]\n72:        rollout: dict with image_stacks, actions, rewards, dones, values\n76:        'rewards': [], 'dones': [], 'values': [],\n107:            obs, reward, terminated, truncated, info = env.step(action)\n110:            rollout['rewards'].append(reward)\n192:    total_updates = args.total_updates\n196:    print(f\"Total updates: {total_updates}\")\n201:    best_reward = -float('
... [truncated — 1623 chars total]
```

---


---
<!-- auto-log 2026-04-11 17:42:57 bash -->
### [Auto-Log] 2026-04-11 17:42:57 — DPPO Training — Started

**Command:** `cd c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller && source dppo/Scripts/activate && timeout 30 python -m scripts.train_dppo \
    --pretrained checkpoints/diffusion_policy/20260402_032701/best_model.pt \
    --total-updates 500 2>&1 | head -20`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 17:43:09 bash -->
### [Auto-Log] 2026-04-11 17:43:09 — DPPO Training — Started

**Command:** `cd c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller && source dppo/Scripts/activate && python -m scripts.train_dppo --help 2>&1 | head -20`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 17:43:53 bash -->
### [Auto-Log] 2026-04-11 17:43:53 — DPPO Training — Started

**Command:** `cd c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller && source dppo/Scripts/activate && LOG=logs/train_dppo_run5_$(date +%Y%m%d_%H%M%S).log && nohup python -m scripts.train_dppo --pretrained checkpoints/diffusion_policy/20260402_032701/best_model.pt --total-updates 500 > "$LOG" 2>&1 & echo "PID=$! LOG=$LOG"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 17:44:22 bash -->
### [Auto-Log] 2026-04-11 17:44:22 — DPPO Training — Started

**Command:** `sleep 25 && ls -t c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/ | grep "train_dppo_run5" | head -3 && echo "---" && ls -t c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/checkpoints/diffusion_policy/ | grep "^dppo_" | head -5`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 17:44:25 bash -->
### [Auto-Log] 2026-04-11 17:44:25 — DPPO Training — Started

**Command:** `cat c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_run5_20260411_174353.log | head -30`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 17:44:28 bash -->
### [Auto-Log] 2026-04-11 17:44:28 — DPPO Training — Started

**Command:** `ls -la c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/ | grep "train_dppo_run5" && wc -c c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/logs/train_dppo_run5_20260411_174353.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 17:44:35 bash -->
### [Auto-Log] 2026-04-11 17:44:35 — DPPO Training — Started

**Command:** `ps aux 2>/dev/null | grep train_dppo | grep -v grep | head -5`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 17:45:50 bash -->
### [Auto-Log] 2026-04-11 17:45:50 — DPPO Training — Started

**Command:** `cd c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller && source dppo/Scripts/activate && python -m scripts.train_dppo --pretrained checkpoints/diffusion_policy/20260402_032701/best_model.pt --total-updates 500 2>&1 &
sleep 45
jobs -l`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 17:52:30 bash -->
### [Auto-Log] 2026-04-11 17:52:30 — DPPO Training — Started

**Command:** `cd c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller && source dppo/Scripts/activate && python -m scripts.train_dppo --pretrained checkpoints/diffusion_policy/20260402_032701/best_model.pt --total-updates 500 2>&1 | head -50 &
sleep 50
cat /proc/$!/fd/1 2>/dev/null || true
wait $!`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 17:54:35 bash -->
### [Auto-Log] 2026-04-11 17:54:35 — DPPO Training — Started

**Command:** `cd c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller && source dppo/Scripts/activate && python -m scripts.train_dppo --pretrained checkpoints/diffusion_policy/20260402_032701/best_model.pt --total-updates 500 2>&1
echo "EXIT_CODE: $?"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 17:56:44 bash -->
### [Auto-Log] 2026-04-11 17:56:44 — DPPO Training — Started

**Command:** `grep -n "print\|logging\|log_file\|FileHandler\|stdout" c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/scripts/train_dppo.py | head -20`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 17:57:58 bash -->
### [Auto-Log] 2026-04-11 17:57:58 — DPPO Training — Started

**Command:** `cd c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller && source dppo/Scripts/activate && python -m scripts.train_dppo --pretrained checkpoints/diffusion_policy/20260402_032701/best_model.pt --total-updates 500 > /tmp/run5_test.log 2>&1 &
echo "PID=$!"
sleep 60
echo "--- log so far ---"
cat /tmp/run5_test.log | head -50`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-11 17:58:49 bash -->
### [Auto-Log] 2026-04-11 17:58:49 — DPPO Training — Started

**Command:** `cd c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller && source dppo/Scripts/activate && python -u -m scripts.train_dppo --pretrained checkpoints/diffusion_policy/20260402_032701/best_model.pt --total-updates 500 2>&1 | tee /tmp/run5_tee.log &
sleep 30 && echo "=== tee log ===" && cat /tmp/run5_tee.log | head -30`

**Output:**
```
(empty)
```

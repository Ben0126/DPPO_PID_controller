# DPPO PID Controller — Project Context

## What This Is

Vision-based quadrotor control research. A Diffusion Policy (Phase 3a supervised pre-training)
is fine-tuned with D²PPO (Dispersive PPO) advantage-weighted RL to overcome covariate shift.

**Core contribution:** Dispersive Loss prevents feature collapse in high-speed visual drone control.
**Target venues:** CoRL 2025 / ICRA 2026 / RSS 2026

---

## Research Pipeline Status

| Phase | Description | Status |
|-------|-------------|--------|
| v4.0 Ph.0 | INDI Hover Gate | Done — tilt 0.00°, omega 0.000 rad/s |
| v4.0 Ph.1 | CTBR PPO Expert | Done — RMSE 0.0649m, 0/50 crashes (`20260419_142245`) |
| v4.0 Ph.2 | FPV Data Collection v4.0 | Done — `data/expert_demos_v4.h5` (1000 ep, 3.9GB, 0 crashes) |
| v4.0 Ph.3a | Flow Matching supervised pre-training | Done — best val=0.0630 (`flow_policy_v4/20260420_034314`) |
| **v4.0 Ph.3b** | **ReinFlow RL Fine-tuning** | **27 runs through 2026-05-17. H4 BC (IMU-dominant fusion, score 0.165) is v4.0 SOTA. All RL runs systematically destroy BC. Run 28 (positive-advantage mask) in progress.** |
| ~~v4.0 Ph.3c~~ | ~~DAgger Recovery~~ | DENIED (2026-05-12): recovery data poisons hover BC (49/50 crash, RMSE 2.44m). |
| **v4.0 Ph.3d** | **H4 Architecture + Hierarchical Metric** | Done — H4 IMU encoder (feature_dim 512, V/I grad ratio 3.22×); new 飛→穩→準 eval metric; RMSE bias and AWR mode-collapse identified |
| v4.0 Ph.4 | Hardware deployment (Jetson Orin Nano) | Future |
| v3.3 ref | DPPO v3.3 best result | Done — Run 1: RMSE 0.1039m, 50/50 crashes |

---

## Directory Structure

All source code lives in the inner git repo: `DPPO_PID_controller/`

```
DPPO_PID_controller/          ← git repo root (cd here before running any script)
  configs/
    quadrotor.yaml            ← physics params, reward weights, env config
    diffusion_policy.yaml     ← model arch + training + dppo hyperparams (edit this for new runs)
    ppo_expert.yaml           ← PPO expert training config
  envs/
    quadrotor_dynamics.py     ← 6-DOF physics (200Hz inner loop, RK4)
    quadrotor_env.py          ← state-based RL env (50Hz outer loop, 15D obs)
    quadrotor_visual_env.py   ← FPV wrapper (64×64 RGB synthetic renderer)
  models/
    diffusion_policy.py       ← VisionDiffusionPolicy (10,929,256 params)
    conditional_unet1d.py     ← ConditionalUnet1D backbone
    diffusion_process.py      ← DDIM noise schedule (cosine, 100 train / 10 infer steps)
    vision_encoder.py         ← CNN encoder (6ch stacked frames → 256D feature)
    ppo_expert.py             ← PPO expert + RunningMeanStd normalization
  scripts/
    train_ppo_expert.py       ← Phase 1 training
    collect_data.py           ← Phase 2 expert data collection
    train_diffusion.py        ← Phase 3a supervised training (~14h on RTX 3090)
    train_dppo.py             ← Phase 3b D²PPO fine-tuning (~10-11h per run)
    evaluate_ppo_expert.py    ← PPO expert evaluation (50 episodes)
    evaluate_rhc.py           ← RHC closed-loop eval (diffusion vs PPO expert)
  utils/
    training_metrics.py       ← JSON metric logging (training_metrics/)
    visualization.py          ← Trajectory/reward plots
  checkpoints/
    ppo_expert/               ← PPO expert checkpoints
    diffusion_policy/         ← Supervised + DPPO checkpoints
  data/
    expert_demos.h5           ← 1000 episodes, 500k steps; DO NOT DELETE
  docs/
    dev_log.md                ← Phase 1 detailed history
    dev_log_phase2_3.md       ← Phase 2–3 history (read this for full context)
```

---

## Environment Setup

```bash
# Always work from the inner git repo
cd DPPO_PID_controller

# Activate venv (Windows bash)
source dppo/Scripts/activate

# Verify GPU (must show CUDA available)
python check_device.py
```

---

## Key Commands

```bash
# --- Phase 3b: D²PPO fine-tuning ---
python -m scripts.train_dppo \
    --pretrained checkpoints/diffusion_policy/20260402_032701/best_model.pt

# --- Phase 3a: Supervised training (only if re-running from scratch) ---
python -m scripts.train_diffusion --config configs/diffusion_policy.yaml

# --- Hierarchical evaluation (新 SOTA metric: 飛→穩→準) ---
# Auto-detects H3a / H4 architecture from checkpoint state_dict.
dppo/Scripts/python.exe -m scripts.evaluate_hierarchical \
    --n-episodes 20 \
    --ckpts \
        "H4_BC:checkpoints/flow_policy_v4/20260514_175219/best_model.pt" \
        "Run25_best:checkpoints/reinflow_v4/reinflow_v4_20260515_023519/best_reinflow_model.pt"

# --- RHC closed-loop evaluation (legacy RMSE; WARNING: RMSE is biased toward short-lived policies) ---
python -m scripts.evaluate_rhc \
    --diffusion-model checkpoints/diffusion_policy/<timestamp>/best_model.pt \
    --ppo-model checkpoints/ppo_expert/20260401_103107/best_model.pt \
    --ppo-norm checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz

# --- PPO expert evaluation ---
python -m scripts.evaluate_ppo_expert

# --- Expert data collection (Phase 2 is done; only re-run if data is lost) ---
python -m scripts.collect_data

# --- TensorBoard ---
tensorboard --logdir logs/diffusion_policy/
```

---

## Key Checkpoints

| Artifact | Path |
|----------|------|
| v4.0 CTBR PPO Expert | `checkpoints/ppo_expert_v4/20260419_142245/best_model.pt` |
| v4.0 Flow Matching BC (original arch, historical) | `checkpoints/flow_policy_v4/20260420_034314/best_model.pt` |
| v4.0 H3a hover-only BC (IMU 128D) | `checkpoints/flow_policy_v4/20260512_170638/best_model.pt` |
| **v4.0 H4 BC (current SOTA, IMU 512D)** | **`checkpoints/flow_policy_v4/20260514_175219/best_model.pt`** |
| v4.0 Run 23 (H3a + RL hover-only) | `checkpoints/reinflow_v4/reinflow_v4_20260514_055001/best_reinflow_model.pt` |
| v4.0 Run 25 (H4 + RL, AWR mode-collapse) | `checkpoints/reinflow_v4/reinflow_v4_20260515_023519/best_reinflow_model.pt` |
| v4.0 Run 26 (H4 + Linear IAE reward) | `checkpoints/reinflow_v4/reinflow_v4_20260516_052606/best_reinflow_model.pt` |
| v3.3 DPPO Run 1 (best v3.x result, 0.1039m RMSE — known biased) | `checkpoints/diffusion_policy/dppo_v33_20260413_033647/best_dppo_v33_model.pt` |

---

## Architecture

**Current (Phase 3a / 3b baseline):**
```
FPV image stack (T_obs=2 frames, 6×64×64 uint8)
  → VisionEncoder CNN → 256D feature vector
  → ConditionalUnet1D (down_dims=[256,512]) + timestep embed (128D) → cond_dim=384
  → DDIM 10 steps → action sequence (T_pred=8 × 4 motor thrusts ∈ [-1,1])
  → Execute first T_action=4 steps → re-observe (RHC loop)

D²PPO loss: L = E[ exp(β × A_norm) × ||ε_θ(a,τ,s) − ε||² ]
Value net:  ValueNetwork(feature_dim=256, hidden_dim=256) → scalar V(s)
```

**Architecture H4 (Current SOTA, 2026-05-15+):**
```
FPV image stack (6×64×64) → VisionEncoder → 256D vision_feat (456k params)
6D IMU [ω,a]              → IMUEncoder MLP(6→1024→512) → 512D imu_feat (532k params, DOMINANT)
cat([256D, 512D])          → 768D global_cond  (IMU 67%)
768D + timestep(128D)      → 896D cond → ConditionalUnet1D → velocity field v_θ

[Training only] 512D imu_feat → tilt_head Linear(512,1) → predicted tilt rad

V/I gradient ratio: 46.8× (Original) → 9.9× (H3a) → 3.22× (H4)
Inference: 2-step Euler (14ms, recommended; 1-step is suboptimal post-RL)
```

**Architecture v3.1 (Phase 3c, historical):**
```
FPV image stack (6×64×64) → VisionEncoder → 256D vision_feat
6D IMU [ω,a]              → IMUEncoder MLP(6→64→32) → 32D imu_feat
cat([256D, 32D])           → 288D global_cond
```

---

## Critical Hyperparameters — Phase 3b Run 2 (Ended)

| Param | Value | Why |
|-------|-------|-----|
| `advantage_beta` | **0.1** | Run 1 used 1.0 → max weight 20×, caused collapse; 0.1 → max 1.35× |
| `learning_rate` | **5e-6** | Run 1 used 3e-5 → overwrote pretrained weights in ~100 updates |
| `n_rollout_steps` | **4096** | Doubled from 2048 → lower-variance GAE advantage estimates |
| `n_epochs` | **3** | Reduced from 5 → less per-update gradient drift |
| `value_lr` | **3e-4** | Reduced from 1e-3 → value net converges alongside policy |

Config file: `configs/diffusion_policy.yaml` — section `dppo:`

## v3.1 New Commands

```bash
# Phase 2 v3.1: re-collect data with IMU + depth
python -m scripts.collect_data \
    --model checkpoints/ppo_expert/20260401_103107/best_model.pt \
    --norm  checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz \
    --output data/expert_demos_v31.h5 --v31

# Phase 3a v3.1: supervised pre-training
python -m scripts.train_diffusion_v31 --config configs/diffusion_policy.yaml

# Phase 3c v3.1: DPPO fine-tuning
python -m scripts.train_dppo_v31 \
    --pretrained checkpoints/diffusion_policy/v31_<timestamp>/best_model.pt
```

## v3.3 New Commands（物理 IMU + 歸一化，P6 修復後）

```bash
# Phase 2 v3.3: re-collect data with normalized physics-based IMU + depth
python -m scripts.collect_data \
    --model checkpoints/ppo_expert/20260401_103107/best_model.pt \
    --norm  checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz \
    --output data/expert_demos_v33.h5 --v33

# Phase 3a v3.3: supervised pre-training
python -m scripts.train_diffusion_v33 --config configs/diffusion_policy.yaml

# Phase 3c v3.3: DPPO fine-tuning
python -m scripts.train_dppo_v33 \
    --pretrained checkpoints/diffusion_policy/v33_<timestamp>/best_model.pt

# Phase 4 v3.3: RHC evaluation
python -m scripts.evaluate_rhc_v33 \
    --diffusion-model checkpoints/diffusion_policy/dppo_v33_<timestamp>/best_dppo_v33_model.pt \
    --ppo-model checkpoints/ppo_expert/20260401_103107/best_model.pt \
    --ppo-norm  checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz
```

---

## v4.0 DAgger Recovery Training (Hypothesis 2)

### Hardware Constraints (this machine)

| Resource | Total | Free | Implication |
|----------|-------|------|-------------|
| RAM | 33 GB | ~12 GB | Max hover episodes = 500 (500 ep ≈ 6 GB images). 1000 hover + 500 recovery = 17.6 GB → heavy swap, very slow. |
| GPU VRAM | 24 GB | ~22 GB | batch_size=512 OOMs during backward with recovery mix. Use batch_size=256. |

### Correct Command Protocol

**CRITICAL — always use the direct Python path + Bash tool `run_in_background=true`. Do NOT use `nohup ... &` or `| head -N`.**

```bash
# Step 1: Collect recovery demos (DONE — data/expert_demos_v4_recovery.h5)
dppo/Scripts/python.exe -m scripts.collect_data_v4_recovery \
    --model  checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
    --norm   checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz \
    --output data/expert_demos_v4_recovery.h5 \
    --n-episodes 500 --tilt-max 30.0 --perturb-vel 2.0

# Step 2: BC mixed training — Bash tool MUST use run_in_background=true
dppo/Scripts/python.exe -m scripts.train_flow_v4 \
    --recovery-h5 data/expert_demos_v4_recovery.h5 \
    --recovery-episodes 500 \
    --hover-episodes 500

# Step 2 smoke test (5 epochs only)
dppo/Scripts/python.exe -m scripts.train_flow_v4 \
    --recovery-h5 data/expert_demos_v4_recovery.h5 \
    --recovery-episodes 500 \
    --hover-episodes 500 --quick

# Step 3: BC gate eval
dppo/Scripts/python.exe -m scripts.evaluate_rhc_v4 \
    --flow-model checkpoints/flow_policy_v4/<timestamp>/best_model.pt \
    --ppo-model  checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
    --ppo-norm   checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz
```

### Monitor Training Progress

**Do NOT check task output file — Python buffers stdout and the file stays empty until completion. Use TensorBoard event API instead.**

```bash
# Check TensorBoard scalars (works even while training)
source dppo/Scripts/activate && python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('logs/flow_policy_v4/<timestamp>'); ea.Reload()
for t in ea.Tags().get('scalars', []):
    ev = ea.Scalars(t); print(f'{t}: n={len(ev)}, last={ev[-1].value:.4f}')
"

# GPU utilization
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# Check running process
ps aux | grep python | grep -v grep
cat /proc/<PID>/cmdline | tr '\0' ' '
```

### BC Gate Decision Matrix

| Result | Next Step |
|--------|-----------|
| BC crash < 50/50 (any improvement) | Step 3: restart RL with `curriculum.enabled: true, swift_perturbation_prob: 0.2` |
| BC crash = 50/50 unchanged | Hypothesis 3: fix IMU encoder fusion (cross-attention or larger image) |

### Step 3 RL Config (if BC gate passes)

```yaml
# reinflow_v4.yaml additions for Step 3
rl:
  loss_type: clipped
  sde_noise_std: 0.1
  clip_epsilon: 0.2
  learning_rate: 1.0e-5
  n_epochs: 4
curriculum:
  enabled: true
  swift_perturbation_prob: 0.2
```

---

## Known Failure Modes

1. **Covariate shift** — supervised-only diffusion always crashes (100% crash rate, RMSE 0.286m).
   D²PPO closed-loop training is mandatory. Do NOT evaluate supervised model without RHC context.

2. **Policy collapse** — per-step reward goes: positive peak → near zero → negative.
   Caused by β too large OR LR too high. Fix: reduce both. Watch value loss — must drop below 1.0
   within first ~50 updates before policy updates are meaningful.

3. **Value network lag** — value loss > 10 at update 10 means random V(s) estimates.
   Never draw conclusions from policy behavior in the first 20 updates of a new run.

4. **Per-step vs per-episode reward** — training logs show *per-step* reward (not per-episode).
   Healthy hover = **+0.3–0.6/step**. Collapse = **< 0/step** (crash_penalty −10 dominates).

5. **BC mixed training CUDA OOM** — batch_size=512 causes CUDA OOM during backward when training on 500 hover + 500 recovery episodes. Fix: `batch_size: 256` in `configs/flow_policy_v4.yaml`. The 512→256 change is committed; do not revert.

6. **Cygwin background process anti-patterns** — these silently fail in the Bash tool:
   - `nohup python ... > log.txt 2>&1 &` → log stays empty; process may die
   - `python ... | head -N` → SIGPIPE kills Python after N lines
   - `source dppo/Scripts/activate && python ...` → activation may not propagate
   **Always use** `dppo/Scripts/python.exe -m ...` with Bash tool `run_in_background=true`.

7. **Competing training processes** — multiple simultaneous training runs share GPU VRAM → CUDA OOM or degraded performance. Always kill existing training processes before launching new ones: `ps aux | grep python | grep -v grep`, then `kill <PID>`.

8. **PPO Clipped Surrogate (Hypothesis 1 denial)** — SDE noise σ amplifies policy gradients when crash rate is high. σ=0.1 → sensitivity 50×; σ=0.3 → still clip_fraction 0.70+. PPO peak reward 0.5884 < weighted MSE 0.6948. Root cause: noisy rollout from 50/50 crashes poisons advantages — no optimizer can fix a broken training distribution.

---

## Results Summary (Hierarchical Metric — 飛→穩→準, 2026-05-15+)

| Rank | Checkpoint | Score | Survive | IAE_steady | Term err | Steps avg |
|------|-----------|-------|---------|------------|----------|-----------|
| 1 | **H4 BC** (current SOTA) | **0.165** | 44.0% | 1.473m | 2.510m | **202** |
| 2 | Run25_u50 (BC + warmup) | 0.158 | 43.2% | 1.650m | 3.022m | 197 |
| 3 | H3a BC | 0.151 | 43.6% | 1.656m | 2.839m | 197 |
| 4 | Run25_u200 | 0.128 | 26.6% | 0.976m | 1.694m | 121 |
| 5 | Run23_RL | 0.093 | 18.6% | 0.770m | 1.272m | 86 |
| 6 | Run25_best | 0.086 | 17.1% | 0.713m | 1.119m | 79 |
| 7 | Run19_RL (old "best") | 0.078 | 15.6% | 0.710m | 1.113m | 71 |
| ref | PPO Expert (state-based) | ~0.85 | 100% | 0.065m | 0.065m | 500 |

**v4.0 current SOTA:** **H4 BC** (score 0.165, 202 steps avg, no RL fine-tuning).
**Inference:** 14ms with n_steps=2 (recommended; 1-step is suboptimal post-RL).
**Latency target:** ✓ met (still under 20ms @ 50Hz control period).

### Five Major Findings (2026-05-13~17)

1. **RMSE bias confirmed.** `evaluate_rhc_v4.py:92` divides by `ep_length` not `max_episode_steps` → short-lived policies get artificially low RMSE. Past 24 runs misranked.
2. **Disturbance not crash cause.** Eval with disturbance OFF: same ~73 step crash. Run 24 hypothesis denied.
3. **Phase lag secondary.** T_action 4→1: +13%. n_inference_steps 1→3: +35%. Total +53% steps, still 50/50.
4. **H4 IMU dominance = real v4.0 SOTA.** feature_dim 128→512, grad ratio 9.9→3.22, BC steps 130→202 (+55%).
5. **AWR mode-collapse.** PPO advantage normalization absorbs sparse crash_penalty (constant offset). Weighted MSE forces policy to imitate own crash trajectories. All RL runs degrade systematically. Fix in test: positive-advantage mask (Run 28).

See `docs/dev_log_v4_h4_hierarchical.md` for full diagnostic chain.

---

## Compute Efficiency Rules

**Always maximise hardware utilisation. Never leave GPU or CPU cores idle when there is work to do.**

| Concern | Rule |
|---------|------|
| GPU first | Any tensor operation that can run on CUDA must run on CUDA. Never do per-sample augmentation on CPU (PIL/numpy) when it can be done as a batched GPU tensor op after `.to(device)`. |
| DataLoader workers | Always set `num_workers ≥ 4` (+ `persistent_workers=True`, `pin_memory=True`) so CPU data loading overlaps with GPU forward/backward. `num_workers=0` is only acceptable for debugging. |
| Non-blocking transfers | Use `.to(device, non_blocking=True)` for all tensor transfers to overlap PCIe transfer with GPU compute. |
| Augmentation placement | GPU tensor augmentation (brightness/contrast/noise) belongs in the **training loop after `.to(device)`**, not in `Dataset.__getitem__`. PIL-based CPU augmentation is ~9× slower and must not be used. |
| Batch size | RTX 3090 has 24 GB VRAM. Default batch_size=256 uses only ~2.7 GB. Prefer 512+ unless memory errors occur. |

**Lesson learned (2026-04-05):** PIL-based ColorJitter in `Dataset.__getitem__` inflated epoch time from ~100 s to ~900 s (9× slowdown). Replaced with on-GPU tensor ops — zero PIL, zero overhead.

---

## Python & CUDA Info

- Python 3.9 (venv at `DPPO_PID_controller/dppo/`)
- PyTorch with CUDA 12.8 (RTX 3090)
- stable-baselines3 >= 2.0.0, gymnasium >= 0.28.0
- See `requirements.txt` for full list

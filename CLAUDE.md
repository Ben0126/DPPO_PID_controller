# DPPO PID Controller — Project Context

## What This Is

Vision-based quadrotor control research. A Diffusion Policy (Phase 3a supervised pre-training)
is fine-tuned with D²PPO (Dispersive PPO) advantage-weighted RL to overcome covariate shift.

**Core contribution (original thesis):** Dispersive Loss prevents feature collapse in high-speed visual drone control.
**⚠️ Status (2026-06-18): this thesis FAILED its pre-registered test.** The P2 2×2 ablation (Dispersive×E2E, 3 seeds, frozen protocol) found Dispersive gives **no** survival/Tier1 gain above seed noise (D1E1 vs D0E1 = +1.1pp, pooled std 4.2pp); with a frozen encoder it is byte-identically a no-op (D1E0 ≡ D0E0). Project pivots to a **negative result + diagnosis** — see Major Findings #10 and `RESEARCH_PLAN_v6.md`.
**✅ Faithful re-run (2026-06-23) CONFIRMS the negative result and is now rebuttal-proof.** Re-running the full 2×2 with the *official-code-faithful* Dispersive (InfoNCE-L2 on the **`flow_net` mid-block**, λ=**0.5**, τ=0.5, with the `/d` per-dim normalisation the paper's Algorithm 1 omits — NOT the legacy off-path `vis_pooled`/λ=0.05/log-distance) gives D1E1 vs D0E1 = **−2.2pp Tier1 (pooled std 6.3pp), −2.1pp survival** → still no gain. The legacy "byte-identical no-op" (C2) is **OVERTURNED**: under the faithful flow_mid placement `flow_net` trains even with a frozen encoder, so D1E0 ≠ D0E0 (MD5 differs ×3 seeds) and Dispersive-on-frozen is mildly *harmful* (Tier1 87.8→74.4). Artifacts: `evaluation_results/p2f_ablation_leaderboard.json`, `docs/experiment_report_faithful_dispersive.md`. **§5/§9 of the draft rewritten — DONE (2026-06-23, paper v0.4 `docs/paper_negative_result_draft.md`):** Abstract/§1/§3/§5/§6.1/§6.2/§7/§8/§9 re-synced to faithful P2f; §2 Related Work replaced with the transfer-boundary version (`research_output/related_work_draft.md`, refs renumbered, new [22]–[30]); legacy `vis_pooled`/λ=0.05 retained only as the §6.1 off-path probe; exported clean HTML+PDF via `scripts/export_paper.py`. Paper can now go to deep-science-writer for prose polish.
**✅ Phase 7 write-up DONE (2026-06-27, paper v0.5 `docs/paper_negative_result_draft.md`).** Both 2×2s now folded into the paper: §6.3 reframed (no longer ends on the "needs a competent teacher" speculation) + **new §6.4 "The decisive test: Teacher × Observation 2×2"** (Table 6, Figure 6 `docs/figures/teacher_obs_2x2.png`) reporting the v7 decisive result — **FLOOR NOT BROKEN** (T1O1 cond-IAE 2.93 vs T0O0 2.69, Δ+0.24m ∈ pooled std 0.23m; coverage T1 buys survival +8/+14pp but precision 0; coverage×sensing **interaction negative**). Diagnosis upgraded "coverage/teacher-limited" → **triple exclusion (representation/coverage/sensing) → robustness–precision *capacity* conflict** (claim-strength aligned per Remi: conflict observed, capacity = leading-but-untested explanation). **Title upgraded** to "Representation Collapse Is Not the Bottleneck — and Neither Is Coverage or Sensing: A Negative Result and Capacity Diagnosis…". Abstract/§1/§7/§8/§9 re-synced; §5↔§6.4 protocol-alignment note added; **Phase 6 scale-invariant ablation folded into §6.1 (Table 5)+§7+§8** (closes the last §6.1 "scale-sensitive criterion" rebuttal). Reviewed (remi) + prose-polished (deep-science-writer Phase-4; numbers/conclusions unchanged); re-exported HTML+PDF (6 figs base64, Unicode OK). See Major Findings #11–#12 and `docs/experiment_report_p2to_decisive.md`. **Figures renumbered to appearance order (2026-06-27)** — Fig 1 single_seed (§4), 2 ablation_forest (§5), 3 rank_survival (§6.1), 4 crosshair (§6.3), 5 sensing (§6.3), 6 teacher_obs (§6.4); `make_paper_figures.py` functions + cross-refs updated, re-exported. **Capacity-conflict literature anchors added (2026-06-27):** refs [31] Sener & Koltun (multi-task as multi-objective, NeurIPS 2018) + [32] Yu et al. PCGrad (conflicting gradients, NeurIPS 2020), cited in §6.4/§7 (web-verified arXiv 1810.04650 / 2001.06782). **→ paper v0.5 fully integrated; no deferred write-up items remain.**
**Target venues:** CoRL 2025 / ICRA 2026 / RSS 2026

---

## Research Pipeline Status

| Phase | Description | Status |
|-------|-------------|--------|
| v4.0 Ph.0 | INDI Hover Gate | Done — tilt 0.00°, omega 0.000 rad/s |
| v4.0 Ph.1 | CTBR PPO Expert | Done — RMSE 0.0649m, 0/50 crashes (`20260419_142245`) |
| v4.0 Ph.2 | FPV Data Collection v4.0 | Done — `data/expert_demos_v4.h5` (1000 ep, 3.9GB, 0 crashes) |
| v4.0 Ph.3a | Flow Matching supervised pre-training | Done — best val=0.0630 (`flow_policy_v4/20260420_034314`) |
| ~~v4.0 Ph.3b~~ | ~~ReinFlow RL Fine-tuning~~ | Done (concluded) — 27 runs all fail (AWR mode-collapse). H4 BC tops composite score but P0 shows it's artifact-prone (Tier1 13%, 17.4% oracle); see Results Summary. |
| ~~v4.0 Ph.3c~~ | ~~DAgger Recovery~~ | DENIED (2026-05-12): recovery data poisons hover BC. |
| **v4.0 Ph.3d** | **H4 Architecture + Hierarchical Metric** | Done — H4 IMU 512D, V/I ratio 3.22×, new 飛→穩→準 metric. |
| **v5.0 BC** | **Dynamic Task Tag + Dispersive Loss Pre-training** | Done — best val_flow **0.0663** (`checkpoints/flow_policy_v5/20260604_141454/best_model.pt`) |
| **v5.0 RL** | **ReinFlow RL Fine-tuning + Advantage Masking** | Done — best reward **-0.3417**, eval score **0.130** (`checkpoints/reinflow_v5/20260604_193923/best_reinflow_model.pt`) |
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
# --- Phase 3e: v5.0 Joint E2E training (Visual encoder unfrozen, mixed hover + recovery demos) ---
dppo/Scripts/python.exe -m scripts.train_flow_v5 \
    --config configs/flow_policy_v5.yaml \
    --recovery-h5 data/expert_demos_v4_recovery.h5 \
    --recovery-episodes 500 \
    --hover-episodes 500

# --- Hierarchical evaluation (新 SOTA metric: 飛→穩→準) ---
# Auto-detects H3a / H4 / v5 architecture from checkpoint state_dict.
dppo/Scripts/python.exe -m scripts.evaluate_hierarchical \
    --n-episodes 30 \
    --ckpts \
        "H4_BC:checkpoints/flow_policy_v4/20260514_175219/best_model.pt" \
        "Joint_E2E_v5:checkpoints/flow_policy_v5/20260603_171316/best_model.pt"

# --- Phase 3b: D²PPO fine-tuning ---
python -m scripts.train_dppo \
    --pretrained checkpoints/diffusion_policy/20260402_032701/best_model.pt

# --- Phase 3a: Supervised training (only if re-running from scratch) ---
python -m scripts.train_diffusion --config configs/diffusion_policy.yaml

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
| **v4.0 H4 BC (current BC Score SOTA, IMU 512D)** | **`checkpoints/flow_policy_v4/20260514_175219/best_model.pt`** |
| **v5.0 BC Training Checkpoint (task cond + dispersive)** | **`checkpoints/flow_policy_v5/20260604_141454/best_model.pt`** |
| **v5.0 ReinFlow RL Best Checkpoint (Advantage Masking)** | **`checkpoints/reinflow_v5/20260604_193923/best_reinflow_model.pt`** |
| v5.0 Joint E2E Training Baseline (unfrozen encoder) | `checkpoints/flow_policy_v5/20260603_171316/best_model.pt` |
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

**Architecture H4 (Current BC Score SOTA, 2026-05-15+):**
```
FPV image stack (6×64×64) → VisionEncoder → 256D vision_feat (456k params)
6D IMU [ω,a]              → IMUEncoder MLP(6→1024→512) → 512D imu_feat (532k params, DOMINANT)
cat([256D, 512D])          → 768D global_cond  (IMU 67%)
768D + timestep(128D)      → 896D cond → ConditionalUnet1D → velocity field v_θ

[Training only] 512D imu_feat → tilt_head Linear(512,1) → predicted tilt rad

V/I gradient ratio: 46.8× (Original) → 9.9× (H3a) → 3.22× (H4)
Inference: 2-step Euler (14ms, recommended; 1-step is suboptimal post-RL)
```

**Architecture v5.0 (Current Survival SOTA, 2026-06-03+):**
```
FPV image stack (6×64×64) → VisionEncoderV5 (CNN) → spatial_map (256, 4, 4) & pooled (256D)
6D IMU [ω,a]              → IMUEncoder MLP(6→1024→512) → 512D imu_feat
pooled & spatial_map      → CrossAttentionIMU2Vision (Q=Linear(imu_feat), K=V=spatial_map) → attended (256D)
cat([attended, imu_feat])  → 768D global_cond
768D + timestep(128D)      → 896D cond → ConditionalUnet1D → velocity field v_θ (Flow Matching)

[Training only - State Aux Loss] pooled (256D) → StatePredictor (256→256→15) → state_pred (15D)
L = L_flow + λ_state × MSE(state_pred, state_gt) + λ_tilt × MSE(tilt_pred, tilt_gt)
Inference: Visual encoder is UNROZEN and trained端到端 (End-to-End) alongside flow_net on mixed data.
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

## Results Summary — P0 Frozen Protocol (2026-06-17)

> ⚠️ **METRIC RESET — old composite scores are NOT comparable.** The eval metric changed 3× (RMSE → linear-clip hierarchical → exp-decay hierarchical). Any "Score" from a different protocol cannot be ranked against another. **P0 freezes one protocol** — `scripts/evaluate_frozen_p0.py` (30 ep, base_seed 12345, σ=2.0 exp-decay, paired identical init across models, bootstrap 95% CI) — and adds artifact-robust columns. Canonical artifact: `evaluation_results/frozen_p0_leaderboard.json`.

**Frozen leaderboard** (sorted by composite score; `cond-*` = precision over episodes surviving ≥250 steps ONLY; `n_cond` = how many of 30 episodes that is):

| Model | Score (95% CI) | Survive | Tier1% | cond-IAE (n) | all-IAE | %Oracle |
|-------|----------------|---------|--------|--------------|---------|---------|
| **PPO Oracle** (state-based) | 0.967 [.966,.967] | 100% | 100% | 0.068m (30) | 0.068m | **100%** |
| H4_BC | 0.168 [.154,.182] | 40.8% | 13.3% | 2.520m (4) ⚠️ | 1.428m | 17.4% |
| v5_RL_best | 0.129 [.117,.139] | 28.8% | 13.3% | 3.340m (4) ⚠️ | 1.289m | 13.3% |
| v5_BC | 0.124 [.107,.143] | 55.9% | 70.0% | 2.724m (21) | 2.553m | 12.8% |
| Joint_E2E_v5 | 0.106 [.095,.118] | 62.2% | 80.0% | 3.077m (24) | 2.994m | 11.0% |

⚠️ `n_cond=4` (H4_BC, v5_RL_best): only 4/30 episodes flew past the 250-step threshold, so their cond-IAE is unreliable and their **low all-IAE is a short-survival artifact** (crashed too early to drift), NOT precision.

**SOTA STATUS (P0-corrected):**
- **Oracle is now measured, not assumed.** State-based PPO scores **0.9668** under the frozen protocol (100% survive, IAE 0.068m) — this replaces the legacy hard-coded **0.85**. Best vision policy reaches only **17.4% of oracle** → nothing is deployable (~5× the oracle's 0.068m hover error; no sub-meter closed-loop vision hover exists — reliable cond-IAE for n≥21 is 2.7–3.1m for all).
- **"H4 BC = SOTA" — OVERTURNED.** H4 tops the composite score but survives only 40.8% with Tier1 13.3%. The multiplicative `SR × precision` composite still rewards "die early but precise" over "fly long"; it cannot separate the two.
- **"v5_RL_best = 51% precision gain" — OVERTURNED (artifact).** Its low all-IAE (1.289m) is from crashing early (n_cond 4/30); its *conditional* IAE is 3.340m — the WORST of the group. No precision breakthrough.
- **Honest single number = Tier1 pass%** (fraction of episodes flying ≥ half the 500-step horizon): **Joint_E2E_v5 80% > v5_BC 70% ≫ H4_BC ≈ v5_RL_best 13%.** **Survival frontier = Joint_E2E_v5.**
- **Inference:** 14ms with n_steps=2 (✓ under 20ms @ 50Hz). Latency was never the bottleneck.

See `memory/project_p0_frozen_eval.md` for the corrected leaderboard rationale and the proof of the short-survival artifact.

### Major Findings (2026-05-13 ~ 2026-06-17)

1. **RMSE bias confirmed.** `evaluate_rhc_v4.py:92` divides by `ep_length` not `max_episode_steps` → short-lived policies get artificially low RMSE. Past 24 runs misranked.
2. **Disturbance not crash cause.** Eval with disturbance OFF: same ~73 step crash. Run 24 hypothesis denied.
3. **Phase lag secondary.** T_action 4→1: +13%. n_inference_steps 1→3: +35%. Total +53% steps, still 50/50.
4. **H4 IMU dominance = real v4.0 SOTA.** feature_dim 128→512, grad ratio 9.9→3.22, BC steps 130→202 (+55%). **(P0 caveat: "SOTA" was composite-score-only — H4 survives just 40.8% / Tier1 13.3%; the IMU-dominance win is on hover tracking, not survival.)**
5. **AWR mode-collapse.** PPO advantage normalization absorbs sparse crash_penalty (constant offset). Weighted MSE forces policy to imitate own crash trajectories. All 27 RL runs degrade systematically.
6. **Encoder-action alignment solved via Joint E2E Training (v5.0 Joint E2E).** By unfreezing the visual encoder and training on 50% hover + 50% recovery data, the representation space aligned with action generation, boosting survival rate to **60.1%** (+55% relative increase over H4 BC).
7. **Refactored hierarchical scoring metric.** The linear clipping `max(0, 1 - error/2)` was replaced with a smooth exponential decay `exp(-error)` to reward longer flights, though a discontinuity cliff at $SR=0.5$ still exists and heavily penalizes normal steady-state drift.
8. **Positive-Advantage Masking solves AWR mode-collapse.** Completely masking out negative advantage updates (`positive_advantage_mask: true`) prevents the policy from imitating crash rollouts, enabling stable RL fine-tuning (`v5_RL_best`). ~~Reduced steady-state drift by 51% (1.224m vs 2.505m).~~ **(P0 OVERTURNED: the "51% drift reduction" is a short-survival artifact. Under frozen paired eval v5_RL_best survives only 28.8% / n_cond 4/30; its conditional IAE is 3.340m — WORSE than v5_BC's 2.724m. The low all-IAE just means it crashed before it could drift.)** The curriculum still confirms a real Robustness-Precision capacity conflict (expanding to `pos=0.57m` collapsed survival to 4.8%).

9. **P0 frozen protocol + measured oracle (2026-06-17).** Froze one eval metric to end the 3× churn; added paired init, conditional-IAE, Tier1%, bootstrap CI; and replaced the hard-coded 0.85 oracle with a **measured 0.9668** (state-based PPO through the same `evaluate_frozen_p0`). Net result: prior composite/IAE rankings were artifact-driven; on the honest axis (Tier1%/survival) **Joint_E2E_v5 is the frontier**, and **nothing is deployable** (best 17.4% of oracle).

10. **Dispersive Loss does NOT help (core hypothesis falsified, 2026-06-18).** P2 2×2 ablation (Dispersive {OFF,ON} × E2E {OFF,ON}, 3 seeds, frozen protocol — `scripts/run_p2_ablation.py` + `evaluate_p2_ablation.py`): D1E1 vs D0E1 Tier1 **+1.1pp within 4.2pp pooled std**, survival identical (65.0%) → no gain above seed noise. **Mechanistic confirmation:** D1E0 ≡ D0E0 **MD5-byte-identical across all 3 seeds** (dispersive acts on `vis_pooled`; with a frozen encoder its gradient is discarded → exact no-op). The only (small) survival mover is **E2E** (Tier1 87.8→92.2pp), but survival/precision don't improve; all 4 P2 cells (Tier1 88–93%) beat the prior frontier Joint_E2E_v5 (80%) via the **recipe** (H4-transfer + task-cond + recovery mix), not dispersive. Precision still cond-IAE 2.6–3.2m (~13% oracle) → nothing deployable. **Conclusion: representation collapse is not the binding constraint; project pivots to negative-result + diagnosis (`RESEARCH_PLAN_v6.md` Phase 3).** **Phase 3a feature-collapse diagnosis (2026-06-18, `scripts/measure_feature_collapse.py` → `evaluation_results/p2_feature_collapse.json`, `docs/experiment_report_feature_collapse.md`):** measuring `vis_pooled` (D=256) on a fixed 4000-img batch over all 12 ckpts gives a result *sharper* than "the mechanism is inert" — (i) D1E0 ≡ D0E0 to 4 d.p. (eff_rank 30.3, corroborates the MD5 no-op); (ii) collapse is REAL — naive E2E (D0E1) drops eff_rank 30.3→9.0 and pushes mean pairwise cosine 0.07→0.96; (iii) Dispersive *games its objective* — D1E1 minimises disp_loss (−1.23→−8.14) only by inflating feat_norm ~287× (11.4→3281), and intrinsic rank gets WORSE (9→2, 99.8% variance on 2 dims), because `vis_pooled` feeds only the aux `StatePredictor`, not the action path (`cat([attended, imu_feat])`); (iv) survival is DECOUPLED from `vis_pooled` rank — a 15× rank swing (30→2) leaves survival flat (66→65%). **Phase 3c survival-mover attribution (`docs/experiment_report_survival_movers.md`):** gain over prior frontier Joint_E2E_v5 decomposes as **recipe (H4-transfer+task-cond+recovery mix) → frozen D0E0 = +7.8pp Tier1 / +3.9pp survive (holds with FROZEN encoder, so NOT from E2E) ≫ E2E (+4.4pp Tier1, survival-neutral) ≫ Dispersive (~0)**; precision is the unmoved axis (cond-IAE ~2.8–2.9m all configs). **Phase 3b precision diagnosis (`scripts/measure_ood_coverage.py` + `measure_image_distance_info.py`, `docs/experiment_report_ood_coverage.md` + `experiment_report_image_distance_info.md`): precision is INFORMATION-GATED, not data-gated.** Closed-loop drifts to median 2.83m where BC has <0.2% of its mass; but the data-gated fix is futile because (a) `--pos-range` doesn't change coverage (env anchors `target=init_pos` for hover), (b) the PPO expert itself can't recover from >2m offset (3m→0/20 survive), and (c) the FPV image can't encode metric range past 2m — the only range cue is crosshair `size` (4 quantised levels; DR-OFF saturates at 2px so renders at 2.0/2.5/3.0m are byte-identical; DR-ON ridge image→distance R² far≥1.5m = 0.12). **Phase 3b sensing ablation — "change the OBSERVATION" is REFUTED as stated (2026-06-21, `scripts/measure_higher_res_gate.py` + `scripts/run_p3b_rangecue.py`, `docs/experiment_report_sensing_ablation.md`).** Two results overturn the info-gated implication: **(A) free higher-res gate** (image→distance ridge, dual form, 3 res × 2 targets) shows the far-range info loss is a **renderer TARGET ARTIFACT, not the pixel count** — production crosshair far R² ≈0 at 64/128/256px (resolution alone useless), but a non-saturating perspective target restores far R² to **0.42 at the SAME 64px** (resolution a minor lever: 128→0.50, 256→0.45). **(B) range-cue intervention** (fold the metric pos-error the FPV lacks into task-cond, D0E1 recipe, 3 seeds, frozen P0): even handed the **oracle** metric range, precision barely moves and is fragile — scalar_clean cond-IAE 2.91→**2.43m** (~0.5m, costs −6.7pp survive/−13pp Tier1, still ~36× oracle 0.068m); σ=0.15m sensor noise erases it (scalar_noised 2.81m ≈ control); the richer **full 3D pos-error cue reproducibly COLLAPSES survival** (pos3d_clean Tier1 92→**7%**, n_cond 5/0/1, across all 3 seeds). **→ precision is NOT sensing-gated: supplying range (even oracle, even full position) does not restore it and a richer cue harms. The binding constraint is the absence of learned far-range RECOVERY behaviour in the 1–3m band — coverage/teacher-competence, not the observation channel.** Moving precision needs a competent far-range teacher to generate 1–3m coverage; a better sensor/policy over the existing data will not. Wider-init retrain remains NOT recommended (`scripts/run_p3b_retrain.py`; expert can't label that band → cond-IAE stays ~2.8m). **Faithful re-run (P2f, 2026-06-23, `scripts/run_p2_ablation.py --faithful` + `evaluate_p2_ablation.py` → `evaluation_results/p2f_ablation_leaderboard.json`, `docs/experiment_report_faithful_dispersive.md`):** the original P2 Dispersive was unfaithful on 3 axes (off-path `vis_pooled`, λ=0.05, hand-rolled log-distance) while §2 claimed "exactly as specified"; a 2nd trap was the paper's Algorithm 1 dropping the official `/z.shape[1]` (`/d`) normalisation → τ=0.5 saturates to zero-gradient on GroupNorm flat features (another silent no-op). Faithful version (InfoNCE-L2 on `flow_net` mid-block, λ=0.5, τ=0.5, `/d`) **reproduces the negative result** — D1E1 vs D0E1 = **−2.2pp Tier1 (pooled std 6.3pp), −2.1pp survival**, cond-IAE 2.7–3.2m (~13% oracle) — closing the "you used it wrong" rebuttal. **But C2 ("D1E0 ≡ D0E0 byte-identical no-op") is OVERTURNED:** the faithful placement trains `flow_net` even under `--freeze-vision`, so D1E0 ≠ D0E0 (MD5 differs ×3 seeds) and Dispersive-on-frozen is mildly **harmful** (Tier1 87.8→74.4, variance 3.1→8.7pp) rather than inert — a richer result that replaces the MD5 no-op claim; draft §5/§9 must be rewritten.

11. **Teacher × Observation does NOT break the precision floor (v7 H_v7 REFUTED, 2026-06-26).** The constructive follow-up (`RESEARCH_PLAN_v7.md`) pre-registered that the cond-IAE ≈2.8m floor breaks **only if BOTH** far-range recovery labels (Teacher coverage, T1) **and** a range-encoding perspective observation (O1) are supplied. Phases 0–3 removed each as a candidate (competent PID-CTBR far teacher: 100% survive 1–4m, cond-IAE 0.14–0.18m; perspective far-R² 0.05→0.40; far-coverage 0.4%→11.7%). **Decisive frozen-P0 T×O 2×2** (`scripts/run_p2to_ablation.py` + `scripts/evaluate_p2to_ablation.py`, 4 cells × 3 seeds, cond-IAE PRIMARY; new `train_flow_v5.py --hover-h5` + `evaluate_frozen_p0.py --target-render` so O1 trains **and** evals on perspective; OOM-fixed `FlowDatasetV5` two-pass pre-alloc, byte-identical & equivalence-tested): cond-IAE T0O0 **2.69±0.15m** / T0O1 2.48±0.14m / T1O0 2.71±0.22m / **T1O1 2.93±0.18m** (36–43× the 0.0675m oracle; composite 20–23% oracle). **Verdict: T1O1 vs T0O0 Δ=+0.24m (pooled std 0.23m) → not significant, not ≤1.5m → FLOOR NOT BROKEN.** **Structure = the result:** coverage (T1) buys **survival** (+8pp crosshair / +14pp perspective, Tier1→~100%) but moves precision 0; sensing (O1) nudges precision within noise; the **interaction is NEGATIVE** — best precision is T0O1 (2.48m), adding far-recovery on top (→T1O1) *degrades* it to 2.93m. ⇒ the binding constraint is a **Robustness–Precision Capacity Conflict** (now RL-free / pre-registered / 3-seed; was RL-confounded in Finding #8), **NOT** coverage and **NOT** sensing — both removed and the floor held. Paper pivots to negative-deepening (§6/§7 decisive exclusion experiment). Artifacts `evaluation_results/p2to_ablation_{manifest,leaderboard}.json`, `docs/experiment_report_p2to_decisive.md`.

12. **Scale-invariant on-path regularizer: gaming is real & removable, but control is decoupled (v7 Phase 6 / Direction 4 DONE, 2026-06-27).** Closes the last §6.1 rebuttal ("the faithful InfoNCE-L2 only failed because it games a *scale-sensitive* criterion via norm inflation; a scale-invariant one might help"). Held the P2f **D1E1 recipe fixed** and varied **only** `--dispersive-form` on `flow_net` mid-block (new `_dispersive_loss_cosine` = unit-sphere InfoNCE with **`/d` dropped**; `_dispersive_loss_vicreg` = variance-hinge + off-diag covariance², float32-guarded; default `infonce` byte-identical), 6 new runs (cosine/vicreg ×3 seeds), off/infonce reused from P2f. Scripts `scripts/run_p6_form_ablation.py` + `scripts/evaluate_p6_form_ablation.py` (control frozen-P0 **+** flow_mid geometry, directional verdict, `--reverdict` for GPU-free recompute). **[A] objective-gaming REMOVED:** infonce GAMES (feat_norm **8.93× off** 9.5→84.8, eff_rank **collapses 221→36** = 3.5% of 1024 dims); cosine/vicreg are **clean** (feat_norm **1.33×/1.36×**, eff_rank **221→769/867** = 75%/85% — genuine high-rank dispersion, the *intended* effect, no norm cheat) → the gaming is an artifact of the scale-sensitive L2 criterion. **[B] control NOT improved (`any_control_improved=False`):** all four forms in one band (survival 60–65%, Tier1 82–92%, cond-IAE **2.9–3.1m**, 11–12% oracle); both scale-invariant forms even *regress* cond-IAE slightly (cosine +0.21m, vicreg +0.13m, just past pooled std). A **24× flow_mid rank swing (36→867) buys ~0 control** → on-path representation geometry is **DECOUPLED** from survival/precision (on-path analogue of Finding #10's `vis_pooled` rank ⟂ survival). **⇒ Dispersive negative result reinforced: fixing feature collapse (now demonstrably working) does not move closed-loop control regardless of the criterion.** Artifacts `evaluation_results/p6_form_ablation_{manifest,leaderboard}.json`, `docs/experiment_report_p6_scale_invariant.md`. Fold into §5/§6.1 in Phase 7.

See [docs/dev_log_v4_h4_hierarchical.md](docs/dev_log_v4_h4_hierarchical.md), [docs/experiment_report_joint_e2e.md](docs/experiment_report_joint_e2e.md), [docs/experiment_report_feature_collapse.md](docs/experiment_report_feature_collapse.md), [docs/experiment_report_survival_movers.md](docs/experiment_report_survival_movers.md), [docs/experiment_report_ood_coverage.md](docs/experiment_report_ood_coverage.md), [docs/experiment_report_image_distance_info.md](docs/experiment_report_image_distance_info.md) [docs/experiment_report_sensing_ablation.md](docs/experiment_report_sensing_ablation.md), [docs/experiment_report_faithful_dispersive.md](docs/experiment_report_faithful_dispersive.md), [docs/experiment_report_p2to_decisive.md](docs/experiment_report_p2to_decisive.md) and [docs/experiment_report_p6_scale_invariant.md](docs/experiment_report_p6_scale_invariant.md) for full diagnostic chains and reports.

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

# Research Plan v6 — Path A: Simulation Representation Learning for Visual Drone Hover

**Version:** 6.0
**Date:** 2026-06-17
**Supersedes:** the SOTA framing in RESEARCH_PLAN.md (v5.0). Those composite-score
rankings were proven artifact-prone by P0 — see `memory/project_p0_frozen_eval.md`.
**Target venues:** ICRA / workshop (simulation-only; no real-robot claim).

---

## 0. Why v6 exists (the re-plan)

The project chased a single scalar ("hierarchical composite score") that changed
formula 3× (RMSE → linear-clip → exp-decay). P0 froze the protocol and added
artifact-robust columns. Two headline claims collapsed:

- **"H4 BC = SOTA"** — true only on the composite score; H4 survives 40.8% with
  Tier1 13.3%. The multiplicative `SR × precision` score rewards *die-early-but-precise*.
- **"v5_RL_best = 51% precision gain"** — a **short-survival artifact**: low all-IAE
  came from crashing before the drone could drift (n_cond 4/30); its *conditional*
  IAE is 3.340 m, the worst of the group.

**Honest standing (frozen protocol, measured oracle 0.9668):**

| Model | Score | Survive | **Tier1%** | cond-IAE (n) | %Oracle |
|-------|-------|---------|------------|--------------|---------|
| PPO Oracle (state) | 0.967 | 100% | **100%** | 0.068 m (30) | 100% |
| H4_BC | 0.168 | 40.8% | 13.3% | 2.520 m (4)⚠️ | 17.4% |
| v5_RL_best | 0.129 | 28.8% | 13.3% | 3.340 m (4)⚠️ | 13.3% |
| v5_BC | 0.124 | 55.9% | 70.0% | 2.724 m (21) | 12.8% |
| Joint_E2E_v5 | 0.106 | 62.2% | **80.0%** | 3.077 m (24) | 11.0% |

**Nothing is deployable** (best 17.4% of oracle). The survival frontier is
Joint_E2E_v5. **v6 stops optimizing the composite score and optimizes survival /
Tier1% directly**, with precision reported only conditional on survival.

### Frozen evaluation protocol (DO NOT change once published)
`scripts/evaluate_frozen_p0.py` — 30 ep, `base_seed=12345`, σ=2.0 exp-decay,
paired identical init across models, bootstrap 95% CI, conditional-IAE over
episodes surviving ≥250 steps, **measured** PPO-oracle normaliser. Canonical
artifact: `evaluation_results/frozen_p0_leaderboard.json`.

**Primary metric for all v6 experiments: Tier1 pass-rate (fraction of episodes
flying ≥ 250 / 500 steps) and survival.** Composite score is reported only as a
flagged secondary. cond-IAE is reported only when `n_cond ≥ ~15`.

---

## Phase 0 — Freeze the metric ✓ DONE (2026-06-17)
- `scripts/evaluate_frozen_p0.py` with paired seeding + conditional precision.
- **Oracle normalized rigorously:** state-based PPO rolled through the *same*
  `evaluate_frozen_p0` (`--oracle-ckpt/--oracle-norm`) → measured composite
  **0.9668** (100% survive, IAE 0.068 m), replacing the hard-coded 0.85.

---

## Phase 1 — Baseline matrix (lower/upper bounds on the frozen protocol)

Goal: bracket the vision policies between an honest lower bound (naive BC) and the
oracle upper bound, all scored by the **identical** frozen protocol.

| Baseline | Input | Role | Status |
|----------|-------|------|--------|
| **PPO Oracle** (state-based) | 15D state | Upper bound (%Oracle ref = 0.9668) | ✓ DONE |
| **BC-vision-only** | RGB stack (no IMU, no flow) | Naive lower bound: regress action seq with MSE | ✓ DONE 2026-06-19 (3 seeds {0,1,2}) — **Score 0.089±0.018, Survive 54.7±3.4%, Tier1 61.1±13.4%, %Oracle 9.2±1.8%** (in the flow-policy band; seed 0's Tier1 80% was luck) |
| **PPO-from-pixels** | RGB stack | End-to-end RL-from-pixels reference | ✓ DONE 2026-06-19 (3 seeds {0,1,2}, 1M steps) — **COLLAPSE-PRONE: Score 0.114±0.017, Survive 30.1±15.6%, Tier1 15.6±22.0%, %Oracle 11.8±1.8%** (2/3 seeds collapse n_cond 0/30; seed 1 survives Tier1 46.7%) |
| H4_BC / v5_BC / Joint_E2E_v5 / v5_RL_best | RGB+IMU | Existing flow policies | ✓ in leaderboard |

### 1.1 BC-vision-only (runnable now)
Same `VisionEncoder` (6→256) as the flow policies, **no IMU, no flow** — a 2-layer
MLP head regresses the `T_pred×action_dim` sequence with MSE. Isolates "what does a
plain feed-forward vision→action map achieve", i.e. the value added by IMU fusion +
flow modelling. Exposes `predict_action(images, imu=None, ...)` so it scores through
the frozen rollout with zero eval changes.

```bash
# Train (hover+recovery mix, same data as the flow policies)
dppo/Scripts/python.exe -m scripts.train_bc_vision_only \
    --recovery-h5 data/expert_demos_v4_recovery.h5 \
    --recovery-episodes 500 --hover-episodes 500 \
    --tag bc_vision_only_s0 --seed 0
# Eval under the frozen protocol (generic baseline rollout)
dppo/Scripts/python.exe -m scripts.evaluate_baselines_frozen \
    --ckpts "BC_vis:checkpoints/bc_vision_only/bc_vision_only_s0/best_model.pt" \
    --oracle-ckpt checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
    --oracle-norm checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz
```

### 1.2 PPO-from-pixels (BUILT + smoke-passed 2026-06-19)
Trains PPO end-to-end on `QuadrotorVisualEnv(QuadrotorEnvV4)` (CNN actor-critic over
the 6×64×64 stack, CTBR + INDI inner loop unchanged). `models/ppo_pixel.py` reuses the
flow policies' `VisionEncoder` (6→256); actor and critic each own a **separate**
encoder (mirrors `PPOExpert`'s separate-net design so per-network grad-clip stays
valid). The actor exposes the flow `predict_action(images, imu=None, n_steps=None,
task_cond=None) -> (B, action_dim, T_pred)` contract (reactive single action tiled to
T_pred), so it scores through `scripts.evaluate_baselines_frozen --kind ppo_pixels`.
**Pixel-specific implementation notes:** (i) T_obs=2 frame stacking maintained across
the vectorised envs, auto-reset aware; (ii) the rollout buffer is stored **uint8 on
CPU**, minibatches moved to GPU + /255 inside `update()` (float32-on-GPU ≈ 13 GB
otherwise). **Budget** (`configs/ppo_from_pixels.yaml`): 1M timesteps (documentation
budget vs the state expert's 3M), n_steps 2048, 8 envs, lr 1e-4, n_epochs 4. Smoke
(`--quick`, 2 updates) ran end-to-end at ~43 s/update → **full run ≈ 1 h**.
**Prior:** strong evidence vision RL from scratch collapses (27 ReinFlow runs, AWR
mode-collapse) — this documents *how far* naive pixel-PPO gets, not expected to be
competitive.
```bash
dppo/Scripts/python.exe -m scripts.train_ppo_from_pixels --tag ppo_px_s0 --seed 0
dppo/Scripts/python.exe -m scripts.evaluate_baselines_frozen --kind ppo_pixels \
    --ckpts "PPO_px:checkpoints/ppo_from_pixels/ppo_px_s0/best_model.pt" \
    --oracle-ckpt checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
    --oracle-norm checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz
```
**Result (2026-06-19, `ppo_px_s0`, 1M steps ≈ 1 h):** COLLAPSED as predicted. Training
eval survive 0% throughout (mean_len 53–73, best 73.1); rollout ep_len stuck ~85–100;
VL 1600→250. Frozen eval (`baselines_frozen_ppopx.json`): Score 0.100, Survive 15.2%,
**Tier-1 0%, n_cond 0/30** (nothing flew past 250 steps), %Oracle 10.4%. Its low
*all*-IAE 0.70m is a pure short-survival artifact (crashes ~75 steps before it can
drift) — a clean in-the-wild demonstration of the artifact §4's protocol catches.

---

## Phase 2 — Core ablation: Dispersive × E2E (2×2, ≥3 seeds)

**Research question (v6 core):** does Dispersive Loss improve *survival* (not just a
metric artifact), and does it require an end-to-end-trainable encoder?

Two factors, held everything else fixed (init = H4 transfer, data = 500 hover + 500
recovery, 80 epochs, lr 1e-4, batch 256, task-conditioned, `configs/flow_policy_v5.yaml`):

| Factor | ON | OFF |
|--------|----|-----|
| **Dispersive** | `--lambda-disp 0.05` (repels `vis_pooled` features) | `--lambda-disp 0.0` |
| **E2E** | vision encoder trainable | `--freeze-vision` (encoder frozen at H4 transfer) |

### The 2×2 cells

| Cell | Dispersive | E2E | Prediction |
|------|-----------|-----|------------|
| **D0E0** | OFF | OFF (frozen) | Frozen-feature floor |
| **D0E1** | OFF | ON | ≈ Joint_E2E_v5 (survival frontier) |
| **D1E0** | ON | OFF (frozen) | **≈ D0E0** — dispersive acts on `vis_pooled`, which is frozen ⇒ no-op |
| **D1E1** | ON | ON | Dispersive's real test: does repelling features lift survival above D0E1? |

> **Mechanistic note (predicted, and itself a result):** the dispersive term is
> applied to the vision encoder's pooled features (`flow_policy_v5._dispersive_loss(vis_pooled)`).
> With `--freeze-vision` that gradient is zero, so **D1E0 must collapse onto D0E0**.
> The 2×2 therefore isolates the *interaction*: Dispersive can only act through a
> trainable encoder. The decisive comparison is **D1E1 vs D0E1**.
>
> **⚠️ This note applies to the LEGACY off-path `vis_pooled` placement only.** The
> faithful re-run (§ Faithful re-run below) puts Dispersive on the `flow_net` mid-block,
> which trains regardless of `--freeze-vision` — so under the faithful implementation
> **D1E0 ≠ D0E0** (the no-op prediction is overturned). The decisive comparison
> **D1E1 vs D0E1** is unchanged.

### Design rules
- **≥3 seeds** per cell (`--seed 0,1,2`), report **mean ± std of Tier1% and survival**.
- **Primary axis = Tier1% / survival** (frozen protocol). Composite score secondary
  and flagged; cond-IAE only when `n_cond ≥ ~15`.
- Decision: Dispersive is supported **iff** D1E1 beats D0E1 on Tier1%/survival by
  more than the across-seed std (not on composite score, not on all-IAE).

### Run it
```bash
# One cell now (sweep driver, single cell — keeps the manifest consistent):
dppo/Scripts/python.exe -m scripts.run_p2_ablation \
    --h4-ckpt checkpoints/flow_policy_v4/20260514_175219/best_model.pt \
    --recovery-h5 data/expert_demos_v4_recovery.h5 \
    --recovery-episodes 500 --hover-episodes 500 \
    --cells D1E1 --seeds 0

# Full sweep (12 runs, sequential, skips finished cells):
dppo/Scripts/python.exe -m scripts.run_p2_ablation \
    --h4-ckpt checkpoints/flow_policy_v4/20260514_175219/best_model.pt \
    --recovery-h5 data/expert_demos_v4_recovery.h5 \
    --cells D0E0 D0E1 D1E0 D1E1 --seeds 0 1 2

# Aggregate by cell (mean±std over seeds, Tier1%/survival primary):
dppo/Scripts/python.exe -m scripts.evaluate_p2_ablation \
    --manifest evaluation_results/p2_ablation_manifest.json \
    --oracle-ckpt checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
    --oracle-norm checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz
```

### Compute budget
~80 epochs/run, batch 256, 500+500 episodes in RAM (≈12 GB — the known-safe mix;
batch 256 avoids the documented OOM). One run ≈ 1.5–2.5 h on the RTX 3090; 12 runs
sequential ≈ 1 day. Runs are launched one-at-a-time by the sweep driver (no GPU
contention — see Known Failure Mode #7).

### RESULTS ✓ DONE (2026-06-18, all 12 cells, frozen protocol, oracle 0.9668)
`evaluation_results/p2_ablation_leaderboard.json`

| Cell | Tier1% (mean±std) | Survival | Composite | %Oracle |
|------|-------------------|----------|-----------|---------|
| D0E0 (Disp✗ E2E✗ frozen) | 87.8 ± 3.1 | 66.1 ± 3.7 | 0.130 ± 0.016 | 13.4% |
| D1E0 (Disp✓ E2E✗ frozen) | **87.8 ± 3.1** | **66.1 ± 3.7** | 0.130 ± 0.016 | 13.4% |
| D0E1 (Disp✗ E2E✓) | 92.2 ± 3.1 | 65.0 ± 2.8 | 0.120 ± 0.010 | 12.4% |
| D1E1 (Disp✓ E2E✓) | 93.3 ± 2.7 | 65.0 ± 2.5 | 0.128 ± 0.013 | 13.2% |

1. **Dispersive Loss — NOT SUPPORTED (core hypothesis fails).** D1E1 vs D0E1:
   Tier1 **+1.1 pp** (pooled std **4.2 pp** → within noise), survival **identical**
   (65.0 both). Even where dispersive *can* act (trainable encoder), it produces no
   survival/Tier1 gain above seed noise.
2. **Mechanistic prediction confirmed — byte-identical.** D1E0 ≡ D0E0: the
   `best_model.pt` files are **MD5-identical across all 3 seeds**. With a frozen
   encoder the dispersive gradient on `vis_pooled` is discarded, touching no
   trainable parameter ⇒ "Dispersive ON" frozen = "Dispersive OFF" frozen, bit-for-bit.
3. **E2E (trainable encoder) — small, mixed.** Tier1 +4.4 pp (87.8→92.2) but
   survival slightly *down* (66.1→65.0) and composite *down* (0.130→0.120): E2E pushes
   more episodes past the half-horizon without extending mean flight or improving
   precision. All four P2 cells (Tier1 88–93%) **beat the prior frontier Joint_E2E_v5
   (80%)** — but that gain is the *recipe* (H4-transfer init + task-cond + recovery
   mix), not dispersive. Precision unchanged (cond-IAE 2.6–3.2 m, ~13% oracle) →
   **still nothing deployable.**

**Verdict:** representation collapse (what Dispersive targets) is **not** the binding
constraint for visual hover. The paper pivots to a *negative result + diagnosis*.

### Faithful re-run ✓ DONE (2026-06-23, all 12 cells, frozen protocol, oracle 0.9668)
`evaluation_results/p2f_ablation_leaderboard.json` · `docs/experiment_report_faithful_dispersive.md`

The deep-science-writer smoke test found the legacy P2 Dispersive was **unfaithful on
3 axes** (off-path `vis_pooled`, λ=0.05, hand-rolled log-distance) while paper §2 claims
"exactly as specified"; a 2nd trap was the paper's Algorithm 1 dropping the official
`/z.shape[1]` (`/d`) normalisation → τ=0.5 saturates to zero gradient (another silent
no-op). The faithful re-run uses **InfoNCE-L2 on the `flow_net` mid-block, λ=0.5, τ=0.5,
`/d`** (`--faithful`); `p2f_` tags + separate manifest leave legacy P2 untouched.

| Cell | Tier1% (mean±std) | Survival | cond-IAE |
|------|-------------------|----------|----------|
| D0E0 (Disp✗ E2E✗ frozen) | 87.8 ± 3.1 | 66.1 ± 3.7 | 2.9 m |
| D1E0 (Disp✓ E2E✗ frozen) | **74.4 ± 8.7** | 60.4 ± 1.6 | 2.8 m |
| D0E1 (Disp✗ E2E✓) | 92.2 ± 3.1 | 65.0 ± 2.8 | 2.9 m |
| D1E1 (Disp✓ E2E✓) | 90.0 ± 5.4 | 62.9 ± 2.4 | 2.9 m |

1. **Negative result CONFIRMED & rebuttal-proofed.** D1E1 vs D0E1 = **−2.2pp Tier1
   (pooled std 6.3pp), −2.1pp survival** → no gain even where Dispersive can act. Faithful
   to the official recipe, so the "you used it wrong" rebuttal is closed.
2. **C2 ("D1E0 ≡ D0E0 byte-identical no-op") OVERTURNED.** The faithful flow_mid placement
   trains `flow_net` under `--freeze-vision`, so `p2f_D1E0_s* ≠ p2f_D0E0_s*` (MD5 differs
   ×3 seeds). The frozen row is **not inert** — Dispersive-on-frozen is mildly *harmful*
   (Tier1 87.8→74.4, variance 3.1→8.7pp): a richer result that replaces the MD5 no-op.
   **Draft §5/§9 must be rewritten.**
3. E2E remains the only (small) Tier1 mover; precision unmoved (cond-IAE ~2.8–2.9 m, ~13%
   oracle). The Phase-3 binding-constraint conclusion (far-range recovery, not
   representation collapse) is unchanged. Only after this does the paper return to
   deep-science-writer for literature + writing.

---

## Phase 3 (negative-result path — Dispersive did NOT help)
Phase 2 showed no Tier1%/survival lift from Dispersive, so per the pre-registered
rule the project becomes a **negative result + diagnosis**: representation-collapse
is not the binding constraint for visual hover — information asymmetry / OOD coverage
is (the IMU already carries the attitude signal; the encoder is not the bottleneck).
Diagnostic steps: (a) measure the actual feature-collapse the dispersive term is
supposed to fix (rank/variance of `vis_pooled`) with vs without it ✓ DONE; (b) precision
diagnosis ✓ DONE — coverage gap confirmed (3b) then resolved as **information-gated**
(3b-info): the FPV image can't encode metric range past 2 m, so a wider-init retrain is
NOT recommended (would reproduce the 2.8 m floor); (c) write up E2E-recipe vs
prior-frontier as the only survival mover ✓ DONE.

**All three Phase 3 diagnostics complete. Negative-result chain: (3a) representation
collapse is not the bottleneck — Dispersive games its objective and survival is
decoupled from `vis_pooled` rank; (3c) the recipe (not Dispersive, not E2E) is the only
survival mover; (3b) precision is capped by the FPV observation model (monocular 64×64
can't resolve range past 2 m), not by representation or data coverage.**

### 3a — Feature-collapse measurement ✓ DONE (2026-06-18)
`scripts/measure_feature_collapse.py` → `evaluation_results/p2_feature_collapse.json`;
full write-up `docs/experiment_report_feature_collapse.md`. `vis_pooled` (D=256),
fixed 4000-image hover+recovery batch, mean over 3 seeds:

| Cell | eff_rank | n_eff_99 | feat_norm | disp_loss | Survival |
|------|---------:|---------:|----------:|----------:|---------:|
| D0E0 / D1E0 (frozen) | 30.3 | 91 | 1.80 | −0.81 | 66.1 % |
| D0E1 (E2E, Disp✗) | 9.0 | 45 | 11.4 | −1.23 | 65.0 % |
| D1E1 (E2E, Disp✓) | **2.0** | **2** | **3281** | **−8.14** | 65.0 % |

Result is **sharper than the pre-registered "mechanism is inert" prediction**:

1. **Frozen no-op confirmed at the feature level.** D1E0 ≡ D0E0 to 4 d.p. (complements the MD5 byte-identity).
2. **Collapse is real.** Naive E2E (D0E1) collapses the H4 features: eff_rank 30→9, mean pairwise cosine 0.07→**0.96**. There *was* something for Dispersive to fix.
3. **Dispersive gambles its objective and makes collapse worse.** D1E1 minimises the literal dispersive loss (−1.23→−8.14) only by **inflating feature norm ~287×** (11.4→3281); intrinsic rank gets *worse* (9→**2**, 99.8 % variance on 2 dims). The low cosine (0.013) is a 2-D-at-huge-radius artifact, not high-dim spread. It most cheaply inflates the pooled-only `fc` layer — and `vis_pooled` feeds only the aux `StatePredictor`, **not** the action path (`cat([attended, imu_feat])`).
4. **Survival is decoupled from `vis_pooled` rank.** Effective rank swings **15×** (30→2) while survival is flat within seed noise (66→65 %). Strongest statement of the v6 thesis: representation collapse is *not* the binding constraint for visual hover.

### 3c — Survival-mover attribution ✓ DONE (2026-06-18)
`docs/experiment_report_survival_movers.md` (from the two leaderboard JSONs, same frozen protocol). Decomposes the gain over the prior frontier Joint_E2E_v5 (Tier1 80.0 %, survival 62.2 %, cond-IAE 3.08 m):

| Step | Δ Tier1 | Δ Survival | cond-IAE |
|------|--------:|-----------:|---------:|
| Recipe (→ frozen cell D0E0) | **+7.8 pp** | **+3.9 pp** | 2.93 m |
| + E2E (D0E0→D0E1) | +4.4 pp | −1.1 pp | 2.91 m |
| + Dispersive (D0E1→D1E1) | +1.1 pp (∈ noise) | 0.0 | 2.81 m |

**Mover ranking: recipe (H4-transfer + task-cond + recovery mix) ≫ E2E (Tier1-only bump, survival-neutral) ≫ Dispersive (~0).** The recipe gain holds with a *frozen* encoder, so it is not from E2E. **Precision is the unmoved axis:** cond-IAE stays ~2.8–2.9 m (~13 % oracle, n_cond 24–29 = reliable) across every config → no representation/training lever moved precision → motivates 3b (is precision OOD-coverage-gated?).

### 3b — OOD-coverage probe (cheap, no retrain) ✓ DONE (2026-06-18)
`scripts/measure_ood_coverage.py` → `evaluation_results/p3b_ood_coverage.json`; write-up `docs/experiment_report_ood_coverage.md`. Compares position-error magnitude ‖pos−target‖ in BC training (states[:,0:3]) vs D0E1 closed-loop steady state (27/30 surviving, frozen seeds):

| pos-err | TRAIN (n=480k) | CLOSED-LOOP steady (n=4.6k) |
|---------|---------------:|----------------------------:|
| p50 | 0.066 m | **2.833 m** (= cond-IAE) |
| p99 | 0.621 m | 6.574 m |
| max | 4.073 m | 7.718 m |
| frac >1 m | 0.20 % | 90.0 % |
| frac >3 m | 0.005 % | 45.6 % |

**97.3 %** of steady-state samples are above the training **p99** (0.62 m); **25.0 %** are above the training **max** (zero-coverage region); steady median = **4.56× train p99**. The policy spends ~90 % of steady state beyond 1 m, where the at-target BC data has <0.2 % of its mass — a real coverage gap. This is a **necessary condition, not sufficiency**: it had to be checked whether *filling* the gap would help (data-gated) or whether the observation can't encode range at all (information-gated). Resolved in 3b-info below: **information-gated**.

### 3b-info — DATA-gated vs INFORMATION-gated: RESOLVED = information-gated ✓ DONE (2026-06-18)
`scripts/measure_image_distance_info.py` → `evaluation_results/p3b_image_distance_info{,_nodr}.json`; write-up `docs/experiment_report_image_distance_info.md`. Three findings collapse the data-gated path:
1. **`--pos-range` creates NO position-error coverage** — `quadrotor_env_v4.reset()` anchors `target=init_pos` for hover, so widening init just relocates the hover point. Measured: the collected wide (±3 m) `expert_demos_v4_recovery_wide3.h5` has identical coverage to ±1 m (both 0.4 % of steps >1 m). The naive retrain would train on unchanged coverage.
2. **The PPO expert can't recover from >2 m offset** (target≠init, 20 trials): 1 m→0.066 m 20/20, 2 m→0.066 m 20/20, **3 m→1.56 m 0/20**. Even a *correct* offset collection caps at ~2 m, below the 2–3 m precision regime.
3. **The FPV image doesn't encode metric distance** — the only range feature is crosshair `size=max(2,min(6,int(6/(d+0.5))+dr))`. DR-ON: adjacent-distance d-prime **<0.2 everywhere**; ridge decode image→distance R² near(<1 m) **0.41**, far(≥1.5 m) **0.12**. DR-OFF (noiseless ceiling): size = **2 px for all d≥2 m** → renders at 2.0/2.5/3.0 m are **byte-identical** (d′=0). The policy's 2.83 m steady drift sits where the image carries zero range info.

**Verdict (3b-info): the FPV observation does not encode metric range past 2 m.** This was read as "information-gated → change the observation". The sensing ablation below tested that implication by intervention and **refutes it as stated**.

### 3b-sensing — Higher-res gate + range-cue INTERVENTION: "change the observation" REFUTED ✓ DONE (2026-06-21)
`scripts/measure_higher_res_gate.py` + `scripts/run_p3b_rangecue.py` (→ `train_flow_v5.py --range-cue`, `evaluate_frozen_p0.py --cue-noise`); artifacts `evaluation_results/p3b_higher_res_gate.json`, `p3b_rc_{clean,noised}{,_s12}_frozen.json`; write-up `docs/experiment_report_sensing_ablation.md`.

**(A) Free higher-res gate** (image→distance ridge, dual form, 3 res × 2 targets, DR on): the far-range info loss is a **renderer TARGET ARTIFACT, not the pixel count**. Production saturating crosshair far R²(≥1.5 m) ≈ 0 at 64/128/256 px (resolution alone useless); a non-saturating **perspective target restores far R² to 0.42 at the SAME 64 px** (128→0.50, 256→0.45 — resolution a minor secondary lever). So the 3b-info "64×64 can't encode range" is partly an artifact of the crosshair size formula; even the perspective target's far R² (0.45) ≪ near (0.88), i.e. a better sensor would improve, not solve, far range.

**(B) Range-cue intervention** (fold the metric pos-error the FPV lacks into the v5 task-cond slot — `states[:, :3]`, no re-collection; D0E1 recipe; **3 seeds, frozen P0**; control reuses `p2_D0E1`):

| arm | cue | cond-IAE (mean±std) | survive % | Tier1 % |
|-----|-----|---------------------:|----------:|--------:|
| control | — | **2.906 ± 0.075** | 65.0 ± 2.8 | 92.2 ± 3.1 |
| scalar_clean | ‖pos_err‖ σ=0 | **2.430 ± 0.242** | 58.3 ± 4.9 | 78.9 ± 15.9 |
| scalar_noised | ‖pos_err‖ σ=0.15 | 2.805 ± 0.258 | 57.9 ± 10.3 | 66.7 ± 31.4 |
| pos3d_clean | 3D σ=0 | n/a (collapse) | 40.6 ± 3.0 | **6.7 ± 7.2** |
| pos3d_noised | 3D σ=0.15 | 2.975 ± 0.242 | 53.9 ± 3.2 | 67.8 ± 9.6 |

Even the **oracle** range cue buys only ~0.5 m (2.91→2.43 m, still ~36× oracle 0.068 m) and costs survival; σ=0.15 m noise erases it; the **richer 3D cue reproducibly collapses survival** (Tier1 92→7 %, all 3 seeds). "More sensing → better" is falsified.

**Corrected verdict: precision is NOT sensing-gated.** Supplying metric range (even oracle, even full position) does not restore precision and a richer cue harms survival. The binding constraint is the absence of learned far-range **recovery behaviour** in the 1–3 m band — a coverage/teacher-competence gap (the expert can't recover from >2 m, so it can't label that band), not the observation channel. Moving precision needs a competent far-range teacher to generate 1–3 m coverage; a better sensor or a better policy over the existing data will not. Wider-init retrain stays NOT recommended (cond-IAE prediction unchanged ~2.8 m).

---

## Phase 4 — Paper write-up (negative result) — IN PROGRESS

Draft: `docs/paper_negative_result_draft.md` (v0.1, started 2026-06-18; last updated
2026-06-19). Title: *"Representation Collapse Is Not the Bottleneck: A Negative Result
and Diagnosis for Vision-Based Quadrotor Hover."* Target ICRA / robot-learning
workshop, simulation-only.

**Structure complete (all sections drafted):** Abstract · §1 Introduction ·
**§2 Related Work** · §3 Task/Policy/Dispersive mechanism · §4 Frozen protocol ·
§5 2×2 ablation · §6 Diagnosis (6.1 representation / 6.2 survival-movers / 6.3 sensing) ·
§7 Discussion · §8 Limitations · §9 Conclusion · Reproducibility · **References**.
Every number traces to a frozen-protocol artifact or diagnostic JSON.

**Figures ✓ DONE (2026-06-18; Fig 4 added 2026-06-21):** Fig 1 `rank_survival_decoupling.png`
(§6.1, 15× rank swing vs flat survival), Fig 2 `crosshair_distance_saturation.png` (§6.3,
range-cue saturation + image→distance R² — now framed as a *measurement*, not the
verdict), Fig 3 `single_seed_swing.png` (§4, single-seed unreliability), Fig 4
`sensing_ablation.png` (§6.3, higher-res gate + range-cue intervention → precision is
coverage/teacher-competence-gated, not sensing-gated). Numbers read straight from the
artifacts by `scripts/make_paper_figures.py` (re-runnable); output in `docs/figures/`.

**§2 Related Work + References [1]–[21] ✓ DONE (2026-06-19):** grounded via NotebookLM
(notebook `generative-rl-flow-policy-rese`). Four threads — generative visuomotor
policies (Diffusion Policy, Flow Matching, Rectified Flow, π₀, Riemannian FM), RL
fine-tuning of generative policies (DPPO, ReinFlow, DDPO, DPOK, QSM), representation
collapse & dispersion (Dispersive Loss [Wang & He], D²PPO [Zou et al.], REPA, VICReg,
Barlow Twins, DM1/MP1), and learning-based vision quadrotor control (Loquercio
*Learning High-Speed Flight in the Wild*, Kaufmann *Swift*). D²PPO's RoboMimic gains
(+22.7 %/+26.1 %) are cited as the exact claim §5–§6 falsify on this task.
- **Direct D²PPO-claim-vs-our-null contrast ✓ DONE (2026-06-19):** the §2
  representation-collapse paragraph now juxtaposes D²PPO's +22.7 %/+26.1 % RoboMimic
  gains against our +1.1 pp Tier-1 (inside 4.2 pp seed std) + frozen-encoder
  byte-identical no-op, attributing the gap to setting (50 Hz closed-loop flight vs.
  quasi-static manipulation pre-training) and forward-pointing to the §6 diagnosis.
- **6 formerly-† refs (Fan/DPOK, QSM, REPA, VICReg, Barlow Twins, DM1) ✓ VERIFIED
  (2026-06-19)** against publisher of record (arXiv / official proceedings); all †
  markers removed from the draft. Corrections applied: DPOK title now leads "DPOK:" +
  arXiv:2305.16381; QSM → *ICML* 2024, arXiv:2312.11752; REPA → *ICLR* 2025 (Oral),
  arXiv:2410.06940; VICReg arXiv:2105.04906; Barlow Twins arXiv:2103.03230; DM1 authors
  = Zou et al. (same group as D²PPO [14]), arXiv:2510.07865. No unverified entries remain.
- **GOTCHA:** the notebook source *"Resolving Policy Collapse and Representation Decay in
  Generative Robot Control"* is a user/AI-written Markdown synthesis note, **NOT a
  citable paper** — deliberately excluded from References.

**Draft status (Draft v0.3, 2026-06-19):** Title FINALIZED; abstract + intro POLISHED;
venue DEFERRED (header "Target: ICRA / robot-learning workshop (venue TBD)").
**P1 baselines DONE (3 seeds) + folded into Table 1** — BC-vision-only
(9.2 ± 1.8 % oracle, Tier-1 61.1 ± 13.4 %) and PPO-from-pixels (collapse-prone:
Tier-1 15.6 ± 22.0 %, 2/3 seeds collapse to n_cond 0/30, 1/3 reaches Tier-1 46.7 %)
added as rows ᴮ (mean ± std over seeds {0,1,2}) plus a diagnosis paragraph: plain
vision-BC sits in the IMU+flow policies' band (→ policy sophistication ≠ bottleneck)
and pixel-PPO's collapsed-seed low all-IAE is a clean in-the-wild short-survival-artifact
demo for §4. **Seed 0 was unrepresentative** — its BC Tier-1 80 % and uniform PPO
collapse (Tier-1 0 %) were single-seed luck; the 3-seed means are the reported numbers.
Artifacts table + Reproducibility updated; seed driver `scripts/run_baseline_seeds_1_2.sh`,
aggregate `evaluation_results/baselines_frozen_seeds_aggregate.json`.

**TODO before submission:** finalize venue (user-deferred). All prior content TODOs
cleared — refs verified, abstract/intro polished, P1 baselines trained (3 seeds, mean ±
std) + folded.

---

## Artifacts & scripts
| File | Purpose |
|------|---------|
| `scripts/evaluate_frozen_p0.py` | Frozen protocol + measured oracle (P0) |
| `scripts/train_flow_v5.py` | Flow BC; `--lambda-disp`, `--freeze-vision`, `--seed`, `--tag` |
| `scripts/run_p2_ablation.py` | Sequential 2×2 × seeds sweep + manifest |
| `scripts/evaluate_p2_ablation.py` | Aggregate cells by Tier1%/survival |
| `scripts/measure_feature_collapse.py` | Phase 3a: `vis_pooled` rank/variance/dispersion per cell |
| `evaluation_results/p2_feature_collapse.json` | Phase 3a collapse metrics (per-run + by-cell) |
| `docs/experiment_report_feature_collapse.md` | Phase 3a diagnosis write-up |
| `docs/experiment_report_survival_movers.md` | Phase 3c: recipe vs E2E vs Dispersive survival attribution |
| `scripts/measure_ood_coverage.py` | Phase 3b: training vs closed-loop pos-error coverage probe |
| `evaluation_results/p3b_ood_coverage.json` | Phase 3b OOD-coverage artifact |
| `docs/experiment_report_ood_coverage.md` | Phase 3b OOD-coverage diagnosis write-up |
| `scripts/measure_image_distance_info.py` | Phase 3b: FPV image→distance information probe (data vs info gated) |
| `evaluation_results/p3b_image_distance_info{,_nodr}.json` | Phase 3b image-distance-info artifacts (DR on/off) |
| `docs/experiment_report_image_distance_info.md` | Phase 3b info-gated verdict write-up |
| `scripts/run_p3b_retrain.py` | Phase 3b wider-init retrain driver (built; NOT recommended to run — info-gated) |
| `scripts/measure_higher_res_gate.py` | Phase 3b-sensing: free higher-res / target-saturation gate (image→distance R², dual ridge) |
| `evaluation_results/p3b_higher_res_gate.json` | Phase 3b-sensing higher-res gate artifact |
| `scripts/run_p3b_rangecue.py` | Phase 3b-sensing: range-cue positive-control ablation driver (5 arms; control reuses p2_D0E1) |
| `evaluation_results/p3b_rc_{clean,noised}{,_s12}_frozen.json` | Phase 3b-sensing range-cue frozen-P0 results (3 seeds) |
| `docs/experiment_report_sensing_ablation.md` | Phase 3b-sensing write-up — "change the observation" REFUTED; precision is coverage/competence-gated |
| `docs/paper_negative_result_draft.md` | **Negative-result paper draft** — full structure: P0 protocol + P2 ablation + 3a/3b/3c diagnostics + §2 Related Work + References [1]–[21] + Fig 1–4. §6.3 revised (2026-06-21): "sensing-gated" → coverage/teacher-competence-gated, via the higher-res gate + range-cue intervention |
| `scripts/make_paper_figures.py` | Regenerates the paper figures from the JSON artifacts (publication quality) |
| `docs/figures/rank_survival_decoupling.png` | Fig 1 — rank↔survival decoupling (§6.1) |
| `docs/figures/crosshair_distance_saturation.png` | Fig 2 — FPV range-cue saturation + image→distance decode (§6.3, measurement) |
| `docs/figures/sensing_ablation.png` | Fig 4 — higher-res gate + range-cue intervention (§6.3, verdict: coverage/teacher-competence-gated) |
| `scripts/train_bc_vision_only.py` | BC-vision-only lower-bound baseline |
| `scripts/evaluate_baselines_frozen.py` | Generic frozen rollout for non-flow baselines |
| `evaluation_results/frozen_p0_leaderboard.json` | Canonical frozen leaderboard |
| `evaluation_results/p2_ablation_manifest.json` | Sweep run → checkpoint map |

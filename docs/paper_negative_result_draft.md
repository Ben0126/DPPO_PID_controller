# Representation Collapse Is Not the Bottleneck: A Negative Result and Diagnosis for Vision-Based Quadrotor Hover

**Draft v0.3 — 2026-06-19.** Target: ICRA / robot-learning
workshop (venue TBD). **Simulation-only; no real-robot claim.** This draft is assembled from the
frozen-protocol leaderboard and the Phase 3 diagnostic reports; every number is
reproducible from the cited script/artifact.

---

## Abstract

Diffusion / flow-matching visual policies are prone to *representation collapse* when
fine-tuned end-to-end, and **Dispersive Loss** — a contrastive regulariser that repels
intermediate features — has been proposed to prevent it and thereby improve
high-frequency visual control. We pre-registered and tested this hypothesis on a
vision-based quadrotor hover task (monocular 64×64 FPV + IMU, flow-matching policy,
50 Hz closed loop). Under a **frozen evaluation protocol** (paired initial conditions,
conditional-on-survival precision, bootstrap CIs, across-seed mean ± std, a *measured*
state-based oracle), a
**2×2 ablation** (Dispersive × end-to-end encoder, three seeds per cell) finds that
**Dispersive Loss yields no survival or task-precision gain above seed noise** (+1.1 pp
Tier-1 within a 4.2 pp pooled std), and with a frozen encoder it is **byte-identical**
to its absence.
Reaching even this null verdict required a protocol hardened against **single-seed
noise**: our own from-pixels PPO baseline swings between a 0 % and 47 % Tier-1 pass-rate
on the training seed alone, so we report every retrained model as an **across-seed
mean ± std** rather than a single leaderboard row.
We then diagnose *why*: (i) Dispersive does not cure collapse — it **games its own
objective**, inflating feature norm ~287× while the intrinsic rank gets *worse*
(effective rank 9→2), and closed-loop survival is **decoupled** from that rank (a 15×
rank swing leaves survival flat); (ii) the only real survival mover is a
transfer/conditioning/recovery **recipe**, not Dispersive; (iii) precision is gated
neither by representation nor by sensing. Although the 64×64 FPV cannot encode metric
range beyond ~2 m, a renderer gate shows that loss is a fixable target artifact (a
non-saturating target restores it at 64 px), and a positive-control **intervention**
settles it: *handing the policy the oracle metric position error barely moves precision*
(~36× the state oracle) and a richer cue **collapses survival** — so precision is capped
by the absence of a learned far-range recovery behaviour (a coverage / teacher-competence
gap), not by the observation channel. We conclude that for this class of task neither
representation collapse nor sensing is the binding constraint. We release the protocol,
the intervention, and all diagnostics.

---

## 1. Introduction

End-to-end visual control with generative policies (diffusion [3], flow matching [4])
has advanced rapidly, but a recurring failure mode is *representation collapse*: when
the visual encoder is trained jointly with the action head, its features can degenerate
onto a low-dimensional subspace, losing the information the controller needs.
**Dispersive Loss** [13] addresses this by adding a contrastive term that repels a
batch's intermediate features, and has been reported to improve generative-policy
control [14].

We asked a simple, falsifiable question on a concrete task — **vision-based quadrotor
hover** — does Dispersive Loss improve closed-loop control, and does it need an
end-to-end-trainable encoder to do so? The task is a good stress test: the policy runs
at 50 Hz from a monocular 64×64 FPV stream plus IMU, with a flow-matching action head;
covariate shift and feature collapse are both plausible failure modes.

Our answer is **no**, and the value of the paper is the *diagnosis* of why. We make
three contributions:

1. **A frozen evaluation protocol** for visual hover that is robust to the
   short-survival *and* single-seed artifacts that silently mislead precision metrics and
   leaderboard rows for policies that crash early or train unstably (paired seeds,
   conditional-on-survival precision, bootstrap CIs, across-seed mean ± std, and a
   *measured* rather than assumed oracle).
2. **A pre-registered 2×2 falsification** (three seeds per cell) of the Dispersive-Loss
   hypothesis, including a byte-identical no-op control.
3. **A mechanistic diagnosis** separating the three candidate bottlenecks
   (representation, data coverage, sensing) and localising the real one — including a
   positive-control *intervention* that hands the policy the oracle sensing signal and
   shows it does **not** restore precision, isolating the teacher's far-range
   incompetence as the binding constraint.

This is a negative result, but a constructive one: it redirects effort from
representation regularisation to the observation model.

---

## 2. Related Work

**Generative visuomotor policies.** Denoising diffusion models [1] and their
deterministic DDIM sampler [2] made expressive, multimodal generative modelling
practical, and Diffusion Policy [3] adapted this to visuomotor imitation by casting
action prediction as conditional action diffusion. Flow matching [4] and rectified
flow [5] provide a simulation-free, ODE-based alternative that trains a velocity field
along straight transport paths; both have since been used as robot-control policies —
e.g. the π₀ vision-language-action flow model [6] and the Riemannian flow-matching
policy for motion learning [7]. Our policy is a flow-matching action head over a fused
vision–IMU conditioning vector, placing it in this family.

**RL fine-tuning of generative policies.** Behaviour-cloned generative policies inherit
the demonstrator's covariate shift, motivating RL fine-tuning. DPPO [8] treats the
multi-step denoising chain as an MDP and fine-tunes a pre-trained diffusion policy with
PPO, reporting stable, on-manifold exploration for continuous control. ReinFlow [9]
extends this to flow policies by injecting *learnable, bounded noise* into the otherwise
deterministic flow ODE, turning it into a discrete-time SDE with closed-form transition
probabilities so that exact log-likelihoods — and hence policy gradients — are available
at very few denoising steps; Q-score matching [12] is a related value-based route. This
interactive, continuous-control setting should be distinguished from RL fine-tuning of
*image-generation* diffusion models (DDPO [10]; DPOK / Fan et al. [11]), which optimise
a non-interactive sampler against a reward and do not face closed-loop dynamics. Our RL
stage (ReinFlow with positive-advantage masking) sits squarely in the former group.

**Representation collapse and dispersive regularization.** Generative models trained
with a pure regression objective lack explicit pressure on their internal features,
which can collapse semantically distinct observations onto near-identical
representations. Dispersive Loss [13] counters this with a "contrastive loss without
positive pairs" — keeping only the repulsion term so that a batch's intermediate
features spread out, with no positive pairs, augmentation, or extra encoder. D²PPO [14]
imports this regulariser into diffusion-policy *pre-training*, reporting large
manipulation gains (≈ +22.7 % success after pre-training and +26.1 % after PPO
fine-tuning on RoboMimic) attributed to relieving representation collapse — the precise
claim our task is designed to test. Related anti-collapse mechanisms include
representation alignment to a pretrained encoder (REPA [15]) and the classic
variance/covariance and redundancy-reduction objectives of VICReg [16] and Barlow
Twins [17]; concurrent work also pairs dispersive-style regularisation with MeanFlow for
one-step manipulation (DM1 [18], MP1 [19]). We adopt Dispersive Loss exactly as
specified [13, 14] and find, on our task, no closed-loop benefit (§5–§6): against
D²PPO's reported +22.7 % / +26.1 % manipulation-success gains on RoboMimic [14], we
measure a +1.1 pp Tier-1 change that sits inside the 4.2 pp across-seed std, with
survival and precision unmoved — and, with a frozen encoder, a byte-identical no-op.
We attribute the gap to the setting (interactive 50 Hz closed-loop flight vs.
quasi-static manipulation pre-training) and diagnose it mechanistically in §6.

**Learning-based vision quadrotor control.** End-to-end learned controllers have pushed
agile flight well past classical sense-map-plan pipelines: Loquercio et al. [20] fly a
quadrotor through cluttered wild environments at high speed from onboard vision, trained
in simulation via *privileged learning* (imitating a state-privileged teacher), and the
Swift system [21] beats human champions at drone racing with deep RL on a vision-based
policy. A common thread is decoupling perception from control via a privileged teacher —
the same paradigm as our state-based PPO oracle (§3). Relative to these
agility/navigation results, our task isolates a different axis: closed-loop *hover
precision* under an information-poor monocular observation — where, by intervention
(§6.3), the binding constraint turns out to be the privileged teacher's own
incompetence in the operating regime, not the sensing channel.

---

## 3. Task, Policy, and the Dispersive Mechanism

**Task.** Hover a quadrotor at a target. The simulator is a 6-DOF rigid body (200 Hz
RK4 inner loop, INDI attitude control) with a 50 Hz outer control loop. The agent
observes a 2-frame stack of 64×64 RGB synthetic FPV images and a 6-D IMU vector
(gyro + specific force); it outputs a short sequence of collective-thrust/body-rate
(CTBR) commands, executing the first and re-observing (receding horizon).

**Policy (flow matching, v5).** A CNN vision encoder produces a pooled feature
`vis_pooled ∈ R^256` and a spatial map `R^{256×4×4}`. An IMU MLP produces
`imu_feat ∈ R^512`. An IMU-to-vision cross-attention (IMU query, spatial keys/values)
produces `attended ∈ R^256`. The flow network conditions on
`global_cond = [attended; imu_feat] ∈ R^768`. A training-only auxiliary head predicts
the 15-D privileged state from `vis_pooled`.

**Dispersive Loss.** Applied to `vis_pooled`:
`L_disp = −mean_{i≠j} log(‖x_i − x_j‖ + ε)`, added with weight λ=0.05. It pushes the
batch's pooled features apart. **Note (important for §6.1):** `vis_pooled` feeds only
the auxiliary state head; the *action* path conditions on `attended` (from the spatial
map) and `imu_feat`. Dispersive therefore reaches the action path only indirectly,
through the shared convolutional trunk.

**Oracle.** A state-based PPO controller reading the privileged 15-D state is the
performance upper bound.

---

## 4. Frozen Evaluation Protocol

The project's metric had changed three times (RMSE → linear-clip hierarchical →
exp-decay hierarchical), making cross-run numbers incomparable; a **short-survival
artifact** kept inflating precision for policies that crash early (a policy that dies at
step 24 was credited with a "precise" 0.6 m hover from its few alive steps); and, as we
demonstrate below with our own PPO baseline, **single-seed evaluation is itself
unreliable** — a collapse-prone policy can swing from Tier-1 0 % to 47 % on the training
seed alone. We freeze one protocol (`evaluate_frozen_p0.py`) and do not change it:

- **Paired initial conditions.** Episode *i* uses `seed = 12345 + i` to seed the env,
  the global RNG (visual domain randomisation), and torch (flow noise), so every model
  sees byte-identical initial conditions — a paired comparison.
- **Primary axes: survival and Tier-1 pass-rate** (fraction of 30 episodes flying ≥
  250/500 steps). These are robust to the short-survival artifact.
- **Conditional precision.** Integrated absolute error (IAE) of position is reported
  **only over episodes that survived ≥ 250 steps** (`cond-IAE`, with its support
  `n_cond`); the naive all-episode IAE is reported but flagged.
- **Bootstrap 95% CIs** over episodes.
- **Across-seed mean ± std** for every model we *retrain* in this work — Table 1's P1
  baselines and Table 2's 2×2 cells — over three training seeds. The legacy flow-policy
  checkpoints predate this work and are single-seed; that is a stated limitation, and the
  seed sensitivity we document below is exactly why a single training run is not a safe
  basis for a leaderboard row.
- **Measured oracle.** The state-based PPO oracle is rolled through the *same* protocol;
  its composite score (**0.9668**, 100 % survival, 0.068 m hover IAE) is the
  normaliser (`%Oracle`), replacing a previously hard-coded constant.

Under this protocol, **three single-seed claims fail to survive.** Two come from the
prior leaderboard: "H4 = SOTA" (true only on a multiplicative composite that rewards
*die-early-but-precise*; H4 survives 40.8 %, Tier-1 13.3 %) and "RL gives 51 % precision
gain" (a short-survival artifact: its conditional IAE is the *worst* of the group,
3.34 m, on n_cond = 4/30). The third we generate ourselves and catch with the seed
protocol: on a single seed our PPO-from-pixels baseline "uniformly collapses"
(Tier-1 0 %), but over three seeds it is only *collapse-prone* — a high-variance Tier-1
of **15.6 ± 22.0 %**, with one seed reaching 47 % — so the single-seed verdict
simultaneously over- and under-states the behaviour. The lesson is structural, not
incidental: on this task one training seed is not a safe basis for a leaderboard row,
which is why every model we retrain is reported as a seed mean ± std.
**No vision policy exceeds ~17 % of oracle; none is deployable.** (Table 1.)

**Table 1 — Frozen leaderboard** (flow policies: `evaluation_results/frozen_p0_leaderboard.json`;
P1 baselines ᴮ, 3 seeds: `evaluation_results/baselines_frozen_leaderboard.json`,
`evaluation_results/baselines_frozen_ppopx.json`, per-seed
`…_{bc,ppopx}_s{1,2}.json`, aggregate `…_seeds_aggregate.json`).

| Model | Score | Survive | Tier-1 | cond-IAE (n) | %Oracle |
|-------|------:|--------:|-------:|-------------:|--------:|
| PPO Oracle (state) | 0.967 [.966,.967] | 100 % | 100 % | 0.068 m (30) | 100 % |
| H4_BC | 0.168 [.154,.182] | 40.8 % | 13.3 % | 2.520 m (4)⚠ | 17.4 % |
| v5_RL_best | 0.129 [.117,.139] | 28.8 % | 13.3 % | 3.340 m (4)⚠ | 13.3 % |
| v5_BC | 0.124 [.107,.143] | 55.9 % | 70.0 % | 2.724 m (21) | 12.8 % |
| PPO-from-pixels ᴮ | 0.114 ± 0.017 | 30.1 ± 15.6 % | 15.6 ± 22.0 % | 2.83 m (1/3)⚠⚠ | 11.8 ± 1.8 % |
| Joint_E2E_v5 | 0.106 [.095,.118] | 62.2 % | 80.0 % | 3.077 m (24) | 11.0 % |
| BC-vision-only ᴮ | 0.089 ± 0.018 | 54.7 ± 3.4 % | 61.1 ± 13.4 % | 3.30 ± 0.36 m (18) | 9.2 ± 1.8 % |

⚠ n_cond = 4: cond-IAE unreliable; the low all-IAE is a short-survival artifact.
⚠⚠ PPO-from-pixels is collapse-prone, not uniformly collapsed: 2 of 3 seeds survive
0/30 past 250 steps (n_cond = 0, cond-IAE undefined; their low all-IAE 0.70–0.87 m is a
pure short-survival artifact, crashing ~75 steps), while 1 seed reaches Tier-1 46.7 %
(cond-IAE 2.83 m). The cond-IAE shown is that single surviving seed (1/3).
ᴮ P1 baseline (this work): **mean ± std over 3 seeds {0,1,2}** (population std, matching
Table 2); the bracketed [.,.] on the other rows are bootstrap 95 % CIs over a single
checkpoint's episodes. BC-vision-only = no IMU, no flow (plain vision→action MLP);
PPO-from-pixels = end-to-end RL from pixels (1 M steps, CTBR).

**The two P1 baselines (3 seeds each) sit within the flow-policy band and reinforce the
diagnosis.** A plain vision→action regressor (BC-vision-only: no IMU, no flow) already
reaches **9.2 ± 1.8 % of oracle at Tier-1 61.1 ± 13.4 %** — *on par with* the IMU-fused
flow policies v5_BC (12.8 %, Tier-1 70 %) and Joint_E2E_v5 (11.0 %, Tier-1 80 %), well
within the wide across-seed spread. Flow modelling and IMU fusion thus buy little in
survival on this task: **policy sophistication is not the bottleneck.** The other
baseline, PPO-from-pixels, is the collapse-prone case dissected above; here it doubles as
a clean short-survival-artifact demonstration. On its two collapsed seeds the deceptively
low *all*-IAE (0.70–0.87 m) is a textbook instance of what the §4 protocol is built to
expose — they "look precise" only because they crash (~75 steps) before they can drift —
whereas averaging over seeds raises the all-IAE to 1.41 ± 0.89 m once the surviving
seed's drift is counted. Both baselines stay far below the 17 % ceiling; none is
deployable.

![Figure 3](figures/single_seed_swing.png)

**Figure 3.** Per-seed Tier-1 pass-rate (left) and survival (right) for the two P1
baselines, three training seeds each (dots, labelled s0–s2), with the across-seed
mean ± std (diamond). PPO-from-pixels swings from 0 % (seeds 0, 2 — would read as
"uniformly collapsed") to 47 % (seed 1) on Tier-1 from the training seed alone
(15.6 ± 22.0 %), so any single-seed leaderboard row simultaneously over- and
under-states it; BC-vision-only is comparatively stable (61.1 ± 13.4 %). This seed
sensitivity is why every retrained model in this work is reported as a three-seed
mean ± std (`scripts/make_paper_figures.py`,
`evaluation_results/baselines_frozen_seeds_aggregate.json`).

---

## 5. The Core Ablation: Dispersive × E2E (2×2, 3 seeds)

We vary two factors, holding everything else fixed (H4-transfer init, 500 hover + 500
recovery demos, 80 epochs, lr 1e-4, batch 256, task-conditioned):

- **Dispersive**: λ=0.05 (ON) vs 0.0 (OFF), on `vis_pooled`.
- **E2E**: vision encoder trainable (ON) vs frozen at the transferred init (OFF).

**Mechanistic prediction (pre-registered).** Dispersive acts on `vis_pooled`; with a
frozen encoder its gradient touches no trainable parameter, so **D1E0 must equal D0E0**.
The decisive comparison is therefore **D1E1 vs D0E1** (Dispersive ON vs OFF, both E2E).

**Table 2 — 2×2 ablation** (`evaluate_p2_ablation.py`, 3 seeds, frozen protocol).

| Cell | Disp / E2E | Tier-1 (mean±std) | Survival | cond-IAE | %Oracle |
|------|-----------|------------------:|---------:|---------:|--------:|
| D0E0 | OFF / frozen | 87.8 ± 3.1 | 66.1 ± 3.7 | 2.93 m | 13.4 % |
| D1E0 | ON / frozen  | **87.8 ± 3.1** | **66.1 ± 3.7** | 2.93 m | 13.4 % |
| D0E1 | OFF / E2E    | 92.2 ± 3.1 | 65.0 ± 2.8 | 2.91 m | 12.4 % |
| D1E1 | ON / E2E     | 93.3 ± 2.7 | 65.0 ± 2.5 | 2.81 m | 13.2 % |

**Findings.**
- **Dispersive is not supported.** D1E1 vs D0E1: Tier-1 **+1.1 pp inside the 4.2 pp
  pooled across-seed std**; survival identical (65.0); precision unchanged.
- **Byte-identical no-op.** D1E0 ≡ D0E0: the trained `best_model.pt` files are
  **MD5-identical across all three seeds** — direct confirmation that, with a frozen
  encoder, "Dispersive ON" is bit-for-bit "Dispersive OFF".
- The decision rule (Dispersive supported iff D1E1 beats D0E1 on Tier-1/survival by more
  than the pooled std) returns **NOT supported**.

All four cells beat the prior frontier Joint_E2E_v5 (Tier-1 80 %), but §6.2 shows that
gain is the *recipe*, not Dispersive. Precision is unmoved (cond-IAE 2.8–2.9 m, ~13 %
oracle): **nothing is deployable.**

---

## 6. Diagnosis: Which Bottleneck?

Three candidate explanations for the flat result: representation (the thing Dispersive
targets), data coverage, or sensing. We test the first two with observation-only
measurements (no retraining), and the third with both a renderer gate and a
positive-control **intervention** that hands the policy the very signal it is presumed to
lack (§6.3).

### 6.1 Representation: Dispersive games its objective; survival ⟂ rank

We push a fixed 4000-image hover+recovery batch through each checkpoint's encoder and
measure the geometry of `vis_pooled` (D=256). (`measure_feature_collapse.py`.)

**Table 3 — feature geometry** (mean over 3 seeds).

| Cell | eff_rank | dims for 99% var | feat_norm | L_disp | mean cos |
|------|---------:|-----------------:|----------:|-------:|---------:|
| D0E0 / D1E0 (frozen) | 30.3 | 91 | 1.80 | −0.81 | 0.071 |
| D0E1 (E2E) | 9.0 | 45 | 11.42 | −1.23 | **0.956** |
| D1E1 (E2E + Disp) | **2.0** | **2** | **3281** | **−8.14** | 0.013 |

Three things follow. **(i)** The frozen no-op is confirmed at the feature level (D1E0 ≡
D0E0 to 4 d.p.). **(ii)** Collapse is *real*: naive E2E (D0E1) drops effective rank
30→9 and drives mean pairwise cosine to 0.96 (features nearly colinear) — there *was*
something for Dispersive to fix. **(iii)** Dispersive does not fix it; it **games its
objective**. It minimises `L_disp` (−1.23→−8.14) and inflates pairwise distance ~1270×,
but does so purely by inflating the feature norm ~287× (11.4→3281); the *intrinsic*
dimensionality gets **worse** (effective rank 9→2; 99.8 % of variance on 2 dims). The
"healthy-looking" low cosine (0.013) is a 2-D-at-huge-radius artifact, not high-
dimensional spread. Because `vis_pooled` feeds only the auxiliary head (§3), the
cheapest way to satisfy the distance objective is to inflate the pooled-specific
projection, leaving the action-relevant spatial pathway comparatively untouched.

**The decisive plot is rank vs survival (Figure 1).** Effective rank swings **15×**
(30→9→2) while closed-loop survival stays flat (66.1→65.0→65.0 %, within ~3 pp std).
**Survival is decoupled from `vis_pooled` rank.** Representation collapse — neither preventing it
(D0E0 rank 30) nor worsening it (D1E1 rank 2) — moves the closed-loop outcome. This is
the strongest statement of the result: representation collapse is *not* the binding
constraint.

![Figure 1](figures/rank_survival_decoupling.png)

**Figure 1.** `vis_pooled` effective rank (bars, log axis) collapses ~15× across the
2×2 cells (30.3 frozen → 9.0 E2E → 2.0 E2E+Dispersive), while closed-loop survival and
Tier-1 pass-rate (lines, right axis; error bars = across-seed std) stay flat. Survival
is decoupled from the representation geometry Dispersive targets.

### 6.2 The only survival mover is the recipe, not Dispersive (or E2E)

Decomposing the gain over the prior frontier as controlled steps
(`experiment_report_survival_movers.md`):

| Step | Δ Tier-1 | Δ Survival | cond-IAE |
|------|---------:|-----------:|---------:|
| prior frontier Joint_E2E_v5 → **recipe** (frozen cell D0E0) | **+7.8 pp** | **+3.9 pp** | 2.93 m |
| + E2E (D0E0 → D0E1) | +4.4 pp | −1.1 pp | 2.91 m |
| + Dispersive (D0E1 → D1E1) | +1.1 pp (∈ noise) | 0.0 | 2.81 m |

The dominant lift is the **recipe** (H4-transfer init + task-conditioning + recovery
mix), and it holds with a **frozen** encoder — so it is not from E2E. E2E adds a small
Tier-1-only bump that neither extends survival nor improves precision. Dispersive adds
nothing. **Mover ranking: recipe ≫ E2E ≫ Dispersive ≈ 0.**

### 6.3 Precision is coverage/teacher-competence-limited, not sensing-limited

cond-IAE is ~2.8 m (~13 % oracle) for *every* configuration. Why won't precision move?

**Coverage probe** (`measure_ood_coverage.py`). Comparing position-error magnitude in
the BC data vs the closed-loop steady state: training is overwhelmingly at-target
(95.8 % of timesteps within 0.2 m; p99 = 0.62 m), while the surviving policy spends
~90 % of its steady state beyond 1 m and 46 % beyond 3 m. **97.3 %** of steady-state
samples lie above the training p99; **25 %** above the training maximum (zero-coverage).
The policy lives where it has ~no labels — a real coverage gap, *consistent* with either
a data limit or a sensing limit.

**Is the coverage gap data-fixable, and does the FPV even encode range?** Three
observation-only checks bear on it:
1. **Widening the recovery init does not change coverage.** The env anchors
   `target = init_pos` for hover, so a wider `--pos-range` just relocates the hover
   point; a freshly collected ±3 m dataset has *identical* position-error coverage to
   the ±1 m one (both 0.4 % of timesteps beyond 1 m).
2. **The expert itself cannot recover from > 2 m offset** (target ≠ init): 1 m and 2 m
   recover to 0.066 m (20/20), but 3 m crashes (0/20). Even a correct offset collection
   tops out at ~2 m — below the precision-limiting regime.
3. **The FPV image does not encode metric range** (`measure_image_distance_info.py`;
   **Figure 2**). The only range cue is the target crosshair *size*,
   `size = max(2, min(6, ⌊6/(d+0.5)⌋ + dr))`. With domain randomisation, adjacent-
   distance separability is d′ < 0.2 everywhere, and a linear decode of distance from
   the image has R² = 0.41 near (< 1 m) but only **0.12 far (≥ 1.5 m)**. With DR off
   (noiseless ceiling) the size **saturates at 2 px for all d ≥ 2 m**, so renders at
   2.0, 2.5, and 3.0 m are **byte-identical** (d′ = 0). The crosshair *position* encodes
   only normalised direction (range-invariant).

The policy's 2.83 m steady drift sits exactly where the observation carries **no
recoverable metric range** (Figure 2). It is tempting to stop here and call precision
*information-gated by the observation model* — the conclusion of an earlier draft. But a
measurement that the FPV does not encode range does not establish that the missing range
is the *binding* cause of imprecision. We test that implication directly with a renderer
gate and a positive-control intervention, and **both refute it**.

![Figure 2](figures/crosshair_distance_saturation.png)

**Figure 2.** *Left:* the only FPV range cue (target crosshair size) saturates at the
2 px floor beyond ~2 m — under domain randomisation (red band = min–max) the size is
uninformative even nearer, and with DR off (dashed) renders at 2.0/2.5/3.0 m are
byte-identical. The policy's steady-state drift (green, median 2.83 m) sits inside the
saturated region. *Right:* a linear decode of distance from the image succeeds near
(R²=0.41, <1 m) but fails far (R²=0.12, ≥1.5 m). This is a *measurement* of the
observation, not the binding cause of imprecision (Figure 4).

**The information loss is a renderer artifact, not the pixel count (gate).** Before
treating "richer sensing" as the fix, we ask whether a more capable monocular renderer
would even *carry* the far-range information (`measure_higher_res_gate.py`; **Figure 4,
left**). Decoding distance from the image across {64, 128, 256} px × {production
saturating crosshair, perspective non-saturating target}, the saturating crosshair's
far-range R² is ≈ 0 at *every* resolution (−0.01 / 0.11 / 0.04) — raising resolution
alone does nothing — whereas a perspective (optical-expansion) target lifts far R² to
**0.42 at the same 64 px** (0.50 / 0.45 at 128 / 256 px). So the §6.3 measurement is
partly an artifact of this synthetic renderer's quantised crosshair, fixable without any
resolution change; even so, the perspective target's far R² (0.45) ≪ its near R² (0.88),
so a richer sensor would *improve*, not solve, far-range precision.

**Even the oracle range cue does not move precision (intervention).** The decisive test
supplies the missing channel directly: we fold the metric body-frame position error the
FPV cannot encode (`states[:, :3]`, computed from the existing data — no re-collection)
into the policy's conditioning, retrain the D0E1 frontier recipe unchanged, and re-eval
under the frozen protocol (3 seeds; control = the no-cue D0E1; **Figure 4, right**).
Handed the *oracle* metric range (scalar ‖pos-err‖, σ=0), cond-IAE moves only
2.91 → **2.43 m** — still ~36× the 0.068 m state oracle, and bought with −6.7 pp survival
/ −13 pp Tier-1. A realisable sensor (σ=0.15 m) erases even that (2.81 m ≈ control). The
*richer* full 3-D position cue is actively harmful: it **collapses survival across all
three seeds** (Tier-1 92 → 7 %, surviving to the 250-step threshold in ≈2/30 episodes),
so its conditional precision is an artifact, not a win. Supplying range tells the drone
*how far off* it is, but not *what to do* about it — and no demonstration in the 1–3 m
band teaches that.

**Verdict: precision is not sensing-gated.** Range information — even oracle, even full
position — does not restore precision, and a richer cue harms survival. The binding
constraint is the absence of a learned far-range *recovery behaviour* in the 1–3 m band:
the BC data has no labels there (coverage gap, above) and the teacher cannot generate
them (the expert itself crashes from > 2 m, check 2). Moving precision would require a
*competent far-range teacher* to generate 1–3 m coverage — not a better sensor, not a
better policy over the existing data, and not representation regularisation. A wider-init
BC retrain is therefore still predicted to leave cond-IAE at ~2.8 m and is not pursued.

![Figure 4](figures/sensing_ablation.png)

**Figure 4.** *Left (gate):* far-range (≥1.5 m) image→distance R² for three resolutions ×
two target renderers. The saturating production crosshair is uninformative at every
resolution (raising pixels does nothing); a non-saturating perspective target restores
~half the far-range information already at 64 px — the loss is a target artifact, not the
pixel count. *Right (intervention):* closed-loop precision (cond-IAE, 3 seeds) when the
policy is *handed* the metric position error the FPV lacks. Even the oracle scalar range
(σ=0) leaves precision ~36× the state oracle (green dashed, 0.068 m); sensor noise erases
the gain; the richer 3-D cue (σ=0) collapses survival (hatched; conditional IAE
unreliable). Precision is coverage/teacher-competence-gated, not sensing-gated.

---

## 7. Discussion

**What generalises.** (a) The frozen protocol and the short-survival artifact it fixes
are reusable for any survival-vs-precision visual control benchmark. (b) The
"objective-gaming" pathology of a *scale-sensitive* dispersion regulariser on an
*off-path* feature is a general caution: a distance-maximising contrastive term can be
trivially satisfied by norm inflation, *worsening* intrinsic rank; one should regularise
the feature the policy actually conditions on, and use a scale-invariant criterion.
(c) The rank-vs-survival decoupling argues that "prevent representation collapse" is not
automatically the right lever for closed-loop control — the binding constraint must be
located empirically.

**What is specific.** The precision diagnosis (§6.3) is specific to a setting whose
privileged teacher saturates inside the operating regime: our state-based PPO oracle
itself crashes from > 2 m offset, so it cannot label the 1–3 m band where the policy
drifts. A task with a teacher competent over the full error range — or a renderer whose
distance cue is not the quantised crosshair we found to be artifactually saturating —
could place the bottleneck elsewhere.

**Constructive implication.** Improving metric precision here is neither a representation
nor a sensing problem: our intervention shows that even the *oracle* range, handed
directly to the policy, barely moves precision and a richer cue collapses survival
(§6.3). The lever is the *teacher and its coverage* — a controller that recovers from
multi-metre offsets, generating 1–3 m demonstrations to imitate. A better sensor would
help only once such recovery behaviour exists to be conditioned on; absent it, higher
resolution, stereo/depth, representation regularisation, and wider naive BC coverage are
all predicted not to move the ~2.8 m floor.

---

## 8. Limitations

- **Simulation only**; no real-robot validation. The renderer is synthetic and its
  distance encoding is a design choice (a deliberately information-poor monocular cue) —
  though the §6.3 intervention controls for this by handing the policy the oracle range
  directly, so the precision conclusion does not hinge on the renderer's information
  content.
- The precision verdict is **conditional on the teacher**: our state-based PPO oracle
  itself crashes from > 2 m offset, so "coverage/teacher-competence-gated" is established
  for *this* teacher; a controller competent over the full error range is the untested
  lever and the recommended next step.
- **One task** (hover) and one policy family (flow matching with IMU-vision cross-
  attention). The Dispersive null result is established for `vis_pooled`; we did not
  exhaustively sweep λ, the feature it is applied to, or alternative dispersion
  criteria — though §6.1 explains mechanistically why those are unlikely to rescue it
  here.
- T_obs = 2 frames could in principle yield range via motion parallax; we argue (and the
  flat empirical precision across all configs supports) that the 28 ms baseline and slow
  drift make this far weaker than the static size channel.
- Three seeds per cell; effects are reported relative to the pooled across-seed std.

---

## 9. Conclusion

On a vision-based quadrotor hover task, under a frozen, artifact-robust protocol with a
measured oracle, **Dispersive Loss does not improve survival or precision** above seed
noise, and is a byte-identical no-op without an end-to-end encoder. The mechanism does
not even cure the collapse it targets — it games its scale-sensitive objective and
worsens intrinsic rank — and closed-loop survival is decoupled from that representation
geometry entirely. The real levers are elsewhere: a transfer/conditioning/recovery
recipe accounts for the survival gains, and **task precision is capped by the teacher's
incompetence in the operating regime**, not by sensing: a positive-control intervention
that hands the policy the oracle metric position error barely moves precision (~36× the
state oracle) and a richer cue collapses survival, and the FPV's apparent range-blindness
is itself a fixable renderer artifact. Neither representation collapse nor the sensing
channel is the binding constraint for this task; the missing far-range recovery behaviour
is. We release the protocol, the ablation, the intervention, and all diagnostics so the
result and its diagnosis are reproducible.

---

## Reproducibility / Artifacts

| Component | Script | Artifact |
|-----------|--------|----------|
| Frozen protocol + measured oracle | `scripts/evaluate_frozen_p0.py` | `evaluation_results/frozen_p0_leaderboard.json` |
| §4 P1 baselines (BC-vision-only, PPO-from-pixels), 3 seeds | `scripts/train_bc_vision_only.py`, `scripts/train_ppo_from_pixels.py` (+ `models/ppo_pixel.py`, `configs/ppo_from_pixels.yaml`), `scripts/evaluate_baselines_frozen.py`; seed driver `scripts/run_baseline_seeds_1_2.sh` | seed 0: `evaluation_results/baselines_frozen_leaderboard.json`, `…_ppopx.json`; seeds 1–2: `…_{bc,ppopx}_s{1,2}.json`; aggregate (mean±std): `…_baselines_frozen_seeds_aggregate.json` |
| 2×2 ablation sweep / aggregate | `scripts/run_p2_ablation.py`, `scripts/evaluate_p2_ablation.py` | `evaluation_results/p2_ablation_{manifest,leaderboard}.json` |
| §6.1 feature geometry | `scripts/measure_feature_collapse.py` | `evaluation_results/p2_feature_collapse.json` — `docs/experiment_report_feature_collapse.md` |
| §6.2 survival movers | — | `docs/experiment_report_survival_movers.md` |
| §6.3 coverage probe | `scripts/measure_ood_coverage.py` | `evaluation_results/p3b_ood_coverage.json` — `docs/experiment_report_ood_coverage.md` |
| §6.3 image-distance info | `scripts/measure_image_distance_info.py` | `evaluation_results/p3b_image_distance_info{,_nodr}.json` — `docs/experiment_report_image_distance_info.md` |
| §6.3 higher-res gate (Fig 4 left) | `scripts/measure_higher_res_gate.py` | `evaluation_results/p3b_higher_res_gate.json` — `docs/experiment_report_sensing_ablation.md` |
| §6.3 range-cue intervention, 3 seeds (Fig 4 right) | `scripts/run_p3b_rangecue.py` (→ `train_flow_v5.py --range-cue`, `evaluate_frozen_p0.py --cue-noise`) | `evaluation_results/p3b_rc_{clean,noised}{,_s12}_frozen.json` — `docs/experiment_report_sensing_ablation.md` |

Frozen protocol: 30 episodes, base seed 12345, σ = 2.0 exp-decay composite, paired init,
conditional-IAE over episodes surviving ≥ 250/500 steps, bootstrap 95% CI, measured PPO
oracle 0.9668.

---

## References

> Sources retrieved via NotebookLM (notebook *Generative RL & Flow Policy Research*,
> 2026-06-19). All author lists, venues, and identifiers were verified against the
> publisher of record (arXiv / official proceedings) on 2026-06-19; no entries remain
> unverified.

[1] J. Ho, A. Jain, P. Abbeel. "Denoising Diffusion Probabilistic Models." *NeurIPS*, 2020.

[2] J. Song, C. Meng, S. Ermon. "Denoising Diffusion Implicit Models." *ICLR*, 2021.

[3] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, S. Song. "Diffusion Policy:
Visuomotor Policy Learning via Action Diffusion." *Robotics: Science and Systems (RSS)*,
2023. (Extended version: *Int. J. Robotics Research*, 2024.)

[4] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, M. Le. "Flow Matching for Generative
Modeling." *ICLR*, 2023.

[5] X. Liu, C. Gong, Q. Liu. "Flow Straight and Fast: Learning to Generate and Transfer
Data with Rectified Flow." *ICLR*, 2023.

[6] K. Black, N. Brown, D. Driess, et al. (Physical Intelligence). "π₀: A
Vision-Language-Action Flow Model for General Robot Control." arXiv:2410.24164, 2024.

[7] M. Braun, N. Jaquier, L. Rozo, T. Asfour. "Riemannian Flow Matching Policy for Robot
Motion Learning." *IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS)*,
pp. 5144–5151, 2024.

[8] A. Z. Ren, J. Lidard, L. L. Ankile, A. Simeonov, P. Agrawal, A. Majumdar,
B. Burchfiel, H. Dai, M. Simchowitz. "Diffusion Policy Policy Optimization (DPPO)."
*ICLR*, 2025. arXiv:2409.00588.

[9] T. Zhang, C. Yu, S. Su, Y. Wang. "ReinFlow: Fine-tuning Flow Matching Policy with
Online Reinforcement Learning." arXiv:2505.22094, 2025.

[10] K. Black, M. Janner, Y. Du, I. Kostrikov, S. Levine. "Training Diffusion Models with
Reinforcement Learning (DDPO)." arXiv:2305.13301, 2023.

[11] Y. Fan, O. Watkins, Y. Du, et al. "DPOK: Reinforcement Learning for Fine-tuning
Text-to-Image Diffusion Models." *NeurIPS*, 2023. arXiv:2305.16381.

[12] M. Psenka, A. Escontrela, P. Abbeel, Y. Ma. "Learning a Diffusion Model Policy from
Rewards via Q-Score Matching (QSM)." *ICML*, 2024. arXiv:2312.11752.

[13] R. Wang, K. He. "Diffuse and Disperse: Image Generation with Representation
Regularization." arXiv:2506.09027, 2025. (Submitted to *ICLR* 2026.)

[14] G. Zou, W. Li, H. Wu, Y. Qian, Y. Wang, H. Wang. "D²PPO: Diffusion Policy Policy
Optimization with Dispersive Loss." *AAAI*, 2026. arXiv:2508.02644.

[15] S. Yu, S. Kwak, H. Jang, J. Jeong, J. Huang, J. Shin, S. Xie. "Representation
Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think
(REPA)." *ICLR* (Oral), 2025. arXiv:2410.06940.

[16] A. Bardes, J. Ponce, Y. LeCun. "VICReg: Variance-Invariance-Covariance
Regularization for Self-Supervised Learning." *ICLR*, 2022. arXiv:2105.04906.

[17] J. Zbontar, L. Jing, I. Misra, Y. LeCun, S. Deny. "Barlow Twins: Self-Supervised
Learning via Redundancy Reduction." *ICML*, 2021. arXiv:2103.03230.

[18] G. Zou, H. Wang, H. Wu, Y. Qian, Y. Wang, W. Li. "DM1: MeanFlow with Dispersive
Regularization for 1-Step Robotic Manipulation." arXiv:2510.07865, 2025.

[19] J. Sheng, Z. Wang, P. Li, M. Liu. "MP1: MeanFlow Tames Policy Learning in 1-step for
Robotic Manipulation." arXiv:2507.10543, 2025.

[20] A. Loquercio, E. Kaufmann, R. Ranftl, M. Müller, V. Koltun, D. Scaramuzza. "Learning
High-Speed Flight in the Wild." *Science Robotics*, 6(59), 2021.

[21] E. Kaufmann, L. Bauersfeld, A. Loquercio, M. Müller, V. Koltun, D. Scaramuzza.
"Champion-Level Drone Racing using Deep Reinforcement Learning." *Nature*, 620, 2023.

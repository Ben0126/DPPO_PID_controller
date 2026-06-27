# Representation Collapse Is Not the Bottleneck — and Neither Is Coverage or Sensing: A Negative Result and Capacity Diagnosis for Vision-Based Quadrotor Hover

**Draft v0.5 — 2026-06-27.** Target: ICRA / robot-learning
workshop (venue TBD). **Simulation-only; no real-robot claim.** Every number is reproducible
from the cited script/artifact. *Draft note (to be removed at submission):* v0.5 adds the
decisive Teacher × Observation 2×2 (§6.4) and folds in the scale-invariant-regulariser
ablation (§6.1); the diagnosis upgrades from "coverage/teacher-limited" to a triple
exclusion (representation, coverage, sensing) leaving a robustness–precision capacity
conflict, with the Abstract, §6.3, §7, §8 and §9 re-synced accordingly. The core Dispersive
ablation (§5, official-code-faithful P2f: InfoNCE-L2 on the `flow_net` mid-block, λ=0.5,
τ=0.5, `/d`) is unchanged.

---

## Abstract

Diffusion / flow-matching visual policies are prone to *representation collapse* when
fine-tuned end-to-end, and **Dispersive Loss** — a contrastive regulariser that repels
intermediate features — has been proposed to prevent it and thereby improve
high-frequency visual control. We pre-registered and tested this hypothesis on a
vision-based quadrotor hover task (monocular 64×64 FPV + IMU, flow-matching policy,
50 Hz closed loop). Under a **frozen evaluation protocol** (paired initial conditions,
conditional-on-survival precision, bootstrap CIs, across-seed mean ± std, a *measured*
state-based oracle), a **2×2 ablation** (Dispersive × end-to-end encoder, three seeds per
cell) — run with an **official-code-faithful** Dispersive Loss (InfoNCE-L2 on the
generative `flow_net` mid-block, λ=0.5, τ=0.5, including the `/d` per-dimension
normalisation) — finds **no survival or task-precision gain above seed noise** (−2.2 pp
Tier-1 within a 6.3 pp pooled std; survival −2.1 pp). Because the implementation follows
the released code exactly — placement, weight, and per-dimension normalisation — the null
cannot be attributed to a fidelity error. The frozen-encoder condition is an informative
control rather than a no-op: under the faithful mid-block placement the term trains the
generative network **even with a frozen encoder**, where it **destabilises** control
(mean Tier-1 87.8→74.4, variance up) rather than leaving it inert.
Reaching even this null verdict required a protocol hardened against **single-seed
noise**: our own from-pixels PPO baseline swings between a 0 % and 47 % Tier-1 pass-rate
on the training seed alone, so we report every retrained model as an **across-seed
mean ± std** rather than a single leaderboard row.
We then diagnose *why*. **(i)** Dispersive does not cure collapse — at **both** the faithful
`flow_net` mid-block and an off-path placement (§6.1) it **games its own objective**,
inflating feature norm (~9× and ~287× respectively) while the intrinsic rank gets *worse*;
replacing it with two **scale-invariant** criteria (unit-sphere InfoNCE, VICReg) removes the
norm inflation and genuinely raises effective rank (from 3.5 % to 75–85 % of dimensions),
yet still leaves closed-loop control flat — so survival is **decoupled** from feature
geometry whatever criterion enforces dispersion. **(ii)** The only real survival mover is a
transfer/conditioning/recovery **recipe**, not Dispersive. **(iii)** Precision is gated by
**neither representation, sensing, nor coverage**. A renderer gate and a positive-control
**intervention** rule out sensing: although the 64×64 FPV cannot encode metric range beyond
~2 m, that loss is a fixable renderer artifact (a non-saturating target restores it at
64 px), and *handing the policy the oracle metric position error barely moves precision*
(~36× the state oracle) while a richer cue **collapses survival**. We then build the
competent far-range teacher and range-encoding observation this points to — a PID-CTBR
teacher that recovers the 1–3 m band at 100 % survival (cond-IAE 0.14–0.18 m) and a
perspective renderer that restores far-range image→distance R² 0.05→0.40 — and remove
**both** remaining candidate constraints jointly in a pre-registered three-seed
**Teacher × Observation 2×2** under the same frozen protocol. The floor *holds* (cond-IAE
2.93 m, ~43× oracle, no better than the neither-factor control), and the two factors
**negatively interact** on precision: coverage buys survival (+8–14 pp) but, stacked on the
better-sensing cell, *worsens* precision (2.48→2.93 m). We conclude that for this class of
task the binding constraint is none of representation collapse, sensing, or coverage, but a
**robustness–precision conflict** — directly observed as this negative interaction — that we
attribute to limited model capacity, the leading explanation we identify and the next lever
to test directly. We release the protocol, both 2×2 ablations, the intervention, and all
diagnostics.

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
   hypothesis, hardened with an **official-code-faithful** implementation (InfoNCE-L2
   on the `flow_net` mid-block, λ=0.5, `/d`), so the null result controls for
   implementation fidelity, plus a frozen-encoder control.
3. **A mechanistic diagnosis** that *excludes* all three candidate bottlenecks
   (representation, data coverage, sensing) rather than merely ranking them. A
   positive-control *intervention* hands the policy the oracle sensing signal and shows it
   does **not** restore precision (excluding sensing); a decisive, pre-registered
   **Teacher × Observation 2×2** then supplies a *competent* far-range teacher **and** a
   range-encoding observation jointly, and the precision floor still does not move
   (excluding coverage). What remains is a **robustness–precision capacity conflict**.

This is a negative result, but a constructive one: by excluding representation, sensing,
and coverage in turn it redirects effort toward model capacity and the
robustness–precision trade-off — not the auxiliary regularisers, richer sensors, or
data-coverage fixes a reader would reach for first. Because the regulariser is
implemented exactly as released — placement, weight, and per-dimension normalisation
(§3) — the null result cannot be attributed to an implementation error.

---

## 2. Related Work

*Dispersive regularization at a transfer boundary: from manipulation to high-speed visual hover.*

**Generative policies for visuomotor control.** Behaviour cloning treats control as
supervised regression from observations to actions, but robot action distributions are
multimodal, sequentially correlated, and precision-sensitive, which makes plain
regression brittle. Building on denoising diffusion [1] and its deterministic DDIM
sampler [2], Diffusion Policy [3] reframed control as conditional denoising diffusion
over short action sequences, with receding-horizon execution and a GroupNorm visual
encoder trained end-to-end. Flow matching [4] and rectified flow [5] offer a
continuous-time, simulation-free alternative that learns a velocity field from noise to
data, since adopted as a lighter-weight policy backbone — e.g. the π₀ vision-language-
action flow model [6] and the Riemannian flow-matching policy [7]. Our policy is a
flow-matching action head over a fused vision–IMU conditioning vector, placing it in this
family.

Because imitation alone cannot exceed its demonstrations, a second line fine-tunes
pretrained generative policies with reinforcement learning. DPPO [8] casts the denoising
chain as a multi-step MDP and applies PPO; ReinFlow [9] injects a learnable noise network
into a flow policy's otherwise deterministic path, recovering a tractable likelihood that
supports stable RL fine-tuning at very few — or even one — denoising steps; Q-score
matching [12] is a related value-based route. This interactive, closed-loop setting should
be distinguished from RL fine-tuning of *image-generation* diffusion models (DDPO [10];
DPOK [11]), which optimise a non-interactive sampler against a reward. This
generative-policy-plus-RL stack is the substrate on which the regularizer studied here is
layered, and the same stack in which the positive results for that regularizer were
reported.

**Representation regularization in generative models and policies.** A recurring concern
in denoising-based generators is that regression objectives supervise the output but leave
intermediate features under-constrained, allowing them to collapse. The failure has a
well-studied form: Jing et al. [22] show that even contrastive objectives suffer
*dimensional collapse*, with embeddings spanning a lower-dimensional subspace than their
nominal dimension — the property later diagnostics quantify through effective rank.
REPA [15] addresses this by *aligning* a diffusion transformer's hidden states with a
pretrained self-supervised encoder, at the cost of an external encoder and data; the
classic variance/covariance and redundancy-reduction criteria of VICReg [16] and Barlow
Twins [17] pursue the same anti-collapse goal without positive pairs. Dispersive Loss [13]
removes the external-encoder cost differently: it keeps only the *repulsive* half of a
contrastive objective, encouraging a generative network's intermediate representations to
spread out with no positive pairs, no extra parameters, and no external data, validated on
ImageNet class-conditional generation where it consistently improves FID.

Two robotics works carry Dispersive Loss into control. D²PPO [14] identifies "diffusion
representation collapse" — semantically similar observations mapped to indistinguishable
features — as a cause of manipulation failures, adds dispersive regularization treating
every hidden representation in a batch as a negative pair, and reports average gains of
22.7 % in pre-training and 26.1 % after PPO fine-tuning on RoboMimic, with high real-robot
success on a Franka Panda. DMPO [23] extends the idea to one-step MeanFlow policies,
arguing dispersive regularization is necessary for stable single-step generation;
concurrent MeanFlow-plus-dispersive manipulation work includes DM1 [18] and MP1 [19].
Notably, the public DMPO configuration specifies InfoNCE-L2 on the mid-block with weight
0.5 — the settings a faithful re-implementation in a new domain would inherit, and the
settings we adopt in §3 / §5.

Three properties of this literature matter here. First, every validation is on image
generation [13] or quasi-static, table-top manipulation scored by discrete task
success [14, 23]. Second, the mechanism on offer is *observational discriminability*: the
failure dispersive loss is claimed to fix is confusing two similar images. Third, none of
these works publishes a fidelity audit — placement, weight, the per-dimension
normalization in the official InfoNCE-L2 form — or a frozen-encoder control that isolates
whether the term is even on the action-gradient path. Whether the benefit, the mechanism,
or even the on-path behaviour transfers to a setting with a different bottleneck is left
open.

**Learning-based vision quadrotor control.** High-speed visual flight is exactly such a
setting. The dominant recipe couples privileged-learning imitation with sensor
abstraction: Deep Drone Racing [24] maps images to a waypoint and desired speed for a
downstream planner; Deep Drone Acrobatics [25] regresses collective thrust and body rates
from abstracted visual–inertial inputs to fly 3g maneuvers by imitating a privileged
expert; Learning High-Speed Flight in the Wild [20] maps depth and IMU to collision-free
trajectories with zero-shot sim-to-real transfer; and champion-level drone racing was
achieved with deep RL from onboard perception [21]. A consistent theme is that what
unlocks robustness and transfer is *data diversity, sensor abstraction, and privileged
supervision* — not auxiliary representation regularizers. The failure mode here is also
different in kind: a hover or recovery policy does not fail by confusing two similar
frames, it fails by drifting off-distribution and crashing, the classic covariate-shift
problem of behaviour cloning [26].

**Negative results, auxiliary-loss (in)effectiveness, and honest evaluation.**
Representation-centric add-ons do not help uniformly. Most directly, Schneider et al. [27]
benchmark pretrained visual representations in model-based RL and find — "surprisingly" —
that they are no more sample-efficient and no more OOD-robust than features learned from
scratch, concluding that data diversity and architecture, not representation quality,
drive OOD generalization. Establishing a credible negative result also depends on
evaluation discipline: Henderson et al. [28] document how seed variance makes single-run
RL claims unreliable; Agarwal et al. [29] show that few-run benchmarks are statistically
fragile and advocate bootstrap intervals; Patterson et al. [30] catalogue maximization
bias and unfair hyperparameter budgets. Together they motivate the frozen, multi-seed
protocol with interval estimates (§4) on which our null effect is asserted rather than
assumed.

**Positioning: a transfer boundary, not a contradiction.** We take Dispersive Loss
following the official implementation — InfoNCE-L2 on the generative network's mid-block,
weight 0.5, including the per-dimension normalization present in the released code — and
ask whether the benefit reported for image generation [13] and table-top
manipulation [14, 23] transfers to high-speed visual quadrotor hover. Under a frozen,
three-seed protocol with bootstrap intervals (following [28, 29]), it does not: against
D²PPO's reported +22.7 % / +26.1 % manipulation-success gains on RoboMimic [14], we
measure a −2.2 pp Tier-1 change that sits inside a 6.3 pp across-seed std, with survival
and precision unmoved (§5–§6). This is best read not as a contradiction of the
manipulation results but as a *transfer boundary*. Where the bottleneck is observational
discriminability — telling two near-identical grasp scenes apart — spreading intermediate
features can help [14]. Where the bottleneck is behavioural coverage — recovering from a
one-to-three-metre offset before crashing — feature dispersion addresses the wrong axis,
echoing Schneider et al.'s [27] finding that representation quality is not the constraint
governing out-of-distribution behaviour. Our contribution is a faithful, pre-registered,
multi-seed demonstration of where a popular regularizer stops working on this task, and a
diagnosis of why.

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

**Dispersive Loss (faithful, official recipe).** We apply Dispersive Loss exactly as
released in the official code [13, 14]: the InfoNCE-L2 form on the generative network's
intermediate block — here the `flow_net` mid-block (`return_mid=True`),
`L_disp = log E_{i≠j}[exp(−D/τ)]`, `D = ‖z_i − z_j‖² / d` — with weight **λ=0.5** and
**τ=0.5**, including the per-dimension `/d` normalization present in the released code.
(Implementation: `models/flow_policy_v5.py:204–224, 269–287`.) Because the term lands on
`flow_net` mid-features, its gradient reaches the action path **regardless of whether the
visual encoder is trainable** — the property that makes the frozen-encoder control in §5
informative.

**Fidelity of the implementation, and an off-path variant studied in §6.1.** Two
configuration choices determine whether the term is active at all, and we fix both to the
released values. *Placement, weight, and form:* the term must sit on a generative-network
intermediate block (here the `flow_net` mid-features) at λ=0.5 in the InfoNCE-L2 form. A
hand-rolled log-distance variant `−mean_{i≠j} log(‖x_i − x_j‖ + ε)` at λ=0.05 applied to
`vis_pooled` — a feature that feeds **only** the auxiliary state head, not the action path
(which conditions on `attended` and `imu_feat`) — departs from the released recipe on all
three axes. *Normalisation:* the published Algorithm 1 omits the per-dimension `/d` divisor
present in the released code; without it the loss saturates to zero gradient at τ=0.5, so
this divisor is required for the term to carry gradient at all. The headline ablation (§5)
uses the released configuration on both axes. Separately, §6.1 studies the off-path
`vis_pooled` placement as a mechanistic probe, where its off-path position exposes how a
scale-sensitive dispersion term can game its objective.

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

![Figure 1](figures/single_seed_swing.png)

**Figure 1.** Per-seed Tier-1 pass-rate (left) and survival (right) for the two P1
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

- **Dispersive**: the official-code-faithful InfoNCE-L2 term (§3) at λ=0.5 (ON) vs 0.0
  (OFF), on the `flow_net` mid-block.
- **E2E**: vision encoder trainable (ON) vs frozen at the transferred init (OFF).

**Design and power.** With three seeds per cell and a pooled across-seed Tier-1 std of
~6.3 pp, this design is powered to detect only large effects (≳6 pp Tier-1); a smaller
true effect of Dispersive cannot be excluded. We therefore report a null in the sense of
*no effect above seed noise*, not a proof of exact zero, and judge support by whether
D1E1 exceeds D0E1 by more than the pooled std.

**What the frozen-encoder cell tests.** Because the term lands on `flow_net`
mid-features, it trains the generative network *even with a frozen encoder*. The
frozen-encoder cell (E0) is therefore not a trivial no-op: it isolates
Dispersive's effect with the *encoder* held fixed but the *flow network* still adapting.
The decisive hypothesis test is **D1E1 vs D0E1** (Dispersive ON vs OFF, both E2E).

**Table 2 — 2×2 ablation, faithful Dispersive** (`evaluate_p2_ablation.py`, 3 seeds,
frozen protocol; `evaluation_results/p2f_ablation_leaderboard.json`).

| Cell | Disp / E2E | Tier-1 (mean±std) | Survival (mean±std) | cond-IAE | %Oracle |
|------|-----------|------------------:|--------------------:|---------:|--------:|
| D0E0 | OFF / frozen | 87.8 ± 3.1 | 66.1 ± 3.7 | 2.93 m | 13.4 % |
| D1E0 | ON / frozen  | **74.4 ± 8.7** | **60.4 ± 1.6** | 2.81 m | 12.9 % |
| D0E1 | OFF / E2E    | 92.2 ± 3.1 | 65.0 ± 2.8 | 2.91 m | 12.4 % |
| D1E1 | ON / E2E     | 90.0 ± 5.4 | 62.9 ± 2.4 | 2.89 m | 12.2 % |

![Figure 2](figures/ablation_forest.png)

**Figure 2.** The 2×2 ablation as a forest plot (faithful Dispersive, 3 seeds, mean ±
across-seed std). *Left:* Tier-1; the grey band is D0E1 ± the pooled across-seed std
(6.3 pp). The decisive D1E1 (ON/E2E) sits inside it — **no Dispersive effect above seed
noise** (−2.2 pp). The one cell that moves is D1E0 (ON/frozen, 74.4 ± 8.7), where
Dispersive-on-frozen lowers the mean and inflates the variance. *Right:* survival is flat
across all four cells. (`scripts/make_paper_figures.py` →
`evaluation_results/p2f_ablation_leaderboard.json`.)

**Findings.**
- **Dispersive is not supported (Figure 2).** D1E1 vs D0E1: Tier-1 **−2.2 pp inside the
  6.3 pp pooled across-seed std**; survival −2.1 pp; conditional precision unchanged
  (2.89 vs 2.91 m). The decision rule (supported iff D1E1 beats D0E1 by more than the
  pooled std) returns **NOT supported**. The conclusion holds under the off-path,
  low-weight variant of §6.1 as well (+1.1 pp), so it does not hinge on placement or weight.
- **The frozen-encoder condition is an informative control, not a no-op.** Under the
  faithful mid-block placement the encoder-frozen runs *do* train `flow_net`, so the
  checkpoints differ (`p2f_D1E0_s* ≠ p2f_D0E0_s*`, MD5 distinct across all three seeds).
  The frozen row is not inert: Dispersive-on-frozen **destabilises** control (mean
  Tier-1 87.8→74.4, variance 3.1→8.7 pp) — with no trainable encoder to co-adapt, repelling
  the mid-features pushes the mean down and the variance up rather than helping.
- **E2E is the only (small) Tier-1 mover** (D0E0→D0E1, +4.4 pp), and even it neither
  extends survival nor improves precision (§6.2).

All four cells beat the prior frontier Joint_E2E_v5 (Tier-1 80 %), but §6.2 shows that
gain is the *recipe*, not Dispersive. Precision is unmoved (cond-IAE 2.7–3.2 m, ~12–13 %
oracle): **nothing is deployable.** This same 2×2 methodology — frozen P0, three seeds per
cell, the pooled-std decision rule — is re-used unchanged for the second, precision-focused
ablation in §6.4 (Teacher × Observation), so the two falsification tests rest on one
protocol; only the manipulated factors and the primary axis (there, cond-IAE) differ, as
each hypothesis dictates.

---

## 6. Diagnosis: Which Bottleneck?

Three candidate explanations for the flat result: representation (the thing Dispersive
targets), data coverage, or sensing. We test the first two with observation-only
measurements (no retraining), and the third with both a renderer gate and a
positive-control **intervention** that hands the policy the very signal it is presumed to
lack (§6.3).

### 6.1 Representation: Dispersive games its objective; survival ⟂ rank

*Placement note.* We measure feature geometry at **both** placements. We start with the
off-path `vis_pooled` placement (Table 3; legacy P2 checkpoints): because `vis_pooled`
feeds only the auxiliary state head, it cleanly isolates how a scale-sensitive dispersion
term games its objective off the action path. We then confirm the same pathology at the
faithful `flow_net` mid-block, where the official term actually acts (Table 4; the p2f
checkpoints of §5). One placement-dependent detail: off-path, a frozen encoder zeroes the
gradient (so D1E0 ≡ D0E0, the legacy no-op); at the faithful placement the term trains
`flow_net` even when frozen (§5), so D1E0 ≠ D0E0.

We push a fixed 4000-image hover+recovery batch through each checkpoint's encoder and
measure the geometry of `vis_pooled` (D=256). (`measure_feature_collapse.py`, legacy P2
checkpoints.)

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

**The decisive plot is rank vs survival (Figure 3).** Effective rank swings **15×**
(30→9→2) while closed-loop survival stays flat (66.1→65.0→65.0 %, within ~3 pp std).
**Survival is decoupled from `vis_pooled` rank.** Within this off-path probe, neither
preventing collapse (D0E0 rank 30) nor worsening it (D1E1 rank 2) moves the closed-loop
outcome. Together with the faithful on-path null (§5), this is the paper's strongest
evidence that, *for this task*, representation collapse is not the binding constraint. The
same pathology and decoupling hold at the faithful placement (Table 4, below).

![Figure 3](figures/rank_survival_decoupling.png)

**Figure 3.** `vis_pooled` effective rank (bars, log axis) collapses ~15× across the
2×2 cells (30.3 frozen → 9.0 E2E → 2.0 E2E+Dispersive), while closed-loop survival and
Tier-1 pass-rate (lines, right axis; error bars = across-seed std) stay flat. Survival
is decoupled from the representation geometry Dispersive targets.

**The same pathology holds at the faithful placement (Table 4).** The off-path probe
above cannot say whether the faithful `flow_net` mid-features — where the official term
actually acts — behave the same way. Measuring them directly (same fixed 4000-sample
batch, now of (image, IMU, action, timestep `t`) so the mid-features are well defined;
`measure_feature_collapse_flowmid.py`) returns the same verdict, more sharply. Dispersive
ON drives the InfoNCE-L2 objective to its subset floor (`log(1/2048) = −7.62`, identical
across all three seeds) — it fully *wins* its objective — but does so purely by inflating
the mid-feature norm ~9× (9.4→84) while the effective rank **collapses** (221→36 with a
trainable encoder, 249→37 frozen; dims-for-99 %-variance 683→177). Exactly as off-path,
the term games a scale-sensitive objective by norm growth and *worsens* intrinsic rank.
Two further points: (i) naive E2E barely dents the mid-feature rank (D0E0→D0E1,
249→221) — unlike `vis_pooled` (30→9), the action-path mid-features do not strongly
collapse on their own, so there is little for Dispersive to "fix"; (ii) the
frozen-Dispersive cell (D1E0) shows the *same* norm blow-up and rank collapse as D1E1,
a plausible mechanism for its closed-loop destabilisation (§5) — repelling the velocity
field's own mid-features inflates them ~9× with no encoder free to compensate. Across all
four cells the mid-feature rank swings ~7× (249→36) while survival stays within ~3 pp
(§5): the rank-vs-survival decoupling holds at the faithful placement too.

**Table 4 — faithful `flow_net` mid-block geometry** (D=1024, mean±std over 3 seeds;
`measure_feature_collapse_flowmid.py` → `evaluation_results/p2f_feature_collapse_flowmid.json`).

| Cell | eff_rank | dims for 99% var | feat_norm | disp_infonce | mean cos |
|------|---------:|-----------------:|----------:|-------------:|---------:|
| D0E0 (OFF / frozen) | 248.8 ± 9.0 | 722 | 9.35 | −0.283 | 0.174 |
| D1E0 (ON / frozen)  | 36.7 ± 0.5 | 189 | 83.8 | **−7.624** | 0.058 |
| D0E1 (OFF / E2E)    | 221.4 ± 16.2 | 683 | 9.49 | −0.293 | 0.163 |
| D1E1 (ON / E2E)     | **35.9 ± 0.6** | 177 | **84.8** | **−7.624** | 0.056 |

**Scale-invariant criteria cure the collapse but still do not move control (Table 5).** One
escape remains for the faithful term: perhaps it fails only because its *scale-sensitive*
L2-InfoNCE objective is gameable by norm inflation, and a criterion that cannot inflate its
way out would both produce genuine dispersion *and* help control. We test this directly,
holding the D1E1 recipe fixed and varying **only** the dispersion criterion on the same
`flow_net` mid-block (3 seeds each, frozen protocol; `run_p6_form_ablation.py`): a
**unit-sphere InfoNCE** (cosine; norm inflation impossible by construction) and a
**VICReg-style** variance + covariance term. Both behave exactly as intended on the
geometry — feature norm stays O(1) (1.33× / 1.36× the no-dispersion baseline, against the
faithful term's 8.93×) and effective rank *rises* to **75 % / 85 %** of the 1024 dimensions
(against the faithful term's collapse to 3.5 %): they achieve the high-rank dispersion the
regulariser is designed for, without the cheat. Yet closed-loop control does not improve —
all forms sit in one band (survival 60–65 %, Tier-1 82–92 %, cond-IAE 2.9–3.1 m), and the
only metric to cross the pooled seed std does so in the *worse* direction (cond-IAE
+0.13–0.21 m). A **24× swing in mid-block effective rank (36→867) buys ≈ 0 control.** This
closes the "you used a scale-sensitive criterion" escape and is the on-path counterpart of
the `vis_pooled` decoupling above: representation geometry is decoupled from closed-loop
control *regardless of which criterion enforces dispersion*.

**Table 5 — scale-invariant dispersion forms** (D1E1 recipe, only the criterion varies;
3 seeds, geometry from `measure_feature_collapse_flowmid.py`, control from frozen P0;
`evaluation_results/p6_form_ablation_leaderboard.json`,
`docs/experiment_report_p6_scale_invariant.md`).

| Form on `flow_net` mid-block | eff_rank (of 1024) | feat_norm ÷ off | Survival | Tier-1 | cond-IAE |
|------------------------------|-------------------:|----------------:|---------:|-------:|---------:|
| off (D0E1, no dispersion) | 221 (22 %) | 1.00× | 65.0 ± 2.8 | 92.2 ± 3.1 | 2.91 m |
| InfoNCE-L2 (faithful, D1E1) | 36 (3.5 %) | **8.93×** | 62.9 ± 2.4 | 90.0 ± 5.4 | 2.89 m |
| unit-sphere InfoNCE (cosine) | **769 (75 %)** | 1.33× | 64.4 ± 3.3 | 86.7 ± 8.2 | 3.10 m |
| VICReg (var + cov) | **867 (85 %)** | 1.36× | 60.5 ± 2.4 | 82.2 ± 11.0 | 3.03 m |

### 6.2 The only survival mover is the recipe, not Dispersive (or E2E)

Decomposing the gain over the prior frontier as controlled steps
(`experiment_report_survival_movers.md`):

| Step | Δ Tier-1 | Δ Survival | cond-IAE |
|------|---------:|-----------:|---------:|
| prior frontier Joint_E2E_v5 → **recipe** (frozen cell D0E0) | **+7.8 pp** | **+3.9 pp** | 2.93 m |
| + E2E (D0E0 → D0E1) | +4.4 pp | −1.1 pp | 2.91 m |
| + Dispersive, faithful (D0E1 → D1E1) | −2.2 pp (∈ noise) | −2.1 pp | 2.89 m |

(The D0E0/D0E1 cells are Dispersive-OFF, so the +Dispersive step is the only one that
depends on the regulariser: under the official configuration it is −2.2 pp, and under the
off-path low-weight variant of §6.1 it is +1.1 pp — both inside the across-seed std.)

The dominant lift is the **recipe** (H4-transfer init + task-conditioning + recovery
mix), and it holds with a **frozen** encoder — so it is not from E2E. E2E adds a small
Tier-1-only bump that neither extends survival nor improves precision. Dispersive adds
nothing (slightly negative under the official configuration, ∈ noise). **Mover ranking:
recipe ≫ E2E ≫ Dispersive ≈ 0.**

### 6.3 Precision is not sensing-gated — leaving a coverage hypothesis to test

cond-IAE is ~2.8 m (~13 % oracle) for *every* configuration. Why won't precision move?
This subsection rules out **sensing** with a renderer gate and a positive-control
intervention, leaving a single leading candidate — a coverage / teacher-competence gap —
which §6.4 then tests directly and also excludes.

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
   **Figure 4**). The only range cue is the target crosshair *size*,
   `size = max(2, min(6, ⌊6/(d+0.5)⌋ + dr))`. With domain randomisation, adjacent-
   distance separability is d′ < 0.2 everywhere, and a linear decode of distance from
   the image has R² = 0.41 near (< 1 m) but only **0.12 far (≥ 1.5 m)**. With DR off
   (noiseless ceiling) the size **saturates at 2 px for all d ≥ 2 m**, so renders at
   2.0, 2.5, and 3.0 m are **byte-identical** (d′ = 0). The crosshair *position* encodes
   only normalised direction (range-invariant).

The policy's 2.83 m steady drift sits exactly where the observation carries **no
recoverable metric range** (Figure 4). One might conclude that precision is
*information-gated by the observation model*. That inference does not follow: a
measurement that the FPV does not encode range does not establish that the missing range
is the *binding* cause of imprecision. We test that implication directly with a renderer
gate and a positive-control intervention, and **both refute it**.

![Figure 4](figures/crosshair_distance_saturation.png)

**Figure 4.** *Left:* the only FPV range cue (target crosshair size) saturates at the
2 px floor beyond ~2 m — under domain randomisation (red band = min–max) the size is
uninformative even nearer, and with DR off (dashed) renders at 2.0/2.5/3.0 m are
byte-identical. The policy's steady-state drift (green, median 2.83 m) sits inside the
saturated region. *Right:* a linear decode of distance from the image succeeds near
(R²=0.41, <1 m) but fails far (R²=0.12, ≥1.5 m). This is a *measurement* of the
observation, not the binding cause of imprecision (Figure 5).

**The information loss is a renderer artifact, not the pixel count (gate).** Before
treating "richer sensing" as the fix, we ask whether a more capable monocular renderer
would even *carry* the far-range information (`measure_higher_res_gate.py`; **Figure 5,
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
under the frozen protocol (3 seeds; control = the no-cue D0E1; **Figure 5, right**).
Handed the *oracle* metric range (scalar ‖pos-err‖, σ=0), cond-IAE moves only
2.91 → **2.43 m** — still ~36× the 0.068 m state oracle, and bought with −6.7 pp survival
/ −13 pp Tier-1. A realisable sensor (σ=0.15 m) erases even that (2.81 m ≈ control). The
*richer* full 3-D position cue is actively harmful: it **collapses survival across all
three seeds** (Tier-1 92 → 7 %, surviving to the 250-step threshold in ≈2/30 episodes),
so its conditional precision is an artifact, not a win. Supplying range tells the drone
*how far off* it is, but not *what to do* about it — and no demonstration in the 1–3 m
band teaches that.

**Verdict: precision is not sensing-gated.** Range information — even oracle, even full
position — does not restore precision, and a richer cue harms survival. That leaves a
single leading candidate, which the measurements above motivate but do **not** prove: the
absence of a learned far-range *recovery behaviour* in the 1–3 m band. The BC data has no
labels there (the coverage gap above) and the *current* state-PPO expert cannot generate
them (it crashes from > 2 m, check 2). The natural inference — *precision is
coverage / teacher-competence-gated, and would move given a competent far-range teacher* —
is exactly the kind of "fixable-bottleneck" conclusion that a negative-result paper should
not assert from observational probes alone. **We therefore do not stop here.** §6.4 builds
the very thing this inference calls for — a teacher competent across the full 1–4 m band
*and* the coverage and range-encoding observation it implies — and tests, decisively,
whether precision then moves. (It does not: the coverage hypothesis is excluded, leaving a
capacity conflict.)

![Figure 5](figures/sensing_ablation.png)

**Figure 5.** *Left (gate):* far-range (≥1.5 m) image→distance R² for three resolutions ×
two target renderers. The saturating production crosshair is uninformative at every
resolution (raising pixels does nothing); a non-saturating perspective target restores
~half the far-range information already at 64 px — the loss is a target artifact, not the
pixel count. *Right (intervention):* closed-loop precision (cond-IAE, 3 seeds) when the
policy is *handed* the metric position error the FPV lacks. Even the oracle scalar range
(σ=0) leaves precision ~36× the state oracle (green dashed, 0.068 m); sensor noise erases
the gain; the richer 3-D cue (σ=0) collapses survival (hatched; conditional IAE
unreliable). Precision is **not sensing-gated**; the coverage / teacher-competence
hypothesis this points to is built and tested decisively — and also excluded — in §6.4.

### 6.4 The decisive test: a competent teacher and a range-encoding observation still do not break the floor (Teacher × Observation 2×2)

§6.1 excluded representation and §6.3 excluded sensing, leaving one leading hypothesis:
precision is **coverage / teacher-competence-gated** and would move given a teacher
competent in the 1–3 m band. A negative-result paper should not rest on that inference
from observational probes — so we *build the fix the hypothesis demands* and test it under
the same frozen protocol. This mirrors §5: a pre-registered 2×2, three seeds per cell,
frozen P0, the same pooled-std decision rule; only the manipulated factors and the primary
axis differ (here precision, cond-IAE, is primary, because precision is the question).

**The two factors, each built to remove a candidate constraint.**

- **Teacher coverage (T).** A cascade **PID-CTBR** controller (gentle recovery gains)
  recovers the full **1–4 m band at 100 % survival, cond-IAE 0.14–0.18 m** (~2× the 0.068 m
  oracle, ~18× tighter than the 2.8 m closed-loop floor) — a genuinely *competent*
  far-range teacher, where the state-PPO expert crashes beyond 2 m (§6.3, check 2). It
  collects far-range recovery demonstrations whose position-error mass lands **11.7 %** in
  1–3 m (p99 = 2.59 m), versus the ~0.4 % of the hover data — a ~29× coverage jump, with
  zero crashes. **T0** = hover-only data; **T1** = hover + this far-range recovery set.
- **Observation (O).** **O0** is the production saturating crosshair; **O1** is the
  non-saturating perspective target of the §6.3 gate, now integrated into the production
  env, which restores far-range (≥ 1.5 m) image→distance R² from **0.05 to 0.40** (DR-on,
  in-env). O1 policies are *trained and evaluated* on the perspective observation.

Every cell is the **D0E1 frontier recipe** (Dispersive OFF, encoder E2E, H4-transfer init,
task-conditioned, 500 hover + 500 recovery), so T and O are the only varying factors; the
factors enter purely through which Phase-3 dataset is read. 12 BC runs (4 cells × seeds
{0,1,2}), each scored through the *identical* `evaluate_frozen_p0` protocol (30 paired
episodes, base seed 12345, σ = 2.0, n_inf = 2), with O1 cells handed their own perspective
renderer at eval time (`evaluate_frozen_p0.py --target-render`).

**Decision rule (pre-registered, mirroring §5).** The floor is "broken" iff **all** hold:
(1) `cond-IAE(T1O1) < cond-IAE(T0O0) − pooled_std` (significant vs the neither-factor
control); (2) `cond-IAE(T1O1) ≤ 1.5 m` (an absolute target, ~2× off the floor); (3)
`survival(T1O1) ≥ survival(T0O0) − pooled_std` (a survival guard, because the §6.3 pos3d
cue produced a "precision win" that was really a survival collapse); cond-IAE is trusted
only when `n_cond ≥ 15` in both cells.

**Table 6 — Teacher × Observation 2×2** (`evaluate_p2to_ablation.py`, 3 seeds, frozen
protocol; `evaluation_results/p2to_ablation_leaderboard.json`; measured oracle 0.068 m,
0.9668 composite). cond-IAE is the **primary** axis (lower is better); %Oracle is the
secondary composite score, which rewards survival and precision jointly and so is not
monotone in survival.

| Cell | Teacher / Obs | **cond-IAE (mean±std)** | Survival | Tier-1 | %Oracle | n_cond |
|------|---------------|------------------------:|---------:|-------:|--------:|-------:|
| T0O0 | none / crosshair | **2.69 ± 0.15 m** (40× oracle) | 83.2 ± 5.8 % | 98.9 ± 1.6 % | 20.1 % | 29.7 |
| T0O1 | none / perspective | **2.48 ± 0.14 m** (36×) | 76.7 ± 10.5 % | 86.7 ± 9.8 % | 22.1 % | 26.0 |
| T1O0 | far-range / crosshair | **2.71 ± 0.22 m** (40×) | **91.4 ± 1.5 %** | **100.0 ± 0.0 %** | 23.1 % | 30.0 |
| T1O1 | far-range / perspective | **2.93 ± 0.18 m** (43×) | 90.6 ± 3.3 % | 98.9 ± 1.6 % | 20.8 % | 29.7 |

![Figure 6](figures/teacher_obs_2x2.png)

**Figure 6.** The Teacher × Observation 2×2 (3 seeds, mean ± across-seed std). *Left:*
cond-IAE (primary; lower = more precise) as a 2×2 grid; the green dashed line is the
0.068 m state oracle and the whole grid sits at 36–43× it. *Right:* survival per cell.
Coverage (T1, bottom row) lifts survival but not precision; the decisive T1O1-vs-T0O0
contrast (arrow) is **+0.24 m — worse, and inside the 0.23 m pooled std** —
**floor not broken**; the negative coverage×sensing interaction (T0O1 2.48 → T1O1 2.93 m)
is annotated. (`scripts/make_paper_figures.py` →
`evaluation_results/p2to_ablation_leaderboard.json`.)

**Verdict — floor NOT broken.** The decisive T1O1-vs-T0O0 comparison fails every breaking
criterion: cond-IAE **2.93 vs 2.69 m → Δ = +0.24 m** against a **pooled std of 0.23 m** (not
significant — and in the *wrong* direction); 2.93 m ≫ the 1.5 m absolute target; only the
survival guard passes (90.6 % ≥ 83.2 − 6.7 %). The floor holds across the **entire** 2×2
(2.48–2.93 m, 20–23 % oracle); n_cond ≥ 22 everywhere, so this is not a short-survival
artifact. **Even with the competent far-range teacher *and* the range-encoding observation
supplied jointly, conditional hover precision does not move.**

**What the grid shows — robustness ↑, precision pinned, a negative interaction.** The
result is not flat noise; it has clean, interpretable structure that *strengthens* the
diagnosis (`experiment_report_p2to_decisive.md`):

1. **Coverage (T1) buys robustness, not precision.** Far-range recovery labels lift
   survival **+8.2 pp** on crosshair (83.2→91.4 %) and **+13.9 pp** on perspective
   (76.7→90.6 %) and pin Tier-1 at ~99–100 %, exactly the expected coverage benefit — yet
   move precision by **0** (T1O0 2.71 m ≈ T0O0 2.69 m).
2. **Sensing (O1) is a weak precision lever and is not free.** On the hover-only row it
   improves cond-IAE 2.69→2.48 m (≈ 0.21 m, ~1× the pooled std — marginal) while drifting
   survival 83.2→76.7 % and Tier-1 98.9→86.7 %.
3. **The interaction is NEGATIVE.** The single best precision cell is **T0O1 (2.48 m)**;
   adding far-range recovery on top of it (→ T1O1) *degrades* precision to **2.93 m
   (+0.45 m)**. The very factor that buys survival pulls the policy toward wide-range
   corrective behaviour that widens steady-state hover error.

**This is a Robustness–Precision Capacity Conflict, made quantitative and RL-free.** A v5
curriculum-RL run showed the same conflict but was confounded by RL dynamics (advantage
masking); here it appears in a clean, three-seed **supervised** 2×2 whose decision rule was
pre-registered before data collection (`RESEARCH_PLAN_v7.md`): within this policy's capacity
one can have wide-range survival **or** tight hover precision, and pushing coverage trades
the latter for the former. Two claim strengths should be kept distinct: the
robustness–precision *conflict* is **directly observed** (the negative interaction is
measured — survival-buying coverage costs precision, +0.45 m > the 0.23 m pooled std),
whereas *capacity* is the **leading explanation** we advance for it, **not** a directly
manipulated result — we did not vary model capacity here, and a precision-specialised head
or larger backbone is the direct test (§8). The tension is the task-level signature of a
phenomenon well documented in multi-task learning: competing objectives optimised under a
fixed parameter budget trade off rather than co-improve — the multi-objective / Pareto view
of Sener & Koltun [31], who note that a weighted sum of losses only works when tasks do not
compete — and their gradients can directly conflict, so that a step improving one objective
worsens the other (the negative-cosine "conflicting gradients" of Yu et al. [32]). Our
coverage factor behaves as exactly such a competing objective: the gradient that buys
survival is the one that worsens precision. Together with §6.1 (representation) and
§6.3 (sensing), this is a **triple exclusion**: removing each of the three leading
candidate constraints — separately and, for coverage × sensing, jointly — leaves the
precision floor intact. The binding constraint is therefore **none of representation
collapse, sensing, or coverage**, but a capacity / robustness–precision conflict.

---

## 7. Discussion

**What generalises.** *(a)* The frozen protocol and the short-survival artifact it fixes
are reusable for any survival-vs-precision visual control benchmark. *(b)* The
"objective-gaming" pathology of a *scale-sensitive* dispersion regulariser is a general
caution: a distance-maximising contrastive term can be trivially satisfied by norm
inflation, *worsening* intrinsic rank (~287× off-path, ~9× on-path; §6.1). We checked the
two obvious escapes and both fail to rescue control. *Placement:* regularising the on-path
feature the policy actually conditions on (§5) removes the auxiliary-only projection there
is to inflate, yet still does not help (no gain when the encoder is trainable; mildly
destabilising when frozen). *Criterion:* swapping in a **scale-invariant** form
(unit-sphere InfoNCE or VICReg; §6.1) removes the norm-inflation gaming outright (feature
norm 8.93×→~1.3×) and produces *genuine* high-rank dispersion (effective rank 3.5 %→75–85 %
of dimensions) — the intended effect of the regulariser — yet closed-loop survival and
precision still do not move (if anything cond-IAE regresses ~0.1–0.2 m). So neither the
placement nor the criterion is what holds it back. *(c)* The rank-vs-survival decoupling —
now established at both the off-path `vis_pooled` and the on-path `flow_net` mid-block, and
across both scale-sensitive and scale-invariant criteria — argues that "prevent
representation collapse" is not automatically the right lever for closed-loop control; the
binding constraint must be located empirically.

**Coverage and sensing are jointly insufficient.** The §6.4 2×2 turns the §6.3 precision
inference into a *test* and refutes it: supplying a competent far-range teacher (T1) and a
range-encoding observation (O1) **together** leaves cond-IAE pinned at 2.93 m (≈ 43×
oracle), no better than the neither-factor control. The grid also gives the
robustness–precision trade-off its first clean, RL-free quantification on this task:
coverage reliably buys **survival** (+8.2 pp crosshair, +13.9 pp perspective; Tier-1 →
~100 %) but moves precision by 0, and the coverage × sensing **interaction is negative** —
the best precision cell is T0O1 (2.48 m) and adding far-range recovery on top of it
*worsens* precision to T1O1 (2.93 m). The factor that buys survival is the factor that
costs precision. This reproduces, RL-free and pre-registered, the same conflict a v5
curriculum-RL run showed under confounded dynamics: within this capacity the policy can
spend itself on wide-range survival **or** tight hover, not both — the robustness–precision
instance of the capacity-bound, conflicting-objective trade-off long studied in multi-task
learning [31, 32].

**What is specific.** The earlier reading — that the precision floor is specific to a
teacher saturating inside the operating regime — does **not** survive §6.4. We removed that
specific limitation: a PID-CTBR teacher competent across the full 1–4 m band (100 %
recovery, cond-IAE 0.14–0.18 m) labelled the 1–3 m drift band densely, and precision still
did not move. What remains specific is the *capacity*: the conflict is a statement about
this policy family and parameter budget (≈ 3 M trainable). A higher-capacity or
precision-specialised architecture is the untested lever; it is out of scope for the
pre-registered coverage × sensing question and is the natural next hypothesis.

**Constructive implication.** Improving metric precision here is neither a representation,
a sensing, **nor** a coverage problem — the three fixes a reader would reach for first are
each excluded (§6.1, §6.3, §6.4). The intervention shows even the *oracle* range barely
moves precision and a richer cue collapses survival (§6.3); the decisive 2×2 shows that
even adding a competent far-range teacher's coverage on top does not move it either, and
trades precision for survival (§6.4). The remaining lever is therefore **model capacity and
the robustness–precision trade-off** — e.g. a dedicated precision head or a larger action
backbone that does not have to spend representational budget on wide-range recovery to stay
alive. Higher resolution, stereo/depth, representation regularisation, and wider BC
coverage are all predicted not to move the ~2.8 m floor on this architecture.

---

## 8. Limitations

- **Simulation only**; no real-robot validation. The renderer is synthetic and its
  distance encoding is a deliberate design choice (an information-poor monocular cue) — a
  feature for the information-gated analysis but a limit on external validity: the precision
  floor and its perspective-target remedy are established for this abstraction, not for
  photorealistic imagery. The §6.3 intervention controls for the renderer's information
  content (it hands the policy the oracle range directly), and the §6.4 teacher emits a CTBR
  command forward-compatible with a PX4 offboard interface, but validation under SITL/Gazebo
  rendering and on hardware (Jetson + PX4 MAVLink) remains future work.
- **The verdict is a capacity statement, not a teacher-competence one.** One might expect
  the floor to be conditional on a teacher that saturates beyond 2 m; §6.4 removes that
  possibility by supplying a teacher competent across the full 1–4 m band (100 % recovery,
  cond-IAE 0.14–0.18 m) — precision still does not move. The verdict is instead conditional
  on the **policy capacity**: the robustness–precision conflict is established for this
  policy family (flow matching with IMU-vision cross-attention, ≈ 3 M trainable), but
  *capacity itself was not varied* — that is the leading explanation, not a manipulated
  factor. Directly testing it with a higher-capacity or precision-specialised head is the
  recommended next step, and is out of scope for the pre-registered coverage × sensing
  question.
- **Data-volume asymmetry in the T×O 2×2.** T0 cells use hover-only data (500 ep), T1 cells
  hover + far-range recovery (1000 ep) — intrinsic to "add far-range labels". The asymmetry
  *favours* T1, yet precision does not improve under that favourable tilt, so it does not
  threaten the *negative* conclusion (it would matter only for a positive one). A
  size-matched near-recovery control is the fallback if a reviewer presses.
- **One task** (hover) and one policy family. The headline Dispersive null is established at
  the **official `flow_net` mid-block placement** (λ=0.5, τ=0.5, `/d`); we additionally
  probe the off-path `vis_pooled` placement (§6.1) and, beyond the official InfoNCE-L2 form,
  two **scale-invariant** dispersion criteria (unit-sphere InfoNCE, VICReg; §6.1) that
  remove the objective-gaming yet still do not move control. We did not exhaustively sweep
  λ, τ, or every candidate intermediate block — though §6.1–§6.2 explain mechanistically
  why those are unlikely to rescue it here.
- **Statistical power.** Three seeds per cell — a pooled across-seed Tier-1 std of ~6.3 pp
  (Dispersive 2×2) and a pooled cond-IAE std of 0.23 m (T×O 2×2) — powers these designs to
  detect only large effects. Both results are nulls *in the sense of no effect above seed
  noise*; a smaller true effect, and the directional high-variance frozen-encoder
  degradation of §5, cannot be sharply resolved at this seed count.
- T_obs = 2 frames could in principle yield range via motion parallax; we argue (and the
  flat empirical precision across all configs supports) that the 28 ms baseline and slow
  drift make this far weaker than the static size channel.

---

## 9. Conclusion

On a vision-based quadrotor hover task, under a frozen, artifact-robust protocol with a
measured oracle and an **official-code-faithful** Dispersive implementation (InfoNCE-L2 on
the `flow_net` mid-block, λ=0.5, `/d`), **Dispersive Loss does not improve survival or
precision** above seed noise (−2.2 pp Tier-1, ∈ noise). Even with a frozen encoder — where
the faithful term still trains the generative network — it does not help and is mildly
destabilising, so the frozen-encoder condition is an informative control rather than a
no-op. At **both** the faithful and an off-path placement the mechanism does not even cure
the collapse it targets — it games its scale-sensitive objective and worsens intrinsic
rank — and two **scale-invariant** criteria that *do* cure the collapse (effective rank to
75–85 % of dimensions, no norm inflation) still leave closed-loop control flat: survival is
decoupled from representation geometry whatever criterion enforces it.

We then ran the diagnosis to its end. The survival gains come from a
transfer/conditioning/recovery **recipe**, not Dispersive. And task precision — pinned at
cond-IAE ≈ 2.8 m, ~13 % of the state oracle, across every configuration — is gated by
**none** of the three leading candidates. A positive-control intervention rules out
**sensing** (handing the policy the oracle metric position error barely moves precision,
~36× the state oracle; a richer cue collapses survival; the FPV's range-blindness is itself
a fixable renderer artifact). A decisive, pre-registered **Teacher × Observation 2×2** then
rules out **coverage**: a teacher competent across the full 1–4 m band and a range-encoding
observation, supplied jointly, leave the floor at 2.93 m (no better than the neither-factor
control), and coverage and sensing *negatively interact* — coverage buys survival but costs
precision. The binding constraint for this task is therefore neither representation
collapse, nor sensing, nor coverage, but a **robustness–precision conflict** — directly
observed as that negative interaction — which we attribute to limited model capacity. That
attribution is the leading explanation, not a manipulated result; the next hypothesis to
test directly is model capacity — a precision-specialised head or a larger backbone — which
lies outside the pre-registered coverage × sensing question of this study. We release the
protocol, both 2×2 ablations, the intervention, and all diagnostics so the result and its
diagnosis are reproducible.

---

## Reproducibility / Artifacts

| Component | Script | Artifact |
|-----------|--------|----------|
| Frozen protocol + measured oracle | `scripts/evaluate_frozen_p0.py` | `evaluation_results/frozen_p0_leaderboard.json` |
| §4 P1 baselines (BC-vision-only, PPO-from-pixels), 3 seeds | `scripts/train_bc_vision_only.py`, `scripts/train_ppo_from_pixels.py` (+ `models/ppo_pixel.py`, `configs/ppo_from_pixels.yaml`), `scripts/evaluate_baselines_frozen.py`; seed driver `scripts/run_baseline_seeds_1_2.sh` | seed 0: `evaluation_results/baselines_frozen_leaderboard.json`, `…_ppopx.json`; seeds 1–2: `…_{bc,ppopx}_s{1,2}.json`; aggregate (mean±std): `…_baselines_frozen_seeds_aggregate.json` |
| §5 faithful 2×2 (P2f, headline) | `scripts/run_p2_ablation.py --faithful`, `scripts/evaluate_p2_ablation.py` | `evaluation_results/p2f_ablation_{manifest,leaderboard}.json` — `docs/experiment_report_faithful_dispersive.md` |
| legacy 2×2 ablation (off-path, §6.1 probe) | `scripts/run_p2_ablation.py`, `scripts/evaluate_p2_ablation.py` | `evaluation_results/p2_ablation_{manifest,leaderboard}.json` |
| §6.1 feature geometry — off-path `vis_pooled` (Table 3) | `scripts/measure_feature_collapse.py` | `evaluation_results/p2_feature_collapse.json` — `docs/experiment_report_feature_collapse.md` |
| §6.1 feature geometry — faithful `flow_net` mid-block (Table 4) | `scripts/measure_feature_collapse_flowmid.py` | `evaluation_results/p2f_feature_collapse_flowmid.json` |
| §6.1 scale-invariant dispersion forms (Table 5) | `scripts/run_p6_form_ablation.py` (→ `train_flow_v5.py --dispersive-form {cosine,vicreg}`), `scripts/evaluate_p6_form_ablation.py` | `evaluation_results/p6_form_ablation_{manifest,leaderboard}.json` — `docs/experiment_report_p6_scale_invariant.md` |
| §6.2 survival movers | — | `docs/experiment_report_survival_movers.md` |
| §6.3 coverage probe | `scripts/measure_ood_coverage.py` | `evaluation_results/p3b_ood_coverage.json` — `docs/experiment_report_ood_coverage.md` |
| §6.3 image-distance info | `scripts/measure_image_distance_info.py` | `evaluation_results/p3b_image_distance_info{,_nodr}.json` — `docs/experiment_report_image_distance_info.md` |
| §6.3 higher-res gate (Fig 5 left) | `scripts/measure_higher_res_gate.py` | `evaluation_results/p3b_higher_res_gate.json` — `docs/experiment_report_sensing_ablation.md` |
| §6.3 range-cue intervention, 3 seeds (Fig 5 right) | `scripts/run_p3b_rangecue.py` (→ `train_flow_v5.py --range-cue`, `evaluate_frozen_p0.py --cue-noise`) | `evaluation_results/p3b_rc_{clean,noised}{,_s12}_frozen.json` — `docs/experiment_report_sensing_ablation.md` |
| §6.4 decisive Teacher × Observation 2×2 (Table 6, Fig 6) | `scripts/collect_data_v7_pidctbr.py`, `scripts/run_p2to_ablation.py` (→ `train_flow_v5.py --hover-h5`), `scripts/evaluate_p2to_ablation.py` (→ `evaluate_frozen_p0.py --target-render`) | `evaluation_results/p2to_ablation_{manifest,leaderboard}.json` — `docs/experiment_report_p2to_decisive.md` |

Frozen protocol: 30 episodes, base seed 12345, σ = 2.0 exp-decay composite, paired init,
conditional-IAE over episodes surviving ≥ 250/500 steps, bootstrap 95% CI, measured PPO
oracle 0.9668.

---

## References

> Sources retrieved via NotebookLM (notebook *Generative RL & Flow Policy Research*,
> 2026-06-19). All author lists, venues, and identifiers were verified against the
> publisher of record (arXiv / official proceedings) on 2026-06-19; no entries remain
> unverified. References [31]–[32] (multi-task capacity / conflicting-gradient anchors for
> the §6.4 capacity conclusion) were added and verified against arXiv / NeurIPS proceedings
> on 2026-06-27.

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

[22] L. Jing, P. Vincent, Y. LeCun, Y. Tian. "Understanding Dimensional Collapse in
Contrastive Self-Supervised Learning." *ICLR*, 2022. arXiv:2110.09348.

[23] G. Zou, H. Wang, H. Wu, Y. Qian, Y. Wang, W. Li. "One Step is Enough: Dispersive
MeanFlow Policy Optimization (DMPO)." arXiv:2601.20701, 2026.

[24] E. Kaufmann, M. Gehrig, P. Foehn, R. Ranftl, A. Dosovitskiy, V. Koltun, D. Scaramuzza.
"Deep Drone Racing: Learning Agile Flight in Dynamic Environments." *Conf. on Robot
Learning (CoRL)*, 2018. arXiv:1806.08548.

[25] E. Kaufmann, A. Loquercio, R. Ranftl, M. Müller, V. Koltun, D. Scaramuzza. "Deep Drone
Acrobatics." *Robotics: Science and Systems (RSS)*, 2020.

[26] S. Ross, G. J. Gordon, J. A. Bagnell. "A Reduction of Imitation Learning and Structured
Prediction to No-Regret Online Learning (DAgger)." *AISTATS*, 2011. arXiv:1011.0686.

[27] M. Schneider, R. Krug, N. Vaskevicius, L. Palmieri, J. Boedecker. "The Surprising
Ineffectiveness of Pre-Trained Visual Representations for Model-Based Reinforcement
Learning." *NeurIPS*, 2024. arXiv:2411.10175.

[28] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, D. Meger. "Deep
Reinforcement Learning that Matters." *AAAI*, 2018. arXiv:1709.06560.

[29] R. Agarwal, M. Schwarzer, P. S. Castro, A. C. Courville, M. G. Bellemare. "Deep
Reinforcement Learning at the Edge of the Statistical Precipice." *NeurIPS*, 2021.
arXiv:2108.13264.

[30] A. Patterson, S. Neumann, M. White, A. White. "Empirical Design in Reinforcement
Learning." *Journal of Machine Learning Research*, 25, 2024. arXiv:2304.01315.

[31] O. Sener, V. Koltun. "Multi-Task Learning as Multi-Objective Optimization."
*NeurIPS*, 2018. arXiv:1810.04650.

[32] T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, C. Finn. "Gradient Surgery for
Multi-Task Learning." *NeurIPS*, 2020. arXiv:2001.06782.

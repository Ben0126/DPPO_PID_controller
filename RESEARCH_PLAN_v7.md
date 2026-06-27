# RESEARCH_PLAN_v7 — Breaking the Precision Floor: Competent Far-Range Teacher × Perceivable Observation

**Version:** 7.0
**Date:** 2026-06-24
**Supersedes:** the *diagnosis* of `RESEARCH_PLAN_v6.md`. v6 *localized* the binding
constraint (far-range coverage / teacher-competence) and proved each candidate fix fails
*alone*; v7 is the constructive follow-up that tests the fixes *jointly* and attempts to
break the floor.
**Target venues:** ICRA / robot-learning workshop (simulation-only; no real-robot claim).

---

## 0. Context — why v7 exists

The negative-result paper (`docs/paper_negative_result_draft.md`, v0.4) did more than
falsify Dispersive Loss; it **localized the binding constraint** for vision-based
quadrotor hover. Under a frozen, multi-seed protocol it ruled out the two front-runners:

- **Not representation collapse.** Faithful Dispersive (flow_mid, λ=0.5, τ=0.5, `/d`)
  gives −2.2 pp Tier-1 (∈ noise); survival is *decoupled* from `vis_pooled`/flow_mid rank
  (15× / 7× rank swings, flat survival).
- **Not the sensing channel alone.** A positive-control intervention that *hands the
  policy the oracle metric position error* barely moves precision (cond-IAE 2.91 → 2.43 m,
  still ~36× the 0.068 m oracle) and a richer 3-D cue *collapses* survival.

The paper's verdict: precision is pinned at **cond-IAE ≈ 2.8 m (~13 % oracle)** by the
**absence of a learned far-range recovery behaviour in the 1–3 m band** — a
**coverage / teacher-competence gap**. The root cause is structural and concrete:

1. `envs/quadrotor_env_v4.py:320-321` — in `hover` mode `self.target_position =
   init_pos.copy()`, so **target ≡ init**. Even "recovery" demos (Swift mode, `--pos-range`)
   only relocate the hover point; there are **no position-error labels in the 1–3 m band**.
2. The privileged state-based PPO teacher itself **crashes from > 2 m offset** (3 m → 0/20
   recover), so it *cannot* label that band even if asked.
3. The FPV crosshair **size saturates at 2 px for d ≥ 2 m**
   (`size = max(2,min(6,⌊6/(d+0.5)⌋+dr))`, `quadrotor_visual_env.py:188`) — but the paper's
   higher-res gate showed this is a **renderer target artifact, not the pixel count**: a
   non-saturating perspective target restores far-range R² to 0.42 at the same 64 px.

**v7 tests the one combination the paper never tried: supply BOTH the missing labels AND a
perceivable observation, jointly.** The paper proved each fix *alone* fails. The decisive,
pre-registered question of v7 is whether **(competent far-range teacher) × (perspective
target renderer)** *together* break the 2.8 m floor.

**Intended outcome (either is publishable):**
- **Floor broken** → the paper upgrades from a pure negative result to "we localized the
  constraint *and broke it*" — a constructive positive contribution.
- **Floor not broken even with both** → the negative result deepens into a stronger
  structural claim (e.g. a model-capacity / robustness–precision conflict), with the 2×2 as
  proof that coverage + sensing are *jointly* insufficient.

---

## 1. Central hypothesis (pre-registered)

> **H_v7.** The closed-loop precision floor (cond-IAE ≈ 2.8 m) is broken **only when both**
> (a) the BC dataset contains *competent* 1–3 m recovery labels (target ≠ init), **and**
> (b) the FPV observation can encode metric range (perspective, non-saturating target).
> Neither factor alone suffices (already shown by the paper's sensing intervention and
> higher-res gate).

**Decisive design — a 2×2 (Teacher coverage T × Observation O), 3 seeds/cell, frozen P0.**
This deliberately mirrors the paper's §5 Dispersive×E2E 2×2 so the methodology and
decision rule carry over unchanged.

| Cell | Teacher labels 1–3 m | Renderer | Pre-registered prediction |
|------|----------------------|----------|---------------------------|
| **T0O0** | no (current data) | crosshair (saturating) | current frontier ≈ 2.8 m floor (control) |
| **T1O0** | yes (far-range demos) | crosshair | labels but unperceivable → **still ~2.8 m** (sensing blocks) |
| **T0O1** | no | perspective | perceivable but no labels → **still ~2.8 m** (coverage blocks) |
| **T1O1** | yes | perspective | **decisive** — both present → predict floor **breaks** |

**Decision rule (pre-registered).** The floor is "broken" iff, over 3 seeds:
`cond-IAE(T1O1) < cond-IAE(T0O0) − pooled_std` **AND** `cond-IAE(T1O1) ≤ 1.5 m`
(a meaningful absolute target, ~2× off the floor) **AND** survival/Tier-1 do **not**
collapse (`survival(T1O1) ≥ survival(T0O0) − pooled_std`). The survival guard is mandatory
because the paper's pos3d cue showed precision "wins" that are really survival collapse.
Report cond-IAE only when `n_cond ≥ ~15`.

**Primary metric (unchanged from v6):** Tier-1 pass-rate / survival on the frozen protocol;
cond-IAE conditional on survival ≥ 250 steps; composite score secondary + flagged.

---

## Phase 0 — Cheap gates BEFORE the multi-day pipeline

Per the standing rule (`feedback_evidence_before_training`), run the measurements that could
redirect the pipeline before spending GPU-days. These gates **decide the teacher source and
confirm the renderer**; we proceed to the full pipeline regardless, but the gates determine
*which* branch.

> **✅ ALL THREE GATES PASSED (2026-06-25).** Report: `docs/experiment_report_p0_teacher_renderer_gates.md`.
> - **Gate A — PASS, teacher = PID-CTBR (no PPO retrain).** New `CascadePIDController.compute_ctbr_action`
>   (cascade Levels 1–3 → normalized CTBR, inverse of `_decode_action`) audited on the v4 env via
>   `scripts/audit_recovery_ctbr.py` (`evaluation_results/p0_recovery_audit_ctbr.json`). With a **gentle
>   recovery tune (vel_max=1.0, Kp_pos=0.8)** PID-CTBR recovers the **full 1–4 m band at 100% survival,
>   cond-IAE 0.14–0.18 m** (~2× oracle 0.068 m, ~18× better than the 2.8 m closed-loop floor). The PPO
>   expert is oracle-grade ≤2 m (0.067 m) but **collapses ≥3 m** (8%/4% survive, 0 conditional) → cannot
>   teach the far band. The baseline PID gains (vel_max=2.0) trip the 60° tilt limit on diagonal 2 m
>   offsets ("death valley", 63.7%); easing the approach fixes it. → **the 2.8 m floor is NOT a
>   teacher-incapacity wall; Phase 1 PPO retrain is unnecessary.**
> - **Gate B — PASS, perspective renderer confirmed in-env.** Ported the perspective AA-disk into the
>   production env behind `QuadrotorVisualEnv(target_render="perspective")` (the prototype subclass in
>   `measure_higher_res_gate.py` is deleted → single source of truth). Re-measured far-range (≥1.5 m)
>   image→distance R²: crosshair **0.05 → perspective 0.40** (DR-on, `measure_higher_res_gate` dual form,
>   matches the documented 0.12→0.42); crosshair **0.59 → perspective 1.00** (clean no-DR ceiling,
>   `measure_image_distance_info`). DR-on far info goes from ≈0 to recoverable.
> - **Gate C — PASS (confirmed earlier).** Added the `setpoint_offset` reset mode now via the unused
>   `reset(options=...)` kwarg in `quadrotor_env_v4.py` (target shifted by a caller vector; default
>   unchanged) — reused by the audit and by the Phase-1/3 wide-init collection.

- **Gate A — Teacher competence audit (decides teacher source).** Measure existing
  controllers' recovery from a *setpoint offset* (target ≠ init) at {1, 2, 3, 4} m, ~20
  trials each, reporting recover-rate and final IAE:
  - PPO expert `ppo_expert_v4/20260419_142245` (known: crashes > 2 m — the baseline);
  - **PID waypoint baseline** (`memory/project_pid_baseline.md`: "Waypoint 1.18 m / 0 crash"
    — strong candidate for a ready-made 0-crash far-range teacher);
  - any LQR/MPC controller present in the repo.
  Sub-check: the chosen teacher must emit the **same CTBR action space** the BC policy
  imitates. **Decision:** if any controller recovers 1–3 m at ~0 crash → use it as the
  teacher, **skip Phase 1 retrain**. Else → Phase 1 retrains PPO.

- **Gate B — Perspective renderer far-R² (confirms Direction 3 in-env).** Port the prototype
  `PerspectiveTargetVisualEnv._draw_target()` (`scripts/measure_higher_res_gate.py:69-101`)
  into the production env behind a flag (Phase 2) and confirm with
  `scripts/measure_image_distance_info.py` that far-range (≥ 1.5 m) image→distance R² rises
  from ~0.12 (crosshair) to ≥ ~0.42 at 64 px (and check 128 px).

- **Gate C — Env `target ≠ init` support.** Locate/confirm a setpoint/waypoint `target_type`
  in `envs/quadrotor_env_v4.py` that decouples `target` from `init_pos`. If absent, scope the
  minimal env change (an offset-setpoint reset mode) for both teacher retrain (Phase 1) and
  data collection (Phase 3).

---

## Phase 1 — Build the far-range teacher (retrain branch; skipped if Gate A finds a competent controller)

Goal: a controller that recovers from 1–3 m offset at ~0 crash, **competent to label the
1–3 m band**.

- **Env change:** add a `setpoint_offset` reset mode to `envs/quadrotor_env_v4.py` —
  `target` fixed, `init_pos = target + random_offset` with an offset-range **curriculum**
  (start ≤ 1 m, expand toward 3 m as recover-rate holds). Surgical fix for the `target ≡ init`
  root cause at lines 320-321; gate behind a flag so existing hover collection is untouched.
- **Retrain PPO:** `scripts/train_ppo_expert_v4.py` + `configs/ppo_expert_v4.yaml`
  (`models/ppo_expert.py` unchanged — already reads the 15-D state incl. `pos_error_body`).
  Keep the hover-bias init. 3 M steps, 8 async envs. Curriculum on offset range, not reward
  shape (`quadrotor_env_v4.py:466-507` already has a linear `w_pos` penalty).
- **Gate:** new teacher must recover from 3 m at ~0 crash (re-run Gate A). If even a
  from-scratch RL teacher cannot learn 3 m recovery here, that is itself a **headline finding**
  (the constraint is deeper than coverage) → Phase 7 negative-deepening outcome.

---

## Phase 2 — Perspective renderer integration (Direction 3)

`envs/quadrotor_visual_env.py`:
- Add a `use_perspective_target` flag (env arg / config).
- Add `_draw_target_perspective(image, px, py, target_dist, W, H)` porting the
  optical-expansion radius `(W*focal)*physical_size/(dist+0.1)` + anti-aliased disc from
  `measure_higher_res_gate.py:80-101`. Keep DR mechanism (`_dr_focal_scale`,
  `_dr_crosshair_d` scaled to resolution).
- Dispatch at line ~153: `if self.use_perspective_target: _draw_target_perspective(...)`.
- **Note:** any policy trained on perspective images must be *evaluated* on perspective
  images — see Phase 4 eval flag.

---

## Phase 3 — Re-collect far-range demonstrations (fill the coverage gap)

> **✅ DONE & COVERAGE GATE PASSED (2026-06-25).** Report:
> `docs/experiment_report_p3_dataset_collection.md`; artifact `evaluation_results/p3_coverage_v7.json`.
> - **New collector** `scripts/collect_data_v7_pidctbr.py` (instead of extending the PPO-driven
>   `collect_data_v4_recovery.py`): a **single PID-CTBR teacher** (gentle Gate-A gains
>   `vel_max=1.0, Kp_pos=0.8, omega_max=6.0`) collects **both** hover (T0) and far recovery (T1),
>   so the T-axis is a pure coverage operation with **no teacher swap** (user decision 2026-06-25).
>   Each trajectory is **dual-rendered** (crosshair + perspective) from the *same* dynamics state
>   and *same* per-frame DR noise (`np.random` state saved/restored) → O is a **byte-clean swap**.
> - **Four datasets** `data/expert_demos_v7_{hover,far}_{crosshair,persp}.h5` — 500 ep each,
>   **100 % survival / 0 crash** in both modes; far offsets span [1.001, 2.995] m. h5 format =
>   `collect_data_v4_recovery`; `episode_type` drives `FlowDatasetV5` task-cond (hover→[1,0],
>   recovery→[0,1]).
> - **Coverage HARD GATE PASS** (`scripts/check_coverage_v7.py`): far `frac in [1,3] m` = **0.117**
>   (p99 2.59 m) vs the paper's ~0.4 % `frac>1 m` — a ~29× jump; hover `frac>1 m` = **0.000**
>   (max 0.274 m) → T0 has no far coverage, T1 fills it. Crosshair vs perspective **states
>   byte-identical** for both modes (clean O-swap verified).
> - **Next:** Phase 4 — the decisive frozen-P0 T×O 2×2 (12 BC runs = 4 cells × 3 seeds,
>   one-at-a-time; O1 arms train **and** eval on the perspective renderer).

- Extend `scripts/collect_data_v4_recovery.py` with `--setpoint-offset-range` (target ≠ init,
  1–3 m) and `--use-perspective-target`, driving the Phase-1 teacher (or the Gate-A
  controller). Output four datasets matching the 2×2:
  `expert_demos_v7_{near,far}_{cross,persp}.h5`.
  *(Implemented instead as the standalone `scripts/collect_data_v7_pidctbr.py`; files named
  `expert_demos_v7_{hover,far}_{crosshair,persp}.h5`.)*
- **Coverage check (cheap, mandatory before retraining):** `scripts/measure_ood_coverage.py`
  on the new `far` datasets → position-error mass must land in 1–3 m (the paper's `frac>1 m`
  ~0.4 % should jump). If not filled, the env change is wrong — stop and fix.
  *(Done via `scripts/check_coverage_v7.py`, which reuses `measure_ood_coverage.pos_err_from_h5`;
  GATE PASS.)*
- Hardware: RAM ~12 GB free → 500 hover + 500 recovery max; `batch_size=256` (OOM at 512).

---

## Phase 4 — The decisive 2×2: Teacher × Observation (3 seeds, frozen P0)

> **⛔ DONE — H_v7 REFUTED, floor NOT broken, negative result DEEPENED (2026-06-26).** Report:
> `docs/experiment_report_p2to_decisive.md`; artifacts `evaluation_results/p2to_ablation_{manifest,leaderboard}.json`,
> 12 ckpts `checkpoints/flow_policy_v5/p2to_*_s{0,1,2}/`.
> - **12 BC runs done** (4 cells × 3 seeds, one-at-a-time). Tooling: `train_flow_v5.py --hover-h5`
>   (override hover-pool path → O1 trains on perspective h5), `evaluate_frozen_p0.py --target-render`
>   (= the planned `--use-perspective-target`, generalised; O1 evals on perspective). Driver
>   `run_p2to_ablation.py` + aggregator `evaluate_p2to_ablation.py` (cond-IAE PRIMARY, H_v7 rule).
>   Also fixed a real OOM in `FlowDatasetV5` (two-pass pre-alloc, ~21 GB→~10.7 GB peak,
>   equivalence-tested byte-identical; `T1O1_s2` retrained identically val_flow 0.0108).
> - **2×2 cond-IAE (mean±std, frozen P0; oracle 0.0675 m):** T0O0 **2.69±0.15 m**, T0O1 2.48±0.14 m,
>   T1O0 2.71±0.22 m, **T1O1 2.93±0.18 m** (36–43× oracle, composite 20–23 % oracle). **Verdict:
>   T1O1 vs T0O0 Δ=+0.24 m (pooled std 0.23 m) → NOT significant, NOT ≤1.5 m → FLOOR NOT BROKEN.**
> - **Structure (this is the result):** coverage (T1) buys **survival** (+8 pp crosshair / +14 pp
>   perspective, Tier1→~100 %) but moves precision 0; sensing (O1) nudges precision within noise;
>   the **interaction is NEGATIVE** — best precision is T0O1 (2.48 m), adding far-recovery on top
>   (→T1O1) *degrades* it to 2.93 m. ⇒ **Robustness–Precision Capacity Conflict**, now RL-free /
>   pre-registered / 3-seed (was confounded by RL in v5 Finding #8). Binding constraint = capacity,
>   **not** coverage, **not** sensing. → Phase 7 **negative-deepening** branch.

- Train the **D0E1 frontier recipe** (Dispersive OFF, encoder E2E, H4-transfer init,
  task-conditioned) on each T×O combination via `scripts/train_flow_v5.py`
  (`--transfer-from-h4`, `--lambda-disp 0.0`, `--recovery-h5 <far|near>`, `--seed`,
  `--tag p2to_<cell>_s<seed>`). New sweep driver `scripts/run_p2to_ablation.py`
  (mirror `run_p2_ablation.py`; cells T0O0/T1O0/T0O1/T1O1 × seeds {0,1,2}, sequential).
- **Eval:** `scripts/evaluate_frozen_p0.py` + `scripts/evaluate_p2_ablation.py` aggregation.
  **Add `--use-perspective-target` to the eval env** so O1 cells are scored on their training
  renderer (observation is the manipulated variable; physics init paired by `seed = 12345+i`).
- Headline output: `evaluation_results/p2to_ablation_leaderboard.json` + H_v7 verdict.

---

## Phase 5 — Methodology hardening (Direction 2, cross-cutting)

Enforced as default for every v7 run: 3-seed `evaluate_frozen_p0` + `evaluate_p2_ablation`
aggregation (paired init, conditional-IAE over survival ≥ 250, bootstrap 95 % CI, measured
oracle 0.9668, mean ± std, pooled-std decision rule) for **all** Phases 1/4/6. Pre-register
thresholds (the H_v7 rule) before training; **no single-seed leaderboard rows** (the paper's
PPO-from-pixels swung 0 %↔47 % Tier-1 on one seed). Thin seed-sweep driver pattern reused
from `run_p2_ablation.py`.

---

## Phase 6 — Scale-invariant on-path regularizer ablation (Direction 4, formal)

> **✅ DONE (2026-06-27) — rebuttal CLOSED, negative result REINFORCED.** 6 new runs (cosine/vicreg ×3
> seeds) on the fixed P2f D1E1 recipe, off/infonce reused from P2f, frozen-P0 + flow_mid geometry.
> **[A] objective-gaming REMOVED:** infonce GAMES (feat_norm **8.93× off**, eff_rank **collapses 221→36**
> = 3.5 % of 1024 dims); cosine/vicreg are **clean** (feat_norm **1.33×/1.36×**, eff_rank **221→769/867**
> = 75 %/85 % — genuine high-rank dispersion, the *intended* effect, without the norm cheat). So the gaming
> is an artifact of the scale-sensitive L2 criterion. **[B] control NOT improved:** all four forms in the
> same band (survival 60–65 %, Tier1 82–92 %, cond-IAE **2.9–3.1 m**); `any_control_improved=False` — both
> scale-invariant forms even *regress* cond-IAE slightly (+0.13–0.21 m, just past pooled std). A **24× rank
> swing buys ~0 control** → flow_mid geometry is **decoupled** from survival/precision (on-path analogue of
> §6.1's `vis_pooled` rank ⟂ survival). Artifacts: `evaluation_results/p6_form_ablation_{manifest,leaderboard}.json`,
> `docs/experiment_report_p6_scale_invariant.md`. → fold into §5/§6.1 in Phase 7.

Closes the last rebuttal ("you used a scale-sensitive criterion"). The paper showed faithful
InfoNCE-L2 *games* its objective by norm inflation (feat_norm ~9× on flow_mid, ~287× off-path)
and *worsens* intrinsic rank.

- Implement scale-invariant variants in `models/flow_policy_v5.py` alongside the existing
  `_dispersive_loss` / `_dispersive_loss_infonce`:
  - **(a) unit-sphere InfoNCE** — L2-normalize `mid_feat` to the unit sphere before the
    InfoNCE-L2 distance (cosine-based; norm inflation impossible by construction);
  - **(b) VICReg-style** variance + covariance term on the on-path flow_mid features.
  Expose via `--dispersive-form {infonce,cosine,vicreg}` in `train_flow_v5.py`
  (default `infonce` = unchanged).
- Ablation on the **best v7 recipe** (T1O1 if floor broke, else D0E1), 3 seeds, frozen P0.
  Measure **both** closed-loop (Tier-1/survival/cond-IAE) **and** feature geometry
  (`scripts/measure_feature_collapse_flowmid.py`: eff_rank, feat_norm — confirm *no* norm
  inflation). Test: a scale-invariant criterion prevents objective-gaming; whether it also
  moves closed-loop control is the open question (prior: rank ⟂ survival → likely flat).

---

## Phase 7 — Write-up / integration

> **✅ DONE (2026-06-27) — both 2×2s folded into the paper (v0.5).** Branch taken: floor NOT
> broken → negative result deepened.
> - **§6.3 reframed + new §6.4 "The decisive test: Teacher × Observation 2×2"** added to
>   `docs/paper_negative_result_draft.md`: §6.3 no longer ends on the "needs a competent
>   teacher" speculation but frames it as a hypothesis; §6.4 reports the 4-cell grid
>   (Table 6), the FLOOR-NOT-BROKEN verdict (T1O1 2.93 vs T0O0 2.69, Δ+0.24 m ∈ pooled std
>   0.23 m), and the **negative coverage×sensing interaction** (T0O1 2.48 → T1O1 2.93 m).
>   Conclusion upgraded from "coverage/teacher-limited" to a **triple exclusion → capacity /
>   robustness–precision conflict** (claim strength aligned per Remi: conflict observed,
>   capacity = leading-but-untested explanation). Cites `experiment_report_p2to_decisive.md`.
> - **Title** upgraded (user decision): "Representation Collapse Is Not the Bottleneck —
>   and Neither Is Coverage or Sensing: A Negative Result and Capacity Diagnosis…".
> - **Abstract / §1 / §7 / §8 / §9 re-synced**; §5↔§6.4 protocol alignment note added; **Phase 6
>   scale-invariant ablation folded into §6.1 (Table 5) + §7 + §8** (closes the last §6.1
>   "scale-sensitive criterion" rebuttal).
> - **Figure 6 `docs/figures/teacher_obs_2x2.png`** added via `scripts/make_paper_figures.py`
>   (cond-IAE grid + survival + verdict/interaction annotations; reads `p2to_ablation_leaderboard.json`).
> - **Exported** `docs/paper_negative_result_draft.{html,pdf}` (6 figs base64-embedded, Unicode OK).
> - **Reviewed** (remi → claim-strength + meta-commentary fixes applied) and **prose-polished**
>   (deep-science-writer Phase-4 humanization; manuscript already AI-ism-clean, numbers/conclusions unchanged).
> - **Figures renumbered to appearance order (2026-06-27):** Fig 1 single_seed (§4), 2 ablation_forest
>   (§5), 3 rank_survival (§6.1), 4 crosshair (§6.3), 5 sensing (§6.3), 6 teacher_obs (§6.4); all in-text
>   refs + `make_paper_figures.py` (functions renamed to match, embedded "see Fig. 5" cross-ref fixed)
>   updated; re-exported.
> - **Capacity-conflict literature anchors added (2026-06-27):** [31] Sener & Koltun
>   (multi-task as multi-objective optimization, NeurIPS 2018, arXiv:1810.04650) + [32] Yu et al.
>   PCGrad (conflicting gradients, NeurIPS 2020, arXiv:2001.06782), web-verified, cited in §6.4/§7
>   to anchor the robustness–precision capacity conclusion (closes Remi Minor #6).
> - **⇒ RESEARCH_PLAN_v7 Phases 0–7 ALL COMPLETE; no deferred write-up items remain.**

- ~~If floor broken (T1O1): pivot to "constraint localized *and broken*"…~~ (not taken — H_v7 refuted).
- **Floor not broken (ACTIVE):** deepen the negative result — coverage + sensing are *jointly*
  insufficient → **capacity / robustness–precision conflict** (the Phase-4 2×2 shows it directly:
  coverage buys survival but the coverage×sensing interaction *costs* precision; consistent with v5
  curriculum collapse but now RL-free / pre-registered / 3-seed). The 2×2 is the proof object;
  fold into `docs/paper_negative_result_draft.md` §6/§7 as the decisive exclusion experiment
  ("removed the two leading candidate constraints — coverage, sensing — and the floor held").
- Re-export via `scripts/export_paper.py`; deep-science-writer skill for prose polish once numbers
  are final.
- **Order (user decision 2026-06-26): Phase 6 BEFORE Phase 7.** Phase 6 (scale-invariant on-path
  regulariser ablation) is *orthogonal to the T×O precision result* but is a **pending formal
  ablation the user committed to on 2026-06-24** — it closes the §6.1 "you used a scale-sensitive
  criterion" rebuttal (faithful InfoNCE-L2 games its objective via norm inflation). Run Phase 6
  first (or in parallel), then fold both 2×2s into the write-up in one pass.

---

## Critical files to create / modify

| Area | File | Change |
|------|------|--------|
| Env root-cause (Dir 1) | `envs/quadrotor_env_v4.py:~320` | `setpoint_offset` reset mode (`target ≠ init`, offset curriculum), flagged |
| Teacher retrain (Dir 1) | `scripts/train_ppo_expert_v4.py`, `configs/ppo_expert_v4.yaml` | offset-curriculum PPO retrain (branch only if Gate A fails) |
| Renderer (Dir 3) | `envs/quadrotor_visual_env.py:~153,178-193` | `use_perspective_target` flag + `_draw_target_perspective()` |
| Data (Dir 1+3) | `scripts/collect_data_v4_recovery.py` | `--setpoint-offset-range`, `--use-perspective-target`; 4 datasets |
| Decisive 2×2 (Dir 1+3) | new `scripts/run_p2to_ablation.py` | T×O cells × 3 seeds sweep + manifest |
| Eval (Dir 2) | `scripts/evaluate_frozen_p0.py` | `--use-perspective-target` eval-env flag for O1 arms |
| Regularizer (Dir 4) | `models/flow_policy_v5.py:187-224`, `scripts/train_flow_v5.py` | `--dispersive-form {infonce,cosine,vicreg}` scale-invariant variants |

**Reused as-is (do not reinvent):** `scripts/measure_image_distance_info.py` (Gate B),
`scripts/measure_ood_coverage.py` (Phase 3 coverage check),
`scripts/measure_higher_res_gate.py` (perspective prototype),
`scripts/evaluate_p2_ablation.py` (3-seed aggregation + decision rule),
`scripts/measure_feature_collapse_flowmid.py` (Phase 6 geometry).

---

## Compute budget & operational rules

- **Gates (Phase 0):** minutes–hours (inference/measurement only).
- **Teacher retrain (Phase 1, if needed):** 3 M-step PPO ≈ a few hours.
- **Data re-collection (Phase 3):** 500+500 ep × renderer variants, hours.
- **Decisive 2×2 (Phase 4):** 12 BC runs × ~1.5–2.5 h ≈ ~1 day, **one-at-a-time** (Failure Mode #7).
- **Dir-4 ablation (Phase 6):** ~6–9 runs, ~1 day.
- **Total ≈ 2–3 GPU-days, sequential.**
- **Always** use `dppo/Scripts/python.exe -m ...` with Bash `run_in_background=true`
  (never `nohup`/pipe/`source activate`); monitor via TensorBoard event API, not buffered
  stdout; `batch_size=256`; kill competing python before launching.

---

## Verification (end-to-end)

1. **Gate B:** `python -m scripts.measure_image_distance_info` (perspective) →
   far-R²(≥1.5 m) ≥ ~0.42 (vs ~0.12 crosshair).
2. **Phase 3 coverage:** `python -m scripts.measure_ood_coverage` on `*_far_*.h5` → position
   error mass in 1–3 m. Hard gate before any BC run.
3. **Phase 4 decisive:** `python -m scripts.evaluate_p2_ablation --manifest
   evaluation_results/p2to_ablation_manifest.json ...` → apply the H_v7 decision rule on
   `cond-IAE(T1O1)` vs `cond-IAE(T0O0)` with survival guard; verdict printed.
4. **Phase 6:** `measure_feature_collapse_flowmid.py` on cosine/VICReg runs → feat_norm O(1).
5. All numbers trace to `evaluation_results/*.json`; figures via `scripts/make_paper_figures.py`.

---

## Anticipated rebuttals (reviewer defense)

### R1 — "Why a PID-CTBR teacher / a custom simulator, not PX4's architecture + SITL?"

The question bundles three things: (1) the teacher controller, (2) the simulation platform,
and (3) an implicit misread that the contribution is *a controller*. It is not — the
contribution is a **diagnosis of a vision-based imitation policy's precision limits**; the
teacher is **instrumentation (a label source)**, not the scientific object. Disarm (3) first
and most of the fire is gone. Key points: the cascade controller is **architecturally
identical to PX4's** (position→velocity→attitude→rate) and emits exactly the collective-thrust
+ body-rate command PX4's offboard `SET_ATTITUDE_TARGET` consumes — it is a PX4-*equivalent*
controller, and CTBR was chosen for forward-compatibility with PX4 deployment. The **decisive
Teacher×Observation 2×2 requires a single self-consistent simulator** (hold every factor fixed
but the two of interest); PX4-SITL would confound teacher coverage with a change of both
controller and physics. The **observation-axis intervention** (perspective target restoring
R²_far 0.05→0.40) needs control of the renderer at the *information* level, which a
photorealistic SITL stack does not afford. Teacher quality is judged against the **in-sim
oracle** (state-PPO 0.068 m / 100%); PID-CTBR already reaches **~2× oracle across 1–4 m**, so a
different controller unlocks no headroom. The teacher is a **swappable component** (re-tuned
PX4 rate controller / LQR / MPC would serve identically).

**Paper paragraph (EN, Method / rebuttal):**

> *Our teacher is instrumentation, not a contribution: it supplies competent CTBR recovery
> labels for behavior cloning, and the study's claims concern the conditions under which a
> vision policy's precision floor breaks, independent of the label source. The cascade
> controller we use is architecturally identical to PX4's multicopter stack
> (position→velocity→attitude→rate) and emits exactly the collective-thrust + body-rate command
> that PX4's offboard interface (`SET_ATTITUDE_TARGET`) consumes; it is, functionally, a
> PX4-equivalent controller. We deliberately instantiate it inside a single, self-consistent
> simulator so that our decisive Teacher×Observation ablation holds every factor fixed but the
> two of interest — using PX4-SITL would confound teacher coverage with a change of both
> controller and physics. Critically, the observation-axis intervention (a non-saturating
> perspective target that restores far-range range information, R²_far 0.05→0.40) requires
> control of the renderer at the information level, which a photorealistic SITL stack does not
> afford. The teacher is a swappable component: any competent CTBR controller (a re-tuned PX4
> rate controller, LQR, MPC) would serve identically, and ours already reaches ~2× the in-sim
> oracle across 1–4 m, leaving no headroom a different controller would unlock.*

**論文段落（中，方法/答辯）：**

> *本研究的教師是「儀器」而非貢獻：它只負責替 behavior cloning 提供勝任的 CTBR 恢復標籤，而本文的
> 主張在於 vision policy 的精度地板「在什麼條件下會被打破」，與標籤來源無關。我們採用的串級控制器在
> 架構上與 PX4 多旋翼控制堆疊完全相同（position→velocity→attitude→rate），且輸出的正是 PX4 offboard
> 介面（`SET_ATTITUDE_TARGET`）所接收的 collective-thrust + body-rate 指令——功能上即為 PX4 等價控制器。
> 我們刻意將它實例化於單一、自洽的模擬器內，使決定性的 Teacher×Observation ablation 能固定所有因子、
> 只變動兩個目標因子；改用 PX4-SITL 會使「教師覆蓋」與「控制器+物理的同時改變」相互混淆。關鍵在於觀測
> 軸的介入（非飽和透視目標將遠距資訊 R²_far 從 0.05 提升至 0.40）需要在「資訊層級」操控渲染器，而
> 寫實的 SITL 堆疊無法提供。教師是可替換元件：任何勝任的 CTBR 控制器（重新調校的 PX4 rate controller、
> LQR、MPC）皆可等效替代，而我們的教師在 1–4 m 已達 in-sim oracle 的約 2 倍，換控制器並無額外可解鎖的
> 餘裕。*

### R2 — External validity (own this honestly; do NOT hand-wave)

The real exposure is sim-only + synthetic renderer. State it proactively in Limitations and
frame the CTBR choice as PX4-forward-compatible (deployment is future work, not avoidance).

**Paper paragraph (EN, Limitations):**

> *Our study is conducted in a lightweight simulator with a synthetic 64×64 FPV renderer chosen
> for its controllability. This is a feature for the information-gated analysis but a limit on
> external validity: the precision floor and the perspective-target remedy are established for
> this abstraction, not for photorealistic imagery. Validating the finding under
> PX4-SITL/Gazebo rendering and on hardware (Jetson Orin Nano + PX4 MAVLink, our chosen CTBR
> interface being directly compatible) is future work.*

**論文段落（中，限制）：**

> *本研究於輕量模擬器、以合成 64×64 FPV 渲染器進行；該渲染器因其可控性而被採用。這對 information-gated
> 分析是優點，卻是 external validity 的限制：精度地板與透視目標補救僅在此抽象下成立，未涵蓋寫實影像。
> 在 PX4-SITL/Gazebo 渲染與硬體（Jetson Orin Nano + PX4 MAVLink；我們所選的 CTBR 介面可直接相容）上
> 驗證此發現，列為未來工作。*

**Strongest fallback if pressed:** cite the number — PID-CTBR already reaches ~2× the measured
in-sim oracle (0.068 m) across 1–4 m, far above the 2.8 m closed-loop floor, so swapping in
PX4's controller cannot change any conclusion in this simulator.

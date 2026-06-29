# RESEARCH_PLAN_v9 — Breaking the Hover-Specialist's Own Closed-Loop Precision Wall (~2.48 m)

**Version:** 9.0
**Date:** 2026-06-29
**Supersedes the *diagnosis* of v8.** v8 tested the leading v7 explanation (capacity) and
**refuted it**: scaling `flow_net` 3.3× (13.5 M → 44 M) left cond-IAE at 2.63 m (Finding #13).
v8 left two distinct, un-conflated candidates — **(a) capacity *allocation*** and **(b) a
closed-loop limit**. v9 attacks **(b)**, the deployability path.
**Target venues:** ICRA / robot-learning workshop (simulation-only; no real-robot claim).

---

## 0. Context — the real wall is the hover specialist's *own* 2.48 m, not 2.93 m

The v7 T×O 2×2 (`evaluation_results/p2to_ablation_leaderboard.json`) contains the decisive
observation v8 sharpened:

- **T0O1 (hover-only + perspective) = cond-IAE 2.48 m** — a *pure hover specialist*, with
  **zero allocation conflict** (no recovery data to share budget with) and **zero coverage
  dilution**. It is the best precision any cell reaches.
- **T1O1 (= v8 baseline S) = 2.93 m.** v8 XL (44 M) = 2.63 m.

So the v8 "allocation" candidate **(a)** is real but **upper-bounded by 2.48 m**: a dedicated
precision pathway is, by construction, a hover specialist, and a hover specialist's own
closed-loop floor is 2.48 m. **Allocation can recover 2.93 → 2.48 m (a survival–precision
rebalance) but cannot cross the 1.5 m deployability target.** The irreducible wall is the
**2.48 m of the hover specialist itself**, reached despite:

1. **near-perfect open-loop fit** — Gate A (v8) measured T0O1's open-loop hover action MSE at
   **0.00035** (the oracle hovers at 0.068 m), and
2. **range information present** — T0O1 trains/evals on the *perspective* renderer (far-R²
   0.40), and v7 §6.3 showed even handing the policy the **oracle** metric range barely moves
   it (2.91 → 2.43 m).

**A policy that fits the expert's hover action almost perfectly open-loop, and can perceive its
offset, still settles ~2.48 m off-target closed-loop.** That gap — *open-loop near-perfect,
closed-loop 2.48 m, with the information available* — is a **closed-loop control / BC-objective
signature**, not a representation, coverage, sensing, or raw-capacity one. v9 localizes and
attacks it.

**User decisions (2026-06-29):** (1) v9 attacks the **2.48 m closed-loop wall** (the only path
to sub-meter / deployable). (2) **The Phase-0 cheap gates decide the fix mechanism.**

**Intended outcome (either is publishable):**
- Wall broken → the **first deployable sub-meter closed-loop vision hover** on this task — a
  constructive positive result closing the whole arc.
- Wall holds even for the zero-conflict specialist under the targeted fix → the **deepest
  negative result**: the floor is an intrinsic closed-loop limit of monocular-FPV BC hover,
  not fixable by representation, coverage, sensing, capacity, *or* the targeted closed-loop
  lever — a strong structural claim.

---

## 1. Central hypothesis (pre-registered)

> **H_v9.** The cond-IAE ≈ 2.48 m wall of the hover specialist is a **closed-loop steady-state
> control / BC-objective limit** (the BC policy reaches a *biased equilibrium* ~2.48 m
> off-target despite a near-perfect open-loop action fit and available range information). It is
> broken by the closed-loop / objective fix that Phase 0 identifies — **not** by architecture
> specialization, which only recovers the specialist's own 2.48 m.

**Decision rule (pre-registered, mirrors H_v7/H_v8).** The wall is "broken" iff, over 3 seeds,
on the frozen P0 protocol (perspective render, the T0O1 condition): `cond-IAE(fix) <
cond-IAE(T0O1) − pooled_std` **AND** `cond-IAE(fix) ≤ 1.5 m` **AND** survival does not collapse
(`survival(fix) ≥ survival(T0O1) − pooled_std`), with `n_cond ≥ 15`. The survival guard is
mandatory (v7's pos3d-cue lesson: precision "wins" that are survival collapses).

**Primary metric (unchanged):** cond-IAE (survival ≥ 250 steps); survival/Tier-1 guard; composite secondary.

---

## Phase 0 — Cheap gates: localize the 2.48 m mechanism BEFORE any GPU-days

Per `feedback_evidence_before_training`: these inference-only measurements on the **existing**
T0O1 / S / XL checkpoints decide *which* fix Phase 1 implements. They distinguish three
mechanisms with different fixes.

- **Gate α — Closed-loop equilibrium shape.** Re-run `evaluate_frozen_p0` on T0O1 (and S, XL)
  logging the **per-step position error** trajectory (extend `rollout_episode` to dump
  `positions − targets`; it already accumulates them for the metric). Question: on surviving
  episodes does ‖pos-err‖ **converge to a steady ~2.48 m plateau** (a *biased equilibrium* →
  steady-state control/objective limit) or **drift / oscillate / diverge** (compounding
  instability)? Note the env hovers in `target ≡ init` mode, so the drone **starts at the
  target (zero offset)** — a plateau at 2.48 m means the policy *actively drives itself* to a
  2.48 m equilibrium from a perfect start. New `scripts/measure_equilibrium.py`.
  → plateau ⇒ b2 (steady-state bias) / b3 (authority); divergence ⇒ b1 (compounding).

- **Gate β — Action bias at the equilibrium (open-loop, decisive b2 vs b3).** Render the FPV
  observation **at the 2.48 m-equilibrium state** and **at the true target**, feed both to
  T0O1, and compare the policy's predicted CTBR to the **expert's** CTBR at those same states
  (the PID-CTBR teacher / state-PPO oracle). If at the offset state the policy outputs ≈ the
  *target* (no-correction) action while the expert outputs a strong *corrective* action →
  **the policy under-corrects at offset** = steady-state bias from action imitation (**b2**,
  objective). If the policy ≈ the expert's corrective action yet the drone still doesn't close
  → **control authority / action parameterization** (**b3**). Extend
  `scripts/measure_capacity_fit.py` (it already renders states + runs `predict_action`).

- **Gate γ — Counterfactual bias-correction (does a position-aware nudge collapse the wall?).**
  At inference only, add to T0O1's action a small term proportional to the (oracle, for this
  probe) steady-state pos-error — an integral/bias correction — and re-score cond-IAE on frozen
  P0. If a simple correction **collapses 2.48 m → sub-meter**, the wall is a *correctable
  steady-state bias* and a position-error-aware **objective** is the fix (b2). If it does not
  (or destabilises survival), the limit is deeper (b1/b3 or intrinsic). `evaluate_frozen_p0`
  variant with an oracle-bias-correction hook.

**Gate decision table (pre-registered branch for Phase 1):**

| Phase-0 finding | Mechanism | Phase-1 fix |
|-----------------|-----------|-------------|
| α plateau + β under-correction + γ collapses | **b2** steady-state objective bias | **position-error-aware BC objective** |
| α plateau + β corrective-but-no-close | **b3** control authority | **action reparameterization** (residual head / finer T_action) |
| α divergence | **b1** compounding | **short-horizon closed-loop fine-tune** (near-equilibrium DAgger relabel) |
| γ no-collapse under any | intrinsic closed-loop limit | **negative-deepening** write-up |

---

## Phase 1 — Attack the 2.48 m wall (fix selected by Phase 0), 3 seeds, frozen P0

Each fix is trained on the **T0O1 recipe** (hover-only + perspective; the zero-conflict
specialist) so the result is attributed to the closed-loop lever, not allocation/coverage.

- **b2 — position-error-aware objective.** Augment `compute_loss` (`models/flow_policy_v5.py`)
  with a term that penalizes the *closed-loop steady-state position error*, not just action
  imitation: either (i) a differentiable **short K-step rollout** through the dynamics
  (`envs/quadrotor_dynamics.py`; check RK4 autograd-friendliness — else a learned 1-step
  surrogate) with an L1 pos-error penalty on the terminal state, or (ii) an **offset-state
  re-weighting / expert-corrective-action** term that up-weights imitation of the teacher's
  *corrective* action at offset states. New `--lambda-poserr` flag.
- **b3 — action reparameterization.** A residual precision head that outputs a fine correction
  on top of the base CTBR when near-target (gated), and/or `T_action`/inference-step changes
  that raise corrective bandwidth; trained on T0O1 data.
- **b1 — closed-loop fine-tune.** Short-horizon DAgger: roll the BC policy out, relabel the
  visited near-equilibrium states with the PID-CTBR teacher's corrective action, and fine-tune
  on the relabelled set — **scoped to near-target states only** to avoid the hover-poisoning
  that denied the v4 DAgger and the AWR mode-collapse that sank the v5 RL (Known Failure
  Modes #5/#8; `project_dagger_h2`, `project_temperature_scaling`).

Pre-registered H_v9 decision rule applied vs the **T0O1 2.48 m** baseline.

## Phase 1b — Specialization bridge (confirm allocation is the *bounded* lever, not the wall)

Implement the v8 **(a)** candidate to close it cleanly: a 2-expert `flow_net` MoE
(precision/hover expert + recovery expert) routed by the existing near-target gate
(`is_recovery` from `rollout_episode`), param-matched to a v8 ladder rung. **Pre-registered
prediction:** it recovers T1O1 2.93 → ~2.48 m (T0O1's precision) *with* survival, but **does
not** cross 1.5 m — confirming allocation is a real survival–precision rebalance **bounded by**
the hover-specialist wall, which only the Phase-1 closed-loop fix can cross. (If the MoE
*also* lands at ~2.48 m it corroborates H_v9; if it beats 2.48 m, that itself is a finding.)

---

## Phase 2 — Integration

- **Wall broken (Phase 1):** pivot the paper from a pure negative result to "we localized the
  constraint through a quadruple exclusion *and broke it*" — the first deployable sub-meter
  closed-loop vision hover; new §6.6 / results section; possibly a venue upgrade.
- **Wall holds:** deepen to a **quadruple-plus exclusion** — the 2.48 m floor is intrinsic to
  monocular-FPV BC hover, surviving representation, coverage, sensing, capacity, *and* the
  targeted closed-loop fix on the zero-conflict specialist. Fold into §6/§7.
- remi review → deep-science-writer polish → `scripts/export_paper.py`; new Finding #14 +
  `docs/experiment_report_v9_closedloop.md`.

---

## Critical files

**Reuse / extend**
- `scripts/evaluate_frozen_p0.py` + `scripts/evaluate_hierarchical.py:rollout_episode`
  (per-step positions/targets already accumulated → extend to dump trajectories for Gate α).
- `scripts/measure_capacity_fit.py` (renders states + `predict_action` → extend for Gate β
  action-bias probe).
- `envs/quadrotor_env_v4.py` (`reset(options=…)` `setpoint_offset` from v7 Gate C — to render
  offset-state observations) + `envs/quadrotor_dynamics.py` (RK4 — for the b2 differentiable
  rollout, pending autograd check).
- `models/flow_policy_v5.py:compute_loss` (`--lambda-poserr` b2 term; the MoE for Phase 1b),
  `scripts/train_flow_v5.py` (new flags), `scripts/run_p2cap_ablation.py` driver pattern.
- T0O1/S/XL checkpoints (`checkpoints/flow_policy_v5/p2to_T0O1_s{0,1,2}`, `p2cap_*`).

**New:** `scripts/measure_equilibrium.py` (Gate α), `scripts/run_p2cl_ablation.py` +
`scripts/evaluate_p2cl_ablation.py` (Phase 1 sweep + H_v9 verdict, mirror p2cap),
`docs/experiment_report_v9_closedloop.md`.

**Hard pre-checks before GPU-days (Gate γ is the cheapest decisive one):** if Gate γ's
oracle-bias correction does NOT collapse 2.48 m, do not invest in the b2 objective — the limit
is b1/b3 or intrinsic.

---

## Compute budget & operational rules

- **Phase 0 gates:** inference-only, minutes–hours (the redirect step; run first).
- **Phase 1 fix:** ~3 seeds × ~2–3 h (b2/b3 BC-style) or longer (b1 closed-loop) ≈ ~1 GPU-day.
- **Phase 1b MoE:** ~3 seeds ≈ ~½–1 GPU-day.
- **Total ≈ 1.5–2.5 GPU-days, sequential.**
- **Operational discipline (`feedback_background_launch_quirk`):** the Bash-tool background
  launch reports **spurious `failed`** while the detached python keeps running; **never trust
  the driver-task notification** — verify via `powershell` cmdline / >5 GB RSS before any
  relaunch, else concurrent drivers overlap → RAM OOM. `dppo/Scripts/python.exe -m …`,
  `batch_size=256`, monitor via TensorBoard event API / `final_model.pt` count, kill competing
  python first.

---

## Verification (end-to-end)

1. **Gate α:** `python -m scripts.measure_equilibrium` → T0O1 ‖pos-err‖ trajectory: plateau at
   ~2.48 m (steady-state) vs divergence; equilibrium-offset distribution.
2. **Gate β:** extended `measure_capacity_fit` → policy-vs-expert CTBR at the offset state
   (under-correction b2 vs corrective-no-close b3).
3. **Gate γ:** `evaluate_frozen_p0` + oracle-bias hook → does a position-aware nudge collapse
   2.48 m → sub-meter?
4. **Phase 1:** `run_p2cl_ablation` (3 seeds) → `evaluate_p2cl_ablation` applies the H_v9 rule
   vs T0O1 2.48 m; verdict printed.
5. All numbers trace to `evaluation_results/p2cl_*.json`; figures via `make_paper_figures.py`.

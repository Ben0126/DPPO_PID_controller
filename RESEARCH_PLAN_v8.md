# RESEARCH_PLAN_v8 — Testing (and Attempting to Break) the Robustness–Precision Capacity Conflict

**Version:** 8.0
**Date:** 2026-06-28
**Supersedes:** the *diagnosis* of `RESEARCH_PLAN_v7.md`. v7 ran the decisive Teacher×Observation
2×2 and **refuted** H_v7 (the cond-IAE ≈ 2.8 m precision floor did **not** break when far-range
recovery labels and a range-encoding perspective observation were supplied *jointly*). v7 localized
the binding constraint to a **Robustness–Precision Capacity Conflict** but left `capacity` as the
**leading-but-UNTESTED** explanation (Remi flagged this; paper §6.4/§7 wording aligned to
"conflict observed, capacity = leading untested explanation"). v8 is the test.
**Target venues:** ICRA / robot-learning workshop (simulation-only; no real-robot claim).

---

## 0. Context — why v8 exists

The negative-result paper (`docs/paper_negative_result_draft.md`, v0.5) completed a **triple
exclusion**: the precision floor is **not** representation collapse (Findings #10/#12), **not** the
sensing channel (O1 perspective restores far-R² 0.05→0.40 yet precision stays ~2.8 m), and **not**
coverage/teacher-competence (a competent PID-CTBR far teacher reaches 1–4 m at 100 % survival /
cond-IAE 0.14–0.18 m, yet the BC floor holds). The decisive frozen-P0 T×O 2×2:

| Cell | cond-IAE (mean±std, m) | survival | reading |
|------|------------------------|----------|---------|
| T0O0 | 2.69 ± 0.15 | 0.83 | control (neither factor) |
| **T0O1** | **2.48 ± 0.14** | 0.77 | **best precision** (hover-only + perspective) |
| T1O0 | 2.71 ± 0.22 | 0.91 | coverage, precision unmoved |
| **T1O1** | **2.93 ± 0.18** | 0.91 | both factors → **precision REGRESSES** |

The headline is the **negative interaction**: under the perspective observation, adding far-range
recovery labels (T0O1 → T1O1) *degrades* precision 2.48 → 2.93 m while buying survival. A ~3 M
trainable-parameter shared trunk appears forced to trade "wide-range survival/recovery" against
"dead-centre precise hover" (`FLOOR_BROKEN=false`, Δ=+0.24 m ∈ pooled std 0.23 m;
`evaluation_results/p2to_ablation_leaderboard.json`).

**This is the capacity-conflict claim. It is untested.** v8 tests it with the same pre-registered,
frozen-P0, 3-seed methodology, after cheap gates redirect the pipeline.

**User decisions (2026-06-28):**
1. **Goal = harden the negative result FIRST, then extend toward a floor-break** (Stage 1 → Stage 2).
2. **Intervention = let the Phase-0 cheap gates decide** (distinguish *open-loop fitting
   interference* from *closed-loop compounding error*, and rule out the *data-dilution* confound).

**Intended outcome (either is publishable / hardens the paper):**
- Floor broken by capacity → an extra constructive positive contribution (Stage 2 follow-on).
- Floor holds even when capacity is scaled → upgrade §7 from "capacity = untested" to **"even scaling
  capacity does not break it → the conflict is more fundamental than parameter count"** — the
  strongest version of this negative result.

---

## 1. Central hypothesis (pre-registered)

> **H_v8.** The cond-IAE ≈ 2.8 m floor and the negative coverage×precision interaction are caused
> by insufficient *usable* capacity / parameter sharing in the ~3 M-trainable model. The floor
> breaks (or the T0O1→T1O1 negative interaction vanishes) when the precision and recovery objectives
> are given separate / larger capacity.

**Decision rule (pre-registered, mirrors H_v7).** On the **T1O1 recipe** (where the conflict bites),
over 3 seeds, frozen P0 with perspective eval render: capacity "breaks the floor" iff
`cond-IAE(scaled) < cond-IAE(S) − pooled_std` **AND** `cond-IAE(scaled) ≤ 1.5 m`
**AND** the survival guard holds (`survival(scaled) ≥ survival(S) − pooled_std`), with `n_cond ≥ 15`.
Secondary read: does the T0O1→T1O1 negative interaction shrink with capacity?

**Primary metric (unchanged):** cond-IAE (survival ≥ 250 steps); Tier-1/survival guard; composite secondary.

---

## Phase 0 — Cheap gates BEFORE the GPU-days (decide the main intervention)

Per the standing rule (`feedback_evidence_before_training`): run the measurement that could redirect
the whole pipeline before spending GPU-days. The gates decide *which* capacity intervention (if any).

### Gate A — Open-loop fitting interference vs closed-loop compounding error (decisive, inference-only)

`scripts/measure_capacity_fit.py` (reuses `scripts.evaluate_hierarchical.build_policy` +
`FlowMatchingPolicyV5`): on the **existing 12 p2to checkpoints**, measure each model's **open-loop fit
on a FIXED, seeded hover near-target batch** (paired across models, same `(t, eps)` draw and same
`_fixed_x1` inference noise) — both the flow-matching loss and the open-loop action-prediction error
(2-step inference vs expert CTBR). Compare the **generalist (T1O1) vs the hover-specialist (T0O1)** on
the perspective hover batch (and T1O0 vs T0O0 on crosshair as replication); also report each model's
fit on a **far** batch to confirm the tradeoff direction.

- generalist hover-fit **≈** specialist (Δ within seed pooled-std) → the model **has** the capacity to
  represent precise hover open-loop → the floor is **closed-loop / partial-observability**, not
  open-loop fitting capacity → scaling is **predicted inert** → Stage 1 = a *minimal* scaling confirm,
  reframe toward Stage-2 recurrence/memory.
- generalist hover-fit **measurably worse** than specialist (Δ > pooled-std, and the generalist fits
  far better) → **fitting interference is real** → scaling/specialization is the right lever → Stage 1
  = the capacity scaling ladder.

### Gate B — Rule out the data-dilution confound (necessary control)

T1O1 dilutes hover gradients to ~50 %; precision may drop merely because hover is down-weighted, not
from capacity. We already have the dilution endpoints (T0O1 = 100 % hover 2.48 m; T1O1 = 50/50
2.93 m). Add `--hover-weight` to `scripts/train_flow_v5.py` (loss-side weight or `WeightedRandomSampler`,
default 1.0 = unchanged) and screen 2–3 reweighting points on the T1O1 recipe:

- reweighting toward hover **recovers precision AND keeps T1 survival** → it is a **data-balance**
  problem, not capacity → revise paper §6.4; the floor-break lever is loss-reweighting/curriculum.
- reweighting cannot recover both (precision returns only as survival collapses, or neither moves) →
  the capacity claim survives → proceed to the scaling ladder.

---

## Phase 1 — Decisive capacity experiment (main intervention selected by Gate A/B)

Default branch (Gate A = fitting interference, Gate B = not dilution): **Capacity Scaling Ladder** on
the **T1O1 recipe**, varying **only flow_net capacity**, 3 seeds, frozen P0
(`evaluate_frozen_p0 --target-render perspective`):

- Add `--down-dims` to `scripts/train_flow_v5.py` (CLI override of `unet_cfg['down_dims']`; no config sprawl). Rungs:
  - **S** = (256,512) — current; reuse existing T1O1.
  - **M** = (256,512,768) — **add one depth level**: the first two levels keep H4-transferable shapes
    (`transfer_from_h4` partial-loads them) → most transfer-friendly capacity lever.
  - **L** = (384,768) or (512,1024) — **widen**: flow_net trains from scratch (encoders/tilt still
    transfer); print `val_flow` to rule out a "scaled-from-scratch underfit" confound.
- Print `n_params` and best `val_flow` per checkpoint (capacity & fit evidence).
- Apply the H_v8 decision rule; also report whether the T0O1→T1O1 negative interaction shrinks with
  scale (optional scaled-T0O1 control to measure the interaction directly).

If Gate A = closed-loop: shrink Phase 1 to a **minimal scaling confirm** (M only ×3 seeds), predict the
floor **holds** → write the decisive "not capacity, it's closed-loop" finding and route Stage 2 to
recurrence.

---

## Phase 2 — Integrate into the paper (harden the negative result)

Either outcome folds into `docs/paper_negative_result_draft.md` §6.4/§7 and `CLAUDE.md` Major Finding #13:
- Floor holds → §7 upgrades "capacity = untested" → "capacity tested (scaled flow_net), floor held →
  the conflict is more fundamental than parameter count"; add a capacity-ladder table + figure
  (`scripts/make_paper_figures.py` reads the new leaderboard json).
- Floor broken → upgrade to "localized **and** broke" (constructive positive result); Stage 2 follows.
- remi review → deep-science-writer polish → `scripts/export_paper.py` re-export HTML/PDF.

---

## Stage 2 — Extend toward a floor-break (scoped; branch chosen by Phase-0 gates; potential 2nd paper)

Expand into a v9 plan once v8 results land:
- **fitting/sharing-bound** → **structural specialization**: a precision pathway or a recovery/hover
  two-expert MoE (gated by a near-target indicator), **parameter-matched** to a scaling rung →
  a **Capacity × Specialization** comparison testing "parameter sharing, not parameter count".
- **gradient interference** → **PCGrad / gradient surgery** on hover-loss vs recovery-loss (no extra
  params; anchors refs [31] Sener & Koltun, [32] Yu et al. PCGrad).
- **closed-loop / partial observability** → **recurrence/memory** (GRU/temporal range integration) or
  an action-parameterization / observation reformulation.

---

## Critical files

**Reuse as-is (do not reinvent)**
- `scripts/train_flow_v5.py` — already has the recipe flags (`--recovery-h5`/`--hover-h5`/
  `--transfer-from-h4`/`--lambda-disp 0.0`/`--seed`/`--tag`); add `--down-dims`, `--hover-weight`.
- `scripts/evaluate_frozen_p0.py` (`--target-render perspective`), `scripts/evaluate_p2to_ablation.py`
  (3-seed aggregation + pre-registered verdict), `scripts/evaluate_hierarchical.py:build_policy`
  (architecture auto-detect, reused by Gate A).
- `scripts/run_p2to_ablation.py` (sweep-driver pattern), `make_paper_figures.py`, `export_paper.py`.
- Datasets `data/expert_demos_v7_{hover,far}_{crosshair,persp}.h5`; 12 p2to checkpoints (Gate A).
- `models/flow_policy_v5.py` (`down_dims` is already a constructor arg);
  `models/conditional_unet1d.py` (adding a depth level keeps the first two levels H4-transferable).

**New**
- `scripts/measure_capacity_fit.py` (Gate A, inference-only).
- `train_flow_v5.py`: `--down-dims`, `--hover-weight` (small additions).
- `scripts/run_p2cap_ablation.py` + `scripts/evaluate_p2cap_ablation.py` (capacity sweep driver +
  aggregation, mirrors p2to).

**Known traps:** scaling `down_dims` makes H4 flow_net weights shape-mismatch → `transfer_from_h4`
auto-skips (already supported), so **deepening (M)** preserves more transfer than **widening (L)**;
always print `val_flow` to rule out scaled-from-scratch underfit. Keep `batch_size=256`,
`run_in_background=true`; verify the larger model does not OOM on 24 GB VRAM (drop batch and log if so).

---

## Compute budget & operational rules

- **Gates (Phase 0):** minutes (Gate A inference-only) + a few short BC runs (Gate B).
- **Scaling ladder (Phase 1):** ~3 rungs × 3 seeds × ~2.5–3 h ≈ ~1 day, **one-at-a-time** (Failure Mode #7).
- **Total ≈ 1–2 GPU-days, sequential.**
- **Always** `dppo/Scripts/python.exe -m ...` with Bash `run_in_background=true` (never
  `nohup`/pipe/`source activate`); monitor via the TensorBoard event API, not buffered stdout;
  `batch_size=256`; kill competing python before launching.

---

## Verification (end-to-end)

1. **Gate A:** `python -m scripts.measure_capacity_fit` → prints generalist(T1O1) vs specialist(T0O1)
   hover open-loop fit gap + directional verdict (fitting interference / closed-loop).
2. **Gate B:** T1O1 + `--hover-weight` screen → whether precision and survival recover together.
3. **Phase 1:** `python -m scripts.run_p2cap_ablation` (S/M/L × 3 seeds, one-at-a-time) →
   `python -m scripts.evaluate_p2cap_ablation` applies the H_v8 rule; prints capacity verdict +
   interaction shrinkage.
4. Monitor via TensorBoard event API (`val/flow_loss`, `train/flow_loss`), not buffered stdout.
5. All numbers land in `evaluation_results/p2cap_ablation_{manifest,leaderboard}.json`; figures via
   `scripts/make_paper_figures.py`.
6. Fold conclusions into `docs/paper_negative_result_draft.md` §6.4/§7 + `CLAUDE.md` Finding #13;
   re-export HTML/PDF.

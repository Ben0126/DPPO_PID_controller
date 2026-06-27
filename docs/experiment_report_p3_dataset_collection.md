# Phase 3 — Re-collect Far-Range Demonstrations for the Teacher × Observation 2×2

**Date:** 2026-06-25
**Plan:** `RESEARCH_PLAN_v7.md` Phase 3 (fill the coverage gap; build the 2×2 datasets)
**Depends on:** Phase 0 Gates A & B (`docs/experiment_report_p0_teacher_renderer_gates.md`)
**Goal:** produce the four FPV datasets that form the decisive Teacher (T) × Observation (O)
2×2 — and **hard-gate** that the far datasets actually fill the 1–3 m position-error band
the negative-result paper's BC data lacked, before paying any GPU-days on Phase 4 training.

---

## Design — one teacher, shared trajectories, dual render

Two factors, four datasets:

| Factor | Level 0 | Level 1 |
|--------|---------|---------|
| **T** (teacher coverage) | T0 = hover only (`target == init`, no 1–3 m labels) | T1 = far 1–3 m recovery (`target != init`) |
| **O** (observation) | O0 = crosshair (production, saturating ≥2 m) | O1 = perspective (non-saturating disk, far-range cue) |

Two choices keep both axes clean:

1. **Single teacher = PID-CTBR** (gentle Gate-A tune `vel_max=1.0, Kp_pos=0.8, omega_max=6.0`,
   `compute_ctbr_action`) for **both** hover and far recovery (user decision 2026-06-25). The
   T-axis is therefore a **pure coverage operation** — no teacher swap is folded into T, so T
   cannot be confounded with "we changed the label source". Gate A already showed this teacher
   recovers the full 1–4 m band at 100 % survival / cond-IAE 0.14–0.18 m, so the far labels are
   competent.
2. **Shared trajectory, dual render.** Each (state, target) trajectory is rendered **twice per
   step** — once crosshair, once perspective — from the **same dynamics state** and, crucially,
   the **same per-frame DR noise** (the global `np.random` state is saved before the crosshair
   render and restored before the perspective render). So O0 and O1 differ **only** in the
   target-marker pixels; the trajectory (states / actions / IMU) is byte-identical. O is a
   **clean observation swap**, not a re-collection.

Collector: `scripts/collect_data_v7_pidctbr.py` (`--mode {hover,far}`, two h5 per mode).
h5 format matches `collect_data_v4_recovery.py` (per-episode group `images / actions / states /
imu_data`). `episode_type` drives `FlowDatasetV5`'s task-cond `[is_hover, is_recovery]`: hover
episodes carry `episode_type='hover'` and **no** `init_tilt_deg` (so they are not misclassified
as recovery — `train_flow_v5.py:87`); far episodes carry `episode_type='recovery'` + offset
metadata.

| Cell | Dataset file | episode_type → task-cond |
|------|--------------|--------------------------|
| T0O0 | `data/expert_demos_v7_hover_crosshair.h5` | hover → [1,0] |
| T0O1 | `data/expert_demos_v7_hover_persp.h5` | hover → [1,0] |
| T1O0 | `data/expert_demos_v7_far_crosshair.h5` | recovery → [0,1] |
| T1O1 | `data/expert_demos_v7_far_persp.h5` | recovery → [0,1] |

(At Phase 4 the T1 cells are formed by mixing the hover pool + the far recovery h5 via
`train_flow_v5.py --hover-episodes … --recovery-h5 …`; the four files above are the building
blocks.)

---

## Collection result (500 episodes/mode, seed 12345)

| Mode | Episodes | Steps | Survival | init pos-err (mean / min / max) |
|------|----------|-------|----------|---------------------------------|
| hover (T0) | 500 | 250,000 | **100.0 % (0 crash)** | 0.000 / 0.000 / 0.000 m |
| far (T1)   | 500 | 250,000 | **100.0 % (0 crash)** | 1.986 / 1.001 / 2.995 m |

Far offsets uniformly span the 1–3 m band (mean 1.986 m); both modes have **zero crashes**, so
the far episodes are genuine **successful** recovery trajectories (start 1–3 m off target → fly
back → converge), not survival-by-luck. Files ~2.0–2.1 GB each (8.2 GB total).

---

## Coverage HARD GATE (go/no-go) — `scripts/check_coverage_v7.py`

Position-error magnitude `‖states[:,0:3]‖` (the 15D obs is `[pos_error_body(3), …]`, so this is
exactly `‖pos − target‖`), reusing the validated machinery from `measure_ood_coverage.py`.
Artifact: `evaluation_results/p3_coverage_v7.json`.

| Cell | n steps | p50 | p99 | max | frac > 1 m | frac in [1,3] m | gate |
|------|---------|-----|-----|-----|-----------|-----------------|------|
| T0O0 hover_crosshair | 250 k | 0.043 | 0.146 | 0.274 | **0.000** | 0.000 | PASS |
| T0O1 hover_persp     | 250 k | 0.043 | 0.146 | 0.274 | **0.000** | 0.000 | PASS |
| T1O0 far_crosshair   | 250 k | 0.192 | 2.592 | 2.995 | **0.117** | 0.117 | PASS |
| T1O1 far_persp       | 250 k | 0.192 | 2.592 | 2.995 | **0.117** | 0.117 | PASS |

**Clean O-swap check:** crosshair vs perspective states **byte-identical** for both modes
(`states_identical = True`). Pixel-level: the two renders of the same step differ in only
~12–25 % of pixels (the marker region) — if the per-frame DR noise were *not* RNG-locked, ~100 %
of pixels would differ; the small fraction confirms the noise is shared and only the marker
changes.

**Verdict: OVERALL GATE PASS.**

- The far datasets fill the band the paper localised as empty: the paper's `frac > 1 m` ≈ 0.4 %
  jumps to **11.7 %** (~29×), with a smooth histogram all the way to 3 m.
- The hover datasets have **0.0 %** mass beyond 1 m (max 0.274 m) → T0 genuinely lacks far
  coverage. The T-axis is therefore a clean coverage contrast (T1 adds 1–3 m labels; T0 has none).

---

## Bottom line for Phase 4

- Four datasets ready: `data/expert_demos_v7_{hover,far}_{crosshair,persp}.h5` — 0-crash, 500 ep
  each, single PID-CTBR teacher, dual-rendered from identical trajectories.
- Coverage gate (the only pre-Phase-4 hard gate) **PASSES**: T1 has 1–3 m, T0 does not, O is a
  byte-clean swap.
- **Next (Phase 4):** the decisive frozen-P0 **T × O 2×2** — train the D0E1 frontier recipe on
  each cell, 3 seeds, one-at-a-time (Failure Mode #7); O1 arms must train **and** eval on the
  perspective renderer (`--use-perspective-target` on the eval env). Apply the H_v7 decision
  rule: the 2.8 m floor is "broken" iff `cond-IAE(T1O1) < cond-IAE(T0O0) − pooled_std` **AND**
  `≤ 1.5 m` **AND** survival does not collapse.

### Artifacts
- `data/expert_demos_v7_{hover,far}_{crosshair,persp}.h5`
- `evaluation_results/p3_coverage_v7.json`
- `scripts/collect_data_v7_pidctbr.py`, `scripts/check_coverage_v7.py`

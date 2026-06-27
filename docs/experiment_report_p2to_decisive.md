# Phase 4 — The Decisive Teacher × Observation 2×2 (frozen-P0, 3 seeds)

**Date:** 2026-06-26
**Plan:** `RESEARCH_PLAN_v7.md` Phase 4 (the decisive 2×2)
**Depends on:** Phase 0 Gates A/B (`experiment_report_p0_teacher_renderer_gates.md`),
Phase 3 datasets + coverage gate (`experiment_report_p3_dataset_collection.md`, `evaluation_results/p3_coverage_v7.json`)
**Artifacts:** `evaluation_results/p2to_ablation_manifest.json`, `evaluation_results/p2to_ablation_leaderboard.json`
**Verdict:** ⛔ **PRECISION FLOOR NOT BROKEN — the negative result is deepened, not overturned.**

---

## 1. Pre-registered hypothesis H_v7

The paper localised the binding constraint behind the cond-IAE ≈ 2.8 m hover-precision floor
(~13 % of the 0.068 m state-PPO oracle) as the **far-range coverage / teacher-competence gap** —
not representation collapse (refuted in §5), not sensing alone (the perspective renderer is only
an information gate). The pre-registered claim:

> **H_v7:** the floor breaks **only when BOTH** (a) BC has competent 1–3 m recovery labels
> (Teacher coverage, **T1**) **AND** (b) the FPV observation can encode metric range
> (perspective target, **O1**). Each factor *alone* was already shown inert.

**Decisive design** (mirrors the paper's §5 Dispersive×E2E 2×2): a **Teacher × Observation 2×2**,
3 seeds/cell, evaluated under the frozen P0 protocol. The floor is declared **broken iff ALL**:

1. `cond-IAE(T1O1) < cond-IAE(T0O0) − pooled_std` (significant vs the neither-factor control),
2. `cond-IAE(T1O1) ≤ 1.5 m` (absolute target, ~2× the floor),
3. `survival(T1O1) ≥ survival(T0O0) − pooled_std` (survival guard — blocks the "precision win
   that is really a survival collapse" failure mode the paper's pos3d-cue arm exhibited),

and cond-IAE is only trusted when `n_cond ≥ 15` in both cells.

---

## 2. Design — 4 cells, one recipe, two factors

Every cell is the **D0E1 frontier recipe** (Dispersive OFF `--lambda-disp 0.0`, vision encoder
**E2E** / no `--freeze-vision`, **H4-transfer** warm start
`flow_policy_v4/20260514_175219`, task-conditioned), `--hover-episodes 500
--recovery-episodes 500`. Only the two factors vary, and they vary purely by **which Phase-3 h5
the data comes from** (the crosshair/perspective observation is rendered *into* the h5 at
collection time; `FlowDatasetV5` reads `f[key]['images']` with no render flag):

| Cell | hover pool (`--hover-h5`) | recovery (`--recovery-h5`) | eval render |
|------|---------------------------|----------------------------|-------------|
| **T0O0** (neither) | `expert_demos_v7_hover_crosshair` | — (hover-only) | crosshair |
| **T0O1** (sensing only) | `expert_demos_v7_hover_persp` | — (hover-only) | perspective |
| **T1O0** (coverage only) | `expert_demos_v7_hover_crosshair` | `expert_demos_v7_far_crosshair` | crosshair |
| **T1O1** (both) | `expert_demos_v7_hover_persp` | `expert_demos_v7_far_persp` | perspective |

12 BC runs = 4 cells × seeds {0,1,2}, trained **one-at-a-time** (Failure Mode #7) via
`scripts/run_p2to_ablation.py`; evaluated via `scripts/evaluate_p2to_ablation.py`, which runs each
checkpoint through the exact `evaluate_frozen_p0` protocol (30 ep, base_seed 12345, paired init,
σ=2.0 exp-decay, n_inf=2) and — critically — passes each cell its **own `--target-render`** so the
O1 policies are scored on the **same** perspective observation they trained on.

**Design note (data-volume asymmetry, pre-registered):** T0 = hover-only (500 ep), T1 = hover+far
(1000 ep). This is intrinsic to "add far-range labels". A size-matched near-recovery control was
identified as the fallback if a reviewer objects; the headline here uses the Phase-3 hover-only T0.

### Two tooling additions (this phase)
- `scripts/train_flow_v5.py --hover-h5`: overrides the config's hover-pool `dataset_path`
  (default `expert_demos_v4.h5`, old PPO crosshair) so an O1 cell trains on perspective hover
  imagery. Config default untouched.
- `scripts/evaluate_frozen_p0.py --target-render {crosshair,perspective}`: threads the renderer
  into the eval `QuadrotorVisualEnv` (`physical_size` stays 0.5 m as in training). Default
  `crosshair`, so all prior P0/P2/P2f evals are byte-unchanged. *(This is the `--use-perspective-target`
  eval flag from the plan, implemented as the more general `--target-render` to match the existing
  `QuadrotorVisualEnv(target_render=…)` constructor argument.)*

### Reproducibility note — a real OOM trap, fixed behaviour-preservingly
The last cell (`T1O1_s2`) OOM'd twice at `FlowDatasetV5.__init__`: `np.stack(img_buf)` allocates
the full ~10.7 GB image array **while the Python list still holds its ~10.7 GB of inputs** → a ~2×
(~21 GB) transient peak that exceeds the process commit headroom on this ~12 GB-free machine.
(The earlier T1 cells survived this only because the system had more free RAM ~20 h earlier — a
non-deterministic race.) Fixed by a **two-pass** `FlowDatasetV5`: pass-1 counts windows from
shapes/attrs (no image load), the image array is pre-allocated once (`np.empty`), pass-2 writes
each window in directly → peak ~10.7 GB, **byte-identical output** (equivalence-tested: all of
`img/imu/act/tilt/state/task` `array_equal` between old `np.stack` and new construction). The
re-run trained identically: `T1O1_s2` val/flow = 0.0108 vs siblings 0.0108 / 0.0109. The fix is a
pure memory-layout optimisation; it changes no numerical behaviour and benefits every caller.

---

## 3. Result — the 2×2 (mean ± std over 3 seeds, frozen P0)

**Measured oracle (state-based PPO, same protocol):** composite **0.9668**, cond-IAE **0.0675 m**,
100 % survival.

### PRIMARY axis — conditional-IAE (hover precision, LOWER is better)

|              | **O0 crosshair** | **O1 perspective** |
|--------------|------------------|--------------------|
| **T0 hover-only** | **2.69 ± 0.15 m** (40× oracle) | **2.48 ± 0.14 m** (36×) |
| **T1 +far recov** | **2.71 ± 0.22 m** (40×) | **2.93 ± 0.18 m** (43×) |

### Survival (HIGHER is better)

|              | O0 crosshair | O1 perspective |
|--------------|--------------|----------------|
| T0 hover-only | 83.2 ± 5.8 % | 76.7 ± 10.5 % |
| T1 +far recov | **91.4 ± 1.5 %** | **90.6 ± 3.3 %** |

### Tier1 pass-rate

|              | O0 crosshair | O1 perspective |
|--------------|--------------|----------------|
| T0 hover-only | 98.9 ± 1.6 % | 86.7 ± 9.8 % |
| T1 +far recov | **100.0 ± 0.0 %** | 98.9 ± 1.6 % |

| Cell | cond-IAE | survival | Tier1 | score (95 %-style) | %Oracle | mean n_cond |
|------|----------|----------|-------|--------------------|---------|-------------|
| T0O0 | 2.69 ± 0.15 m | 83.2 % | 98.9 % | 0.195 | 20.1 % | 29.7 |
| T0O1 | 2.48 ± 0.14 m | 76.7 % | 86.7 % | 0.214 | 22.1 % | 26.0 |
| T1O0 | 2.71 ± 0.22 m | 91.4 % | 100.0 % | 0.223 | 23.1 % | 30.0 |
| T1O1 | 2.93 ± 0.18 m | 90.6 % | 98.9 % | 0.202 | 20.8 % | 29.7 |

(n_cond ≥ 22 every cell → cond-IAE is trustworthy throughout; no short-survival artifact.)

---

## 4. H_v7 verdict — FLOOR NOT BROKEN

Decisive comparison **T1O1 (both) vs T0O0 (neither)**:

| Criterion | Value | Pass? |
|-----------|-------|-------|
| [1] cond significant `T1O1 < T0O0 − pooled_std` | 2.931 m vs 2.690 m → Δ **+0.241 m** (pooled std 0.232 m) | ❌ (T1O1 is *worse*) |
| [2] cond absolute `T1O1 ≤ 1.5 m` | 2.931 m | ❌ |
| [3] survival guard `T1O1 ≥ T0O0 − pooled_std` | 90.6 % ≥ 83.2 − 6.7 % | ✅ |
| [4] n_cond ≥ 15 both cells | 29.7 / 29.7 | ✅ |

**⇒ FLOOR NOT BROKEN.** Even with BOTH the competent 1–3 m far-range recovery labels **and** the
far-range-capable perspective observation, conditional hover precision stays at **2.93 m**
(43× the oracle's 0.068 m) — if anything *slightly worse* than the neither-factor control. The
floor holds across the **entire** 2×2 (2.48–2.93 m, composite 20–23 % oracle).

---

## 5. What the 2×2 actually shows — robustness ↑, precision pinned, and a negative interaction

The grid is not flat noise; it has clean, interpretable structure that **strengthens** the paper.

1. **Coverage (T1) buys robustness, not precision.** Adding far-range recovery labels lifts
   survival **+8.2 pp** on crosshair (83.2→91.4 %) and **+13.9 pp** on perspective (76.7→90.6 %),
   and pins Tier1 at ~99–100 %. This is exactly the expected coverage benefit (more recovery
   labels → fly longer). **But it moves precision by 0 →** T1O0 2.71 m ≈ T0O0 2.69 m.

2. **Sensing (O1) alone nudges precision within noise, and costs a little survival.** On the
   hover-only row, perspective improves cond-IAE 2.69→2.48 m (≈0.21 m, ~1× pooled std — marginal)
   while survival drifts 83.2→76.7 % and Tier1 98.9→86.7 % (within seed noise). Perspective is a
   weak precision lever and not free.

3. **The interaction is NEGATIVE — the robustness factor *costs* precision.** The single best
   precision cell is **T0O1 (2.48 m)**; adding far-range recovery on top of it (→ T1O1) *degrades*
   precision to **2.93 m (+0.45 m)**. The very factor that buys survival (far-recovery labels)
   pulls the policy toward wide-range corrective behaviour that widens steady-state hover error.

**This is the Robustness–Precision Capacity Conflict, made quantitative and RL-free.** The v5
curriculum collapse (Major Finding #8) showed the same conflict but was confounded by RL dynamics
(AWR/advantage masking). Here it appears in a clean, pre-registered, 3-seed **supervised** 2×2:
within this model's capacity you can have wide-range survival **or** tight hover precision, and
pushing on coverage trades the latter for the former.

---

## 6. Why this is a *stronger* negative result (for the paper §6/§7)

Phases 0–4 form an exclusion argument that closes the two most plausible "fixable" explanations
for the floor:

- **Coverage is not the wall** — Gate A built a teacher (PID-CTBR, gentle gains) that recovers the
  full 1–4 m band at 100 % survival / cond-IAE 0.14–0.18 m; Phase 3 produced far datasets with
  11.7 % of mass in 1–3 m (29× the paper's ~0.4 %), 0 crashes. The labels are competent and dense.
- **Sensing is not the wall** — Gate B restored far-range image→distance R² 0.05→0.40 (perspective,
  in-env); O1 cells train and eval on it.
- **Yet precision does not move.** With coverage **and** sensing both supplied, cond-IAE is pinned
  at ~2.5–2.9 m (36–43× oracle). The binding constraint is therefore **neither coverage nor
  sensing**, separately or jointly — it is a deeper **capacity / robustness–precision conflict**,
  consistent with (and now decoupled from) the v5 curriculum collapse.

The 2×2 is the proof object: it not only fails to break the floor, it exhibits the conflict
directly (the survival-buying factor is the precision-costing factor). This is the
"floor-not-broken ⇒ deepen the negative result" branch of Phase 7.

### Caveats / scope (own them)
- Single architecture / capacity (v5 FlowMatchingPolicyV5, ~3 M trainable). The conflict is a
  statement about *this* capacity; a larger-capacity or precision-specialised head is untested and
  is the natural next hypothesis — but is **out of scope** for the v7 pre-registered question,
  which asked specifically about coverage × sensing.
- Data-volume asymmetry (T0 500 ep vs T1 1000 ep) favours T1; precision still does not improve
  under that favourable tilt, so the asymmetry does not threaten the *negative* conclusion (it
  would only matter for a positive one). The size-matched near-recovery control remains the
  fallback if a reviewer presses.
- One simulator / renderer; external-validity caveats are owned in `RESEARCH_PLAN_v7.md` R1/R2.

---

## 7. Bottom line

- **H_v7 is refuted.** Teacher coverage × Observation does **not** break the 2.8 m precision floor;
  T1O1 = 2.93 m ≈ 43× oracle, no better than T0O0.
- **The 2×2 deepens the negative result:** coverage (T1) reliably buys **survival** (+8–14 pp) but
  precision is **pinned**, and coverage+sensing together **negatively interact** on precision
  (T0O1 2.48 → T1O1 2.93 m). The binding constraint is a **capacity / robustness–precision
  conflict**, not coverage and not sensing.
- **Next (Phase 7 write-up):** fold this 2×2 into the paper as the decisive exclusion experiment
  in §6/§7 — "we removed the two leading candidate constraints (coverage, sensing) and the floor
  held; the residual is a capacity/robustness–precision conflict." Re-export via
  `scripts/export_paper.py`; deep-science-writer for prose once numbers are frozen.

### Artifacts
- `evaluation_results/p2to_ablation_manifest.json` (12 runs: cell/seed/render/h5/ckpt)
- `evaluation_results/p2to_ablation_leaderboard.json` (cell aggregates + per-run + H_v7 verdict)
- `checkpoints/flow_policy_v5/p2to_{T0O0,T0O1,T1O0,T1O1}_s{0,1,2}/best_model.pt` (12 ckpts)
- `scripts/run_p2to_ablation.py`, `scripts/evaluate_p2to_ablation.py`
- `scripts/train_flow_v5.py` (`--hover-h5` + memory-robust two-pass `FlowDatasetV5`),
  `scripts/evaluate_frozen_p0.py` (`--target-render`)

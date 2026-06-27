# Phase 0 Gates — Far-Range Teacher (Gate A) & Perspective Renderer (Gate B)

**Date:** 2026-06-25
**Plan:** `RESEARCH_PLAN_v7.md` Phase 0 (cheap gates before the multi-day T×O 2×2 pipeline)
**Goal:** before paying GPU-days, decide *which* teacher generates the 1–3 m recovery
coverage (Gate A) and confirm in-env that a non-saturating renderer carries far-range
metric distance (Gate B). Both are read-cheap (minutes, no training).

---

## Gate A — Far-range setpoint-offset recovery audit (decides the teacher)

### What was built

- **`controllers/pid_controller.py`** — behaviour-preserving refactor: cascade Levels 1–3
  extracted into `_outer_loop(state, target) → (F_total, omega_cmd)`; `compute_action`
  (per-motor, base-env) reuses it unchanged. New `compute_ctbr_action(state, target,
  f_c_max, omega_max)` stops at the rate-setpoint layer (skips the motor mixer) and emits
  a normalized CTBR action `[F_c_norm, wx, wy, wz] ∈ [-1,1]` — the exact inverse of
  `QuadrotorEnvV4._decode_action` (`F_c=(a+1)·0.5·F_c_max`, `omega=a[1:4]·omega_max`).
  Verified inverse-consistent and hover-self-consistent (`F_total=mg → F_c_norm=-0.385 ≈
  the PPO −0.39 hover bias`).
- **`envs/quadrotor_env_v4.py`** — `reset(options=...)` now honours
  `options['setpoint_offset']` (shifts the hover/waypoint target by a caller vector; default
  path unchanged). Reused by the audit and by the Phase-1/3 wide-init collection.
- **`scripts/audit_recovery_ctbr.py`** — pins the drone at a near-origin hover
  (`hover_anchor_prob=1.0`, so a 4 m horizontal target stays inside the |pos|<5 m bound),
  shifts the target by a horizontal offset of {1,2,3,4} m at a frozen random azimuth (PID &
  PPO see the **same** offset/seed → paired), and rolls each teacher closed-loop on the base
  env (no rendering). Reuses the frozen-P0 metric machinery (`compute_hierarchical_metrics`,
  `_aggregate_frozen`). Artifact: `evaluation_results/p0_recovery_audit_ctbr.json`.

### Pre-registered pass test
Over the 1–3 m band a teacher passes iff **survival ≥ 80% AND conditional steady-IAE ≤ 1.5 m**
(it actually flies back and converges, not merely survives while drifting) — aligned with the
v7 T1O1 ≤1.5 m target.

### Results (20 trials/distance, paired seeds, survive threshold 250/500 steps)

| Teacher | 1 m | 2 m | 3 m | 4 m |
|---|---|---|---|---|
| **PID-CTBR** (gentle: vel_max=1.0, Kp_pos=0.8) | 100% / 0.141 m | 100% / 0.156 m | 100% / 0.165 m | 100% / 0.179 m |
| PID-CTBR (baseline gains: vel_max=2.0, Kp_pos=1.5) | 83% / 0.41 m | **63.7% / 0.41 m** | 90% / 0.41 m | 86.5% / 0.40 m |
| PPO expert (`20260419_142245`) | 100% / 0.068 m | 100% / 0.067 m | **8.4% / —** | **3.8% / —** |

(survival% / conditional steady-IAE; "—" = 0 episodes survived ≥250 steps.)

**Verdict: PID-CTBR PASS · PPO FAIL.**

### Findings

1. **A competent full-band far-range teacher EXISTS and is cheap (no retrain).** Gentle
   PID-CTBR recovers the whole 1–4 m band at **100% survival, cond-IAE 0.14–0.18 m** — ~2×
   the oracle (0.068 m) and **~18× better than the 2.8 m closed-loop precision floor**. → the
   2.8 m floor is **not** a teacher-incapacity wall; the v7 T-axis is feasible.
2. **PPO cannot teach the far band.** Oracle-grade ≤2 m (0.067 m) but a hard cliff between
   2 m and 3 m (100% → 8%). Confirms the memory note ("PPO can't recover >2 m"). → **Phase 1
   PPO retrain is unnecessary; the teacher source is PID-CTBR.**
3. **The baseline-gain "2 m death valley" is an aggressive-approach artifact, not incapacity.**
   At 2 m the position loop saturates `vel_cmd` at `vel_max=2.0` on both diagonal axes
   (combined 2.83 m/s); braking then demands a tilt past the 60° termination → catastrophic
   ~54-step crashes clustered at diagonal azimuths. Easing `vel_max/Kp_pos` keeps the approach
   inside the tilt envelope → 100% everywhere. (Recommended teacher gains for Phase 1/3
   collection: `vel_max≈1.0, Kp_pos≈0.8`.)
4. **(Optional) hybrid teacher.** PPO is 2× more precise than PID-CTBR ≤2 m (0.067 vs 0.15 m).
   A composite "PPO ≤2 m + PID-CTBR ≥3 m" teacher would give near-oracle labels in-band and
   ~0.15 m labels far-band; not required to pass the gate.

---

## Gate B — Perspective renderer far-R² (confirms the observation channel in-env)

### What was built

- **`envs/quadrotor_visual_env.py`** — `_draw_target` now dispatches on
  `target_render ∈ {"crosshair","perspective"}` (constructor flag; default `"crosshair"` is
  byte-identical to production). `_draw_target_perspective` is the non-saturating, perspective,
  anti-aliased disk: apparent `radius = (W·focal)·physical_size/(dist+0.1)`, clipped
  `[0.5, 0.45W]`, sub-pixel AA — no 2 px saturation floor, so the far-range range cue survives.
  `make_visual_env` passes the flag through.
- **`scripts/measure_image_distance_info.py`** — `--target-render` / `--physical-size` flags.
- **`scripts/measure_higher_res_gate.py`** — deleted the inline `PerspectiveTargetVisualEnv`
  subclass; now toggles the **production** `target_render` flag (single source of truth, DRY).

### Results — far-range (≥1.5 m) image→distance ridge R²

| Renderer | DR | far-R² | near-R² | overall | script |
|---|---|---|---|---|---|
| crosshair (production) | OFF | 0.59 | 0.69 | 0.93 | image_distance_info (primal, n=300) |
| **perspective** | OFF | **1.00** | 0.96 | 0.999 | image_distance_info |
| crosshair (production) | ON | **−0.04** | 0.43 | 0.54 | image_distance_info |
| **perspective** | ON | **0.35** | 0.89 | 0.84 | image_distance_info |
| crosshair (production) | ON | 0.05 | 0.38 | 0.49 | higher_res_gate (dual, n=200) |
| **perspective** | ON | **0.40** | 0.87 | 0.85 | higher_res_gate |

**Verdict: PASS.** Under realistic DR noise the saturating crosshair carries **≈0** far-range
metric distance (−0.04 to 0.05 R²); the perspective disk lifts it to **0.35–0.40**, matching
the documented 0.12→0.42. In the clean no-DR limit the perspective target is a **perfect**
range encoder (R²=1.00 vs crosshair 0.59). DR jitter (±2 px crosshair size + per-frame noise)
swamps the crosshair's 4 quantised size levels but only mildly degrades the disk's continuous
1/dist radius — that robustness is the whole point.

The `higher_res_gate` dedup is verified: running it through the production flag reproduces the
pre-dedup figures (crosshair ≈0.05, perspective ≈0.40).

---

## Bottom line for Phase 1+

- **Teacher = PID-CTBR** (gentle gains), via `setpoint_offset` reset. No PPO retrain.
- **Observation = perspective renderer** (`target_render="perspective"`), now a first-class
  env flag ready for data collection / training / eval (O1 arms must train *and* eval on it).
- Both axes of the decisive **Teacher × Observation 2×2** are now buildable; proceed to Phase
  1/3 (wide-init PID-CTBR recovery collection × {crosshair, perspective}) → the frozen-P0 T×O
  evaluation that tests whether the 2.8 m precision floor breaks only when both are present.

### Artifacts
- `evaluation_results/p0_recovery_audit_ctbr.json`
- `evaluation_results/p0_gateB_{crosshair,perspective}_{nodr,dr}.json`,
  `evaluation_results/p0_gateB_higher_res_64.json`

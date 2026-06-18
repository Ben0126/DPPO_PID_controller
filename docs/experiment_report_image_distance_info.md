# Experiment Report — Precision is Information-Gated, Not Data-Gated (Phase 3b)

**Date:** 2026-06-18
**Scripts:** `scripts/measure_image_distance_info.py` (+ `scripts/measure_ood_coverage.py`)
**Artifacts:** `evaluation_results/p3b_image_distance_info.json` (DR on),
`evaluation_results/p3b_image_distance_info_nodr.json` (DR off)

Phase 3b set out to test whether the ~2.8 m precision floor is **data-gated** (fill
the OOD coverage gap → precision improves) or **information-gated** (the FPV
observation doesn't encode metric distance → no data can fix it). Two cheap
measurements settled it as **information-gated**, and surfaced two facts that make the
naive "wider-init retrain" both unnecessary and, as first set up, impossible.

---

## 1. Two facts that break the naive retrain

**(a) `--pos-range` does not create position-error coverage.** The recovery env uses
`target_type="hover"`, and `quadrotor_env_v4.reset()` sets
`self.target_position = init_pos.copy()` — the target is **anchored to the (moved)
init position**. Widening `--pos-range` just relocates the hover point; the position
*error* stays ~0. Measured directly: the wide (±3 m) collection
`expert_demos_v4_recovery_wide3.h5` has the **same** coverage as the ±1 m data
(both 0.4 % of timesteps beyond 1 m, max ~3–4 m from velocity transients only).
Recovery here comes from tilt+velocity, not position offset.

**(b) The PPO expert itself can't recover from >2 m position offset.** Rolling the
state-based oracle from a target offset (target ≠ init, zero velocity), 20 trials each:

| offset | final err | survived |
|-------:|----------:|---------:|
| 1.0 m | 0.066 m | 20/20 |
| 2.0 m | 0.066 m | 20/20 |
| 3.0 m | 1.561 m | **0/20** |

So even a *correct* position-offset collection could only generate clean labels up to
~2 m — but the policy's precision-limiting regime is 2–3 m, beyond the teacher.

---

## 2. The decisive measurement: the FPV image doesn't encode distance

The only distance-dependent feature in `quadrotor_visual_env._render_fpv` is the
target crosshair **size**: `size = max(2, min(6, int(6/(dist+0.5)) + dr))`, with the
per-episode DR jitter `dr ~ U{-2..+2}` and per-frame Gaussian pixel noise σ=5. The
crosshair *position* encodes only normalised direction (distance-invariant), and is
drawn only when the target is in front. Holding the drone level with the target
straight ahead, rendering 300 samples/distance:

**DR ON (what the policy trains/tests on):**

| distance | crosshair size (px) | — |
|---------:|--------------------:|---|
| 0.10 m | 6–6 | distinct |
| 0.50 m | 4–6 | overlaps |
| 1.00 m | 2–6 | overlaps |
| 1.50 m | 2–5 | overlaps |
| 2.00 m | 2–4 | overlaps |
| 2.50 m | 2–4 | overlaps |
| 3.00 m | 2–3 | overlaps |

- **Adjacent-distance d-prime is < 0.2 everywhere** (image separability / noise; d'≈1
  is "barely distinguishable"). Distances are not separable in image space.
- **Ridge decode image → distance R²:** overall 0.55, **near (<1 m) 0.41**,
  **far (≥1.5 m) 0.12** — distance is weakly recoverable near and essentially
  unrecoverable far.

**DR OFF (noiseless ceiling):** crosshair size collapses to exactly **2 px for every
distance ≥ 2 m**, so the renders at 2.0 / 2.5 / 3.0 m are **byte-identical**
(d-prime = 0 between them). Even with zero noise, the observation model carries
**no** distance information beyond 2 m; the only signal is ~4 quantisation steps
(size 6→4→3→2) over 0–2 m.

---

## 3. Conclusion

The policy's steady-state drift (median **2.83 m**, Phase 3b probe) sits exactly in
the band where the FPV image carries **no recoverable metric distance**. Therefore:

- **Precision is information-gated by the observation model, not data-gated.** The
  ~2.8 m / ~13 %-of-oracle floor that is invariant across *every* v5/v6 config
  (Phase 3c) is explained: no encoder, fusion, dispersive term, or training-data
  change can decode a distance the 64×64 FPV crosshair does not encode past 2 m.
- **The wider-init retrain is futile for precision** and, as first configured (just
  `--pos-range`), produced zero coverage change anyway. Running it would burn ~6 h to
  reproduce the 2.8 m floor.
- This is the **stronger negative result** the plan anticipated: the bottleneck is the
  *sensing*, not the representation (3a) or the data coverage. The direction channel
  (crosshair position) is enough to stay pointed at the target (survival), but not to
  null metric range (precision).

**Caveats.** T_obs=2 frames could in principle give range via motion parallax, but the
inter-frame baseline (28 ms) and slow drift make this far weaker than the static size
channel; IMU integrates to position drift, not absolute range. The measurement fixes a
level attitude and a straight-ahead target to isolate the size channel — the only
metric-range source; attitude/direction variation adds confounds, not range signal.

**Implication for the paper / next steps.** To move precision one must change the
*observation*, not the policy: a richer FPV (higher resolution, an explicit
range/optical-expansion cue, or stereo/depth), or accept that monocular 64×64 FPV
caps metric hover precision and report that as the finding. A wider-init BC retrain is
**not** recommended; if run for completeness, the prediction is cond-IAE stays ~2.8 m.

**Reproduce:**
```bash
dppo/Scripts/python.exe -m scripts.measure_image_distance_info            # DR on
dppo/Scripts/python.exe -m scripts.measure_image_distance_info --no-dr    # noiseless ceiling
```

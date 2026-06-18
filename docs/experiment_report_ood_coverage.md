# Experiment Report — Is Precision OOD-Coverage-Gated? (Phase 3b probe)

**Date:** 2026-06-18
**Script:** `scripts/measure_ood_coverage.py`
**Artifact:** `evaluation_results/p3b_ood_coverage.json`
**Policy rolled out:** D0E1 seed 0 (`checkpoints/flow_policy_v5/p2_D0E1_s0/best_model.pt`,
the clean E2E frontier), frozen seeds (base 12345, 30 ep, n_inf=2).

This is `RESEARCH_PLAN_v6.md` Phase 3 diagnostic (b), run as a **cheap measurement
first** (no retraining): does the ~2.8 m precision floor come from the policy living
in a region the BC data barely covers? It compares the distribution of position-error
magnitude ‖pos − target‖ in **training** vs **closed-loop steady state**.

The 15D obs is `[pos_error_body(3), rot_6d(6), vel_body(3), omega(3)]`
(`quadrotor_env_v4._get_observation`), so ‖states[:, 0:3]‖ is exactly the
position-error the encoder/flow-net was trained on (body-frame norm = world norm).

---

## 1. Result

| pos-err quantile | TRAIN (BC data, n=480k) | CLOSED-LOOP steady (n=4.6k) |
|------------------|------------------------:|----------------------------:|
| p50 | **0.066 m** | **2.833 m** |
| p90 | 0.074 m | 5.162 m |
| p95 | 0.197 m | 5.793 m |
| p99 | 0.621 m | 6.574 m |
| max | 4.073 m | 7.718 m |

| fraction of samples beyond X m | TRAIN | CLOSED-LOOP steady |
|--------------------------------|------:|-------------------:|
| > 1.0 m | 0.20 % | 90.0 % |
| > 2.0 m | 0.025 % | 66.5 % |
| > 3.0 m | 0.005 % | 45.6 % |

**OOD headline:**
- **97.3 %** of closed-loop steady-state samples sit **above the training p99** (0.62 m).
- **25.0 %** sit **above the training maximum** (4.07 m) — a quarter of the time the
  policy is in a region with **literally zero** BC training samples.
- steady-state **median 2.83 m = 4.56× the training p99** (and matches the D0E1
  cond-IAE 2.81 m exactly — this is the same steady-state window the leaderboard scores).

---

## 2. Interpretation

The BC training distribution is **overwhelmingly at-target**: 95.8 % of all timesteps
are within 0.2 m of the target, p99 = 0.62 m. The recovery mix adds only a
vanishingly thin tail beyond 1 m (0.2 % of mass) because the PPO expert pulls
straight back to target — by construction it spends almost no time *at* large error.

The closed-loop policy, by contrast, spends ~90 % of its steady state **beyond 1 m**
and ~46 % **beyond 3 m** — i.e. in the part of state space where it has essentially no
demonstrations to imitate. A BC policy can only reproduce expert behaviour where the
expert was observed; in the 1–7 m band there is ~no expert signal, so the learned
velocity field there is undefined / extrapolated, and the drone has no learned
"pull-back-from-far" behaviour. It drifts out and parks.

**Verdict: the precision floor is OOD-coverage-gated.** The precision-limiting region
(1–3 m steady drift) is exactly the region the BC data does not cover. This is a clean
covariate-shift / coverage-gap diagnosis and is consistent with *every* v5/v6 config
sharing the same ~2.8 m cond-IAE regardless of encoder/representation choices
(Phase 3c): no representation lever can fix a region with no training labels.

---

## 3. Scope, caveats, and the next (expensive) step

**This establishes a necessary condition, not sufficiency.** The probe proves the
coverage gap exists and that the policy lives in it. It does **not** prove that
*filling* the gap would fix precision — an alternative limit is information: 64×64 FPV
images may not resolve fine position at 2–3 m even with labels. The two hypotheses are
separable by the intervention the probe now justifies:

**Phase 3b retrain (justified, not yet run):** collect recovery demos with a **wider
init distribution** (larger `--pos-range` / `--tilt-max` / `--perturb-vel` in
`scripts.collect_data_v4_recovery`, so the expert is observed *transiting through* the
1–4 m band), retrain the D0E1 recipe on the wider mix, and re-measure cond-IAE under
the frozen protocol.
- If cond-IAE drops → precision was coverage-gated (fixable with data).
- If cond-IAE stays ~2.8 m despite coverage → precision is information-gated (a vision
  limit), an even stronger negative result.

Either outcome is publishable. Design choices to fix before launching (multi-hour:
~data collection + ~2 h/seed BC retrain + eval): the widened init ranges, whether to
also widen the *hover* anchor, and how many seeds.

**Reproduce:**
```bash
dppo/Scripts/python.exe -m scripts.measure_ood_coverage \
    --ckpt checkpoints/flow_policy_v5/p2_D0E1_s0/best_model.pt --label D0E1_s0 --n-episodes 30
```

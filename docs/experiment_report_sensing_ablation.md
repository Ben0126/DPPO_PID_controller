# Experiment Report — Is Precision Sensing-Gated? Higher-Res Gate + Range-Cue Intervention (Phase 3b)

**Date:** 2026-06-21
**Scripts:** `scripts/measure_higher_res_gate.py` (free info gate),
`scripts/run_p3b_rangecue.py` + `scripts/train_flow_v5.py` (`--range-cue`) + `scripts/evaluate_frozen_p0.py` (`--cue-noise`)
**Artifacts:** `evaluation_results/p3b_higher_res_gate.json`,
`evaluation_results/p3b_rc_{clean,noised}{,_s12}_frozen.json`,
`evaluation_results/p3b_rangecue_manifest{,_s12}.json`

Phase 3b-info (`experiment_report_image_distance_info.md`) concluded precision is
**information-gated**: the 64×64 FPV crosshair encodes no recoverable metric range past
~2 m (far R²=0.12), so "to move precision, change the observation." That was an
*argument from an information measurement*, not an intervention. This report does two
things: (A) a **free gate** that asks whether a more capable monocular renderer would
even carry the far-range information, and (B) the **decisive intervention** — inject the
exact metric position error the FPV lacks and re-measure closed-loop precision. The two
together **refute §6.3 as stated**: changing the observation is *not* sufficient to move
precision, and a richer cue is actively harmful.

---

## A. Higher-res gate — the information loss is a renderer artifact, not the pixel count

`measure_higher_res_gate.py` renders a level drone with the target straight ahead and
decodes target distance from the image (ridge, dual/kernel form so cost is O(n²) in
samples not O(d²) in pixels), stratified NEAR (<1 m) vs FAR (≥1.5 m), across a 3×2 grid:
resolution {64,128,256} × target {production saturating crosshair, perspective
anti-aliased disk whose radius = W·focal·S/dist}. DR on, 200 samples/distance.

| config | **far R² (≥1.5 m)** | near R² (<1 m) |
|--------|--------------------:|---------------:|
| 64 px — production crosshair  | **−0.01** | 0.45 |
| 128 px — production crosshair | 0.11 | 0.50 |
| 256 px — production crosshair | 0.04 | 0.23 |
| 64 px — perspective target  | **0.42** | 0.88 |
| 128 px — perspective target | **0.50** | 0.92 |
| 256 px — perspective target | **0.45** | 0.91 |

Two confounds cleanly separated:

1. **Raising resolution alone does nothing.** The production crosshair's far R² ≈ 0 at
   64 / 128 / 256 px. More pixels cannot recover far range while the target's size
   formula `size=max(2,min(6,int(6/(d+0.5))+dr))` saturates (cap 6, floor 2, flat ≥2 m).
   → "just use a higher-res camera" is **refuted**.
2. **De-saturating the target marker recovers ~half the far range — at 64 px.** Swapping
   the saturating crosshair for a perspective (optical-expansion) target lifts far R²
   from ≈0 to **0.42 at the SAME 64 px**; resolution is a small secondary lever
   (64→128: 0.42→0.50, then plateaus at 256).

So the §6.3 statement "monocular 64×64 FPV fundamentally cannot encode 2–3 m range" is
**too strong / partly an artifact** of this synthetic renderer's crosshair size
quantisation. The precise claim: far-range info is destroyed by the **target
representation**, and a perspective target restores ~half of it without any resolution
increase. **Practical consequence:** the cheapest information-restoring "better sensor"
retrain can be done at 64 px (same image size, same RAM — no 4× blow-up).
**Caveat:** even the perspective target's far R² (0.45) ≪ its near R² (0.88) —
optical expansion weakens as 1/dist² at range, so a better sensor would improve, not
solve, far-range precision. The range-cue intervention below tests the *upper bound*
(hand the policy the exact range) and shows even that is insufficient.

---

## B. Range-cue intervention — even the oracle cue does not rescue precision

The decisive test. We fold the metric body-frame position error the FPV cannot encode
(`states[:, :3]`, raw metres, computed from existing h5 — **no re-collection**) into the
v5 task-cond slot, so the model concats it into `global_cond` unchanged (`task_dim` widens
2→3 for scalar, 2→5 for pos3d). At eval the cue is rebuilt from the live env state each
step (matched scale/noise). Every arm is the **D0E1 frontier recipe** (dispersive OFF,
encoder E2E ON, H4-transfer init, 500 hover + 500 recovery), differing ONLY in the cue.
**3 seeds, frozen P0 (30 ep, paired init, σ=2.0). Control (no cue) reuses `p2_D0E1`.**

| arm | cue | **cond-IAE (mean±std)** | survive % | Tier1 % | n_cond s0/1/2 |
|-----|-----|------------------------:|----------:|--------:|:-------------:|
| **control** (D0E1) | — | **2.906 ± 0.075** | 65.0 ± 2.8 | 92.2 ± 3.1 | 27/27/29 |
| scalar_clean  | ‖pos_err‖, σ=0    | **2.430 ± 0.242** | 58.3 ± 4.9 | 78.9 ± 15.9 | 26/17/28 |
| scalar_noised | ‖pos_err‖, σ=0.15 | 2.805 ± 0.258 | 57.9 ± 10.3 | 66.7 ± 31.4 | 24/7/29 |
| pos3d_clean   | pos_err 3D, σ=0   | **n/a (collapse)** | 40.6 ± 3.0 | **6.7 ± 7.2** | 5/0/1 |
| pos3d_noised  | pos_err 3D, σ=0.15| 2.975 ± 0.242 | 53.9 ± 3.2 | 67.8 ± 9.6 | 20/24/17 |

(Oracle, state-based PPO under the same protocol: cond-IAE **0.068 m**, 100 % survive.)

**Findings (3-seed robust):**

1. **The oracle range cue buys a small, fragile precision dent — not a rescue.**
   scalar_clean lowers cond-IAE 2.906→2.430 m (~0.5 m, ~16 %; per-seed 2.27/2.25/2.77 —
   2/3 seeds improve, 1 negligible), and pays for it with **−6.7 pp survival and −13 pp
   Tier1**. 2.43 m is still **~36× the oracle's 0.068 m** — not sub-metre, not deployable.
   Handing the policy the *exact* metric range it "lacks" closes only ~1/6 of the gap.
2. **A realisable sensor erases it.** Adding σ=0.15 m noise (scalar_noised → 2.805 m)
   brings precision back to control within seed noise. A real ToF/baro/optical-expansion
   range sensor at this accuracy gives ≈0 precision benefit.
3. **A richer cue is actively, reproducibly harmful.** The full body-frame position error
   (pos3d_clean) **collapses survival across all 3 seeds** — Tier1 92→**7 %**, n_cond
   5/0/1 (it almost never flies to 250 steps), cond-IAE undefined. Noising it
   (pos3d_noised) restores flight but yields no precision gain. "More sensing → better"
   is falsified.

**Caveat.** cond-IAE is computed over surviving episodes only, and arms survive slightly
different episode subsets (different n_cond), so cross-arm cond-IAE is not perfectly
paired; control and scalar arms both have adequate n_cond (≈24–28) so the comparison is
sound, but scalar_clean's lower survival means its conditional set is marginally easier.
This caveat only *weakens* an already-small effect.

---

## C. Unified conclusion — precision is coverage/competence-gated, not sensing-gated

The intervention overturns the §6.3 implication. Putting it with the rest of the Phase 3
chain:

- 3a (feature collapse): representation is **not** the constraint (dispersive is a no-op;
  `vis_pooled` rank decouples from survival).
- 3b-coverage: the closed-loop policy parks at ~2.8 m, where BC has <0.2 % of its mass —
  a real coverage gap; **and** the PPO expert itself cannot recover from >2 m offset.
- 3b-info + **gate (A)**: the FPV's far-range information loss is a renderer **target
  artifact**, recoverable at 64 px with a perspective target — i.e. the observation
  *could* carry range.
- 3b-rangecue **(B, this report)**: but **even handed the exact metric range/position,
  precision does not recover** (≤0.5 m, fragile) and a richer cue **destroys survival**.

**Therefore the binding constraint on precision is not the observation's range channel.**
It is the absence of any learned *far-range recovery behaviour* in the 1–3 m band: the
BC policy has no demonstrations there (coverage gap) and the teacher cannot generate them
(the expert can't recover from >2 m). Range tells the drone *how far off* it is, but not
*what to do* about it — and there is no data teaching that. This is a **stronger, cleaner
negative result** than "sensing is the bottleneck": no representation, encoder, dispersive
term, data-coverage tweak, observation channel, or oracle cue moves the ~2.8 m floor.
Moving precision would require a **competent far-range teacher** (a controller that
recovers from multi-metre offsets) to generate coverage — not a better sensor and not a
better policy over the existing data.

**For the paper.** §6.3 should be reframed from "precision is information-gated; change
the observation" to: *the FPV's range-info loss is a fixable renderer artifact, yet
supplying metric range (even oracle, even full position) does not restore precision and a
richer cue harms survival → precision is gated by the teacher/coverage competence in the
1–3 m band, not by sensing.* The wider-init retrain remains **not recommended** (the
expert can't label that band; prediction unchanged: cond-IAE stays ~2.8 m).

---

## Reproduce

```bash
# A. free higher-res / target-saturation gate
dppo/Scripts/python.exe -m scripts.measure_higher_res_gate --n-samples 200

# B. range-cue ablation (3 seeds; control reuses p2_D0E1)
dppo/Scripts/python.exe -m scripts.run_p3b_rangecue --seeds 0 1 2
# eval, matched cue-noise per arm (clean σ=0 / noised σ=0.15)
dppo/Scripts/python.exe -m scripts.evaluate_frozen_p0 \
    --ckpts "scalar_clean_s0:checkpoints/flow_policy_v5/p3b_rc_scalar_clean_s0/best_model.pt" \
    --n-episodes 30 --cue-scale 3.0 --cue-noise 0.0
```

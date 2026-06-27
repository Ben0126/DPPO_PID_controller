# Experiment Report — Phase 6: Scale-invariant on-path regularizer ablation

**Date:** 2026-06-27
**Plan:** `RESEARCH_PLAN_v7.md` Phase 6 / Direction 4 (formal ablation, user committed 2026-06-24; ordered before Phase 7 on 2026-06-26)
**Artifacts:** `evaluation_results/p6_form_ablation_manifest.json`, `evaluation_results/p6_form_ablation_leaderboard.json`
**Scripts:** `models/flow_policy_v5.py` (`_dispersive_loss_cosine`, `_dispersive_loss_vicreg`), `scripts/train_flow_v5.py` (`--dispersive-form`), `scripts/run_p6_form_ablation.py`, `scripts/evaluate_p6_form_ablation.py`

---

## TL;DR / Verdict

The last open §6.1 rebuttal was: *"your faithful InfoNCE-L2 Dispersive Loss games its objective by
inflating feature norm — that is an artifact of a **scale-sensitive** criterion; a scale-invariant one
might behave differently and actually help control."* Phase 6 tests this directly by holding the entire
**P2f D1E1 faithful recipe fixed** and varying **only the dispersive FORM** on the `flow_net` mid-block
(3 seeds each, frozen-P0 eval, crosshair):

| | **[A] Objective-gaming (geometry)** | **[B] Closed-loop control** |
|---|---|---|
| **Result** | **REMOVED by scale-invariance** | **NOT improved — decoupled from geometry** |
| infonce (faithful) | GAMES: feat_norm **8.93× off**, eff_rank **collapses 221→36** | survival 62.9% / Tier1 90.0% / cond-IAE 2.894 m |
| cosine (unit-sphere) | clean: feat_norm **1.33× off**, eff_rank **221→769 (UP)** | 64.4% / 86.7% / 3.103 m — *regressed* |
| vicreg (var+cov) | clean: feat_norm **1.36× off**, eff_rank **221→867 (UP)** | 60.5% / 82.2% / 3.028 m — *regressed* |

- **[A] is the headline and it is decisive:** a scale-invariant criterion **eliminates the norm-inflation
  gaming** (feat_norm stays O(off) instead of blowing up ~9×) **and does not worsen intrinsic rank — it
  *improves* it dramatically** (effective rank rises from 22% of the 1024-dim space to **75% (cosine) /
  85% (vicreg)**, vs the faithful term's collapse to **3.5%**). So the gaming was indeed an artifact of the
  scale-sensitive L2 criterion, and removing it yields *genuine* high-rank feature dispersion — the stated
  goal of Dispersive Loss — **without the cheat**.
- **[B] is the clean negative-result reinforcement:** a **24× swing in flow_mid effective rank**
  (36 → 867) and a **6.6× swing in feat_norm** (85 → 13) move closed-loop survival/Tier1/precision by
  **~0 in the improving direction**; the only metric crossing the pooled seed std does so in the *worse*
  direction (cond-IAE +0.13–0.21 m). **No scale-invariant form improves control.** → **Criterion scale
  changes only geometry; survival and precision are decoupled from flow_mid representation geometry**,
  exactly as §6.1 found for the off-path `vis_pooled` (rank ⟂ survival). This closes the rebuttal *and*
  deepens the negative result.

---

## Motivation

§6.1 of the negative-result paper showed the **faithful** Dispersive Loss (InfoNCE-L2 on the `flow_net`
mid-block, `/d`, τ=0.5, λ=0.5) minimises its own objective (`disp_infonce` −0.29 → −7.62) primarily by
**inflating feature norm** (~9× on flow_mid, ~287× on the legacy off-path `vis_pooled`) while making the
*intrinsic* rank **worse** — "objective gaming." A reviewer can dismiss this as a property of a
scale-sensitive distance, not of Dispersive Loss in general. Direction 4 was pre-registered to close that
door with two scale-invariant criteria that **cannot** lower their loss by growing norms.

## Design (pre-registered)

Base recipe = **P2f D1E1**, fixed for every arm — `--transfer-from-h4
checkpoints/flow_policy_v4/20260514_175219`, E2E (no `--freeze-vision`), `--recovery-h5
data/expert_demos_v4_recovery.h5 --hover/recovery-episodes 500`, faithful flow_mid, λ=0.5, τ=0.5. Only
`--dispersive-form` varies, so the new arms line up **directly** against the existing §6.1 numbers.

| form | definition (on flow_mid mid-block features `z`, flattened d=1024) | source |
|------|------------------------------------------------------------------|--------|
| **off** | λ = 0 | reuse **P2f D0E1** (not re-trained) |
| **infonce** | faithful InfoNCE-L2: `D = ‖zᵢ−zⱼ‖²/d`, `L = log E[exp(−D/τ)]` | reuse **P2f D1E1** (not re-trained) |
| **cosine** | unit-sphere InfoNCE: `z ← z/‖z‖`, `D = ‖zᵢ−zⱼ‖²` (= 2−2cos), `L = log E[exp(−D/τ)]` — **`/d` dropped** so the angular repulsion stays in force; norm inflation impossible by construction | **new** ×3 seeds |
| **vicreg** | VICReg-style: `var = mean_j max(0, γ−std_j)` (γ=1) `+ cov = (1/d)·Σ_{i≠j} Cov(z)²_{ij}`; both ≥0, minimised when spread+decorrelated (same "more-dispersed→lower-loss" sign as infonce) | **new** ×3 seeds |

New training = cosine×3 + vicreg×3 = **6 runs** (off/infonce reused → 6 runs saved). Sequential, one GPU at
a time (Known Failure Mode #7), batch 256, 80 epochs.

### Implementation notes (correctness)
- **cosine drops `/d`.** On the unit sphere `‖zᵢ−zⱼ‖² ≤ 4`, so dividing by d≈1024 would shrink `D` to
  ~4/d and saturate `exp(−D/τ)→1` (vanishing gradient). Without `/d` the repulsion acts; verified
  numerically scale-invariant (`L(x)=L(1000·x)`, |Δ|=2.4e-7).
- **vicreg forces float32.** The off-diagonal covariance-square sum over ~d² terms overflows fp16 under
  AMP autocast → cast to float32 inside the loss (verified: a rank-1, large-norm fp16 input yields a
  finite value, not `inf`).
- **`--dispersive-form infonce` is the default and is byte-identical to the old code path** (the existing
  `compute_loss(... lambda_dispersive>0, dispersive_target='flow_mid' ...)` call reproduces the exact
  previous `loss_dispersive` to machine precision). P2f checkpoints/manifests are untouched (new `p6_*`
  tags + new manifest).

## Results

All values mean ± std over **3 seeds**, frozen-P0 protocol (30 episodes, base_seed 12345, σ=2.0,
n_inference_steps=2, crosshair render — identical to P2f). Measured oracle composite = **0.9668**.

### [A] Feature geometry — faithful `flow_mid` (paired 4000-image batch, d=1024)

| form | effective rank | rank ratio | mean feat_norm | feat_norm ÷ off | `disp_infonce` |
|------|---------------:|-----------:|---------------:|----------------:|---------------:|
| off (D0E1)      | 221.4 ± 16.2 | 21.6 % | 9.49 ± 0.28 | 1.00× | −0.293 |
| **infonce (D1E1)** | **35.9 ± 0.6** | **3.5 %** | **84.79 ± 0.51** | **8.93×** | **−7.624** |
| **cosine**      | **769.4 ± 14.8** | **75.1 %** | **12.61 ± 0.30** | **1.33×** | −0.588 |
| **vicreg**      | **866.9 ± 4.3** | **84.7 %** | **12.88 ± 0.01** | **1.36×** | −0.622 |

> infonce **games**: drives its own `disp_infonce` to −7.62 by **inflating norm 8.93×** and **collapsing
> rank** (221→36, i.e. to 3.5 % of 1024 dims). cosine/vicreg **do not inflate** norm (≈1.3× off) and
> **raise** effective rank to 75–85 % — genuine dispersion without the norm cheat. Their `disp_infonce`
> barely moves from off (−0.59/−0.62 vs −0.29) precisely because they optimise a *different*,
> scale-invariant objective rather than the L2-norm-gameable one.

### [B] Closed-loop control — frozen P0 (crosshair)

| form | survival % | Tier1 % | cond-IAE (m) | n_cond | composite (% oracle) |
|------|-----------:|--------:|-------------:|-------:|---------------------:|
| off (D0E1)      | 65.0 ± 2.8 | 92.2 ± 3.1 | 2.906 ± 0.075 | 27.7 | 0.1197 (12.4 %) |
| infonce (D1E1)  | 62.9 ± 2.4 | 90.0 ± 5.4 | 2.894 ± 0.069 | 27.0 | 0.1183 (12.2 %) |
| cosine          | 64.4 ± 3.3 | 86.7 ± 8.2 | 3.103 ± 0.140 | 26.0 | 0.1108 (11.5 %) |
| vicreg          | 60.5 ± 2.4 | 82.2 ± 11.0 | 3.028 ± 0.104 | 24.7 | 0.1054 (10.9 %) |

> All four forms sit in the same band: survival 60–65 %, Tier1 82–92 %, cond-IAE **2.9–3.1 m (≈ 35–43×
> the 0.068 m oracle, 11–12 % oracle)**. Decisive comparisons vs infonce (> pooled std = "moved";
> **improvement** = survival/Tier1 ↑ or cond-IAE ↓):
> - **cosine vs infonce:** Tier1 −3.3 pp (pooled 9.8, flat) · survival +1.5 pp (pooled 4.1, flat) ·
>   cond-IAE **+0.209 m (pooled 0.156 → REGRESSED, worse precision)**.
> - **vicreg vs infonce:** Tier1 −7.8 pp (pooled 12.3, flat) · survival −2.3 pp (pooled 3.4, flat) ·
>   cond-IAE **+0.134 m (pooled 0.124 → REGRESSED, worse precision)**.
>
> **`any_control_improved = False`.** The only metric to cross the pooled seed std does so in the *worse*
> direction. A 24× rank swing + 6.6× norm swing buys **no** control improvement.

*(Cross-check: the reused off/infonce numbers reproduce the P2f leaderboard's D0E1/D1E1 cells exactly,
confirming the Phase-6 harness is byte-identical to P2f.)*

## Interpretation

1. **The rebuttal is closed.** The norm-inflation "gaming" of the faithful term **is** an artifact of the
   scale-sensitive L2-InfoNCE criterion: two scale-invariant criteria remove it entirely (feat_norm O(1))
   while achieving the *intended* effect — high-rank, decorrelated mid-features (eff_rank to 75–85 % vs
   the faithful term's 3.5 %). So "Dispersive Loss did nothing" cannot be deflected as "you used it wrong /
   you used a degenerate criterion."
2. **The negative result is deepened, not weakened.** Despite genuinely fixing feature collapse on the
   on-path generative representation (the exact thing Dispersive Loss is designed to do, now demonstrably
   working), **closed-loop survival and precision do not improve.** Geometry of the flow_mid representation
   is **decoupled** from control — the on-path analogue of the §6.1 `vis_pooled` finding (rank ⟂ survival).
   Representation collapse is **not** the binding constraint, regardless of which (faithful or
   scale-invariant) criterion enforces dispersion.
3. **Orthogonal to the T×O precision result (Phase 4).** Phase 6 concerns the §5/§6.1 Dispersive axis; it
   does not touch the cond-IAE ≈ 2.8 m floor conclusion (Robustness–Precision capacity conflict). It is a
   hardening of the Dispersive negative result for §5/§6.1.

## Conclusion

> **Scale-invariance removes the objective-gaming (feat_norm 8.93×→1.3×, eff_rank 3.5 %→75–85 %) but does
> NOT move closed-loop control — it even slightly regresses precision. The criterion's scale only affects
> geometry; survival/precision are decoupled from flow_mid feature geometry. The §6.1 "you used a
> scale-sensitive criterion" rebuttal is closed, and the Dispersive negative result is reinforced.**

## Reproduce

```bash
# Train the two NEW scale-invariant forms (off/infonce reused from P2f), 3 seeds, sequential:
dppo/Scripts/python.exe -m scripts.run_p6_form_ablation \
    --h4-ckpt checkpoints/flow_policy_v4/20260514_175219/best_model.pt \
    --recovery-h5 data/expert_demos_v4_recovery.h5 \
    --recovery-episodes 500 --hover-episodes 500 \
    --forms cosine vicreg --seeds 0 1 2

# Evaluate (control frozen-P0 + flow_mid geometry; off/infonce pulled from the P2f manifest):
dppo/Scripts/python.exe -m scripts.evaluate_p6_form_ablation \
    --manifest evaluation_results/p6_form_ablation_manifest.json \
    --oracle-ckpt checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
    --oracle-norm checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz

# (Recompute ONLY the directional verdict from the cached leaderboard, no GPU:)
dppo/Scripts/python.exe -m scripts.evaluate_p6_form_ablation \
    --reverdict evaluation_results/p6_form_ablation_leaderboard.json
```

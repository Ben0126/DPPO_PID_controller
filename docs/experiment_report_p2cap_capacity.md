# Experiment Report — P2CAP: Testing the Robustness–Precision *Capacity* Conflict

**Plan:** `RESEARCH_PLAN_v8.md` (Phase 0 gates + Phase 1 capacity scaling ladder)
**Date:** 2026-06-28 (Gate A + param audit DONE; ladder RUNNING)
**Status:** Phase 0 Gate A complete; Phase 1 ladder in progress (M×3 + XL×3, ~12–18 h).

---

## 0. Question

v7 localized the cond-IAE ≈ 2.8 m precision floor to a **Robustness–Precision Capacity
Conflict** (decisive T×O 2×2: adding far-recovery labels under the perspective observation
*degrades* precision T0O1 2.48 → T1O1 2.93 m — a **negative coverage×precision interaction**).
But "capacity" was the **leading-but-UNTESTED** explanation. v8 tests it.

---

## 1. Phase 0, Gate A — open-loop fitting interference vs closed-loop compounding (DONE)

`scripts/measure_capacity_fit.py` → `evaluation_results/p2cap_gateA_fit.json`. On the existing
12 p2to checkpoints, measured paired open-loop fit on a FIXED seeded batch (n=3000; same (t,eps)
flow-loss draw and same `_fixed_x1` 2-step inference noise across models), mirroring the
`FlowDatasetV5` windowing. Generalist (T1, hover+far) vs hover-specialist (T0, hover-only), on
the hover near-target batch (precision-relevant) and the far batch (tradeoff direction).

| cell | render | hover act_mse | far act_mse |
|------|--------|---------------|-------------|
| T0O0 | crosshair   | 0.00033 | 0.00540 |
| T1O0 | crosshair   | 0.00035 | 0.00032 |
| T0O1 | perspective | 0.00035 | 0.00787 |
| T1O1 | perspective | 0.00038 | 0.00033 |

**Decisive (perspective, generalist T1O1 − specialist T0O1 on hover):**
`d_hover = +0.000023` (pooled seed std 0.000007 → **3.2× std, resolvable**), `rel = +6.4%`;
`d_far = −0.0075` (generalist fits far ~24× better). Crosshair replication: `+4.0%` (2× std),
`d_far = −0.0051`.

**Reading.** Adding far-recovery labels imposes a **real but small** open-loop fitting cost on
precise hover (~4–6%, statistically resolvable but tiny in absolute terms — both models fit hover
near-perfectly open-loop), while the generalist fits the far band ~20–24× better. So the ~13.5 M
model has **near-enough capacity to represent BOTH behaviours open-loop**. Crucially the open-loop
hover cost (~6%) is **far smaller** than the closed-loop precision regression (+0.45 m / +18%) →
the floor is **predominantly a closed-loop / partial-observability phenomenon**, not open-loop
fitting capacity. Branch fired: `FITTING_INTERFERENCE` (real, weak) → the scaling ladder is
justified, **with the pre-registered prior that scaling will NOT break the floor**.

## 2. Parameter-count audit (correction to the paper) (DONE)

The deployed p2to policy is **13,509,512 trainable parameters** (E2E, no freeze), not "≈ 3 M".
Breakdown: `flow_net` 12.06 M (89.2%), imu_encoder 0.53 M, vision_encoder 0.46 M, cross_attn
0.40 M, state_predictor 0.07 M, tilt_head 0.0005 M. Even with the vision encoder frozen it is
13.05 M trainable. **`docs/paper_negative_result_draft.md` line ~823 ("≈ 3 M trainable") is a
factual error (~4.5×) and must be corrected in Phase 2.** The capacity-conflict prior is
correspondingly weaker — 13.5 M is not obviously capacity-starved for a 4-DoF × 8-step action head.

## 3. Phase 1 — capacity scaling ladder (RUNNING)

`scripts/run_p2cap_ablation.py` + `scripts/evaluate_p2cap_ablation.py`. Holds the **T1O1 frontier
recipe** fixed (hover_persp + far_persp, transfer-from-h4, λ_disp 0, task-cond, perspective eval)
and varies **only** flow_net `--down-dims`, 3 seeds/rung, frozen P0:

| rung | down_dims | flow_net | total | H4 transfer |
|------|-----------|----------|-------|-------------|
| S | (256,512) | 12.1 M | 13.5 M | reused == existing p2to_T1O1 (no retrain) |
| M | (256,512,768) | 33.6 M | 35.0 M | deepen +1 level (first 2 levels transfer; 54 tensors) |
| XL | (512,1024) | 42.6 M | 44.0 M | widen (flow_net from scratch; 37 tensors) |

VRAM verified: XL @ batch 256 peaks **0.99 GB** (flow_net = 1D convs on length-8 sequences → tiny
activations). M `--quick` smoke: builds 35 M, partial-transfer 54/skip 40, val_flow 0.247→0.030.

**Pre-registered H_v8 verdict** (per scaled rung R∈{M,XL} vs S baseline): floor "broken by
capacity" iff `cond-IAE(R) < cond-IAE(S) − pooled_std` AND `≤ 1.5 m` AND survival guard holds AND
`n_cond ≥ 15`. Secondary: does cond-IAE trend monotonically down with capacity?

### Result (DONE 2026-06-29) — FLOOR NOT BROKEN BY CAPACITY

`evaluation_results/p2cap_ablation_leaderboard.json` (frozen-P0, perspective render, 3 seeds/rung,
oracle composite 0.9668):

| rung | flow_net | total | cond-IAE (m) | survival | Tier1 | %oracle |
|------|----------|-------|--------------|----------|-------|---------|
| S  | (256,512)     | 13.5 M | **2.931 ± 0.176** | 90.6 ± 3.3% | 98.9% | 20.8% |
| M  | (256,512,768) | 35.0 M | **2.670 ± 0.262** | 86.7 ± 2.3% | 100.0% | 23.5% |
| XL | (512,1024)    | 44.0 M | **2.633 ± 0.192** | 86.6 ± 6.7% | 94.4% | 24.5% |

**H_v8 verdict: `FLOOR_BROKEN_BY_CAPACITY = False`.** cond-IAE improves *monotonically* with capacity
(2.93 → 2.67 → 2.63 m) — the XL gain is statistically resolvable (Δ −0.298 m > pooled std 0.261 m),
M is not (Δ −0.261 < 0.316) — but the improvement is **small and saturating** (M→XL only −0.04 m) and
stays **~39× the 0.0675 m oracle (≤24.5% oracle composite)**, far from the pre-registered ≤1.5 m
"broken" target. So scaling flow_net **3.3×** buys ~0.3 m of precision and then plateaus.

**Survival even dips −4 pp** as capacity grows (90.6 → 86.6%): the extra precision is bought with a
little survival — the robustness–precision tradeoff is *shifted slightly toward precision, not
dissolved*. Capacity moves the operating point along the conflict frontier; it does not remove the
frontier. Neither end is deployable.

**Open-loop corroboration (two independent signals):** (i) all three rungs converge to the SAME best
`val_flow ≈ 0.0104` (S 0.0103–0.0108, M 0.0104, XL 0.0104) — 3.3× capacity does not lower the training
loss, i.e. 13.5 M already fits the data; (ii) Gate A showed the 13.5 M generalist fits hover+far
open-loop to within ~6% of the specialist. The model is **not fitting-capacity-bound**.

**Conclusion.** Capacity — the leading-but-UNTESTED explanation from v7 — is now **tested and is at most
a minor lever**: a 3.3× flow_net scale-up yields a small, saturating, survival-costing precision gain
and the cond-IAE ≈ 2.8 m floor **holds** (best 2.63 m, ~39× oracle). The binding constraint is therefore
**more fundamental than parameter count**, consistent with a **closed-loop / partial-observability**
bottleneck (compounding error under monocular FPV) rather than representational capacity. v8 hardens the
negative result: §7 upgrades "capacity = leading untested explanation" → "capacity tested by scaling,
floor held."

---

## Artifacts
- `evaluation_results/p2cap_gateA_fit.json` (Gate A, 12 ckpts, per-run + verdict)
- `evaluation_results/p2cap_ablation_{manifest,leaderboard}.json` (ladder; leaderboard pending)
- `scripts/measure_capacity_fit.py`, `scripts/run_p2cap_ablation.py`, `scripts/evaluate_p2cap_ablation.py`
- `scripts/train_flow_v5.py` (`--down-dims`, `--hover-weight`), `scripts/evaluate_hierarchical.py`
  (`detect_arch` now infers `down_dims` from the checkpoint → capacity-agnostic eval)
- ladder ckpts `checkpoints/flow_policy_v5/p2cap_{M,XL}_s{0,1,2}/` (S reuses p2to_T1O1_s{0,1,2})

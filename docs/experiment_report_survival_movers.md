# Experiment Report — What Actually Moved Survival (Phase 3c)

**Date:** 2026-06-18
**Sources:** `evaluation_results/p2_ablation_leaderboard.json` (P2, 3 seeds),
`evaluation_results/frozen_p0_leaderboard.json` (prior frontier), both scored by the
**same** frozen protocol (`scripts/evaluate_frozen_p0.py`: 30 ep, base_seed 12345,
σ=2.0 exp-decay, paired init, conditional-IAE, measured oracle 0.9668).

This is `RESEARCH_PLAN_v6.md` Phase 3 diagnostic (c): attribute the survival
improvement over the prior frontier to its real cause. The short version: the
**recipe** (H4-transfer init + task-conditioning + recovery mix) is the dominant
survival lift; **E2E** (a trainable encoder) adds a small Tier1-only bump;
**Dispersive** adds nothing; and **precision moves with none of them**.

---

## 1. The numbers

Primary axis = **Tier1 pass-rate** (fraction of 30 episodes flying ≥250/500 steps)
and **survival**. Precision = cond-IAE over surviving episodes (reliable here:
n_cond 25–29 per cell). P2 cells are mean ± std over seeds 0/1/2.

| Config | Recipe | E2E | Disp | **Tier1%** | Survival | cond-IAE (n) | %Oracle |
|--------|:------:|:---:|:----:|-----------:|---------:|-------------:|--------:|
| **Joint_E2E_v5** (prior frontier) | partial | ✓ | ✗ | 80.0 | 62.2 % | 3.077 (24) | 11.0 % |
| **D0E0 / D1E0** (P2) | ✓ | ✗ (frozen) | ✗/✓ | 87.8 ± 3.1 | 66.1 ± 3.7 | 2.93 (25–27) | 13.4 % |
| **D0E1** (P2) | ✓ | ✓ | ✗ | 92.2 ± 3.1 | 65.0 ± 2.8 | 2.91 (27–29) | 12.4 % |
| **D1E1** (P2) | ✓ | ✓ | ✓ | 93.3 ± 2.7 | 65.0 ± 2.5 | 2.81 (27–29) | 13.2 % |

"Recipe" = H4-transfer warm-start init + task-conditioning (`task_dim=2`) + 500
hover/500 recovery mix, shared by all four P2 cells. Joint_E2E_v5 (`20260603_171316`)
was the unfrozen-encoder baseline *before* the full recipe was assembled.

---

## 2. Decomposition of the survival lift

Reading the table as a sequence of controlled steps:

**(i) Recipe — the dominant mover.** Prior frontier → the P2 **frozen** cell D0E0:
Tier1 **80.0 → 87.8 (+7.8 pp)**, survival **62.2 → 66.1 (+3.9 pp)**. Crucially this
cell has a **frozen** encoder, so the gain is *not* from end-to-end training — it is
the shared recipe (H4-transfer init + task-cond + recovery mix). Even D0E0's
worst-of-3 seed (83.3 % Tier1) clears the prior frontier, so the floor genuinely
lifted (not a seed artifact). *Caveat:* the prior frontier is a single run vs a
3-seed mean; the point estimate should be read as "the recipe lifted the floor by
~+8 pp Tier1", not to sub-pp precision.

**(ii) E2E — a small Tier1-only bump.** Within the matrix, D0E1 vs D0E0 (the clean,
identical-everything-else contrast): Tier1 **+4.4 pp** (87.8 → 92.2) but survival
**−1.1 pp** (66.1 → 65.0) and cond-IAE flat (2.93 → 2.91). A trainable encoder
pushes a few more episodes past the half-horizon **without extending mean flight or
improving precision** — it redistributes, it doesn't deepen.

**(iii) Dispersive — nothing.** D1E1 vs D0E1: Tier1 **+1.1 pp** inside the 4.2 pp
pooled across-seed std; survival identical (65.0). See
`docs/experiment_report_feature_collapse.md` for the mechanistic reason (it inflates
`vis_pooled` norm and worsens intrinsic rank, on a feature off the action path).

**Survival-mover ranking:** recipe (≈ +7.8 pp Tier1 / +3.9 pp survival, frozen) ≫
E2E (+4.4 pp Tier1, survival-neutral) ≫ Dispersive (~0).

---

## 3. The axis nothing moved: precision

cond-IAE is essentially constant across the entire study — prior frontier 3.08 m,
P2 cells 2.81–2.93 m, **all ≈ 13 % of the 0.068 m oracle** (n_cond 24–29, so these
are reliable, not short-survival artifacts). No lever exercised here — init,
task-cond, recovery mix, encoder trainability, or dispersive regularisation — moved
closed-loop hover precision off the ~2.8 m floor. **Nothing is deployable**
(deployable ≈ sub-0.5 m; oracle is 0.068 m).

This separation is the report's payload: the recipe buys **survival** (staying
airborne longer / past the half-horizon) but not **precision** (how tightly it holds
position once airborne). Survival and precision are decoupled levers, and only
survival responded to any of the v5/v6 interventions.

---

## 4. Conclusion & handoff to Phase 3b

For the negative-result paper, the honest causal story is:

1. **Dispersive Loss — no effect** (the falsified core thesis; Phase 2 + 3a).
2. **The recipe (H4-transfer + task-cond + recovery mix) is the real survival
   lever** (+~8 pp Tier1 over the prior frontier, even with a frozen encoder).
3. **E2E adds a marginal Tier1-only bump** that neither extends survival nor
   improves precision.
4. **Precision is stuck at ~13 % of oracle regardless** — it is *not* gated by any
   representation/training lever tested, which is exactly the question Phase 3b (b)
   probes: is precision instead gated by **OOD coverage** of the recovery-init
   distribution? (`RESEARCH_PLAN_v6.md` Phase 3b.)

**Reproduce:** numbers come straight from the two leaderboard JSONs; no rerun needed.

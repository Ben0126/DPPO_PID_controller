# Experiment Report — Faithful Dispersive Re-run (P2f, 2026-06-23)

**One-line result:** Re-running the Dispersive×E2E 2×2 with an *official-code-faithful*
Dispersive Loss **reproduces the negative result** (D1E1 vs D0E1 = **−2.2pp Tier1**,
pooled std 6.3pp; survival −2.1pp) and **overturns the legacy C2 "byte-identical
no-op"** (faithful placement trains `flow_net` even with a frozen encoder, so
D1E0 ≠ D0E0 and Dispersive-on-frozen is mildly *harmful*).

Canonical artifact: `evaluation_results/p2f_ablation_leaderboard.json`
(manifest `evaluation_results/p2f_ablation_manifest.json`).

---

## 1. Why this re-run was needed (method-fidelity confound)

The deep-science-writer smoke test of the negative-result paper found the legacy P2
Dispersive implementation was **unfaithful on three axes**, while paper §2 claims it is
applied "exactly as specified":

| Axis | Legacy P2 (`vis_pooled`) | Faithful ([13] Wang&He / [14] D²PPO, official `raywang4/DispLoss`) |
|------|--------------------------|-------------------------------------------------------------------|
| Placement | off-action-path `vis_pooled` (feeds only the aux `StatePredictor`) | generative-net intermediate = **`flow_net` mid-block** |
| Weight λ | 0.05 | **0.5** |
| Form | hand-rolled `−mean(log‖fi−fj‖)` | **InfoNCE-L2**: `log E[exp(−D/τ)]`, `D = ‖zi−zj‖²/d`, τ=0.5 |

A whole negative-result paper resting on "the faithfully-implemented method fails"
cannot use an implementation that is unfaithful on placement, scale, and form.

### Second trap caught during the smoke (the `/d` normalisation)
The paper's printed **Algorithm 1 omits the `/z.shape[1]` (per-dimension) normalisation**
present in the official code. Without `/d`, on GroupNorm'd flat mid-features
`D = ‖zi−zj‖²` is O(d); `exp(−D/τ)→0` at τ=0.5 ⇒ the loss **saturates to zero
gradient** — i.e. a *second* silent no-op. The faithful implementation restores `/d`
([models/flow_policy_v5.py:222-224](../models/flow_policy_v5.py#L222-L224)); the smoke
then showed `loss_dispersive` live and non-zero (−2.23→−2.77 over training) and 40/40
`flow_net` parameters receiving gradient.

Both traps would each have manufactured a *fake* faithful no-op; the evidence-before-
training gate caught both before any multi-hour GPU was spent.

---

## 2. Implementation (opt-in, backward-compatible)

- `_dispersive_loss_infonce(features, tau)` — InfoNCE-L2 with `/d`
  ([models/flow_policy_v5.py:204-224](../models/flow_policy_v5.py#L204-L224)).
- Applied to the `flow_net` mid-block via `return_mid=True`
  ([models/flow_policy_v5.py:269-287](../models/flow_policy_v5.py#L269-L287)); legacy
  `vis_pooled` path retained for reproducing the old P2.
- Sweep driver `--faithful` switch sets λ=0.5, `--dispersive-target flow_mid
  --dispersive-tau 0.5`, `p2f_` tag prefix and a separate manifest
  ([scripts/run_p2_ablation.py:59-82,116-133](../scripts/run_p2_ablation.py#L59-L82)).

**Structural consequence:** with the dispersive term on `flow_net` mid-features, the
gradient flows into `flow_net` **regardless of `--freeze-vision`**. So under the
faithful placement the legacy "D1E0 ≡ D0E0 byte-identical" no longer holds.

---

## 3. Protocol

Identical to P2 except the three faithful changes. 12 runs = full 2×2 × seeds {0,1,2},
sequential (one at a time, no GPU contention), shared H4-transfer init, 500 hover + 500
recovery, 80 epochs, lr 1e-4, batch 256, task-conditioned (`configs/flow_policy_v5.yaml`).
Each run ≈ 2.8 h; 12 runs ≈ 27 h wall. Eval = frozen P0 (`evaluate_frozen_p0`, 30 ep,
base_seed 12345, σ=2.0, paired identical init, conditional-IAE), oracle measured 0.9668.

---

## 4. Results

### 4.1 Tier1% (primary axis, mean±std over 3 seeds)
|                       | Dispersive OFF | Dispersive ON |
|-----------------------|----------------|---------------|
| **E2E OFF (frozen)**  | 87.8 ± 3.1 (D0E0) | 74.4 ± 8.7 (D1E0) |
| **E2E ON**            | 92.2 ± 3.1 (D0E1) | 90.0 ± 5.4 (D1E1) |

### 4.2 Survival (mean±std)
|                       | Dispersive OFF | Dispersive ON |
|-----------------------|----------------|---------------|
| **E2E OFF (frozen)**  | 66.1 ± 3.7 | 60.4 ± 1.6 |
| **E2E ON**            | 65.0 ± 2.8 | 62.9 ± 2.4 |

Precision: cond-IAE 2.7–3.2 m every cell (~12–13% of oracle 0.068 m). Nothing deployable.

### 4.3 Decisive comparison
**D1E1 vs D0E1 = −2.2pp Tier1 (pooled std 6.3pp), −2.1pp survival → INCONCLUSIVE /
slightly negative.** Dispersive gives no survival/Tier1 lift even where it *can* act
(trainable encoder) — same verdict as legacy P2 (which found +1.1pp / 4.2pp), now under
the faithful, official-code implementation.

---

## 5. Two conclusions for the paper

1. **Negative result CONFIRMED and rebuttal-proofed.** The faithful re-run closes the
   "you didn't use Dispersive correctly" reviewer rebuttal: with InfoNCE-L2 on the
   generative mid-block, λ=0.5, τ=0.5, `/d` — exactly the official recipe — Dispersive
   still does not move survival/Tier1 in visual drone hover.

2. **Legacy C2 ("byte-identical no-op") OVERTURNED — §5/§9 must be rewritten.** Under the
   faithful placement `flow_net` trains even with a frozen encoder, so
   `p2f_D1E0_s* ≠ p2f_D0E0_s*` (MD5 differs across all 3 seeds). The frozen row is no
   longer inert: Dispersive-on-frozen is mildly **harmful** (Tier1 87.8→74.4, variance
   3.1→8.7) — the faithful term, with no trainable encoder to co-adapt, pushes
   mid-features apart in a way that destabilises control. This is a *richer* result than
   the legacy MD5 no-op and should replace the C2 claim.

E2E remains the only (small) positive Tier1 mover; the precision axis is unmoved, so the
overall thesis — *representation collapse is not the binding constraint; the binding
constraint is the absence of learned far-range recovery in the 1–3 m band* — stands.

---

## 6. Provenance
- Sweep: `scripts/run_p2_ablation.py --faithful --cells D0E0 D0E1 D1E0 D1E1 --seeds 0 1 2`
- Eval: `scripts/evaluate_p2_ablation.py --manifest evaluation_results/p2f_ablation_manifest.json`
- Checkpoints: `checkpoints/flow_policy_v5/p2f_{D0E0,D0E1,D1E0,D1E1}_s{0,1,2}/best_model.pt`
- See also [experiment_report_feature_collapse.md](experiment_report_feature_collapse.md),
  [experiment_report_survival_movers.md](experiment_report_survival_movers.md).

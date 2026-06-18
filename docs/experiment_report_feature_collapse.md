# Experiment Report — Feature-Collapse Diagnosis of `vis_pooled` (Phase 3a)

**Date:** 2026-06-18
**Script:** `scripts/measure_feature_collapse.py`
**Artifact:** `evaluation_results/p2_feature_collapse.json`
**Inputs:** the 12 P2 ablation checkpoints (Dispersive×E2E, 3 seeds), a fixed
4000-image batch (50 % hover + 50 % recovery, `seed=12345`), `vis_pooled` (D=256).

This is `RESEARCH_PLAN_v6.md` Phase 3 diagnostic (a): *measure the feature
collapse the Dispersive Loss is supposed to fix, to show what the mechanism
actually does.* The result is sharper than the pre-registered prediction ("the
mechanism is inert"): the mechanism is **not** inert — it **games its own
objective and makes intrinsic collapse worse** — yet survival is unchanged
because the policy does not act on the feature it regularises.

---

## 1. Results (mean ± std over 3 seeds)

| Cell | Disp / E2E | **eff_rank** | n_eff_99 | top-2 var | **feat_norm** | pair_dist | disp_loss | mean_cos |
|------|-----------|-------------:|---------:|----------:|--------------:|----------:|----------:|---------:|
| D0E0 | OFF / frozen | **30.29 ± 0.00** | 91 | 0.27 | 1.80 | 2.29 | −0.807 | 0.071 |
| D1E0 | ON  / frozen | **30.29 ± 0.00** | 91 | 0.27 | 1.80 | 2.29 | −0.807 | 0.071 |
| D0E1 | OFF / E2E    | **8.97 ± 0.59**  | 45 | 0.58 | 11.42 | 3.53 | −1.228 | **0.956** |
| D1E1 | ON  / E2E    | **2.02 ± 0.00**  | **2** | **0.998** | **3281 ± 262** | **4486 ± 333** | **−8.135** | 0.013 |

`eff_rank` = exp(entropy of centered-covariance eigenvalues), D=256 (max).
`n_eff_99` = # dims for 99 % cumulative variance. `top-2 var` = variance share of
the 2 leading eigenvectors. `feat_norm` = mean ‖vis_pooled‖. `disp_loss` =
`−mean_{i≠j} log(‖xᵢ−xⱼ‖+ε)`, the quantity Dispersive minimises.

---

## 2. Three findings

### (1) D1E0 ≡ D0E0 — byte-identical no-op, confirmed at the feature level
All collapse metrics are identical to 4 decimal places (Δeff_rank = 0.0000,
Δdisp_loss = 0.000000). With `--freeze-vision` the dispersive gradient on
`vis_pooled` touches no trainable parameter, so "Dispersive ON, frozen" produces
the *same features* as "Dispersive OFF, frozen". This corroborates the MD5
byte-identity of the checkpoints (`RESEARCH_PLAN_v6.md` Phase 2 finding #2) with an
independent forward-pass measurement.

### (2) Collapse is real — and naive E2E causes it
Unfreezing the encoder *without* dispersive (D0E1) collapses the H4-transferred
features hard: effective rank **30.3 → 9.0**, dims-for-99 %-variance **91 → 45**,
and mean pairwise **cosine 0.071 → 0.956** — i.e. after E2E the pooled features
become almost colinear (a single dominant direction). So the project's premise
that end-to-end fine-tuning can collapse the representation is **empirically
true**; there *was* something for Dispersive to fix.

### (3) Dispersive does NOT fix collapse — it games its objective
D1E1 drives the literal dispersive loss far down (−1.23 → −8.14) and inflates mean
pairwise distance ~1270× (3.5 → 4486). But it achieves this **purely by inflating
the feature norm ~287×** (1.80 → 3281), not by spreading information across
dimensions. The intrinsic dimensionality gets **worse, not better**:

- effective rank **9.0 → 2.0** (the *lowest* of all four cells),
- **99.83 % of variance** now lives on just **2 dimensions** (n_eff_99: 45 → 2).

The "healthy-looking" low cosine (0.956 → 0.013) is an artifact: random vectors on
a 2-D plane at huge radius are near-orthogonal on average. Dispersive optimised a
distance surrogate that is trivially satisfiable by scaling, so it produced a
representation that is *numerically* maximally dispersed but *intrinsically* more
collapsed. **Mechanistically it most cheaply inflates the pooled-specific `fc`
layer**, which feeds only the auxiliary head (see §3), leaving the
action-relevant spatial pathway comparatively untouched.

---

## 3. Why this leaves survival unchanged (the architecture closes the loop)

In `FlowMatchingPolicyV5._encode` the action conditioning is
`cat([attended, imu_feat])`, where `attended = cross_attn(imu_feat, vis_spatial)`.
**`vis_pooled` is not in the action-conditioning path at all** — it feeds only the
training-only `StatePredictor` auxiliary head (`_encode_full` → `compute_loss`).
Dispersive therefore regularises a branch the policy does not act on. Its gradient
reaches the shared conv trunk only indirectly and most cheaply discharges into the
pooled-only `fc` layer (norm inflation), so the spatial → cross-attention → action
path is largely spared.

The decisive cross-table — **`vis_pooled` collapse vs closed-loop survival**
(survival/Tier1 from `evaluation_results/p2_ablation_leaderboard.json`):

| Cell | eff_rank (vis_pooled) | Survival | Tier1% |
|------|----------------------:|---------:|-------:|
| D0E0 / D1E0 | 30.3 | 66.1 % | 87.8 % |
| D0E1 | 9.0  | 65.0 % | 92.2 % |
| D1E1 | 2.0  | 65.0 % | 93.3 % |

Effective rank of `vis_pooled` swings **15×** (30.3 → 2.0) while survival is **flat
within seed noise** (66.1 → 65.0, std ≈ 3 pp). **Survival is decoupled from
`vis_pooled` rank.** This is the strongest statement of the v6 thesis:
representation collapse (what Dispersive targets) is *not* the binding constraint
for visual hover — neither preventing it nor worsening it moves the closed-loop
outcome.

---

## 4. Conclusion

Dispersive Loss on `vis_pooled` is a textbook **objective-gaming** regulariser in
this setting: it minimises a scale-sensitive distance surrogate by norm inflation,
*worsening* intrinsic rank, and it acts on a feature the policy does not condition
its actions on. Combined with the P2 survival result (no Tier1/survival lift) and
the MD5 / feature-identity no-op under a frozen encoder, the core thesis
("Dispersive Loss prevents feature collapse and thereby improves high-speed visual
control") is falsified along its full mechanistic chain:

1. frozen encoder → exact no-op (D1E0 ≡ D0E0, MD5 + features);
2. trainable encoder → Dispersive acts but **worsens** intrinsic collapse (rank 9→2)
   via norm inflation, not spreads information;
3. and in any case survival is **invariant** to `vis_pooled` rank (15× swing, flat
   survival), because the feature sits off the action-conditioning path.

**Reproduce:**
```bash
dppo/Scripts/python.exe -m scripts.measure_feature_collapse --n-samples 4000
```

**Caveats / scope.** Metrics are on `vis_pooled` over a held-out hover+recovery
image batch (no augmentation), matching where Dispersive was applied. A natural
follow-up is to repeat on `vis_spatial` / `attended` (the actual action-path
features) to confirm the spatial pathway is the one that is comparatively
preserved; the survival decoupling already bounds how much that could matter.

"""
Aggregate the P2 ablation by cell, on the P0-frozen PRIMARY axis (Tier1% / survival).

Reads the sweep manifest (scripts.run_p2_ablation), runs each finished checkpoint
through ``evaluate_frozen`` (the exact P0 frozen protocol — same seeds, env, metric),
then groups by 2x2 cell and reports mean +- std across seeds for:

  * Tier1 pass-rate  (PRIMARY — fraction of episodes flying >= 250/500 steps)
  * survival         (PRIMARY)
  * composite score  (secondary, flagged — artifact-prone; do NOT rank on this alone)

The decisive comparison is D1E1 vs D0E1 (Dispersive ON vs OFF, both E2E). D1E0 is
expected to ~= D0E0 because dispersive acts on the frozen vis_pooled (no-op).

Usage:
  dppo/Scripts/python.exe -m scripts.evaluate_p2_ablation \
      --manifest evaluation_results/p2_ablation_manifest.json \
      --oracle-ckpt checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
      --oracle-norm checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz
"""
import os
import sys
import json
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.evaluate_frozen_p0 import evaluate_frozen, evaluate_oracle_frozen


def _mean_std(xs):
    a = np.asarray(xs, dtype=float)
    return (float(a.mean()), float(a.std())) if len(a) else (float('nan'), float('nan'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='evaluation_results/p2_ablation_manifest.json')
    parser.add_argument('--n-episodes', type=int, default=30)
    parser.add_argument('--base-seed', type=int, default=12345)
    parser.add_argument('--survive-threshold', type=int, default=250)
    parser.add_argument('--quadrotor-config', default='configs/quadrotor_v4.yaml')
    parser.add_argument('--flow-config', default='configs/flow_policy_v4.yaml')
    parser.add_argument('--n-inference-steps', type=int, default=2)
    parser.add_argument('--sigma', type=float, default=2.0)
    parser.add_argument('--oracle-ckpt', default=None)
    parser.add_argument('--oracle-norm', default=None)
    parser.add_argument('--output', default='evaluation_results/p2_ablation_leaderboard.json')
    args = parser.parse_args()

    with open(args.manifest, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # ---- evaluate the measured oracle (for %Oracle) ----
    oracle_score = None
    if args.oracle_ckpt and args.oracle_norm:
        print(f"\n=== PPO_Oracle ({args.oracle_ckpt}) ===")
        oagg = evaluate_oracle_frozen(
            args.oracle_ckpt, args.oracle_norm, args.n_episodes, args.base_seed,
            args.survive_threshold, args.quadrotor_config, args.sigma)
        oracle_score = oagg['score_mean']
        print(f"  measured oracle composite = {oracle_score:.4f}")

    # ---- evaluate every finished checkpoint ----
    per_run = {}
    for tag, rec in sorted(manifest.get('runs', {}).items()):
        ckpt = rec.get('ckpt')
        if rec.get('status') not in ('done', 'skipped_existing'):
            print(f"  SKIP {tag}: status={rec.get('status')}")
            continue
        if not ckpt or not os.path.exists(os.path.join(ROOT, ckpt)):
            print(f"  SKIP {tag}: checkpoint missing ({ckpt})")
            continue
        print(f"\n=== {tag} ({ckpt}) ===")
        agg = evaluate_frozen(ckpt, args.n_episodes, args.base_seed,
                              args.survive_threshold, args.quadrotor_config,
                              args.flow_config, args.n_inference_steps, args.sigma)
        per_run[tag] = {
            'cell': rec['cell'], 'seed': rec['seed'], 'ckpt': ckpt,
            'tier1': agg['tier1_pass_rate'], 'survival': agg['survival_mean'],
            'score': agg['score_mean'], 'n_cond': agg['n_conditional'],
            'iae_cond': agg['iae_steady_cond'],
        }

    # ---- group by cell ----
    cells = {}
    for r in per_run.values():
        cells.setdefault(r['cell'], []).append(r)

    cell_agg = {}
    for cell, runs in cells.items():
        t_m, t_s = _mean_std([r['tier1'] for r in runs])
        s_m, s_s = _mean_std([r['survival'] for r in runs])
        c_m, c_s = _mean_std([r['score'] for r in runs])
        cell_agg[cell] = {
            'n_seeds': len(runs),
            'tier1_mean': t_m, 'tier1_std': t_s,
            'survival_mean': s_m, 'survival_std': s_s,
            'score_mean': c_m, 'score_std': c_s,
            'seeds': sorted(r['seed'] for r in runs),
        }

    # ---- per-run table ----
    print("\n" + "=" * 96)
    print(f"{'Tag':<14}{'Cell':>6}{'Seed':>5}{'Tier1%':>9}{'Surv%':>8}{'Score':>8}"
          f"{'n_cond':>8}{'condIAE':>9}")
    print("-" * 96)
    for tag in sorted(per_run):
        r = per_run[tag]
        print(f"{tag:<14}{r['cell']:>6}{r['seed']:>5}{r['tier1']*100:>8.1f}%"
              f"{r['survival']*100:>7.1f}%{r['score']:>8.3f}{r['n_cond']:>6}/{args.n_episodes:<2}"
              f"{r['iae_cond']:>8.3f}m")
    print("=" * 96)

    # ---- 2x2 grid (PRIMARY = Tier1% mean±std over seeds) ----
    def fmt(cell, key_m, key_s, scale=100, suffix='%'):
        if cell not in cell_agg:
            return f"{'--':>16}"
        a = cell_agg[cell]
        return f"{a[key_m]*scale:>6.1f}±{a[key_s]*scale:<4.1f}{suffix} (n={a['n_seeds']})"

    print("\n2x2 ablation — PRIMARY axis: Tier1 pass-rate (mean±std over seeds)")
    print(f"{'':<14}{'Dispersive OFF':>22}{'Dispersive ON':>22}")
    print(f"{'E2E OFF(frozen)':<14}{fmt('D0E0','tier1_mean','tier1_std'):>22}{fmt('D1E0','tier1_mean','tier1_std'):>22}")
    print(f"{'E2E ON':<14}{fmt('D0E1','tier1_mean','tier1_std'):>22}{fmt('D1E1','tier1_mean','tier1_std'):>22}")

    print("\n2x2 ablation — survival (mean±std over seeds)")
    print(f"{'':<14}{'Dispersive OFF':>22}{'Dispersive ON':>22}")
    print(f"{'E2E OFF(frozen)':<14}{fmt('D0E0','survival_mean','survival_std'):>22}{fmt('D1E0','survival_mean','survival_std'):>22}")
    print(f"{'E2E ON':<14}{fmt('D0E1','survival_mean','survival_std'):>22}{fmt('D1E1','survival_mean','survival_std'):>22}")

    # ---- decision line ----
    if 'D1E1' in cell_agg and 'D0E1' in cell_agg:
        d = cell_agg['D1E1']['tier1_mean'] - cell_agg['D0E1']['tier1_mean']
        pooled = np.hypot(cell_agg['D1E1']['tier1_std'], cell_agg['D0E1']['tier1_std'])
        verdict = ('SUPPORTED' if d > pooled else
                   'NOT supported' if d < -pooled else 'INCONCLUSIVE (< pooled std)')
        print(f"\nDispersive (D1E1 vs D0E1) Tier1% delta = {d*100:+.1f}pp "
              f"(pooled std {pooled*100:.1f}pp) -> Dispersive {verdict}")

    out = {'cell_agg': cell_agg, 'per_run': per_run,
           'oracle_score': oracle_score, 'manifest': args.manifest}
    if oracle_score:
        for cell, a in cell_agg.items():
            a['pct_oracle'] = a['score_mean'] / oracle_score * 100.0
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()

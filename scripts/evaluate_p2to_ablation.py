"""
Aggregate the P2TO decisive ablation by cell, PRIMARY axis = conditional-IAE (precision).

RESEARCH_PLAN_v7 Phase 4. Reads the sweep manifest (scripts.run_p2to_ablation),
runs each finished checkpoint through ``evaluate_frozen`` (the exact P0 frozen
protocol), passing the per-cell ``--target-render`` recorded in the manifest
(O0 -> crosshair, O1 -> perspective) so each policy is evaluated under the SAME
observation it was trained on. Then groups by 2x2 cell and reports mean +- std
across seeds for:

  * cond-IAE   (PRIMARY — steady-state IAE over episodes surviving >= threshold)
  * survival   (guard against "precision win that is actually a survival collapse")
  * Tier1%     (secondary)

H_v7 verdict — the cond-IAE ~2.8 m floor is "BROKEN" iff ALL THREE hold:
  1. cond-IAE(T1O1) < cond-IAE(T0O0) - pooled_std  (significant vs the neither-factor control)
  2. cond-IAE(T1O1) <= --abs-target  (default 1.5 m, ~2x the floor)
  3. survival(T1O1) >= survival(T0O0) - pooled_std  (survival guard; pos3d-cue lesson)
and cond-IAE is only trusted when n_cond >= --min-ncond (default 15) in BOTH cells.

The decisive comparison is T1O1 (both factors) vs T0O0 (neither). The full 2x2 grid
is printed so the T-alone and O-alone main effects are visible too.

Usage:
  dppo/Scripts/python.exe -m scripts.evaluate_p2to_ablation \
      --manifest evaluation_results/p2to_ablation_manifest.json \
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

CELL_ORDER = ['T0O0', 'T0O1', 'T1O0', 'T1O1']


def _mean_std(xs):
    a = np.asarray(xs, dtype=float)
    a = a[~np.isnan(a)]
    return (float(a.mean()), float(a.std())) if len(a) else (float('nan'), float('nan'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='evaluation_results/p2to_ablation_manifest.json')
    parser.add_argument('--n-episodes', type=int, default=30)
    parser.add_argument('--base-seed', type=int, default=12345)
    parser.add_argument('--survive-threshold', type=int, default=250)
    parser.add_argument('--quadrotor-config', default='configs/quadrotor_v4.yaml')
    parser.add_argument('--flow-config', default='configs/flow_policy_v4.yaml')
    parser.add_argument('--n-inference-steps', type=int, default=2)
    parser.add_argument('--sigma', type=float, default=2.0)
    parser.add_argument('--oracle-ckpt', default=None)
    parser.add_argument('--oracle-norm', default=None)
    parser.add_argument('--abs-target', type=float, default=1.5,
                        help='H_v7 absolute cond-IAE target for "floor broken" (default 1.5 m)')
    parser.add_argument('--min-ncond', type=int, default=15,
                        help='Min per-cell mean n_cond for cond-IAE to be trusted (default 15)')
    parser.add_argument('--output', default='evaluation_results/p2to_ablation_leaderboard.json')
    args = parser.parse_args()

    with open(args.manifest, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # ---- evaluate the measured oracle (for %Oracle on the score axis) ----
    oracle_score = None
    if args.oracle_ckpt and args.oracle_norm:
        print(f"\n=== PPO_Oracle ({args.oracle_ckpt}) ===")
        oagg = evaluate_oracle_frozen(
            args.oracle_ckpt, args.oracle_norm, args.n_episodes, args.base_seed,
            args.survive_threshold, args.quadrotor_config, args.sigma)
        oracle_score = oagg['score_mean']
        oracle_iae = oagg['iae_steady_cond']
        print(f"  measured oracle composite = {oracle_score:.4f}  cond-IAE = {oracle_iae:.4f}m")

    # ---- evaluate every finished checkpoint, per-cell target_render ----
    per_run = {}
    for tag, rec in sorted(manifest.get('runs', {}).items()):
        ckpt = rec.get('ckpt')
        render = rec.get('render', 'crosshair')
        if rec.get('status') not in ('done', 'skipped_existing'):
            print(f"  SKIP {tag}: status={rec.get('status')}")
            continue
        if not ckpt or not os.path.exists(os.path.join(ROOT, ckpt)):
            print(f"  SKIP {tag}: checkpoint missing ({ckpt})")
            continue
        print(f"\n=== {tag}  cell={rec['cell']} render={render}  ({ckpt}) ===")
        agg = evaluate_frozen(ckpt, args.n_episodes, args.base_seed,
                              args.survive_threshold, args.quadrotor_config,
                              args.flow_config, args.n_inference_steps, args.sigma,
                              target_render=render)
        per_run[tag] = {
            'cell': rec['cell'], 'seed': rec['seed'], 'ckpt': ckpt, 'render': render,
            'tier1': agg['tier1_pass_rate'], 'survival': agg['survival_mean'],
            'score': agg['score_mean'], 'n_cond': agg['n_conditional'],
            'iae_cond': agg['iae_steady_cond'], 'iae_all': agg['iae_steady_all'],
        }

    # ---- group by cell ----
    cells = {}
    for r in per_run.values():
        cells.setdefault(r['cell'], []).append(r)

    cell_agg = {}
    for cell, runs in cells.items():
        i_m, i_s = _mean_std([r['iae_cond'] for r in runs])     # PRIMARY
        s_m, s_s = _mean_std([r['survival'] for r in runs])
        t_m, t_s = _mean_std([r['tier1'] for r in runs])
        c_m, c_s = _mean_std([r['score'] for r in runs])
        ncond = [r['n_cond'] for r in runs]
        cell_agg[cell] = {
            'n_seeds': len(runs),
            'cond_iae_mean': i_m, 'cond_iae_std': i_s,         # PRIMARY
            'survival_mean': s_m, 'survival_std': s_s,
            'tier1_mean': t_m, 'tier1_std': t_s,
            'score_mean': c_m, 'score_std': c_s,
            'n_cond_mean': float(np.mean(ncond)), 'n_cond_min': int(np.min(ncond)),
            'seeds': sorted(r['seed'] for r in runs),
            'render': runs[0]['render'],
        }
        if oracle_score:
            cell_agg[cell]['pct_oracle'] = c_m / oracle_score * 100.0

    # ---- per-run table ----
    print("\n" + "=" * 104)
    print(f"{'Tag':<16}{'Cell':>6}{'Seed':>5}{'Render':>12}{'condIAE':>9}{'n_cond':>8}"
          f"{'Surv%':>8}{'Tier1%':>8}{'Score':>8}")
    print("-" * 104)
    for tag in sorted(per_run):
        r = per_run[tag]
        print(f"{tag:<16}{r['cell']:>6}{r['seed']:>5}{r['render']:>12}{r['iae_cond']:>8.3f}m"
              f"{r['n_cond']:>5}/{args.n_episodes:<2}{r['survival']*100:>7.1f}%"
              f"{r['tier1']*100:>7.1f}%{r['score']:>8.3f}")
    print("=" * 104)

    # ---- 2x2 grids ----
    def fmt(cell, key_m, key_s, scale=1.0, suffix='m', dec=2):
        if cell not in cell_agg:
            return f"{'--':>20}"
        a = cell_agg[cell]
        return f"{a[key_m]*scale:>6.{dec}f}+/-{a[key_s]*scale:<4.{dec}f}{suffix}(n={a['n_seeds']})"

    print("\n2x2 -- PRIMARY axis: conditional-IAE (mean+/-std over seeds, LOWER is better)")
    print(f"{'':<16}{'O0 crosshair':>22}{'O1 perspective':>22}")
    print(f"{'T0 hover-only':<16}{fmt('T0O0','cond_iae_mean','cond_iae_std'):>22}{fmt('T0O1','cond_iae_mean','cond_iae_std'):>22}")
    print(f"{'T1 +far recov':<16}{fmt('T1O0','cond_iae_mean','cond_iae_std'):>22}{fmt('T1O1','cond_iae_mean','cond_iae_std'):>22}")

    print("\n2x2 -- survival (mean+/-std over seeds, HIGHER is better)")
    print(f"{'':<16}{'O0 crosshair':>22}{'O1 perspective':>22}")
    print(f"{'T0 hover-only':<16}{fmt('T0O0','survival_mean','survival_std',100,'%',1):>22}{fmt('T0O1','survival_mean','survival_std',100,'%',1):>22}")
    print(f"{'T1 +far recov':<16}{fmt('T1O0','survival_mean','survival_std',100,'%',1):>22}{fmt('T1O1','survival_mean','survival_std',100,'%',1):>22}")

    print("\n2x2 -- Tier1 pass-rate (mean+/-std over seeds)")
    print(f"{'':<16}{'O0 crosshair':>22}{'O1 perspective':>22}")
    print(f"{'T0 hover-only':<16}{fmt('T0O0','tier1_mean','tier1_std',100,'%',1):>22}{fmt('T0O1','tier1_mean','tier1_std',100,'%',1):>22}")
    print(f"{'T1 +far recov':<16}{fmt('T1O0','tier1_mean','tier1_std',100,'%',1):>22}{fmt('T1O1','tier1_mean','tier1_std',100,'%',1):>22}")

    print("\n2x2 -- mean n_cond (cond-IAE trustworthy only when >= "
          f"{args.min_ncond})")
    for cell in CELL_ORDER:
        if cell in cell_agg:
            a = cell_agg[cell]
            flag = '' if a['n_cond_mean'] >= args.min_ncond else '  <-- below min, cond-IAE UNRELIABLE'
            print(f"  {cell}: mean n_cond={a['n_cond_mean']:.1f}  min={a['n_cond_min']}{flag}")

    # ---- H_v7 decisive verdict: T1O1 vs T0O0 ----
    verdict = {'available': False}
    if 'T1O1' in cell_agg and 'T0O0' in cell_agg:
        a1, a0 = cell_agg['T1O1'], cell_agg['T0O0']
        pooled_iae = float(np.hypot(a1['cond_iae_std'], a0['cond_iae_std']))
        pooled_surv = float(np.hypot(a1['survival_std'], a0['survival_std']))
        d_iae = a1['cond_iae_mean'] - a0['cond_iae_mean']           # negative = improvement
        d_surv = a1['survival_mean'] - a0['survival_mean']

        cond_signif   = a1['cond_iae_mean'] < (a0['cond_iae_mean'] - pooled_iae)
        cond_absolute = a1['cond_iae_mean'] <= args.abs_target
        surv_guard    = a1['survival_mean'] >= (a0['survival_mean'] - pooled_surv)
        ncond_ok      = (a1['n_cond_mean'] >= args.min_ncond and
                         a0['n_cond_mean'] >= args.min_ncond)

        floor_broken = bool(cond_signif and cond_absolute and surv_guard and ncond_ok)
        verdict = {
            'available': True,
            'comparison': 'T1O1 (both factors) vs T0O0 (neither)',
            'cond_iae_T1O1': a1['cond_iae_mean'], 'cond_iae_T0O0': a0['cond_iae_mean'],
            'cond_iae_delta': d_iae, 'pooled_std_iae': pooled_iae,
            'survival_T1O1': a1['survival_mean'], 'survival_T0O0': a0['survival_mean'],
            'survival_delta': d_surv, 'pooled_std_survival': pooled_surv,
            'abs_target': args.abs_target,
            'cond_significant': bool(cond_signif),
            'cond_absolute_met': bool(cond_absolute),
            'survival_guard_ok': bool(surv_guard),
            'ncond_sufficient': bool(ncond_ok),
            'FLOOR_BROKEN': floor_broken,
        }

        print("\n" + "#" * 84)
        print("# H_v7 DECISIVE VERDICT  (cond-IAE ~2.8 m floor)  T1O1 vs T0O0")
        print("#" * 84)
        print(f"  cond-IAE  T1O1={a1['cond_iae_mean']:.3f}m  T0O0={a0['cond_iae_mean']:.3f}m  "
              f"delta={d_iae:+.3f}m  (pooled std {pooled_iae:.3f}m)")
        print(f"  survival  T1O1={a1['survival_mean']*100:.1f}%  T0O0={a0['survival_mean']*100:.1f}%  "
              f"delta={d_surv*100:+.1f}pp (pooled std {pooled_surv*100:.1f}pp)")
        print(f"  [1] cond significant (T1O1 < T0O0 - pooled std) : {cond_signif}")
        print(f"  [2] cond absolute (T1O1 <= {args.abs_target} m)          : {cond_absolute}")
        print(f"  [3] survival guard (T1O1 >= T0O0 - pooled std)  : {surv_guard}")
        print(f"  [4] n_cond sufficient (>= {args.min_ncond} both cells)     : {ncond_ok}")
        print(f"\n  ==> FLOOR {'BROKEN' if floor_broken else 'NOT broken'} "
              f"({'all 3 criteria met' if floor_broken else 'criteria unmet - negative result deepened'})")
        print("#" * 84)
    else:
        print("\n[verdict] need both T1O1 and T0O0 cells to issue the H_v7 verdict")

    out = {'cell_agg': cell_agg, 'per_run': per_run, 'verdict': verdict,
           'oracle_score': oracle_score, 'manifest': args.manifest,
           'protocol': {'n_episodes': args.n_episodes, 'base_seed': args.base_seed,
                        'survive_threshold': args.survive_threshold, 'sigma': args.sigma,
                        'n_inference_steps': args.n_inference_steps,
                        'abs_target': args.abs_target, 'min_ncond': args.min_ncond}}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()

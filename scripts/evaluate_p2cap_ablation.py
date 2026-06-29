"""
Aggregate the P2CAP capacity ladder by rung, PRIMARY axis = conditional-IAE (precision).

RESEARCH_PLAN_v8 Phase 1. Reads the sweep manifest (scripts.run_p2cap_ablation), runs
each finished checkpoint through ``evaluate_frozen`` (the exact P0 frozen protocol) on
the PERSPECTIVE render (the T1O1 recipe's observation), then groups by capacity rung
(S/M/XL) and reports mean +- std across seeds for cond-IAE (PRIMARY), survival (guard),
Tier1% and composite score.

H_v8 verdict — capacity "breaks the floor" iff, for ANY scaled rung R in {M, XL}:
  1. cond-IAE(R) < cond-IAE(S) - pooled_std   (significant vs the 13.5M baseline)
  2. cond-IAE(R) <= --abs-target              (default 1.5 m, ~2x the floor)
  3. survival(R) >= survival(S) - pooled_std  (survival guard)
and cond-IAE trusted only when n_cond >= --min-ncond (default 15) in both rungs.
Also reports whether cond-IAE trends monotonically DOWN with capacity (S->M->XL).

The expected (pre-registered prior, given Gate A: the 13.5M model already fits hover+far
open-loop to within ~6%): the floor does NOT break -> "capacity tested by scaling
flow_net 13.5M -> 44M, floor held -> the conflict is more fundamental than parameter
count" — the strongest version of the negative result.

Usage:
  dppo/Scripts/python.exe -m scripts.evaluate_p2cap_ablation \
      --manifest evaluation_results/p2cap_ablation_manifest.json \
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

RUNG_ORDER = ['S', 'M', 'XL']
RUNG_LABEL = {'S': 'S (256,512) 13.5M', 'M': 'M (256,512,768) 35M', 'XL': 'XL (512,1024) 44M'}
DONE = ('done', 'skipped_existing', 'reused')


def _mean_std(xs):
    a = np.asarray(xs, dtype=float)
    a = a[~np.isnan(a)]
    return (float(a.mean()), float(a.std())) if len(a) else (float('nan'), float('nan'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='evaluation_results/p2cap_ablation_manifest.json')
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
                        help='H_v8 absolute cond-IAE target for "floor broken" (default 1.5 m)')
    parser.add_argument('--min-ncond', type=int, default=15)
    parser.add_argument('--output', default='evaluation_results/p2cap_ablation_leaderboard.json')
    args = parser.parse_args()

    with open(args.manifest, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # ---- measured oracle (for %Oracle on the score axis) ----
    oracle_score = None
    if args.oracle_ckpt and args.oracle_norm:
        print(f"\n=== PPO_Oracle ({args.oracle_ckpt}) ===")
        oagg = evaluate_oracle_frozen(
            args.oracle_ckpt, args.oracle_norm, args.n_episodes, args.base_seed,
            args.survive_threshold, args.quadrotor_config, args.sigma)
        oracle_score = oagg['score_mean']
        print(f"  measured oracle composite = {oracle_score:.4f}  cond-IAE = {oagg['iae_steady_cond']:.4f}m")

    # ---- evaluate every finished checkpoint on the perspective render ----
    per_run = {}
    for tag, rec in sorted(manifest.get('runs', {}).items()):
        ckpt = rec.get('ckpt')
        render = rec.get('render', 'perspective')
        if rec.get('status') not in DONE:
            print(f"  SKIP {tag}: status={rec.get('status')}")
            continue
        if not ckpt or not os.path.exists(os.path.join(ROOT, ckpt)):
            print(f"  SKIP {tag}: checkpoint missing ({ckpt})")
            continue
        print(f"\n=== {tag}  rung={rec['rung']} down_dims={rec.get('down_dims')} render={render}  ({ckpt}) ===")
        agg = evaluate_frozen(ckpt, args.n_episodes, args.base_seed,
                              args.survive_threshold, args.quadrotor_config,
                              args.flow_config, args.n_inference_steps, args.sigma,
                              target_render=render)
        per_run[tag] = {
            'rung': rec['rung'], 'seed': rec['seed'], 'ckpt': ckpt, 'render': render,
            'down_dims': rec.get('down_dims'),
            'tier1': agg['tier1_pass_rate'], 'survival': agg['survival_mean'],
            'score': agg['score_mean'], 'n_cond': agg['n_conditional'],
            'iae_cond': agg['iae_steady_cond'], 'iae_all': agg['iae_steady_all'],
        }

    # ---- group by rung ----
    rungs = {}
    for r in per_run.values():
        rungs.setdefault(r['rung'], []).append(r)

    rung_agg = {}
    for rung, runs in rungs.items():
        i_m, i_s = _mean_std([r['iae_cond'] for r in runs])     # PRIMARY
        s_m, s_s = _mean_std([r['survival'] for r in runs])
        t_m, t_s = _mean_std([r['tier1'] for r in runs])
        c_m, c_s = _mean_std([r['score'] for r in runs])
        ncond = [r['n_cond'] for r in runs]
        rung_agg[rung] = {
            'n_seeds': len(runs),
            'down_dims': runs[0]['down_dims'],
            'cond_iae_mean': i_m, 'cond_iae_std': i_s,          # PRIMARY
            'survival_mean': s_m, 'survival_std': s_s,
            'tier1_mean': t_m, 'tier1_std': t_s,
            'score_mean': c_m, 'score_std': c_s,
            'n_cond_mean': float(np.mean(ncond)), 'n_cond_min': int(np.min(ncond)),
            'seeds': sorted(r['seed'] for r in runs),
        }
        if oracle_score:
            rung_agg[rung]['pct_oracle'] = c_m / oracle_score * 100.0

    # ---- per-run table ----
    print("\n" + "=" * 104)
    print(f"{'Tag':<16}{'Rung':>5}{'Seed':>5}{'condIAE':>9}{'n_cond':>8}"
          f"{'Surv%':>8}{'Tier1%':>8}{'Score':>8}")
    print("-" * 104)
    for tag in sorted(per_run):
        r = per_run[tag]
        print(f"{tag:<16}{r['rung']:>5}{r['seed']:>5}{r['iae_cond']:>8.3f}m"
              f"{r['n_cond']:>5}/{args.n_episodes:<2}{r['survival']*100:>7.1f}%"
              f"{r['tier1']*100:>7.1f}%{r['score']:>8.3f}")
    print("=" * 104)

    # ---- ladder table (capacity ascending) ----
    print("\nCAPACITY LADDER -- PRIMARY axis: conditional-IAE (mean+/-std, LOWER is better)")
    print(f"{'Rung':<22}{'condIAE':>16}{'survival':>14}{'Tier1':>12}{'n_cond':>9}")
    for rung in RUNG_ORDER:
        if rung not in rung_agg:
            continue
        a = rung_agg[rung]
        flag = '' if a['n_cond_mean'] >= args.min_ncond else '  <-- n_cond low'
        print(f"{RUNG_LABEL[rung]:<22}"
              f"{a['cond_iae_mean']:>7.3f}+/-{a['cond_iae_std']:<5.3f}m"
              f"{a['survival_mean']*100:>8.1f}+/-{a['survival_std']*100:<3.1f}%"
              f"{a['tier1_mean']*100:>7.1f}%"
              f"{a['n_cond_mean']:>8.1f}{flag}")

    # ---- H_v8 verdict: each scaled rung vs S baseline ----
    verdict = {'available': False}
    if 'S' in rung_agg:
        base = rung_agg['S']
        scaled_results = {}
        any_broken = False
        for rung in ('M', 'XL'):
            if rung not in rung_agg:
                continue
            a = rung_agg[rung]
            pooled_iae = float(np.hypot(a['cond_iae_std'], base['cond_iae_std']))
            pooled_surv = float(np.hypot(a['survival_std'], base['survival_std']))
            d_iae = a['cond_iae_mean'] - base['cond_iae_mean']     # negative = improvement
            cond_signif   = a['cond_iae_mean'] < (base['cond_iae_mean'] - pooled_iae)
            cond_absolute = a['cond_iae_mean'] <= args.abs_target
            surv_guard    = a['survival_mean'] >= (base['survival_mean'] - pooled_surv)
            ncond_ok      = (a['n_cond_mean'] >= args.min_ncond and
                             base['n_cond_mean'] >= args.min_ncond)
            broken = bool(cond_signif and cond_absolute and surv_guard and ncond_ok)
            any_broken = any_broken or broken
            scaled_results[rung] = {
                'cond_iae': a['cond_iae_mean'], 'cond_iae_delta_vs_S': d_iae,
                'pooled_std_iae': pooled_iae, 'survival': a['survival_mean'],
                'survival_delta_vs_S': a['survival_mean'] - base['survival_mean'],
                'pooled_std_survival': pooled_surv,
                'cond_significant': bool(cond_signif), 'cond_absolute_met': bool(cond_absolute),
                'survival_guard_ok': bool(surv_guard), 'ncond_sufficient': bool(ncond_ok),
                'floor_broken': broken,
            }
        # monotone-down check on cond-IAE across available rungs
        seq = [rung_agg[r]['cond_iae_mean'] for r in RUNG_ORDER if r in rung_agg]
        monotone_down = all(seq[i + 1] <= seq[i] for i in range(len(seq) - 1)) if len(seq) > 1 else None
        verdict = {
            'available': True,
            'baseline_S_cond_iae': base['cond_iae_mean'],
            'baseline_S_survival': base['survival_mean'],
            'abs_target': args.abs_target,
            'scaled': scaled_results,
            'cond_iae_monotone_down_with_capacity': monotone_down,
            'FLOOR_BROKEN_BY_CAPACITY': bool(any_broken),
        }

        print("\n" + "#" * 84)
        print("# H_v8 VERDICT  (does scaling flow_net capacity break the cond-IAE ~2.8 m floor?)")
        print("#" * 84)
        print(f"  baseline S: cond-IAE={base['cond_iae_mean']:.3f}m  survival={base['survival_mean']*100:.1f}%")
        for rung in ('M', 'XL'):
            if rung not in scaled_results:
                continue
            s = scaled_results[rung]
            print(f"  {rung:>3} vs S: cond-IAE={s['cond_iae']:.3f}m "
                  f"(delta {s['cond_iae_delta_vs_S']:+.3f}m, pooled std {s['pooled_std_iae']:.3f}m)  "
                  f"surv delta {s['survival_delta_vs_S']*100:+.1f}pp  -> "
                  f"floor_broken={s['floor_broken']}")
        print(f"  cond-IAE monotone down with capacity: {monotone_down}")
        print(f"\n  ==> FLOOR {'BROKEN BY CAPACITY' if any_broken else 'NOT broken by capacity'} "
              f"({'positive result' if any_broken else 'capacity tested, floor held -> negative result deepened'})")
        print("#" * 84)
    else:
        print("\n[verdict] need the S baseline rung to issue the H_v8 verdict")

    out = {'rung_agg': rung_agg, 'per_run': per_run, 'verdict': verdict,
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

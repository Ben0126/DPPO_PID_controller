"""
Aggregate the P6 scale-invariant FORM ablation by form — RESEARCH_PLAN_v7 Phase 6.

Two axes per form (off / infonce / cosine / vicreg), mean +- std over 3 seeds:

  CONTROL (closed-loop, via the exact P0 frozen protocol, crosshair render):
    * survival     (PRIMARY guard)
    * Tier1%       (PRIMARY)
    * cond-IAE     (precision; reported only when mean n_cond >= --min-ncond)

  GEOMETRY (faithful flow_mid features, the objective-gaming probe):
    * effective_rank  (intrinsic dimensionality — must NOT be worse than off)
    * mean_feat_norm  (the norm-inflation channel — scale-invariant forms must NOT inflate)
    * disp_infonce    (the faithful objective, for reference)

The off (Dispersive OFF == P2f D0E1) and infonce (faithful == P2f D1E1) arms are
NOT re-trained: their checkpoints are pulled from
evaluation_results/p2f_ablation_manifest.json (reuse_from_p2f in the p6 manifest),
so the four forms are evaluated under one identical harness in this run.

Pre-registered verdict (Phase 6):
  [A] objective-gaming REMOVED  (main result): a scale-invariant form (cosine/vicreg)
      must NOT inflate feat_norm vs off (infonce's ~9x must vanish) AND must NOT make
      effective_rank worse than off. This is what Phase 6 sets out to prove.
  [B] control MOVED?  (secondary, open): does any scale-invariant form beat infonce on
      survival / Tier1 / cond-IAE by more than the pooled seed std? Prior: rank ⟂ survival
      (§6.1) -> expected FLAT. If flat, the clean conclusion is "criterion scale only
      changes geometry, not control".

Usage (launch in background with run_in_background=true):
  dppo/Scripts/python.exe -m scripts.evaluate_p6_form_ablation \
      --manifest evaluation_results/p6_form_ablation_manifest.json \
      --oracle-ckpt checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
      --oracle-norm checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz
"""
import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.evaluate_frozen_p0 import evaluate_frozen, evaluate_oracle_frozen
from scripts.measure_feature_collapse_flowmid import (
    build_input_batch, collapse_metrics, build_model, extract_flow_mid,
)

FORM_ORDER = ['off', 'infonce', 'cosine', 'vicreg']
# P2f cell -> reused form name
P2F_CELL_TO_FORM = {'D0E1': 'off', 'D1E1': 'infonce'}


def _mean_std(xs):
    a = np.asarray(xs, dtype=float)
    a = a[~np.isnan(a)]
    return (float(a.mean()), float(a.std())) if len(a) else (float('nan'), float('nan'))


def collect_entries(p6_manifest, p2f_manifest):
    """Return list of {form, seed, ckpt, source} across the 4 forms.

    off/infonce come from the P2f manifest (D0E1/D1E1); cosine/vicreg from p6.
    """
    entries = []
    # --- reused controls from P2f ---
    for tag, rec in p2f_manifest.get('runs', {}).items():
        form = P2F_CELL_TO_FORM.get(rec.get('cell'))
        if form is None:
            continue
        if rec.get('status') not in ('done', 'skipped_existing'):
            continue
        entries.append({'form': form, 'seed': rec['seed'], 'ckpt': rec['ckpt'],
                        'tag': tag, 'source': 'p2f'})
    # --- new scale-invariant forms from p6 ---
    for tag, rec in p6_manifest.get('runs', {}).items():
        if rec.get('status') not in ('done', 'skipped_existing'):
            continue
        entries.append({'form': rec['form'], 'seed': rec['seed'], 'ckpt': rec['ckpt'],
                        'tag': tag, 'source': 'p6'})
    return entries


def run_control(entries, args):
    """Closed-loop frozen-P0 eval (crosshair) for every entry that has a checkpoint."""
    for e in entries:
        ckpt = e['ckpt']
        if not ckpt or not os.path.exists(os.path.join(ROOT, ckpt)):
            print(f"  SKIP control {e['tag']}: checkpoint missing ({ckpt})")
            e['control'] = None
            continue
        print(f"\n=== CONTROL {e['tag']}  form={e['form']} seed={e['seed']}  ({ckpt}) ===")
        agg = evaluate_frozen(ckpt, args.n_episodes, args.base_seed,
                              args.survive_threshold, args.quadrotor_config,
                              args.flow_config, args.n_inference_steps, args.sigma,
                              target_render='crosshair')
        e['control'] = {
            'tier1': agg['tier1_pass_rate'], 'survival': agg['survival_mean'],
            'score': agg['score_mean'], 'n_cond': agg['n_conditional'],
            'iae_cond': agg['iae_steady_cond'], 'iae_all': agg['iae_steady_all'],
        }


def run_geometry(entries, args):
    """Faithful flow_mid geometry (eff_rank / feat_norm / disp_infonce) for every entry.

    One fixed, seeded (img,imu,act,task) batch + one fixed (t,eps) draw, identical for
    every checkpoint (paired), exactly as scripts.measure_feature_collapse_flowmid.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(ROOT, args.geometry_config), 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    T_obs = cfg['vision']['T_obs']; T_pred = cfg['action']['T_pred']
    action_dim = cfg['action']['action_dim']

    print(f"\nBuilding fixed geometry batch: n={args.geo_n_samples}, seed={args.geo_seed} ...")
    images, imu, actions, task = build_input_batch(
        os.path.join(ROOT, args.geo_hover_h5), os.path.join(ROOT, args.geo_recovery_h5),
        args.geo_n_samples, T_obs, T_pred, seed=args.geo_seed)
    N = len(images)
    g = torch.Generator().manual_seed(args.geo_seed)
    t = torch.rand(N, generator=g)
    eps = torch.randn(N, action_dim, T_pred, generator=g)

    model = build_model(cfg, device)
    for e in entries:
        ckpt = e['ckpt']
        if not ckpt or not os.path.exists(os.path.join(ROOT, ckpt)):
            e['geometry'] = None
            continue
        model.load_state_dict(torch.load(os.path.join(ROOT, ckpt), map_location='cpu'))
        model.eval()
        feats = extract_flow_mid(model, images, imu, actions, task, t, eps, device).to(device)
        m = collapse_metrics(feats, tau=args.geo_tau, pair_subset=args.geo_pair_subset)
        e['geometry'] = {
            'effective_rank': m['effective_rank'], 'rank_ratio': m['rank_ratio'],
            'n_eff_dims_99': m['n_eff_dims_99'], 'mean_feat_norm': m['mean_feat_norm'],
            'disp_infonce': m['disp_infonce'], 'mean_pairwise_cos': m['mean_pairwise_cos'],
            'feature_dim': m['feature_dim'],
        }
        print(f"  GEOMETRY {e['tag']:16s} form={e['form']:8s} | "
              f"eff_rank={m['effective_rank']:.2f} feat_norm={m['mean_feat_norm']:.2f} "
              f"disp_infonce={m['disp_infonce']:.4f}")


def aggregate(entries):
    by_form = defaultdict(list)
    for e in entries:
        by_form[e['form']].append(e)

    form_agg = {}
    for form, es in by_form.items():
        ctl = [e['control'] for e in es if e.get('control')]
        geo = [e['geometry'] for e in es if e.get('geometry')]
        a = {'n_seeds': len(es), 'seeds': sorted(e['seed'] for e in es),
             'source': es[0]['source']}
        if ctl:
            t_m, t_s = _mean_std([c['tier1'] for c in ctl])
            s_m, s_s = _mean_std([c['survival'] for c in ctl])
            i_m, i_s = _mean_std([c['iae_cond'] for c in ctl])
            c_m, c_s = _mean_std([c['score'] for c in ctl])
            ncond = [c['n_cond'] for c in ctl]
            a.update({'tier1_mean': t_m, 'tier1_std': t_s,
                      'survival_mean': s_m, 'survival_std': s_s,
                      'cond_iae_mean': i_m, 'cond_iae_std': i_s,
                      'score_mean': c_m, 'score_std': c_s,
                      'n_cond_mean': float(np.mean(ncond)), 'n_cond_min': int(np.min(ncond))})
        if geo:
            er_m, er_s = _mean_std([gm['effective_rank'] for gm in geo])
            fn_m, fn_s = _mean_std([gm['mean_feat_norm'] for gm in geo])
            di_m, di_s = _mean_std([gm['disp_infonce'] for gm in geo])
            a.update({'eff_rank_mean': er_m, 'eff_rank_std': er_s,
                      'feat_norm_mean': fn_m, 'feat_norm_std': fn_s,
                      'disp_infonce_mean': di_m, 'disp_infonce_std': di_s})
        form_agg[form] = a
    return form_agg


def compute_verdicts(form_agg, min_ncond=15):
    """Phase 6 verdicts from the by-form aggregate (no eval; reusable for --reverdict).

    [A] objective-gaming: a scale-invariant form must NOT inflate feat_norm (>2x off) and
        must NOT make eff_rank worse than off. [B] control: pre-registered "effective" =
        IMPROVEMENT beyond pooled std (survival UP / Tier1 UP / cond-IAE DOWN) vs infonce;
        `moved` is direction-agnostic, `improved`/`regressed` carry the sign.
    """
    verdict = {'objective_gaming': {}, 'control': {}}
    off = form_agg.get('off'); inf = form_agg.get('infonce')

    if off and 'feat_norm_mean' in off:
        base_fn = off['feat_norm_mean']; base_er = off['eff_rank_mean']
        for form in ['infonce', 'cosine', 'vicreg']:
            a = form_agg.get(form)
            if not a or 'feat_norm_mean' not in a:
                continue
            fn_ratio = a['feat_norm_mean'] / base_fn if base_fn else float('nan')
            d_er = a['eff_rank_mean'] - base_er
            inflated = fn_ratio > 2.0
            rank_worse = d_er < -(off['eff_rank_std'] + a['eff_rank_std'])
            verdict['objective_gaming'][form] = {
                'feat_norm': a['feat_norm_mean'], 'feat_norm_ratio_vs_off': fn_ratio,
                'eff_rank': a['eff_rank_mean'], 'eff_rank_delta_vs_off': d_er,
                'norm_inflated': inflated, 'rank_worse_than_off': rank_worse,
                'games_objective': bool(inflated or rank_worse),
            }
        cos_clean = not verdict['objective_gaming'].get('cosine', {}).get('games_objective', True)
        vic_clean = not verdict['objective_gaming'].get('vicreg', {}).get('games_objective', True)
        verdict['scale_invariant_removes_gaming'] = bool(cos_clean and vic_clean)

    if inf and 'tier1_mean' in inf:
        for form in ['cosine', 'vicreg']:
            a = form_agg.get(form)
            if not a or 'tier1_mean' not in a:
                continue
            d_t1 = a['tier1_mean'] - inf['tier1_mean']
            d_sv = a['survival_mean'] - inf['survival_mean']
            d_ia = a['cond_iae_mean'] - inf['cond_iae_mean']        # < 0 = better precision
            p_t1 = float(np.hypot(a['tier1_std'], inf['tier1_std']))
            p_sv = float(np.hypot(a['survival_std'], inf['survival_std']))
            p_ia = float(np.hypot(a['cond_iae_std'], inf['cond_iae_std']))
            verdict['control'][form] = {
                'd_tier1': d_t1, 'pooled_tier1': p_t1,
                'd_survival': d_sv, 'pooled_survival': p_sv,
                'd_cond_iae': d_ia, 'pooled_cond_iae': p_ia,
                'control_moved_vs_infonce': bool(abs(d_t1) > p_t1 or abs(d_sv) > p_sv or abs(d_ia) > p_ia),
                'control_improved_vs_infonce': bool(d_t1 > p_t1 or d_sv > p_sv or d_ia < -p_ia),
                'control_regressed_vs_infonce': bool(d_t1 < -p_t1 or d_sv < -p_sv or d_ia > p_ia),
            }
        ctl = list(verdict['control'].values())
        verdict['any_control_moved'] = bool(any(v['control_moved_vs_infonce'] for v in ctl))
        verdict['any_control_improved'] = bool(any(v['control_improved_vs_infonce'] for v in ctl))
        # Clean Phase-6 result: gaming removed AND no scale-invariant form IMPROVES control.
        verdict['clean_phase6_geometry_decoupled_from_control'] = bool(
            verdict.get('scale_invariant_removes_gaming') and not verdict['any_control_improved'])
    return verdict


def print_verdicts(form_agg, verdict):
    og = verdict.get('objective_gaming', {})
    if og:
        print("\n" + "#" * 92)
        print("# [A] OBJECTIVE-GAMING verdict  (scale-invariant forms vs OFF baseline geometry)")
        print("#" * 92)
        for form in ['infonce', 'cosine', 'vicreg']:
            v = og.get(form)
            if not v:
                continue
            print(f"  {form:8s}: feat_norm={v['feat_norm']:.2f} "
                  f"({v['feat_norm_ratio_vs_off']:.2f}x off)  eff_rank={v['eff_rank']:.2f} "
                  f"(d{v['eff_rank_delta_vs_off']:+.2f} vs off)  -> "
                  f"{'GAMES objective' if v['games_objective'] else 'clean (no gaming)'}")
        print(f"\n  ==> scale-invariant removes norm-inflation gaming: "
              f"{verdict.get('scale_invariant_removes_gaming')}")
    ctl = verdict.get('control', {})
    if ctl:
        print("\n" + "#" * 92)
        print("# [B] CONTROL verdict  (vs INFONCE; IMPROVE = survival/Tier1 UP or cond-IAE DOWN, > pooled std)")
        print("#" * 92)
        for form in ['cosine', 'vicreg']:
            v = ctl.get(form)
            if not v:
                continue
            tag = ('IMPROVED' if v['control_improved_vs_infonce'] else
                   'REGRESSED' if v['control_regressed_vs_infonce'] else 'flat')
            print(f"  {form:8s} vs infonce: Tier1 d{v['d_tier1']*100:+.1f}pp (pooled {v['pooled_tier1']*100:.1f}) | "
                  f"Surv d{v['d_survival']*100:+.1f}pp (pooled {v['pooled_survival']*100:.1f}) | "
                  f"condIAE d{v['d_cond_iae']:+.3f}m (pooled {v['pooled_cond_iae']:.3f}) -> {tag}")
        print(f"\n  ==> any scale-invariant form IMPROVES control vs infonce: "
              f"{verdict.get('any_control_improved')}")
        if verdict.get('clean_phase6_geometry_decoupled_from_control'):
            print("  ==> CLEAN PHASE-6 RESULT: the scale-invariant criterion removes the norm-inflation\n"
                  "      gaming (feat_norm O(1), eff_rank UP) yet does NOT improve closed-loop control\n"
                  "      -> criterion scale only changes geometry; survival/precision are decoupled from it.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='evaluation_results/p6_form_ablation_manifest.json')
    parser.add_argument('--p2f-manifest', default='evaluation_results/p2f_ablation_manifest.json',
                        help='Source of the reused off (D0E1) / infonce (D1E1) control checkpoints')
    parser.add_argument('--n-episodes', type=int, default=30)
    parser.add_argument('--base-seed', type=int, default=12345)
    parser.add_argument('--survive-threshold', type=int, default=250)
    parser.add_argument('--quadrotor-config', default='configs/quadrotor_v4.yaml')
    parser.add_argument('--flow-config', default='configs/flow_policy_v4.yaml',
                        help='Config for build_policy in evaluate_frozen (arch auto-detected)')
    parser.add_argument('--n-inference-steps', type=int, default=2)
    parser.add_argument('--sigma', type=float, default=2.0)
    parser.add_argument('--oracle-ckpt', default=None)
    parser.add_argument('--oracle-norm', default=None)
    parser.add_argument('--min-ncond', type=int, default=15,
                        help='Min per-form mean n_cond for cond-IAE to be trusted (default 15)')
    # geometry knobs (mirror measure_feature_collapse_flowmid)
    parser.add_argument('--skip-geometry', action='store_true',
                        help='Skip the flow_mid feature-geometry probe (control-only run)')
    parser.add_argument('--geometry-config', default='configs/flow_policy_v5.yaml')
    parser.add_argument('--geo-hover-h5', default='data/expert_demos_v4.h5')
    parser.add_argument('--geo-recovery-h5', default='data/expert_demos_v4_recovery.h5')
    parser.add_argument('--geo-n-samples', type=int, default=4000)
    parser.add_argument('--geo-pair-subset', type=int, default=2048)
    parser.add_argument('--geo-tau', type=float, default=0.5)
    parser.add_argument('--geo-seed', type=int, default=12345)
    parser.add_argument('--output', default='evaluation_results/p6_form_ablation_leaderboard.json')
    parser.add_argument('--reverdict', default=None,
                        help='Recompute+rewrite ONLY the verdict block of an existing leaderboard '
                             'JSON from its cached form_agg (no eval, no GPU). Use after changing '
                             'the verdict logic without re-rolling 12 checkpoints.')
    args = parser.parse_args()

    if args.reverdict:
        path = os.path.join(ROOT, args.reverdict)
        with open(path, 'r', encoding='utf-8') as f:
            lb = json.load(f)
        verdict = compute_verdicts(lb['form_agg'], args.min_ncond)
        print_verdicts(lb['form_agg'], verdict)
        lb['verdict'] = verdict
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(lb, f, indent=2)
        print(f"\n[reverdict] rewrote verdict in {args.reverdict} (no re-eval)")
        return

    with open(os.path.join(ROOT, args.manifest), 'r', encoding='utf-8') as f:
        p6_manifest = json.load(f)
    with open(os.path.join(ROOT, args.p2f_manifest), 'r', encoding='utf-8') as f:
        p2f_manifest = json.load(f)

    entries = collect_entries(p6_manifest, p2f_manifest)
    print(f"Collected {len(entries)} runs across forms "
          f"{sorted(set(e['form'] for e in entries))}")

    # ---- measured oracle (for %Oracle on the score axis) ----
    oracle_score = None
    if args.oracle_ckpt and args.oracle_norm:
        print(f"\n=== PPO_Oracle ({args.oracle_ckpt}) ===")
        oagg = evaluate_oracle_frozen(
            args.oracle_ckpt, args.oracle_norm, args.n_episodes, args.base_seed,
            args.survive_threshold, args.quadrotor_config, args.sigma)
        oracle_score = oagg['score_mean']
        print(f"  measured oracle composite = {oracle_score:.4f}")

    run_control(entries, args)
    if not args.skip_geometry:
        try:
            run_geometry(entries, args)
        except Exception as ex:
            print(f"\n[geometry] FAILED ({ex}); continuing control-only")
            import traceback; traceback.print_exc()
            for e in entries:
                e.setdefault('geometry', None)
    else:
        for e in entries:
            e['geometry'] = None

    form_agg = aggregate(entries)
    if oracle_score:
        for a in form_agg.values():
            if 'score_mean' in a:
                a['pct_oracle'] = a['score_mean'] / oracle_score * 100.0

    # ---- per-run table ----
    print("\n" + "=" * 112)
    print(f"{'Tag':<18}{'Form':>9}{'Seed':>5}{'Surv%':>8}{'Tier1%':>8}{'condIAE':>9}{'n_cond':>8}"
          f"{'eff_rank':>10}{'feat_norm':>11}")
    print("-" * 112)
    for e in sorted(entries, key=lambda x: (FORM_ORDER.index(x['form']) if x['form'] in FORM_ORDER else 9, x['seed'])):
        c = e.get('control') or {}
        gm = e.get('geometry') or {}
        surv = f"{c['survival']*100:>7.1f}%" if c else f"{'--':>8}"
        t1 = f"{c['tier1']*100:>7.1f}%" if c else f"{'--':>8}"
        iae = f"{c['iae_cond']:>8.3f}m" if c else f"{'--':>9}"
        nc = f"{c['n_cond']:>5}/{args.n_episodes:<2}" if c else f"{'--':>8}"
        er = f"{gm['effective_rank']:>10.2f}" if gm else f"{'--':>10}"
        fn = f"{gm['mean_feat_norm']:>11.2f}" if gm else f"{'--':>11}"
        print(f"{e['tag']:<18}{e['form']:>9}{e['seed']:>5}{surv}{t1}{iae}{nc}{er}{fn}")
    print("=" * 112)

    # ---- by-form summary (mean +- std) ----
    print("\nBY FORM (mean +- std over seeds):")
    print(f"{'Form':<9}{'n':>3}{'Surv%':>14}{'Tier1%':>14}{'condIAE':>14}{'n_cond':>8}"
          f"{'eff_rank':>14}{'feat_norm':>16}")
    for form in FORM_ORDER:
        if form not in form_agg:
            continue
        a = form_agg[form]
        surv = (f"{a['survival_mean']*100:>6.1f}+-{a['survival_std']*100:<4.1f}"
                if 'survival_mean' in a else f"{'--':>12}")
        t1 = (f"{a['tier1_mean']*100:>6.1f}+-{a['tier1_std']*100:<4.1f}"
              if 'tier1_mean' in a else f"{'--':>12}")
        if 'cond_iae_mean' in a:
            flag = '' if a['n_cond_mean'] >= args.min_ncond else '!'
            iae = f"{a['cond_iae_mean']:>6.3f}+-{a['cond_iae_std']:<4.3f}{flag}"
            nc = f"{a['n_cond_mean']:>5.1f}"
        else:
            iae = f"{'--':>12}"; nc = f"{'--':>6}"
        er = (f"{a['eff_rank_mean']:>7.2f}+-{a['eff_rank_std']:<5.2f}"
              if 'eff_rank_mean' in a else f"{'--':>12}")
        fn = (f"{a['feat_norm_mean']:>8.2f}+-{a['feat_norm_std']:<6.2f}"
              if 'feat_norm_mean' in a else f"{'--':>14}")
        print(f"{form:<9}{a['n_seeds']:>3}{surv:>14}{t1:>14}{iae:>14}{nc:>8}{er:>14}{fn:>16}")
    print("  (! after condIAE = mean n_cond below --min-ncond, precision UNRELIABLE)")

    # ---- Phase 6 verdicts (directional: improvement vs regression) ----
    verdict = compute_verdicts(form_agg, args.min_ncond)
    print_verdicts(form_agg, verdict)

    out = {'form_agg': form_agg, 'entries': entries, 'verdict': verdict,
           'oracle_score': oracle_score,
           'manifest': args.manifest, 'p2f_manifest': args.p2f_manifest,
           'protocol': {'n_episodes': args.n_episodes, 'base_seed': args.base_seed,
                        'survive_threshold': args.survive_threshold, 'sigma': args.sigma,
                        'n_inference_steps': args.n_inference_steps, 'render': 'crosshair',
                        'min_ncond': args.min_ncond,
                        'geometry': None if args.skip_geometry else {
                            'config': args.geometry_config, 'n_samples': args.geo_n_samples,
                            'pair_subset': args.geo_pair_subset, 'tau': args.geo_tau,
                            'seed': args.geo_seed,
                            'image_source': f'{args.geo_hover_h5} + {args.geo_recovery_h5} (50/50)'}}}
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.output)), exist_ok=True)
    with open(os.path.join(ROOT, args.output), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()

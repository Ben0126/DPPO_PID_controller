"""
Phase 3b (negative-result diagnosis) — is closed-loop PRECISION gated by OOD
coverage of the BC training distribution?

Background (RESEARCH_PLAN_v6.md Phase 3c): survival improved with the recipe, but
*precision* is stuck at cond-IAE ~2.8 m (~13 % of the 0.068 m oracle) across every
v5/v6 config. Phase 3b asks WHY precision won't move. The OOD-coverage hypothesis:
the surviving policy drifts to a steady-state ~2.8 m off target, but the BC data
(hover + recovery, initial pos within ±1 m, expert pulls straight back) contains
almost no samples at 2–3 m position error — so the policy has never been taught how
to correct from there and simply parks in an under-covered region.

This script tests that CHEAPLY (no retraining): it compares two distributions of
**position-error magnitude ‖pos − target‖**:

  (A) TRAINING coverage — from the BC h5 states. The 15D obs layout is
      [pos_error_body(3), rot_6d(6), vel_body(3), omega(3)] (quadrotor_env_v4
      `_get_observation`), so ‖states[:, 0:3]‖ is exactly the position-error
      magnitude the encoder/flow net was trained on (body-frame norm == world norm).

  (B) CLOSED-LOOP visited — roll out the current frontier policy under the frozen
      seeds and collect the per-step ‖pos − target‖ for SURVIVING episodes, split
      into the steady-state window (errs[T/2:], the cond-IAE samples) and full.

Decision:
  * If the steady-state visited errors sit far beyond the TRAINING p99 / max
    (i.e. the policy lives where it has ~no training data) → precision IS
    OOD-coverage-gated → widening recovery-init (Phase 3b retrain) is justified.
  * If the visited errors fall INSIDE the training coverage → precision is NOT
    OOD-gated; it's a capacity/other limit and a wider-init retrain won't help
    (saves hours of compute).

Usage:
  dppo/Scripts/python.exe -m scripts.measure_ood_coverage \
      --ckpt checkpoints/flow_policy_v5/p2_D0E1_s0/best_model.pt --label D0E1_s0 \
      --n-episodes 30
"""
import os
import sys
import json
import argparse

import numpy as np
import h5py
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.quadrotor_env_v4 import QuadrotorEnvV4
from envs.quadrotor_visual_env import QuadrotorVisualEnv
from scripts.evaluate_hierarchical import build_policy, rollout_episode


def pos_err_from_h5(h5_path, max_ep):
    """Return per-timestep position-error magnitude from a BC h5 (states[:,0:3])."""
    errs = []
    with h5py.File(h5_path, 'r') as f:
        n_ep = int(f.attrs['n_episodes'])
        for ep in range(min(n_ep, max_ep)):
            key = f'episode_{ep}'
            if key not in f:
                continue
            s = f[key]['states'][:]                 # (T, 15)
            errs.append(np.linalg.norm(s[:, 0:3], axis=1))
    return np.concatenate(errs) if errs else np.array([])


def quantiles(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {}
    qs = [50, 75, 90, 95, 99, 99.9]
    out = {f'p{q}': round(float(np.percentile(x, q)), 4) for q in qs}
    out['mean'] = round(float(x.mean()), 4)
    out['max'] = round(float(x.max()), 4)
    out['n'] = int(x.size)
    return out


def frac_exceeding(x, thresholds):
    x = np.asarray(x, dtype=float)
    return {f'>{t}m': round(float((x > t).mean()), 5) for t in thresholds}


def ascii_hist(x, lo=0.0, hi=5.0, bins=20, width=50, label=''):
    x = np.asarray(x, dtype=float)
    edges = np.linspace(lo, hi, bins + 1)
    h, _ = np.histogram(np.clip(x, lo, hi), bins=edges)
    mx = h.max() if h.max() > 0 else 1
    lines = [f"  {label} (n={x.size}, clipped to [{lo},{hi}]m)"]
    for i in range(bins):
        bar = '#' * int(width * h[i] / mx)
        lines.append(f"  {edges[i]:4.1f}-{edges[i+1]:4.1f}m | {bar} {h[i]}")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description='OOD-coverage probe for closed-loop precision')
    ap.add_argument('--ckpt', default='checkpoints/flow_policy_v5/p2_D0E1_s0/best_model.pt',
                    help='Frontier policy to roll out (default: D0E1 seed 0, the clean E2E frontier)')
    ap.add_argument('--label', default='D0E1_s0')
    ap.add_argument('--hover-h5', default='data/expert_demos_v4.h5')
    ap.add_argument('--recovery-h5', default='data/expert_demos_v4_recovery.h5')
    ap.add_argument('--hover-episodes', type=int, default=500)
    ap.add_argument('--recovery-episodes', type=int, default=500)
    ap.add_argument('--n-episodes', type=int, default=30)
    ap.add_argument('--base-seed', type=int, default=12345)
    ap.add_argument('--survive-threshold', type=int, default=250)
    ap.add_argument('--n-inference-steps', type=int, default=2)
    ap.add_argument('--quadrotor-config', default='configs/quadrotor_v4.yaml')
    ap.add_argument('--flow-config', default='configs/flow_policy_v4.yaml')
    ap.add_argument('--out', default='evaluation_results/p3b_ood_coverage.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---------- (A) training coverage ----------
    print("Loading BC training position-error coverage (states[:,0:3]) ...")
    hover_err = pos_err_from_h5(os.path.join(ROOT, args.hover_h5), args.hover_episodes)
    rec_err = pos_err_from_h5(os.path.join(ROOT, args.recovery_h5), args.recovery_episodes)
    train_err = np.concatenate([hover_err, rec_err])
    print(f"  hover steps={hover_err.size:,}  recovery steps={rec_err.size:,}  total={train_err.size:,}")
    train_q = quantiles(train_err)
    print(f"  TRAIN pos-err: p50={train_q['p50']}  p95={train_q['p95']}  "
          f"p99={train_q['p99']}  max={train_q['max']} m")

    # ---------- (B) closed-loop visited ----------
    with open(os.path.join(ROOT, args.flow_config), 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    vis_cfg = cfg['vision']; act_cfg = cfg['action']

    policy, arch = build_policy(os.path.join(ROOT, args.ckpt), cfg,
                               args.n_inference_steps, device)
    base_env = QuadrotorEnvV4(config_path=os.path.join(ROOT, args.quadrotor_config))
    visual_env = QuadrotorVisualEnv(base_env, image_size=vis_cfg['image_size'])
    T_obs = vis_cfg['T_obs']; T_action = act_cfg['T_action']
    max_steps = base_env.max_episode_steps
    surv_thr = min(args.survive_threshold, max_steps)

    print(f"\nRolling out {args.label} ({args.n_episodes} ep, frozen seeds) ...")
    steady_err, full_err_surv = [], []
    n_surv = 0
    for ep in range(args.n_episodes):
        seed = args.base_seed + ep
        roll = rollout_episode(policy, base_env, visual_env, arch,
                               T_obs, T_action, args.n_inference_steps, device, seed=seed)
        errs = np.linalg.norm(roll['positions'] - roll['targets'], axis=1)
        T = roll['ep_length']
        survived = T >= surv_thr
        tag = ''
        if survived:
            n_surv += 1
            half = max(1, T // 2)
            steady_err.append(errs[half:])
            full_err_surv.append(errs)
            tag = '  [surv]'
        print(f"  Ep {ep+1:>2}/{args.n_episodes} seed={seed} steps={T:>3} "
              f"mean_err={errs.mean():.3f}m{tag}")

    steady_err = np.concatenate(steady_err) if steady_err else np.array([])
    full_err_surv = np.concatenate(full_err_surv) if full_err_surv else np.array([])
    steady_q = quantiles(steady_err)
    full_q = quantiles(full_err_surv)

    # ---------- comparison ----------
    p99 = train_q.get('p99', float('nan'))
    tmax = train_q.get('max', float('nan'))
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    out = {
        'ckpt': args.ckpt, 'label': args.label,
        'n_episodes': args.n_episodes, 'n_survived': n_surv,
        'survive_threshold': surv_thr,
        'train_coverage': train_q,
        'train_coverage_frac_exceeding': frac_exceeding(train_err, thresholds),
        'closed_loop_steady': steady_q,
        'closed_loop_steady_frac_exceeding': frac_exceeding(steady_err, thresholds),
        'closed_loop_full_surv': full_q,
        'ood': {
            'train_p99_m': p99,
            'train_max_m': tmax,
            'steady_frac_above_train_p99': round(float((steady_err > p99).mean()), 5)
                if steady_err.size else float('nan'),
            'steady_frac_above_train_max': round(float((steady_err > tmax).mean()), 5)
                if steady_err.size else float('nan'),
            'steady_median_over_train_p99_ratio': round(float(steady_q.get('p50', np.nan) / p99), 3)
                if p99 else float('nan'),
        },
    }
    os.makedirs(os.path.join(ROOT, os.path.dirname(args.out)), exist_ok=True)
    with open(os.path.join(ROOT, args.out), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    # ---------- report ----------
    print("\n" + "=" * 88)
    print(f"OOD-COVERAGE PROBE -- {args.label}  ({n_surv}/{args.n_episodes} survived >={surv_thr})")
    print("=" * 88)
    print(f"{'metric':<34}{'TRAIN (BC data)':>18}{'CLOSED-LOOP steady':>22}")
    for k in ['p50', 'p90', 'p95', 'p99', 'max']:
        print(f"  pos-err {k:<25}{train_q.get(k,'--'):>16}m{steady_q.get(k,'--'):>20}m")
    print(f"\n  TRAIN fraction of samples beyond X m: {out['train_coverage_frac_exceeding']}")
    print(f"  STEADY fraction of samples beyond X m: {out['closed_loop_steady_frac_exceeding']}")
    print(f"\n  >> steady-state samples above TRAIN p99 ({p99}m): "
          f"{out['ood']['steady_frac_above_train_p99']*100:.1f}%")
    print(f"  >> steady-state samples above TRAIN max ({tmax}m): "
          f"{out['ood']['steady_frac_above_train_max']*100:.1f}%")
    print(f"  >> steady median / train p99 ratio: {out['ood']['steady_median_over_train_p99_ratio']}")
    print()
    print(ascii_hist(train_err, label='TRAIN pos-err'))
    print()
    print(ascii_hist(steady_err, label='CLOSED-LOOP steady pos-err'))

    verdict = ('OOD-GATED (precision-limiting region is under-covered) -> wider-init retrain justified'
               if steady_q.get('p50', 0) > p99 else
               'NOT OOD-gated (visited region is within training coverage) -> wider-init unlikely to help')
    print(f"\nVERDICT: {verdict}")
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()

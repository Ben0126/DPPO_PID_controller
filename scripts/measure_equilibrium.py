"""
Gate alpha (RESEARCH_PLAN_v9 Phase 0) — closed-loop equilibrium shape of the hover floor.

v8 sharpened the wall: the hover specialist T0O1 settles at cond-IAE ~2.48 m closed-loop
despite a near-perfect open-loop action fit (Gate A act_mse 0.00035) and available range
(perspective render). In hover mode the env sets target == init_pos, so the drone STARTS at
the target (zero position offset). This gate asks the decisive shape question on the existing
checkpoints (inference only):

    From a perfect (zero-offset) start, does the per-step position error
      (a) rise and then CONVERGE to a steady ~2.48 m plateau  -> a biased EQUILIBRIUM
          (steady-state control / BC-objective limit; Phase-1 fix = b2 objective / b3 authority)
      (b) keep RISING / oscillate / diverge                   -> compounding instability
          (Phase-1 fix = b1 short-horizon closed-loop fine-tune)

Reuses scripts.evaluate_hierarchical.rollout_episode (which already returns per-step
positions/targets) under the exact frozen-P0 env construction (paired seeds, perspective
render). Analyses SURVIVING episodes only (>= survive_threshold steps), where a steady-state
plateau is meaningful.

Usage:
  dppo/Scripts/python.exe -m scripts.measure_equilibrium \
      --ckpts T0O1:checkpoints/flow_policy_v5/p2to_T0O1_s0/best_model.pt \
              S:checkpoints/flow_policy_v5/p2to_T1O1_s0/best_model.pt \
              XL:checkpoints/flow_policy_v5/p2cap_XL_s0/best_model.pt \
      --render perspective --n-episodes 30
"""
import os
import sys
import json
import argparse
import numpy as np
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env_v4 import QuadrotorEnvV4
from envs.quadrotor_visual_env import QuadrotorVisualEnv
from scripts.evaluate_hierarchical import build_policy, rollout_episode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def episode_equilibrium_stats(positions, targets, survive_threshold):
    """Per-episode closed-loop position-error shape. Returns None if too short."""
    T = len(positions)
    if T < survive_threshold:
        return None
    err = np.linalg.norm(np.asarray(positions) - np.asarray(targets), axis=1)  # (T,)
    half = T // 2
    second = err[half:]
    # linear slope over the second half, expressed per-100-steps (drift rate at "steady state")
    t = np.arange(len(second), dtype=float)
    slope = float(np.polyfit(t, second, 1)[0]) * 100.0 if len(second) > 2 else 0.0
    return {
        'ep_len': int(T),
        'step0_err': float(err[0]),
        'peak_err': float(err.max()),
        'first_half_mean': float(err[:half].mean()),
        'equilibrium': float(second.mean()),      # steady-state offset (cond-IAE-like)
        'second_half_std': float(second.std()),
        'slope_per100': slope,                    # ~0 => plateau; >0 => diverging
        'final_err': float(err[-1]),
    }


def main():
    ap = argparse.ArgumentParser(description='Gate alpha: closed-loop equilibrium shape')
    ap.add_argument('--ckpts', nargs='+', required=True,
                    help='label:path entries (e.g. T0O1:checkpoints/.../best_model.pt)')
    ap.add_argument('--render', default='perspective', choices=['crosshair', 'perspective'])
    ap.add_argument('--n-episodes', type=int, default=30)
    ap.add_argument('--base-seed', type=int, default=12345)
    ap.add_argument('--survive-threshold', type=int, default=250)
    ap.add_argument('--quadrotor-config', default='configs/quadrotor_v4.yaml')
    ap.add_argument('--flow-config', default='configs/flow_policy_v4.yaml')
    ap.add_argument('--n-inference-steps', type=int, default=2)
    ap.add_argument('--out', default='evaluation_results/p2cl_gate_alpha_equilibrium.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  render: {args.render}")
    with open(os.path.join(ROOT, args.flow_config), 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    T_obs = cfg['vision']['T_obs']; T_action = cfg['action']['T_action']

    result = {}
    for entry in args.ckpts:
        label, path = entry.split(':', 1)
        ckpt = os.path.join(ROOT, path.lstrip('./'))
        if not os.path.exists(ckpt):
            print(f"  [skip] {label}: missing {ckpt}"); continue
        print(f"\n=== {label}  ({path})  render={args.render} ===")
        policy, arch = build_policy(ckpt, cfg, args.n_inference_steps, device)
        base_env = QuadrotorEnvV4(config_path=args.quadrotor_config)
        visual_env = QuadrotorVisualEnv(base_env, image_size=cfg['vision']['image_size'],
                                        target_render=args.render)
        max_steps = base_env.max_episode_steps
        thr = min(args.survive_threshold, max_steps)

        ep_stats = []
        for ep in range(args.n_episodes):
            seed = args.base_seed + ep
            roll = rollout_episode(policy, base_env, visual_env, arch, T_obs, T_action,
                                   args.n_inference_steps, device, seed=seed)
            s = episode_equilibrium_stats(roll['positions'], roll['targets'], thr)
            tag = 'cond' if s else f"short({roll['ep_length']})"
            if s:
                ep_stats.append(s)
            print(f"  ep{ep+1:>2} seed={seed} steps={roll['ep_length']:>3} "
                  f"{('eq=%.2fm slope/100=%+.3f step0=%.2f' % (s['equilibrium'], s['slope_per100'], s['step0_err'])) if s else tag}")

        if not ep_stats:
            print(f"  [{label}] no surviving episodes — cannot characterise equilibrium")
            result[label] = {'n_cond': 0}
            continue

        def col(k):
            return np.array([e[k] for e in ep_stats], dtype=float)
        eq = col('equilibrium'); slope = col('slope_per100')
        agg = {
            'n_cond': len(ep_stats),
            'render': args.render,
            'step0_err_mean': round(float(col('step0_err').mean()), 4),
            'first_half_mean': round(float(col('first_half_mean').mean()), 4),
            'equilibrium_mean': round(float(eq.mean()), 4),
            'equilibrium_std': round(float(eq.std()), 4),
            'peak_err_mean': round(float(col('peak_err').mean()), 4),
            'second_half_std_mean': round(float(col('second_half_std').mean()), 4),
            'slope_per100_mean': round(float(slope.mean()), 4),
            'slope_per100_std': round(float(slope.std()), 4),
            'frac_plateau': round(float((np.abs(slope) < 0.20).mean()), 3),  # |drift|<0.2m/100steps
        }
        # verdict: plateau (steady-state biased equilibrium) vs diverging (compounding)
        if abs(agg['slope_per100_mean']) < 0.20 and agg['equilibrium_mean'] > 1.5:
            agg['shape'] = 'PLATEAU'      # biased equilibrium -> b2/b3
        elif agg['slope_per100_mean'] >= 0.20:
            agg['shape'] = 'DIVERGING'    # compounding -> b1
        else:
            agg['shape'] = 'CONVERGING_IN'  # still pulling inward at horizon end
        result[label] = agg
        print(f"  -> [{label}] equilibrium={agg['equilibrium_mean']:.2f}±{agg['equilibrium_std']:.2f}m "
              f"step0={agg['step0_err_mean']:.2f}m slope/100={agg['slope_per100_mean']:+.3f} "
              f"frac_plateau={agg['frac_plateau']} => {agg['shape']}")

    out = {'gate': 'alpha — closed-loop equilibrium shape', 'render': args.render,
           'base_seed': args.base_seed, 'survive_threshold': args.survive_threshold,
           'n_episodes': args.n_episodes, 'by_ckpt': result}
    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 80)
    print("Gate alpha — closed-loop equilibrium shape (surviving episodes)")
    print("=" * 80)
    print(f"{'ckpt':6s} {'n':>3s} {'step0':>7s} {'1st-half':>9s} {'equilib':>14s} {'slope/100':>11s} {'shape':>14s}")
    for label, a in result.items():
        if a.get('n_cond', 0) == 0:
            print(f"{label:6s}  no surviving episodes"); continue
        print(f"{label:6s} {a['n_cond']:>3d} {a['step0_err_mean']:>6.2f}m {a['first_half_mean']:>8.2f}m "
              f"{a['equilibrium_mean']:>7.2f}±{a['equilibrium_std']:<4.2f}m {a['slope_per100_mean']:>+8.3f}   {a['shape']:>12s}")
    print("\nReading: PLATEAU + low step0 => the policy drives itself from a perfect start to a")
    print("biased ~2.48 m EQUILIBRIUM -> steady-state control/objective limit (Phase-1 b2/b3).")
    print("DIVERGING => compounding instability (Phase-1 b1 closed-loop fine-tune).")
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()

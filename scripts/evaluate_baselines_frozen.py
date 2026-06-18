"""
Frozen-protocol evaluation for NON-flow baselines (P1 baseline matrix).

Any model that exposes ``predict_action(images, imu=None, n_steps=None,
task_cond=None) -> (B, action_dim, T_pred)`` can be rolled out through the IDENTICAL
P0 frozen protocol (same paired seeds, same env, same σ=2.0 exp-decay metric,
conditional-IAE, measured %Oracle) used by ``scripts.evaluate_frozen_p0`` — so
baseline numbers drop straight into the canonical leaderboard comparison.

Currently wired for BC-vision-only; PPO-from-pixels will slot in once its actor
exposes the same ``predict_action`` contract.

Usage:
  dppo/Scripts/python.exe -m scripts.evaluate_baselines_frozen \
      --ckpts "BC_vis:checkpoints/bc_vision_only/bc_vision_only_s0/best_model.pt" \
      --oracle-ckpt checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
      --oracle-norm checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.quadrotor_env_v4 import QuadrotorEnvV4
from envs.quadrotor_visual_env import QuadrotorVisualEnv
from scripts.evaluate_hierarchical import (
    rollout_episode, compute_hierarchical_metrics, composite_score,
)
from scripts.evaluate_frozen_p0 import _aggregate_frozen, evaluate_oracle_frozen
from scripts.train_bc_vision_only import BCVisionOnly


def build_baseline(kind, ckpt_path, cfg, device):
    """Construct a baseline model that implements the predict_action contract."""
    vis_cfg, act_cfg = cfg['vision'], cfg['action']
    if kind == 'bc_vision':
        model = BCVisionOnly(vision_feature_dim=vis_cfg['feature_dim'],
                             T_obs=vis_cfg['T_obs'], T_pred=act_cfg['T_pred'],
                             action_dim=act_cfg['action_dim']).to(device)
        model.load(ckpt_path, map_location=device)
        model.eval()
        return model, {'era': 'BC-vis', 'task_dim': 0}
    raise ValueError(f"unknown baseline kind: {kind}")


def evaluate_baseline_frozen(kind, ckpt_path, n_episodes, base_seed, survive_threshold,
                             quadrotor_config, flow_config, n_inference_steps, sigma,
                             device_str='cuda'):
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    with open(flow_config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    vis_cfg, act_cfg = cfg['vision'], cfg['action']

    model, arch = build_baseline(kind, ckpt_path, cfg, device)
    base_env = QuadrotorEnvV4(config_path=quadrotor_config)
    visual_env = QuadrotorVisualEnv(base_env, image_size=vis_cfg['image_size'])
    T_obs, T_action = vis_cfg['T_obs'], act_cfg['T_action']
    max_steps = base_env.max_episode_steps
    survive_threshold = min(survive_threshold, max_steps)

    per_ep = []
    for ep in range(n_episodes):
        seed = base_seed + ep
        roll = rollout_episode(model, base_env, visual_env, arch,
                               T_obs, T_action, n_inference_steps, device, seed=seed)
        m = compute_hierarchical_metrics(roll['positions'], roll['targets'],
                                         roll['omegas'], roll['ep_length'], max_steps)
        m.update(composite_score(m, sigma=sigma))
        m['ep_reward'] = roll['ep_reward']
        m['seed'] = seed
        m['survived_threshold'] = bool(roll['ep_length'] >= survive_threshold)
        per_ep.append(m)
        print(f"  Ep {ep+1:>2}/{n_episodes} seed={seed} | steps={roll['ep_length']:>3} | "
              f"survive={m['survival_rate']*100:>5.1f}% | IAE_st={m['iae_steady']:.3f}m | "
              f"score={m['score']:.3f}{'  [cond]' if m['survived_threshold'] else ''}")

    return _aggregate_frozen(per_ep, n_episodes, base_seed, survive_threshold, arch['era'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpts', nargs='+', required=True,
                        help='List of "label:path" pairs (BC-vision-only baselines)')
    parser.add_argument('--kind', default='bc_vision', choices=['bc_vision'])
    parser.add_argument('--n-episodes', type=int, default=30)
    parser.add_argument('--base-seed', type=int, default=12345)
    parser.add_argument('--survive-threshold', type=int, default=250)
    parser.add_argument('--quadrotor-config', default='configs/quadrotor_v4.yaml')
    parser.add_argument('--flow-config', default='configs/flow_policy_v4.yaml')
    parser.add_argument('--n-inference-steps', type=int, default=2)
    parser.add_argument('--sigma', type=float, default=2.0)
    parser.add_argument('--oracle-ckpt', default=None)
    parser.add_argument('--oracle-norm', default=None)
    parser.add_argument('--output', default='evaluation_results/baselines_frozen_leaderboard.json')
    args = parser.parse_args()

    results = {}
    for entry in args.ckpts:
        label, path = entry.split(':', 1)
        print(f"\n{'='*72}\n=== {label}  ({path})\n{'='*72}")
        if not os.path.exists(path):
            print("  SKIP: not found"); continue
        agg = evaluate_baseline_frozen(
            args.kind, path, args.n_episodes, args.base_seed, args.survive_threshold,
            args.quadrotor_config, args.flow_config, args.n_inference_steps, args.sigma)
        agg['label'] = label; agg['path'] = path
        results[label] = agg

    measured_oracle = None
    if args.oracle_ckpt and args.oracle_norm:
        print(f"\n=== PPO_Oracle ({args.oracle_ckpt}) ===")
        oagg = evaluate_oracle_frozen(args.oracle_ckpt, args.oracle_norm, args.n_episodes,
                                      args.base_seed, args.survive_threshold,
                                      args.quadrotor_config, args.sigma)
        oagg['label'] = 'PPO_Oracle'; oagg['path'] = args.oracle_ckpt
        results['PPO_Oracle'] = oagg
        measured_oracle = oagg['score_mean']

    print("\n" + "=" * 104)
    print(f"{'Label':<22}{'Arch':>7}{'Score(95% CI)':>22}{'Survive%':>10}{'Tier1%':>8}"
          f"{'cond-IAE':>10}{'n_cond':>7}{'all-IAE':>9}" + ("%Oracle".rjust(9) if measured_oracle else ""))
    print("-" * 104)
    for r in sorted(results.values(), key=lambda x: -x['score_mean']):
        ci = r['score_ci95']
        line = (f"{r['label']:<22}{r['arch']:>7}{r['score_mean']:>8.3f} [{ci[0]:.3f},{ci[1]:.3f}]"
                f"{r['survival_mean']*100:>9.1f}%{r['tier1_pass_rate']*100:>7.1f}%"
                f"{r['iae_steady_cond']:>9.3f}m{r['n_conditional']:>4}/{r['n_episodes']:<2}"
                f"{r['iae_steady_all']:>8.3f}m")
        if measured_oracle:
            line += f"{r['score_mean']/measured_oracle*100:>8.1f}%"
        print(line)
    print("=" * 104)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()

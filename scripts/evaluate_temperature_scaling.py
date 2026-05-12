"""
Temperature Scaling Test for ReinFlow v4.0

Tests whether reducing the initial noise variance (x1 ~ N(0, sigma^2 I) with sigma < 1)
reduces the crash rate of ReinFlow-finetuned FlowMatchingPolicyV4.

Hypothesis: the primary cause of 50/50 crashes is stochastic sampling noise accumulation
in body rate, not the policy weights themselves. If reducing sigma improves survival rate,
this confirms the hypothesis.

Usage:
    python -m scripts.evaluate_temperature_scaling \
        --flow-model checkpoints/reinflow_v4/reinflow_v4_20260502_162154/best_reinflow_model.pt

    # Single temperature (quick check):
    python -m scripts.evaluate_temperature_scaling \
        --flow-model checkpoints/reinflow_v4/reinflow_v4_20260502_162154/best_reinflow_model.pt \
        --temperatures 0.5 --episodes 50

Results saved to: evaluation_results/temperature_scaling/results.json
"""

import os
import sys
import argparse
import yaml
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env_v4 import QuadrotorEnvV4
from envs.quadrotor_visual_env import QuadrotorVisualEnv
from models.flow_policy_v4 import FlowMatchingPolicyV4


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop — single temperature
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_one_temperature(
    visual_env: QuadrotorVisualEnv,
    base_env: QuadrotorEnvV4,
    policy: FlowMatchingPolicyV4,
    n_episodes: int,
    T_obs: int,
    T_action: int,
    device: torch.device,
    n_inference_steps: int,
    temperature: float,
    seed: int,
) -> dict:

    results = {
        'temperature': temperature,
        'rewards':        [],
        'lengths':        [],
        'crashes':        0,
        'position_rmse':  [],
        'crash_steps':    [],   # episode length for crashed episodes
    }

    visual_env.env.reset(seed=seed)   # seed once; each ep calls reset() internally

    for ep in range(n_episodes):
        obs, _ = visual_env.reset()
        image_buffer = [obs['image']] * T_obs

        ep_reward = 0.0
        ep_length = 0
        positions = []
        targets   = []
        done      = False

        while not done:
            img_stack  = np.concatenate(image_buffer[-T_obs:], axis=0)
            img_tensor = torch.from_numpy(img_stack).float().unsqueeze(0).to(device) / 255.0
            imu_vec    = base_env.get_imu()
            imu_tensor = torch.from_numpy(imu_vec).float().unsqueeze(0).to(device)

            action_seq = policy.predict_action(
                img_tensor, imu_tensor,
                n_steps=n_inference_steps,
                temperature=temperature,
            )
            action_seq = action_seq.squeeze(0).T.cpu().numpy()   # (T_pred, action_dim)

            for a_idx in range(min(T_action, action_seq.shape[0])):
                action = action_seq[a_idx]
                obs, reward, terminated, truncated, info = visual_env.step(action)
                image_buffer.append(obs['image'])
                ep_reward += reward
                ep_length += 1
                positions.append(info['position'].copy())
                targets.append(info['target'].copy())
                done = terminated or truncated
                if done:
                    break

        results['rewards'].append(ep_reward)
        results['lengths'].append(ep_length)

        crashed = ep_length < base_env.max_episode_steps
        if crashed:
            results['crashes'] += 1
            results['crash_steps'].append(ep_length)

        pos_errors = np.array(targets) - np.array(positions)
        rmse = float(np.sqrt(np.mean(np.sum(pos_errors**2, axis=1))))
        results['position_rmse'].append(rmse)

        status = 'CRASH' if crashed else 'OK   '
        print(f"    Ep {ep+1:>3}/{n_episodes} | "
              f"RMSE={rmse:.4f}m | steps={ep_length:>4} | {status}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Temperature Scaling Test — ReinFlow v4.0 ===")
    print(f"  Checkpoint:  {args.flow_model}")
    print(f"  Device:      {device}")
    print(f"  Episodes:    {args.episodes} per temperature")
    print(f"  Temperatures: {args.temperatures}")

    with open(args.flow_config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    vis_cfg  = cfg['vision']
    act_cfg  = cfg['action']
    flow_cfg = cfg['flow']

    base_env   = QuadrotorEnvV4(config_path=args.quadrotor_config)
    visual_env = QuadrotorVisualEnv(base_env, image_size=vis_cfg['image_size'])

    policy = FlowMatchingPolicyV4(
        vision_feature_dim = vis_cfg['feature_dim'],
        imu_feature_dim    = cfg['imu']['feature_dim'],
        time_embed_dim     = cfg['unet']['time_embed_dim'],
        down_dims          = tuple(cfg['unet']['down_dims']),
        T_obs              = vis_cfg['T_obs'],
        T_pred             = act_cfg['T_pred'],
        action_dim         = act_cfg['action_dim'],
        n_inference_steps  = args.n_inference_steps,
        t_embed_scale      = flow_cfg['t_embed_scale'],
    ).to(device)
    policy.load(args.flow_model)
    policy.eval()

    T_obs    = vis_cfg['T_obs']
    T_action = act_cfg['T_action']

    all_results = []

    for sigma in args.temperatures:
        print(f"\n--- sigma = {sigma} ---")
        res = evaluate_one_temperature(
            visual_env, base_env, policy,
            n_episodes=args.episodes,
            T_obs=T_obs,
            T_action=T_action,
            device=device,
            n_inference_steps=args.n_inference_steps,
            temperature=sigma,
            seed=args.seed,
        )
        n = args.episodes
        mean_rmse   = float(np.mean(res['position_rmse']))
        std_rmse    = float(np.std(res['position_rmse']))
        mean_len    = float(np.mean(res['lengths']))
        mean_crash_step = float(np.mean(res['crash_steps'])) if res['crash_steps'] else float('nan')
        print(f"  sigma={sigma:.1f} | RMSE={mean_rmse:.4f}+/-{std_rmse:.4f}m | "
              f"Crashes={res['crashes']}/{n} | MeanLen={mean_len:.1f} | "
              f"AvgCrashStep={mean_crash_step:.1f}")
        all_results.append({
            'temperature':      sigma,
            'mean_rmse':        mean_rmse,
            'std_rmse':         std_rmse,
            'crashes':          res['crashes'],
            'n_episodes':       n,
            'mean_ep_length':   mean_len,
            'mean_crash_step':  mean_crash_step,
            'per_episode': {
                'rewards':       res['rewards'],
                'lengths':       res['lengths'],
                'position_rmse': res['position_rmse'],
                'crash_steps':   res['crash_steps'],
            },
        })

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  {'sigma':>6}  {'RMSE (m)':>10}  {'Crashes':>9}  {'MeanLen':>8}  {'AvgCrashStep':>13}")
    print(f"  {'─'*60}")
    for r in all_results:
        crash_s = f"{r['mean_crash_step']:.1f}" if not np.isnan(r['mean_crash_step']) else "   N/A"
        print(f"  {r['temperature']:>6.2f}  "
              f"{r['mean_rmse']:>7.4f}+/-{r['std_rmse']:.4f}  "
              f"{r['crashes']:>3}/{r['n_episodes']:<3}     "
              f"{r['mean_ep_length']:>8.1f}  {crash_s:>13}")
    print(f"{'='*65}")

    print("\n  Reference (no temperature scaling):")
    print("  ReinFlow Run 10 (sigma=1.0): RMSE=0.3005m, Crashes=50/50, AvgLen~36")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save = {
        'checkpoint': args.flow_model,
        'n_episodes':       args.episodes,
        'seed':             args.seed,
        'n_inference_steps': args.n_inference_steps,
        'results':          all_results,
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(save, f, indent=2)
    print(f"\n  Results saved -> {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temperature scaling ablation for ReinFlow v4")

    parser.add_argument('--flow-model',        type=str, required=True,
                        help='Path to ReinFlow checkpoint (.pt)')
    parser.add_argument('--flow-config',       type=str,
                        default='configs/flow_policy_v4.yaml')
    parser.add_argument('--quadrotor-config',  type=str,
                        default='configs/quadrotor_v4.yaml')
    parser.add_argument('--temperatures',      type=float, nargs='+',
                        default=[1.0, 0.7, 0.5, 0.3],
                        help='Noise scale values to test (sigma in N(0,sigma^2 I))')
    parser.add_argument('--episodes',          type=int, default=50)
    parser.add_argument('--seed',              type=int, default=42)
    parser.add_argument('--n-inference-steps', type=int, default=1)
    parser.add_argument('--output',            type=str,
                        default='evaluation_results/temperature_scaling/results.json')

    args = parser.parse_args()
    main(args)

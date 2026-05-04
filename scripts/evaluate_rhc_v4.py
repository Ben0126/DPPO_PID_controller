"""
Phase 3a Gate: Closed-Loop RHC Evaluation — v4.0 (Flow Matching + CTBR + INDI)

Evaluates FlowMatchingPolicyV4 on QuadrotorEnvV4 with RHC loop.
Uses 1-step Euler inference (n_inference_steps=1) by default.

Usage:
    python -m scripts.evaluate_rhc_v4 \
        --flow-model checkpoints/flow_policy_v4/20260420_034314/best_model.pt \
        --ppo-model  checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
        --ppo-norm   checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz

Gate: RMSE < 0.15m (BC eval, no RL fine-tuning yet)
"""

import os
import sys
import argparse
import yaml
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env_v4 import QuadrotorEnvV4
from envs.quadrotor_visual_env import QuadrotorVisualEnv
from models.flow_policy_v4 import FlowMatchingPolicyV4
from models.ppo_expert import PPOExpert, RunningMeanStd


def evaluate_flow(env: QuadrotorVisualEnv, base_env: QuadrotorEnvV4,
                  policy: FlowMatchingPolicyV4,
                  n_episodes: int, T_obs: int, T_action: int,
                  device: torch.device,
                  n_inference_steps: int = 1) -> Dict:
    """Evaluate FlowMatchingPolicyV4 with RHC in closed loop."""
    results = {
        'rewards': [], 'lengths': [], 'crashes': 0,
        'position_rmse': [], 'inference_times': [],
        'trajectories': [],
    }

    for ep in range(n_episodes):
        obs, _       = env.reset()
        image_buffer = [obs['image']] * T_obs
        ep_reward    = 0.0
        ep_length    = 0
        positions    = []
        targets      = []
        done         = False

        while not done:
            img_stack  = np.concatenate(image_buffer[-T_obs:], axis=0)
            # Normalise [0,255] → [0,1]
            img_tensor = torch.from_numpy(img_stack).float().unsqueeze(0).to(device) / 255.0

            imu_vec    = base_env.get_imu()
            imu_tensor = torch.from_numpy(imu_vec).float().unsqueeze(0).to(device)

            t_start = time.perf_counter()
            with torch.no_grad():
                action_seq = policy.predict_action(
                    img_tensor, imu_tensor, n_steps=n_inference_steps)
            results['inference_times'].append((time.perf_counter() - t_start) * 1000)

            # action_seq: (1, action_dim, T_pred)  →  (T_pred, action_dim)
            action_seq = action_seq.squeeze(0).T.cpu().numpy()

            for a_idx in range(min(T_action, action_seq.shape[0])):
                action = action_seq[a_idx]
                obs, reward, terminated, truncated, info = env.step(action)
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
        if ep_length < env.env.max_episode_steps:
            results['crashes'] += 1

        pos_errors = np.array(targets) - np.array(positions)
        rmse = np.sqrt(np.mean(np.sum(pos_errors**2, axis=1)))
        results['position_rmse'].append(rmse)
        results['trajectories'].append(np.array(positions))

        print(f"  Ep {ep+1:>3}/{n_episodes} | "
              f"reward={ep_reward:.1f} | RMSE={rmse:.4f}m | "
              f"steps={ep_length} | "
              f"{'CRASH' if ep_length < env.env.max_episode_steps else 'OK'}")

    return results


def evaluate_ppo_expert(base_env: QuadrotorEnvV4, agent: PPOExpert,
                        obs_rms: RunningMeanStd, n_episodes: int) -> Dict:
    results = {
        'rewards': [], 'lengths': [], 'crashes': 0,
        'position_rmse': [], 'trajectories': [],
    }
    for ep in range(n_episodes):
        state, _ = base_env.reset()
        ep_reward, ep_length = 0.0, 0
        positions, targets, done = [], [], False
        while not done:
            action = agent.get_action_deterministic(obs_rms.normalize(state))
            state, reward, terminated, truncated, info = base_env.step(action)
            ep_reward += reward
            ep_length += 1
            positions.append(info['position'].copy())
            targets.append(info['target'].copy())
            done = terminated or truncated
        results['rewards'].append(ep_reward)
        results['lengths'].append(ep_length)
        if ep_length < base_env.max_episode_steps:
            results['crashes'] += 1
        pos_errors = np.array(targets) - np.array(positions)
        results['position_rmse'].append(
            np.sqrt(np.mean(np.sum(pos_errors**2, axis=1))))
        results['trajectories'].append(np.array(positions))
    return results


def plot_results(flow_results: Dict, ppo_results: Optional[Dict], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    labels  = ['Flow v4.0 (BC)']
    rewards = [np.mean(flow_results['rewards'])]
    rmses   = [np.mean(flow_results['position_rmse'])]
    crashes = [flow_results['crashes']]
    r_stds  = [np.std(flow_results['rewards'])]
    rmse_stds = [np.std(flow_results['position_rmse'])]

    if ppo_results:
        labels.append('PPO Expert v4')
        rewards.append(np.mean(ppo_results['rewards']))
        rmses.append(np.mean(ppo_results['position_rmse']))
        crashes.append(ppo_results['crashes'])
        r_stds.append(np.std(ppo_results['rewards']))
        rmse_stds.append(np.std(ppo_results['position_rmse']))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(labels, rewards, yerr=r_stds, capsize=5)
    axes[0].set_title('Mean Episode Reward')
    axes[1].bar(labels, rmses, yerr=rmse_stds, capsize=5)
    axes[1].axhline(0.15, color='r', linestyle='--', label='Gate 0.15m')
    axes[1].set_title('Position RMSE (m)')
    axes[1].legend()
    axes[2].bar(labels, crashes)
    axes[2].set_title('Crash Count')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_v4.png'), dpi=150)
    plt.close()

    if flow_results['inference_times']:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(flow_results['inference_times'], bins=50, edgecolor='black')
        ax.axvline(20, color='r', linestyle='--', label='20ms control period')
        ax.set_xlabel('Inference Time (ms)')
        ax.set_title('Flow Matching Inference Time (v4.0)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'inference_time_v4.png'), dpi=150)
        plt.close()

    print(f"Plots saved to: {save_dir}")


def evaluate(args):
    with open(args.flow_config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vis_cfg = cfg['vision']
    act_cfg = cfg['action']
    flow_cfg = cfg['flow']

    print("\n=== Evaluating FlowMatchingPolicyV4 (BC, RHC) ===")
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

    n = args.n_episodes
    flow_results = evaluate_flow(
        visual_env, base_env, policy, n,
        T_obs=vis_cfg['T_obs'],
        T_action=act_cfg['T_action'],
        device=device,
        n_inference_steps=args.n_inference_steps,
    )

    mean_rmse   = np.mean(flow_results['position_rmse'])
    mean_reward = np.mean(flow_results['rewards'])
    crashes     = flow_results['crashes']
    mean_infer  = np.mean(flow_results['inference_times'])

    print(f"\n--- Flow v4.0 BC Results ({n} episodes) ---")
    print(f"  Mean reward:    {mean_reward:.2f} (+/- {np.std(flow_results['rewards']):.2f})")
    print(f"  Position RMSE:  {mean_rmse:.4f} m")
    print(f"  Crashes:        {crashes}/{n}")
    print(f"  Inference time: {mean_infer:.1f} ms (median {np.median(flow_results['inference_times']):.1f} ms)")

    gate_pass = mean_rmse < 0.15 and crashes < n
    print(f"\n  Phase 3a Gate (RMSE < 0.15m): {'PASS' if gate_pass else 'FAIL'}")

    ppo_results = None
    if args.ppo_model:
        print("\n=== Evaluating PPO Expert v4 ===")
        ppo_base  = QuadrotorEnvV4(config_path=args.quadrotor_config)
        state_dim = ppo_base.observation_space.shape[0]
        act_dim   = ppo_base.action_space.shape[0]
        agent     = PPOExpert(state_dim=state_dim, action_dim=act_dim, hidden_dim=256)
        agent.load(args.ppo_model)
        obs_rms = RunningMeanStd(shape=(state_dim,))
        if args.ppo_norm:
            d = np.load(args.ppo_norm)
            obs_rms.load_state_dict(
                {'mean': d['mean'], 'var': d['var'], 'count': float(d['count'])})
        ppo_results = evaluate_ppo_expert(ppo_base, agent, obs_rms, n)
        print(f"  Mean reward:   {np.mean(ppo_results['rewards']):.2f}")
        print(f"  Position RMSE: {np.mean(ppo_results['position_rmse']):.4f} m")
        print(f"  Crashes:       {ppo_results['crashes']}/{n}")

    plot_results(flow_results, ppo_results, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FlowMatchingPolicyV4 (BC gate)")
    parser.add_argument('--flow-model',        type=str, required=True)
    parser.add_argument('--flow-config',       type=str, default='configs/flow_policy_v4.yaml')
    parser.add_argument('--quadrotor-config',  type=str, default='configs/quadrotor_v4.yaml')
    parser.add_argument('--ppo-model',         type=str, default=None)
    parser.add_argument('--ppo-norm',          type=str, default=None)
    parser.add_argument('--n-episodes',        type=int, default=50)
    parser.add_argument('--n-inference-steps', type=int, default=1,
                        help='Euler steps for flow inference (1=single-step OT)')
    parser.add_argument('--output-dir',        type=str,
                        default='evaluation_results/rhc_v4/')
    args = parser.parse_args()
    evaluate(args)

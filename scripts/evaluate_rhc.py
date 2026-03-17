"""
Phase 4: Closed-Loop RHC Evaluation

Evaluates the Vision Diffusion Policy using Receding Horizon Control:
  - Predict T_pred future actions
  - Execute first T_action actions
  - Re-observe and repeat

Compares against PPO expert baseline on the same scenarios.

Usage:
    python -m scripts.evaluate_rhc \
        --diffusion-model checkpoints/diffusion_policy/.../best_model.pt \
        --ppo-model checkpoints/ppo_expert/.../best_model.pt \
        --ppo-norm checkpoints/ppo_expert/.../best_obs_rms.npz
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
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env import QuadrotorEnv
from envs.quadrotor_visual_env import QuadrotorVisualEnv
from models.diffusion_policy import VisionDiffusionPolicy
from models.ppo_expert import PPOExpert, RunningMeanStd


def evaluate_diffusion(env: QuadrotorVisualEnv, policy: VisionDiffusionPolicy,
                       n_episodes: int, T_obs: int, T_action: int,
                       device: torch.device) -> Dict:
    """Evaluate diffusion policy with RHC."""
    results = {
        'rewards': [], 'lengths': [], 'crashes': 0,
        'position_rmse': [], 'inference_times': [],
        'trajectories': [],
    }

    for ep in range(n_episodes):
        obs, _ = env.reset()
        image_buffer = [obs['image']] * T_obs
        ep_reward = 0.0
        ep_length = 0
        positions = []
        targets = []
        done = False

        while not done:
            # Build image stack
            img_stack = np.concatenate(image_buffer[-T_obs:], axis=0)
            img_tensor = torch.from_numpy(img_stack).float().unsqueeze(0).to(device)

            # Predict action sequence (measure inference time)
            t_start = time.perf_counter()
            with torch.no_grad():
                action_seq = policy.predict_action(img_tensor)
            t_end = time.perf_counter()
            results['inference_times'].append((t_end - t_start) * 1000)

            action_seq = action_seq.squeeze(0).cpu().numpy()  # (T_pred, action_dim)

            # Execute T_action steps (RHC)
            for a_idx in range(min(T_action, len(action_seq))):
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

        # Position RMSE
        pos_errors = np.array(targets) - np.array(positions)
        rmse = np.sqrt(np.mean(np.sum(pos_errors**2, axis=1)))
        results['position_rmse'].append(rmse)
        results['trajectories'].append(np.array(positions))

    return results


def evaluate_ppo_expert(env: QuadrotorEnv, agent: PPOExpert,
                        obs_rms: RunningMeanStd, n_episodes: int) -> Dict:
    """Evaluate PPO expert baseline (no visual wrapper)."""
    results = {
        'rewards': [], 'lengths': [], 'crashes': 0,
        'position_rmse': [], 'trajectories': [],
    }

    for ep in range(n_episodes):
        state, _ = env.reset()
        state_norm = obs_rms.normalize(state)
        ep_reward = 0.0
        ep_length = 0
        positions = []
        targets = []
        done = False

        while not done:
            action = agent.get_action_deterministic(state_norm)
            state, reward, terminated, truncated, info = env.step(action)
            state_norm = obs_rms.normalize(state)
            ep_reward += reward
            ep_length += 1
            positions.append(info['position'].copy())
            targets.append(info['target'].copy())
            done = terminated or truncated

        results['rewards'].append(ep_reward)
        results['lengths'].append(ep_length)
        if ep_length < env.max_episode_steps:
            results['crashes'] += 1

        pos_errors = np.array(targets) - np.array(positions)
        rmse = np.sqrt(np.mean(np.sum(pos_errors**2, axis=1)))
        results['position_rmse'].append(rmse)
        results['trajectories'].append(np.array(positions))

    return results


def plot_results(diff_results: Dict, ppo_results: Dict, save_dir: str):
    """Generate comparison plots."""
    os.makedirs(save_dir, exist_ok=True)

    # 1. Reward comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(['Diffusion RHC', 'PPO Expert'],
                [np.mean(diff_results['rewards']), np.mean(ppo_results['rewards'])],
                yerr=[np.std(diff_results['rewards']), np.std(ppo_results['rewards'])],
                capsize=5)
    axes[0].set_title('Mean Episode Reward')
    axes[0].set_ylabel('Reward')

    axes[1].bar(['Diffusion RHC', 'PPO Expert'],
                [np.mean(diff_results['position_rmse']),
                 np.mean(ppo_results['position_rmse'])],
                yerr=[np.std(diff_results['position_rmse']),
                      np.std(ppo_results['position_rmse'])],
                capsize=5)
    axes[1].set_title('Position RMSE (m)')
    axes[1].set_ylabel('RMSE')

    axes[2].bar(['Diffusion RHC', 'PPO Expert'],
                [diff_results['crashes'], ppo_results['crashes']])
    axes[2].set_title('Crash Count')
    axes[2].set_ylabel('Crashes')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison.png'), dpi=150)
    plt.close()

    # 2. 3D trajectory plot (first episode)
    if diff_results['trajectories'] and ppo_results['trajectories']:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        traj_d = diff_results['trajectories'][0]
        traj_p = ppo_results['trajectories'][0]

        ax.plot(traj_d[:, 0], traj_d[:, 1], -traj_d[:, 2],
                label='Diffusion RHC', alpha=0.8)
        ax.plot(traj_p[:, 0], traj_p[:, 1], -traj_p[:, 2],
                label='PPO Expert', alpha=0.8)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Altitude (m)')
        ax.set_title('3D Flight Trajectory')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'trajectory_3d.png'), dpi=150)
        plt.close()

    # 3. Inference time histogram
    if diff_results['inference_times']:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(diff_results['inference_times'], bins=50, edgecolor='black')
        ax.axvline(50, color='r', linestyle='--', label='50ms target')
        ax.set_xlabel('Inference Time (ms)')
        ax.set_ylabel('Count')
        ax.set_title('DDIM Inference Time Distribution')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'inference_time.png'), dpi=150)
        plt.close()

    print(f"Plots saved to: {save_dir}")


def evaluate(args):
    with open(args.diffusion_config, 'r', encoding='utf-8') as f:
        diff_config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vision_cfg = diff_config['vision']
    action_cfg = diff_config['action']

    n_episodes = args.n_episodes

    # --- Evaluate Diffusion Policy ---
    print("\n=== Evaluating Diffusion Policy (RHC) ===")
    base_env = QuadrotorEnv(config_path=args.quadrotor_config)
    visual_env = QuadrotorVisualEnv(base_env, image_size=vision_cfg['image_size'])

    policy = VisionDiffusionPolicy(
        action_dim=action_cfg['action_dim'],
        T_obs=vision_cfg['T_obs'],
        T_pred=action_cfg['T_pred'],
        image_channels=vision_cfg['channels'],
        image_size=vision_cfg['image_size'],
        feature_dim=vision_cfg['feature_dim'],
        time_embed_dim=diff_config['unet']['time_embed_dim'],
        down_dims=tuple(diff_config['unet']['down_dims']),
        num_diffusion_steps=diff_config['diffusion']['num_timesteps'],
        beta_schedule=diff_config['diffusion']['beta_schedule'],
        ddim_steps=diff_config['diffusion']['ddim_steps'],
    ).to(device)
    policy.load(args.diffusion_model)
    policy.eval()

    diff_results = evaluate_diffusion(
        visual_env, policy, n_episodes,
        T_obs=vision_cfg['T_obs'],
        T_action=action_cfg['T_action'],
        device=device,
    )

    print(f"  Mean reward: {np.mean(diff_results['rewards']):.2f} "
          f"(+/- {np.std(diff_results['rewards']):.2f})")
    print(f"  Position RMSE: {np.mean(diff_results['position_rmse']):.4f} m")
    print(f"  Crashes: {diff_results['crashes']}/{n_episodes}")
    print(f"  Mean inference: {np.mean(diff_results['inference_times']):.1f} ms")

    # --- Evaluate PPO Expert (if provided) ---
    ppo_results = {'rewards': [0], 'position_rmse': [0], 'crashes': 0, 'trajectories': []}
    if args.ppo_model:
        print("\n=== Evaluating PPO Expert ===")
        ppo_env = QuadrotorEnv(config_path=args.quadrotor_config)
        state_dim = ppo_env.observation_space.shape[0]
        action_dim = ppo_env.action_space.shape[0]

        ppo_agent = PPOExpert(state_dim=state_dim, action_dim=action_dim,
                              hidden_dim=args.ppo_hidden_dim)
        ppo_agent.load(args.ppo_model)

        obs_rms = RunningMeanStd(shape=(state_dim,))
        if args.ppo_norm:
            norm_data = np.load(args.ppo_norm)
            obs_rms.load_state_dict({
                'mean': norm_data['mean'], 'var': norm_data['var'],
                'count': float(norm_data['count']),
            })

        ppo_results = evaluate_ppo_expert(ppo_env, ppo_agent, obs_rms, n_episodes)

        print(f"  Mean reward: {np.mean(ppo_results['rewards']):.2f} "
              f"(+/- {np.std(ppo_results['rewards']):.2f})")
        print(f"  Position RMSE: {np.mean(ppo_results['position_rmse']):.4f} m")
        print(f"  Crashes: {ppo_results['crashes']}/{n_episodes}")

    # Plot comparison
    plot_results(diff_results, ppo_results, args.output_dir)

    # Performance ratio
    if args.ppo_model and np.mean(ppo_results['rewards']) != 0:
        ratio = np.mean(diff_results['rewards']) / np.mean(ppo_results['rewards'])
        print(f"\n  Diffusion/PPO performance ratio: {ratio:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Policy with RHC")
    parser.add_argument('--diffusion-model', type=str, required=True)
    parser.add_argument('--diffusion-config', type=str, default='configs/diffusion_policy.yaml')
    parser.add_argument('--quadrotor-config', type=str, default='configs/quadrotor.yaml')
    parser.add_argument('--ppo-model', type=str, default=None)
    parser.add_argument('--ppo-norm', type=str, default=None)
    parser.add_argument('--ppo-hidden-dim', type=int, default=256)
    parser.add_argument('--n-episodes', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='evaluation_results/rhc/')
    args = parser.parse_args()
    evaluate(args)

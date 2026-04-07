"""
Phase 4: Closed-Loop RHC Evaluation — Architecture v3.1

Evaluates VisionDPPOv31 (IMU Late Fusion + FCN Aux Depth) with RHC.
IMU is computed from state observations via finite-difference (same as
train_dppo_v31.py) so the visual env does not need changes.

Usage:
    python -m scripts.evaluate_rhc_v31 \
        --diffusion-model checkpoints/diffusion_policy/v31_<timestamp>/best_model.pt \
        --ppo-model checkpoints/ppo_expert/20260401_103107/best_model.pt \
        --ppo-norm  checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz
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
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env import QuadrotorEnv
from envs.quadrotor_visual_env import QuadrotorVisualEnv
from models.vision_dppo_v31 import VisionDPPOv31
from models.ppo_expert import PPOExpert, RunningMeanStd


def _get_imu(obs_state: np.ndarray, prev_v_body: np.ndarray,
             dt: float = 0.02) -> np.ndarray:
    """6D IMU from state observation (same as train_dppo_v31.py)."""
    omega  = obs_state[12:15]
    v_body = obs_state[9:12]
    accel  = np.zeros(3, dtype=np.float32) if prev_v_body is None \
             else ((v_body - prev_v_body) / dt).astype(np.float32)
    return np.concatenate([omega, accel]).astype(np.float32)


def evaluate_v31(env: QuadrotorVisualEnv, policy: VisionDPPOv31,
                 n_episodes: int, T_obs: int, T_action: int,
                 device: torch.device) -> Dict:
    """Evaluate VisionDPPOv31 policy with RHC."""
    results = {
        'rewards': [], 'lengths': [], 'crashes': 0,
        'position_rmse': [], 'inference_times': [],
        'trajectories': [],
    }

    for ep in range(n_episodes):
        obs, _       = env.reset()
        image_buffer = [obs['image']] * T_obs
        prev_v_body  = None
        ep_reward    = 0.0
        ep_length    = 0
        positions    = []
        targets      = []
        done         = False

        while not done:
            img_stack  = np.concatenate(image_buffer[-T_obs:], axis=0)
            img_tensor = torch.from_numpy(img_stack).float().unsqueeze(0).to(device)

            imu_vec    = _get_imu(obs['state'], prev_v_body)
            imu_tensor = torch.from_numpy(imu_vec).float().unsqueeze(0).to(device)
            prev_v_body = obs['state'][9:12].copy()

            t_start = time.perf_counter()
            with torch.no_grad():
                action_seq = policy.predict_action(img_tensor, imu_tensor)
            results['inference_times'].append((time.perf_counter() - t_start) * 1000)

            action_seq = action_seq.squeeze(0).cpu().numpy()  # (T_pred, 4)

            for a_idx in range(min(T_action, len(action_seq))):
                action = action_seq[a_idx]
                obs, reward, terminated, truncated, info = env.step(action)
                image_buffer.append(obs['image'])
                ep_reward  += reward
                ep_length  += 1
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


def evaluate_ppo_expert(env: QuadrotorEnv, agent: PPOExpert,
                        obs_rms: RunningMeanStd, n_episodes: int) -> Dict:
    results = {
        'rewards': [], 'lengths': [], 'crashes': 0,
        'position_rmse': [], 'trajectories': [],
    }
    for ep in range(n_episodes):
        state, _ = env.reset()
        ep_reward, ep_length = 0.0, 0
        positions, targets, done = [], [], False
        while not done:
            action = agent.get_action_deterministic(obs_rms.normalize(state))
            state, reward, terminated, truncated, info = env.step(action)
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
        results['position_rmse'].append(
            np.sqrt(np.mean(np.sum(pos_errors**2, axis=1))))
        results['trajectories'].append(np.array(positions))
    return results


def plot_results(diff_results: Dict, ppo_results: Dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(['v3.1 RHC', 'PPO Expert'],
                [np.mean(diff_results['rewards']), np.mean(ppo_results['rewards'])],
                yerr=[np.std(diff_results['rewards']), np.std(ppo_results['rewards'])],
                capsize=5)
    axes[0].set_title('Mean Episode Reward')

    axes[1].bar(['v3.1 RHC', 'PPO Expert'],
                [np.mean(diff_results['position_rmse']),
                 np.mean(ppo_results['position_rmse'])],
                yerr=[np.std(diff_results['position_rmse']),
                      np.std(ppo_results['position_rmse'])],
                capsize=5)
    axes[1].set_title('Position RMSE (m)')

    axes[2].bar(['v3.1 RHC', 'PPO Expert'],
                [diff_results['crashes'], ppo_results['crashes']])
    axes[2].set_title('Crash Count')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_v31.png'), dpi=150)
    plt.close()

    if diff_results['inference_times']:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(diff_results['inference_times'], bins=50, edgecolor='black')
        ax.axvline(50, color='r', linestyle='--', label='50ms target')
        ax.set_xlabel('Inference Time (ms)')
        ax.set_title('DDIM Inference Time Distribution (v3.1)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'inference_time_v31.png'), dpi=150)
        plt.close()

    print(f"Plots saved to: {save_dir}")


def evaluate(args):
    with open(args.diffusion_config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_cfg = cfg['vision']
    action_cfg = cfg['action']

    print("\n=== Evaluating VisionDPPOv31 (RHC) ===")
    base_env   = QuadrotorEnv(config_path=args.quadrotor_config)
    visual_env = QuadrotorVisualEnv(base_env, image_size=vision_cfg['image_size'])

    policy = VisionDPPOv31(
        action_dim=action_cfg['action_dim'],
        T_obs=vision_cfg['T_obs'],
        T_pred=action_cfg['T_pred'],
        image_channels=vision_cfg['channels'],
        image_size=vision_cfg['image_size'],
        time_embed_dim=cfg['unet']['time_embed_dim'],
        down_dims=tuple(cfg['unet']['down_dims']),
        num_diffusion_steps=cfg['diffusion']['num_timesteps'],
        beta_schedule=cfg['diffusion']['beta_schedule'],
        ddim_steps=cfg['diffusion']['ddim_steps'],
        use_depth_decoder=False,   # stripped for inference
    ).to(device)
    policy.load(args.diffusion_model)
    policy.eval()

    diff_results = evaluate_v31(
        visual_env, policy, args.n_episodes,
        T_obs=vision_cfg['T_obs'],
        T_action=action_cfg['T_action'],
        device=device,
    )

    n = args.n_episodes
    print(f"\n--- v3.1 Results ({n} episodes) ---")
    print(f"  Mean reward:    {np.mean(diff_results['rewards']):.2f} "
          f"(+/- {np.std(diff_results['rewards']):.2f})")
    print(f"  Position RMSE:  {np.mean(diff_results['position_rmse']):.4f} m")
    print(f"  Crashes:        {diff_results['crashes']}/{n}")
    print(f"  Inference time: {np.mean(diff_results['inference_times']):.1f} ms "
          f"(median {np.median(diff_results['inference_times']):.1f} ms)")

    ppo_results = {'rewards': [0], 'position_rmse': [0], 'crashes': 0, 'trajectories': []}
    if args.ppo_model:
        print("\n=== Evaluating PPO Expert ===")
        ppo_env   = QuadrotorEnv(config_path=args.quadrotor_config)
        state_dim = ppo_env.observation_space.shape[0]
        act_dim   = ppo_env.action_space.shape[0]
        agent     = PPOExpert(state_dim=state_dim, action_dim=act_dim,
                              hidden_dim=args.ppo_hidden_dim)
        agent.load(args.ppo_model)
        obs_rms = RunningMeanStd(shape=(state_dim,))
        if args.ppo_norm:
            d = np.load(args.ppo_norm)
            obs_rms.load_state_dict(
                {'mean': d['mean'], 'var': d['var'], 'count': float(d['count'])})
        ppo_results = evaluate_ppo_expert(ppo_env, agent, obs_rms, n)
        print(f"  Mean reward:   {np.mean(ppo_results['rewards']):.2f}")
        print(f"  Position RMSE: {np.mean(ppo_results['position_rmse']):.4f} m")
        print(f"  Crashes:       {ppo_results['crashes']}/{n}")

    plot_results(diff_results, ppo_results, args.output_dir)

    if args.ppo_model and np.mean(ppo_results['rewards']) != 0:
        ratio = np.mean(diff_results['rewards']) / np.mean(ppo_results['rewards'])
        print(f"\n  Diffusion/PPO performance ratio: {ratio:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VisionDPPOv31 with RHC")
    parser.add_argument('--diffusion-model',  type=str, required=True)
    parser.add_argument('--diffusion-config', type=str,
                        default='configs/diffusion_policy.yaml')
    parser.add_argument('--quadrotor-config', type=str,
                        default='configs/quadrotor.yaml')
    parser.add_argument('--ppo-model',        type=str, default=None)
    parser.add_argument('--ppo-norm',         type=str, default=None)
    parser.add_argument('--ppo-hidden-dim',   type=int, default=256)
    parser.add_argument('--n-episodes',       type=int, default=50)
    parser.add_argument('--output-dir',       type=str,
                        default='evaluation_results/rhc_v31/')
    args = parser.parse_args()
    evaluate(args)

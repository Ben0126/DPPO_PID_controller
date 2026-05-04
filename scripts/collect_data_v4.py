"""
Phase 2 v4.0: Expert Data Collection (CTBR + INDI)

Rolls out the CTBR PPO v4 expert in the visual environment and saves
(image, ctbr_action, state, imu, depth) trajectories to HDF5.

The visual rendering is identical to v3.3 — QuadrotorVisualEnv wraps
QuadrotorEnvV4 directly (only dynamics + target_position are used for rendering).

Usage:
    python -m scripts.collect_data_v4 \
        --model  checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
        --norm   checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz \
        --output data/expert_demos_v4.h5
"""

import os
import sys
import argparse
import numpy as np
import h5py
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env_v4 import QuadrotorEnvV4
from envs.quadrotor_visual_env import QuadrotorVisualEnv
from models.ppo_expert import PPOExpert, RunningMeanStd


def collect_data(args):
    base_env = QuadrotorEnvV4(config_path=args.quadrotor_config)
    env = QuadrotorVisualEnv(base_env, image_size=args.image_size, dr_enabled=True)

    state_dim  = base_env.observation_space.shape[0]
    action_dim = base_env.action_space.shape[0]

    agent = PPOExpert(state_dim=state_dim, action_dim=action_dim,
                      hidden_dim=args.hidden_dim)
    agent.load(args.model)
    print(f"Loaded PPO expert v4 from: {args.model}")

    obs_rms = RunningMeanStd(shape=(state_dim,))
    if args.norm:
        nd = np.load(args.norm)
        obs_rms.load_state_dict({
            'mean':  nd['mean'],
            'var':   nd['var'],
            'count': float(nd['count']),
        })
        print(f"Loaded obs normalization from: {args.norm}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    total_steps = 0
    ep_lengths  = []
    crashes     = 0

    with h5py.File(args.output, 'w') as hf:
        for ep in tqdm(range(args.n_episodes), desc="Collecting episodes"):
            obs, _ = env.reset()
            state_norm = obs_rms.normalize(obs['state'])

            images    = []
            actions   = []
            states    = []
            imu_data  = []
            depth_maps = []
            done = False

            while not done:
                action = agent.get_action_deterministic(state_norm)

                images.append(obs['image'])           # (3, H, W) uint8
                actions.append(action)                # (4,) float32, CTBR [-1,1]
                states.append(obs['state'])           # (15,) float32

                imu_vec = base_env.get_imu()          # (6,) normalized physics IMU
                imu_data.append(imu_vec.astype(np.float32))
                depth_maps.append(env._render_depth())  # (1, H, W) uint8

                obs, reward, terminated, truncated, info = env.step(action)
                state_norm = obs_rms.normalize(obs['state'])
                done = terminated or truncated

                if terminated:
                    crashes += 1

            ep_grp = hf.create_group(f'episode_{ep}')
            ep_grp.create_dataset('images',
                data=np.array(images, dtype=np.uint8),
                compression='gzip', compression_opts=4)
            ep_grp.create_dataset('actions',
                data=np.array(actions, dtype=np.float32))
            ep_grp.create_dataset('states',
                data=np.array(states, dtype=np.float32))
            ep_grp.create_dataset('imu_data',
                data=np.array(imu_data, dtype=np.float32))
            ep_grp.create_dataset('depth_maps',
                data=np.array(depth_maps, dtype=np.uint8),
                compression='gzip', compression_opts=4)

            ep_lengths.append(len(actions))
            total_steps += len(actions)

        hf.attrs['n_episodes']  = args.n_episodes
        hf.attrs['total_steps'] = total_steps
        hf.attrs['image_size']  = args.image_size
        hf.attrs['state_dim']   = state_dim
        hf.attrs['action_dim']  = action_dim
        hf.attrs['action_space'] = 'ctbr'
        hf.attrs['version']     = 'v4'

    print(f"\nData collection complete!")
    print(f"  Episodes:    {args.n_episodes}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Mean ep len: {np.mean(ep_lengths):.1f} steps")
    print(f"  Crashes:     {crashes}/{args.n_episodes}")
    print(f"  Saved to:    {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect CTBR expert demo data (Phase 2 v4.0)")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to PPO v4 expert checkpoint')
    parser.add_argument('--norm', type=str, default=None,
                        help='Path to observation normalization stats')
    parser.add_argument('--quadrotor-config', type=str,
                        default='configs/quadrotor_v4.yaml')
    parser.add_argument('--output', type=str,
                        default='data/expert_demos_v4.h5')
    parser.add_argument('--n-episodes', type=int, default=1000)
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=256)
    args = parser.parse_args()
    collect_data(args)

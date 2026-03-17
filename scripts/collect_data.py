"""
Phase 2: Expert Data Collection

Rolls out a trained PPO expert in the visual environment and saves
(image, action, state) trajectories to HDF5 for Diffusion Policy training.

Usage:
    python -m scripts.collect_data --model checkpoints/ppo_expert/.../best_model.pt \
                                   --norm checkpoints/ppo_expert/.../best_obs_rms.npz
"""

import os
import sys
import argparse
import numpy as np
import h5py
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env import QuadrotorEnv
from envs.quadrotor_visual_env import QuadrotorVisualEnv
from models.ppo_expert import PPOExpert, RunningMeanStd


def collect_data(args):
    # Create visual environment
    base_env = QuadrotorEnv(config_path=args.quadrotor_config)
    env = QuadrotorVisualEnv(base_env, image_size=args.image_size)

    state_dim = base_env.observation_space.shape[0]
    action_dim = base_env.action_space.shape[0]

    # Load PPO expert
    agent = PPOExpert(state_dim=state_dim, action_dim=action_dim,
                      hidden_dim=args.hidden_dim)
    agent.load(args.model)
    print(f"Loaded PPO expert from: {args.model}")

    # Load observation normalization
    obs_rms = RunningMeanStd(shape=(state_dim,))
    if args.norm:
        norm_data = np.load(args.norm)
        obs_rms.load_state_dict({
            'mean': norm_data['mean'],
            'var': norm_data['var'],
            'count': float(norm_data['count']),
        })
        print(f"Loaded obs normalization from: {args.norm}")

    # Collect episodes
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    total_steps = 0
    with h5py.File(args.output, 'w') as hf:
        for ep in tqdm(range(args.n_episodes), desc="Collecting episodes"):
            obs, _ = env.reset()
            state_norm = obs_rms.normalize(obs['state'])

            images = []
            actions = []
            states = []
            done = False

            while not done:
                # Get deterministic action from expert
                action = agent.get_action_deterministic(state_norm)

                images.append(obs['image'])
                actions.append(action)
                states.append(obs['state'])

                obs, reward, terminated, truncated, info = env.step(action)
                state_norm = obs_rms.normalize(obs['state'])
                done = terminated or truncated

            # Save episode to HDF5
            ep_grp = hf.create_group(f'episode_{ep}')
            ep_grp.create_dataset('images', data=np.array(images, dtype=np.uint8),
                                  compression='gzip', compression_opts=4)
            ep_grp.create_dataset('actions', data=np.array(actions, dtype=np.float32))
            ep_grp.create_dataset('states', data=np.array(states, dtype=np.float32))

            total_steps += len(actions)

        # Save metadata
        hf.attrs['n_episodes'] = args.n_episodes
        hf.attrs['total_steps'] = total_steps
        hf.attrs['image_size'] = args.image_size
        hf.attrs['state_dim'] = state_dim
        hf.attrs['action_dim'] = action_dim

    print(f"\nData collection complete!")
    print(f"  Episodes: {args.n_episodes}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect expert demonstration data")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained PPO expert checkpoint')
    parser.add_argument('--norm', type=str, default=None,
                        help='Path to observation normalization stats')
    parser.add_argument('--quadrotor-config', type=str, default='configs/quadrotor.yaml')
    parser.add_argument('--output', type=str, default='data/expert_demos.h5')
    parser.add_argument('--n-episodes', type=int, default=1000)
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=256)
    args = parser.parse_args()
    collect_data(args)

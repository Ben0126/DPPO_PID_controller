"""
Phase 2: Expert Data Collection

Rolls out a trained PPO expert in the visual environment and saves
(image, action, state) trajectories to HDF5 for Diffusion Policy training.

Usage:
    # Standard collection (Phase 2 baseline):
    python -m scripts.collect_data --model checkpoints/ppo_expert/.../best_model.pt \
                                   --norm checkpoints/ppo_expert/.../best_obs_rms.npz

    # v3.1 collection (finite-difference IMU — deprecated, kept for reproduction):
    python -m scripts.collect_data --model checkpoints/ppo_expert/.../best_model.pt \
                                   --norm checkpoints/ppo_expert/.../best_obs_rms.npz \
                                   --output data/expert_demos_v31.h5 \
                                   --v31

    # v3.3 collection (physics-based normalized IMU via env.get_imu() + depth_maps):
    python -m scripts.collect_data --model checkpoints/ppo_expert/.../best_model.pt \
                                   --norm checkpoints/ppo_expert/.../best_obs_rms.npz \
                                   --output data/expert_demos_v33.h5 \
                                   --v33
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

    dt = 1.0 / 50.0  # 50 Hz control loop

    # Mutually exclusive sanity check
    if args.v31 and args.v33:
        raise ValueError("--v31 and --v33 are mutually exclusive; pick one.")

    with_aux = args.v31 or args.v33
    if args.v31:
        print("v3.1 mode: saving imu_data (6D, finite-difference) + depth_maps")
    elif args.v33:
        print("v3.3 mode: saving imu_data (6D, physics-based, normalized) + depth_maps")

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
            imu_data_ep   = []   # v3.1 / v3.2 only
            depth_maps_ep = []   # v3.1 / v3.2 only
            prev_v_body   = None # v3.1 finite-difference history
            done = False

            while not done:
                # Get deterministic action from expert
                action = agent.get_action_deterministic(state_norm)

                images.append(obs['image'])
                actions.append(action)
                states.append(obs['state'])

                # v3.1: finite-difference IMU (deprecated)
                if args.v31:
                    # Angular velocity: state[12:15] (body frame)
                    omega = obs['state'][12:15].copy()
                    # Linear velocity in body frame: state[9:12]
                    v_body = obs['state'][9:12].copy()
                    if prev_v_body is None:
                        accel = np.zeros(3, dtype=np.float32)
                    else:
                        accel = ((v_body - prev_v_body) / dt).astype(np.float32)
                    prev_v_body = v_body
                    imu_data_ep.append(
                        np.concatenate([omega, accel]).astype(np.float32))  # (6,)
                    depth_maps_ep.append(env._render_depth())               # (1,H,W)

                # v3.3: physics-based normalized IMU pulled straight from the env
                if args.v33:
                    imu_vec = base_env.get_imu()                            # (6,)
                    imu_data_ep.append(imu_vec.astype(np.float32))
                    depth_maps_ep.append(env._render_depth())               # (1,H,W)

                obs, reward, terminated, truncated, info = env.step(action)
                state_norm = obs_rms.normalize(obs['state'])
                done = terminated or truncated

            # Save episode to HDF5
            ep_grp = hf.create_group(f'episode_{ep}')
            ep_grp.create_dataset('images', data=np.array(images, dtype=np.uint8),
                                  compression='gzip', compression_opts=4)
            ep_grp.create_dataset('actions', data=np.array(actions, dtype=np.float32))
            ep_grp.create_dataset('states', data=np.array(states, dtype=np.float32))

            if with_aux:
                ep_grp.create_dataset('imu_data',
                    data=np.array(imu_data_ep, dtype=np.float32))       # (T, 6)
                ep_grp.create_dataset('depth_maps',
                    data=np.array(depth_maps_ep, dtype=np.uint8),
                    compression='gzip', compression_opts=4)              # (T, 1, H, W)

            total_steps += len(actions)

        # Save metadata
        hf.attrs['n_episodes'] = args.n_episodes
        hf.attrs['total_steps'] = total_steps
        hf.attrs['image_size'] = args.image_size
        hf.attrs['state_dim'] = state_dim
        hf.attrs['action_dim'] = action_dim
        hf.attrs['v31'] = args.v31
        hf.attrs['v33'] = args.v33

    print(f"\nData collection complete!")
    print(f"  Episodes: {args.n_episodes}")
    print(f"  Total steps: {total_steps:,}")
    if args.v31:
        fmt = "imu_data (finite-diff) + depth_maps"
    elif args.v33:
        fmt = "imu_data (physics, normalized) + depth_maps"
    else:
        fmt = "disabled"
    print(f"  aux fields: {fmt}")
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
    parser.add_argument('--v31', action='store_true',
                        help='Enable v3.1 format: finite-difference IMU + depth_maps (deprecated)')
    parser.add_argument('--v33', action='store_true',
                        help='Enable v3.3 format: physics-based normalized IMU via env.get_imu() + depth_maps')
    args = parser.parse_args()
    collect_data(args)

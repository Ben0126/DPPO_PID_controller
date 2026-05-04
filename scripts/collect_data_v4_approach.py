"""
Phase 2 v4.0 — Approach Demonstrations for Run 13

Collects PPO v4 expert rollouts with the drone initialised at 1.0/1.5/2.0 m
from the hover target, so the expert demonstrates fly-in-and-brake behaviour.
These are merged with the existing hover-only expert_demos_v4.h5 to provide a
broader BC anchor for ReinFlow Run 13's curriculum (pos_end=2.0m).

Per episode the script overrides base_env.initial_pos_range to one of
[1.0, 1.5, 2.0] (round-robin or uniform). Output schema is identical to
collect_data_v4.py so merge_expert_demos.py can concat without conversion.

Usage:
    python -m scripts.collect_data_v4_approach --dry-run --n-episodes 5 \
        --model checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
        --norm  checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz

    python -m scripts.collect_data_v4_approach --n-episodes 300 \
        --model checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
        --norm  checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz \
        --output data/expert_demos_v4_approach.h5
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


APPROACH_RANGES = [1.0, 1.5, 2.0]


def _parse_csv_floats(s: str):
    return [float(x.strip()) for x in s.split(',') if x.strip()]


def collect_data(args):
    base_env = QuadrotorEnvV4(config_path=args.quadrotor_config)
    env = QuadrotorVisualEnv(base_env, image_size=args.image_size, dr_enabled=True)

    pos_ranges = _parse_csv_floats(args.pos_ranges) if args.pos_ranges else APPROACH_RANGES
    vel_ranges = _parse_csv_floats(args.vel_ranges) if args.vel_ranges else None
    print(f"pos_ranges (m): {pos_ranges}")
    print(f"vel_ranges (m/s): {vel_ranges if vel_ranges else f'config default ({base_env.initial_vel_range})'}")

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

    if not args.dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        hf = h5py.File(args.output, 'w')
    else:
        hf = None
        print("[DRY RUN] Not writing to disk; reporting feasibility only.")

    rng = np.random.default_rng(args.seed)
    total_steps = 0
    ep_lengths_by_range = {r: [] for r in pos_ranges}
    crashes_by_range = {r: 0 for r in pos_ranges}

    try:
        for ep in tqdm(range(args.n_episodes), desc="Approach episodes"):
            pos_range_now = pos_ranges[ep % len(pos_ranges)] \
                if args.deterministic_range \
                else float(rng.choice(pos_ranges))
            base_env.initial_pos_range = pos_range_now
            if vel_ranges is not None:
                vel_range_now = vel_ranges[ep % len(vel_ranges)] \
                    if args.deterministic_range \
                    else float(rng.choice(vel_ranges))
                base_env.initial_vel_range = vel_range_now

            obs, _ = env.reset()
            state_norm = obs_rms.normalize(obs['state'])

            images, actions, states, imu_data, depth_maps = [], [], [], [], []
            done = False
            terminated = False

            while not done:
                action = agent.get_action_deterministic(state_norm)

                images.append(obs['image'])
                actions.append(action)
                states.append(obs['state'])
                imu_data.append(base_env.get_imu().astype(np.float32))
                depth_maps.append(env._render_depth())

                obs, reward, terminated, truncated, info = env.step(action)
                state_norm = obs_rms.normalize(obs['state'])
                done = terminated or truncated

            if terminated:
                crashes_by_range[pos_range_now] += 1
            ep_lengths_by_range[pos_range_now].append(len(actions))
            total_steps += len(actions)

            if args.verbose:
                print(f"  ep {ep}: range={pos_range_now}m steps={len(actions)} "
                      f"crashed={terminated}")

            if hf is not None:
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
                ep_grp.attrs['initial_pos_range'] = pos_range_now
                if vel_ranges is not None:
                    ep_grp.attrs['initial_vel_range'] = base_env.initial_vel_range

        if hf is not None:
            hf.attrs['n_episodes']   = args.n_episodes
            hf.attrs['total_steps']  = total_steps
            hf.attrs['image_size']   = args.image_size
            hf.attrs['state_dim']    = state_dim
            hf.attrs['action_dim']   = action_dim
            hf.attrs['action_space'] = 'ctbr'
            hf.attrs['version']      = 'v4_approach'
            hf.attrs['initial_pos_ranges'] = np.array(pos_ranges, dtype=np.float32)
            if vel_ranges is not None:
                hf.attrs['initial_vel_ranges'] = np.array(vel_ranges, dtype=np.float32)
    finally:
        if hf is not None:
            hf.close()

    print("\n=== Approach data collection summary ===")
    overall_crash = sum(crashes_by_range.values())
    for r in APPROACH_RANGES:
        n = len(ep_lengths_by_range[r])
        if n == 0:
            continue
        ml = float(np.mean(ep_lengths_by_range[r]))
        c  = crashes_by_range[r]
        print(f"  range={r}m  episodes={n:3d}  mean_len={ml:5.1f}  crashes={c}/{n}")
    print(f"  TOTAL     episodes={args.n_episodes}  total_steps={total_steps:,}  "
          f"crashes={overall_crash}/{args.n_episodes}")
    if not args.dry_run:
        print(f"  Saved to: {args.output}")

    crash_rate = overall_crash / max(args.n_episodes, 1)
    if crash_rate > 0.20:
        print(f"\n⚠️  Crash rate {crash_rate:.0%} > 20%. Consider trimming to "
              f"{APPROACH_RANGES[:-1]}m only or lowering --max-range.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect approach (1.0-2.0m) CTBR expert demos for Run 13")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--norm', type=str, default=None)
    parser.add_argument('--quadrotor-config', type=str,
                        default='configs/quadrotor_v4.yaml')
    parser.add_argument('--output', type=str,
                        default='data/expert_demos_v4_approach.h5')
    parser.add_argument('--n-episodes', type=int, default=300)
    parser.add_argument('--pos-ranges', type=str, default=None,
                        help='Comma-separated initial_pos_range list (m). Default: 1.0,1.5,2.0')
    parser.add_argument('--vel-ranges', type=str, default=None,
                        help='Comma-separated initial_vel_range list (m/s). '
                             'If set, overrides config initial_vel_range per episode. '
                             'Use 1.0,1.5,2.0 for high-speed incoming demos.')
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic-range', action='store_true',
                        help='Round-robin pos ranges instead of uniform sampling')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run episodes but skip HDF5 write (feasibility check)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-episode result')
    args = parser.parse_args()
    collect_data(args)

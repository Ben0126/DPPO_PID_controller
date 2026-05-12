"""
PID Position Controller Baseline Evaluation

Evaluates a classical 4-level cascade PID controller on the same 50-episode
waypoint-tracking protocol used for PPO Expert and ReinFlow comparisons.
No vision or learned model — uses ground-truth state from the physics engine.

Usage:
    python -m scripts.evaluate_pid_baseline
    python -m scripts.evaluate_pid_baseline --episodes 5 --seed 42   # smoke test
    python -m scripts.evaluate_pid_baseline --Kp-vel 2.0             # softer gains

Results are printed in the same format as evaluate_rhc_v33.py and saved to JSON.
"""

import os
import sys
import argparse
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env import QuadrotorEnv
from controllers.pid_controller import CascadePIDController


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pid(env: QuadrotorEnv, controller: CascadePIDController,
                 n_episodes: int) -> dict:
    results = {
        'rewards':        [],
        'lengths':        [],
        'crashes':        0,
        'position_rmse':  [],
        'step_times_us':  [],   # controller compute time per step (microseconds)
        'trajectories':   [],
    }

    for ep in range(n_episodes):
        obs, _ = env.reset()
        controller.reset()

        ep_reward  = 0.0
        ep_length  = 0
        positions  = []
        targets    = []
        done       = False

        # Initial target comes directly from env after reset
        current_target = env.target_position.copy()

        while not done:
            state = env.dynamics.state   # 13D: [pos(3), quat(4), vel(3), omega(3)]

            t0 = time.perf_counter()
            action = controller.compute_action(state, current_target)
            results['step_times_us'].append((time.perf_counter() - t0) * 1e6)

            obs, reward, terminated, truncated, info = env.step(action)

            current_target = info['target']     # update after step (may change for waypoint mode)
            ep_reward  += reward
            ep_length  += 1
            positions.append(info['position'].copy())
            targets.append(info['target'].copy())
            done = terminated or truncated

        results['rewards'].append(ep_reward)
        results['lengths'].append(ep_length)

        crashed = ep_length < env.max_episode_steps
        if crashed:
            results['crashes'] += 1

        pos_errors = np.array(targets) - np.array(positions)   # (T, 3)
        rmse = np.sqrt(np.mean(np.sum(pos_errors**2, axis=1)))  # mean Euclidean distance
        results['position_rmse'].append(float(rmse))
        results['trajectories'].append(np.array(positions).tolist())

        print(f"  Ep {ep+1:>3}/{n_episodes} | "
              f"reward={ep_reward:7.2f} | RMSE={rmse:.4f} m | "
              f"steps={ep_length:>4} | {'CRASH' if crashed else 'OK   '}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    print("\n=== PID Position Controller Baseline ===")
    print(f"  Config:    {args.config}")
    print(f"  Episodes:  {args.episodes}")
    print(f"  Seed:      {args.seed}")
    print(f"  Gains:     Kp_pos={args.Kp_pos}  Kp_vel={args.Kp_vel}  "
          f"Ki_vel={args.Ki_vel}  Kp_att={args.Kp_att}  Kp_rate={args.Kp_rate}")

    env = QuadrotorEnv(config_path=args.config)
    if args.target_type:
        env.target_type = args.target_type          # override without re-reading YAML
    if args.waypoint_range is not None:
        env.waypoint_range = args.waypoint_range
    print(f"  Target mode: {env.target_type}  waypoint_range={env.waypoint_range:.1f}m")
    env.reset(seed=args.seed)  # seed the RNG once so episodes are reproducible

    controller = CascadePIDController(
        params       = env.dynamics.params,
        Kp_pos       = args.Kp_pos,
        Kp_vel       = args.Kp_vel,
        Ki_vel       = args.Ki_vel,
        vel_int_limit= args.vel_int_limit,
        Kp_att       = args.Kp_att,
        Kp_att_yaw   = args.Kp_att_yaw,
        omega_max    = args.omega_max,
        Kp_rate      = args.Kp_rate,
        vel_max      = args.vel_max,
        dt           = env.dt_outer,
    )

    print()
    results = evaluate_pid(env, controller, args.episodes)
    n = args.episodes

    mean_rmse   = np.mean(results['position_rmse'])
    std_rmse    = np.std(results['position_rmse'])
    mean_reward = np.mean(results['rewards'])
    std_reward  = np.std(results['rewards'])
    mean_len    = np.mean(results['lengths'])
    mean_ct_us  = np.mean(results['step_times_us'])

    print(f"\n{'─'*55}")
    print(f"  Method:         Cascade PID (4-level, NED frame)")
    print(f"  Episodes:       {n}")
    print(f"  Mean reward:    {mean_reward:.2f} (+/- {std_reward:.2f})")
    print(f"  Position RMSE:  {mean_rmse:.4f} m (+/- {std_rmse:.4f} m)")
    print(f"  Crashes:        {results['crashes']}/{n}")
    print(f"  Mean ep length: {mean_len:.1f} steps")
    print(f"  Compute time:   {mean_ct_us:.1f} us/step")
    print(f"{'─'*55}")

    print(f"\n  Comparison table (same env, same eval protocol):")
    print(f"  {'Method':<32} {'RMSE':>8}  {'Crashes':>9}")
    print(f"  {'─'*52}")
    print(f"  {'PPO Expert (CTBR+INDI)':<32} {'0.065m':>8}  {'0/50':>9}")
    print(f"  {'Cascade PID (this run)':<32} {mean_rmse:.4f}m  "
          f"{results['crashes']}/{n}")
    print(f"  {'ReinFlow Run 10 (best RL eval)':<32} {'0.300m':>8}  {'50/50':>9}")
    print(f"  {'BC baseline (supervised only)':<32} {'0.522m':>8}  {'50/50':>9}")

    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_dict = {
        'method':         'CascadePID',
        'n_episodes':     n,
        'mean_reward':    float(mean_reward),
        'std_reward':     float(std_reward),
        'mean_rmse':      float(mean_rmse),
        'std_rmse':       float(std_rmse),
        'crashes':        results['crashes'],
        'mean_ep_length': float(mean_len),
        'mean_ct_us':     float(mean_ct_us),
        'per_episode': {
            'rewards':       results['rewards'],
            'lengths':       results['lengths'],
            'position_rmse': results['position_rmse'],
        },
        'gains': {
            'Kp_pos': args.Kp_pos, 'Kp_vel': args.Kp_vel, 'Ki_vel': args.Ki_vel,
            'Kp_att': args.Kp_att, 'Kp_att_yaw': args.Kp_att_yaw,
            'Kp_rate': args.Kp_rate, 'vel_max': args.vel_max,
            'omega_max': args.omega_max,
        },
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(save_dict, f, indent=2)
    print(f"\n  Results saved → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cascade PID baseline")

    # Environment
    parser.add_argument('--config',        type=str, default='configs/quadrotor.yaml')
    parser.add_argument('--target-type',   type=str, default=None,
                        choices=['hover', 'waypoint'],
                        help='Override target_type from config (hover|waypoint)')
    parser.add_argument('--waypoint-range',type=float, default=None,
                        help='Override waypoint_range (e.g. 2.0m for ReinFlow comparison)')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--output',   type=str,
                        default='evaluation_results/pid_baseline/results.json')

    # PID gains
    parser.add_argument('--Kp-pos',       type=float, default=1.5)
    parser.add_argument('--Kp-vel',       type=float, default=3.0)
    parser.add_argument('--Ki-vel',       type=float, default=0.5)
    parser.add_argument('--vel-int-limit',type=float, default=1.0)
    parser.add_argument('--Kp-att',       type=float, default=8.0)
    parser.add_argument('--Kp-att-yaw',   type=float, default=2.0)
    parser.add_argument('--omega-max',    type=float, default=2.0)
    parser.add_argument('--Kp-rate',      type=float, default=0.15)
    parser.add_argument('--vel-max',      type=float, default=2.0)

    args = parser.parse_args()
    main(args)

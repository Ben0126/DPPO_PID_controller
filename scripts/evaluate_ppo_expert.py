"""
PPO Expert Evaluation Script

Runs N episodes with deterministic policy and reports:
  - Mean/std/median position error (total + per-axis)
  - Phase gate pass/fail (< 0.1m, > 40/50 episodes)
  - Crash rate, episode lengths
  - Per-episode breakdown table

Usage:
    python -m scripts.evaluate_ppo_expert \
        --model checkpoints/ppo_expert/20260401_043637/best_model.pt \
        --norm  checkpoints/ppo_expert/20260401_043637/best_obs_rms.npz
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env import QuadrotorEnv
from models.ppo_expert import PPOExpert, RunningMeanStd


PHASE_GATE = {
    'mean_pos_error':    0.10,   # m
    'episodes_under_01': 40,     # out of n_episodes=50
    'z_axis_error':      0.05,   # m
    'crash_rate':        0.0,    # fraction
}


def evaluate(args):
    env = QuadrotorEnv(config_path=args.quadrotor_config)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOExpert(state_dim=state_dim, action_dim=action_dim,
                      hidden_dim=args.hidden_dim,
                      critic_hidden_dim=args.critic_hidden_dim)
    agent.load(args.model)

    obs_rms = RunningMeanStd(shape=(state_dim,))
    norm_data = np.load(args.norm)
    obs_rms.load_state_dict({
        'mean':  norm_data['mean'],
        'var':   norm_data['var'],
        'count': float(norm_data['count']),
    })

    print(f"\n{'='*64}")
    print(f"  PPO Expert Evaluation")
    print(f"  Model : {args.model}")
    print(f"  Episodes : {args.n_episodes}")
    print(f"{'='*64}\n")

    # -----------------------------------------------------------------------
    # Per-episode metrics
    # -----------------------------------------------------------------------
    ep_rewards      = []
    ep_lengths      = []
    ep_pos_errors   = []   # total 3D error per episode (mean over steps)
    ep_x_errors     = []
    ep_y_errors     = []
    ep_z_errors     = []
    ep_crashes      = []

    header = (f"{'Ep':>4} | {'Reward':>8} | {'Steps':>5} | "
              f"{'PosErr(m)':>9} | {'X':>6} | {'Y':>6} | {'Z':>6} | {'Crash':>5}")
    print(header)
    print('-' * len(header))

    for ep in range(args.n_episodes):
        state, _ = env.reset()
        state_norm = obs_rms.normalize(state)
        ep_reward  = 0.0
        ep_length  = 0
        done       = False

        step_pos_errs = []
        step_x_errs   = []
        step_y_errs   = []
        step_z_errs   = []

        while not done:
            action = agent.get_action_deterministic(state_norm)
            state, reward, terminated, truncated, info = env.step(action)
            state_norm = obs_rms.normalize(state)

            pos_err_vec = info['target'] - info['position']  # (3,)
            step_pos_errs.append(np.linalg.norm(pos_err_vec))
            step_x_errs.append(abs(pos_err_vec[0]))
            step_y_errs.append(abs(pos_err_vec[1]))
            step_z_errs.append(abs(pos_err_vec[2]))

            ep_reward += reward
            ep_length += 1
            done = terminated or truncated

        crashed = ep_length < env.max_episode_steps

        mean_pos = float(np.mean(step_pos_errs))
        mean_x   = float(np.mean(step_x_errs))
        mean_y   = float(np.mean(step_y_errs))
        mean_z   = float(np.mean(step_z_errs))

        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_length)
        ep_pos_errors.append(mean_pos)
        ep_x_errors.append(mean_x)
        ep_y_errors.append(mean_y)
        ep_z_errors.append(mean_z)
        ep_crashes.append(crashed)

        crash_str = 'CRASH' if crashed else '     '
        print(f"{ep+1:>4} | {ep_reward:>8.2f} | {ep_length:>5} | "
              f"{mean_pos:>9.4f} | {mean_x:>6.4f} | {mean_y:>6.4f} | {mean_z:>6.4f} | {crash_str}")

    # -----------------------------------------------------------------------
    # Aggregate stats
    # -----------------------------------------------------------------------
    n   = args.n_episodes
    pos = np.array(ep_pos_errors)
    z   = np.array(ep_z_errors)

    mean_pos  = float(np.mean(pos))
    std_pos   = float(np.std(pos))
    med_pos   = float(np.median(pos))
    under_01  = int(np.sum(pos < 0.10))
    mean_z    = float(np.mean(z))
    crashes   = int(sum(ep_crashes))
    crash_rate = crashes / n

    print(f"\n{'='*64}")
    print(f"  AGGREGATE RESULTS ({n} episodes)")
    print(f"{'='*64}")
    print(f"  Reward          : {np.mean(ep_rewards):.2f} ± {np.std(ep_rewards):.2f}")
    print(f"  Episode length  : {np.mean(ep_lengths):.1f} steps (max {env.max_episode_steps})")
    print(f"  Position error  : {mean_pos:.4f} m  (std {std_pos:.4f}, median {med_pos:.4f})")
    print(f"    X-axis error  : {np.mean(ep_x_errors):.4f} m")
    print(f"    Y-axis error  : {np.mean(ep_y_errors):.4f} m")
    print(f"    Z-axis error  : {mean_z:.4f} m  ← altitude")
    print(f"  Under 0.1m      : {under_01}/{n} episodes")
    print(f"  Crashes         : {crashes}/{n}")

    # -----------------------------------------------------------------------
    # Phase gate
    # -----------------------------------------------------------------------
    print(f"\n{'='*64}")
    print(f"  PHASE GATE ASSESSMENT")
    print(f"{'='*64}")

    gates = [
        ("Mean position error < 0.10m",   mean_pos  < PHASE_GATE['mean_pos_error'],
         f"{mean_pos:.4f}m",  "< 0.10m"),
        (f"Episodes < 0.1m  > {PHASE_GATE['episodes_under_01']}/{n}",
         under_01 >= PHASE_GATE['episodes_under_01'],
         f"{under_01}/{n}",  f">= {PHASE_GATE['episodes_under_01']}/{n}"),
        ("Z-axis error < 0.05m",           mean_z    < PHASE_GATE['z_axis_error'],
         f"{mean_z:.4f}m",   "< 0.05m"),
        ("Crash rate = 0%",                crash_rate == PHASE_GATE['crash_rate'],
         f"{crashes}/{n}",   "0"),
    ]

    all_pass = True
    for name, passed, actual, target in gates:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name:45s} actual={actual}  target={target}")

    print(f"\n  {'>>> PHASE GATE PASSED — ready for Phase 2 <<<' if all_pass else '>>> PHASE GATE NOT MET — adjust and re-train <<<'}")
    print(f"{'='*64}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PPO Expert Phase Gate')
    parser.add_argument('--model',            type=str, required=True,
                        help='Path to best_model.pt')
    parser.add_argument('--norm',             type=str, required=True,
                        help='Path to best_obs_rms.npz')
    parser.add_argument('--quadrotor-config', type=str,
                        default='configs/quadrotor.yaml')
    parser.add_argument('--hidden-dim',        type=int, default=256)
    parser.add_argument('--critic-hidden-dim', type=int, default=None,
                        help='Critic hidden dim (default: same as --hidden-dim)')
    parser.add_argument('--n-episodes',        type=int, default=50)
    args = parser.parse_args()
    evaluate(args)

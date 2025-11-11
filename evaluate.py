"""
Evaluation Script for DPPO PID Controller

This script evaluates a trained PPO agent and visualizes its performance
in controlling a 2nd-order system with adaptive PID gains.
"""

import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dppo_pid_env import make_env


def evaluate_model(
    model_path: str,
    config_path: str = "config.yaml",
    n_episodes: int = 5,
    render: bool = False,
    save_plots: bool = True
):
    """
    Evaluate a trained model and visualize results.

    Args:
        model_path: Path to the trained model
        config_path: Path to configuration file
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment (currently unused)
        save_plots: Whether to save visualization plots
    """
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("DPPO PID Controller - Evaluation Script")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print("=" * 60)

    # Create environment
    print("\n[1/3] Creating evaluation environment...")
    env = make_env(config_path)

    # Wrap in DummyVecEnv for compatibility
    eval_env = DummyVecEnv([lambda: env])

    # Load normalization statistics if they exist
    stats_path = model_path.replace('.zip', '_vec_normalize.pkl')
    if os.path.exists(stats_path):
        print(f"[2/3] Loading normalization statistics: {stats_path}")
        eval_env = VecNormalize.load(stats_path, eval_env)
        eval_env.training = False  # Don't update stats during evaluation
        eval_env.norm_reward = False
    else:
        print("[2/3] No normalization statistics found, evaluating without normalization")

    # Load model
    print(f"[3/3] Loading trained model...")
    model = PPO.load(model_path)

    # Run evaluation episodes
    print("\nRunning evaluation episodes...")
    print("-" * 60)

    episode_rewards = []
    episode_lengths = []
    all_histories = []

    for episode in range(n_episodes):
        obs = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Get action from trained policy
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward[0]
            episode_length += 1

            if done:
                break

        # Store results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Get history from environment
        history = env.get_history()
        all_histories.append(history)

        # Print episode summary
        final_error = abs(history['error'][-1]) if history['error'] else float('nan')
        mean_abs_error = np.mean(np.abs(history['error'])) if history['error'] else float('nan')

        print(f"Episode {episode + 1}/{n_episodes}:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length} steps")
        print(f"  Final Error: {final_error:.4f}")
        print(f"  Mean Abs Error: {mean_abs_error:.4f}")
        print(f"  Final Gains: Kp={history['kp'][-1]:.3f}, Ki={history['ki'][-1]:.3f}, Kd={history['kd'][-1]:.3f}")
        print("-" * 60)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Mean Episode Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print("=" * 60)

    # Visualize results
    if save_plots:
        print("\nGenerating visualizations...")
        output_dir = "./evaluation_results/"
        os.makedirs(output_dir, exist_ok=True)

        # Plot best episode (highest reward)
        best_episode_idx = np.argmax(episode_rewards)
        plot_episode(
            all_histories[best_episode_idx],
            episode_idx=best_episode_idx,
            reward=episode_rewards[best_episode_idx],
            output_dir=output_dir,
            title_prefix="Best"
        )

        # Plot worst episode (lowest reward)
        worst_episode_idx = np.argmin(episode_rewards)
        plot_episode(
            all_histories[worst_episode_idx],
            episode_idx=worst_episode_idx,
            reward=episode_rewards[worst_episode_idx],
            output_dir=output_dir,
            title_prefix="Worst"
        )

        # Plot summary statistics
        plot_summary(episode_rewards, episode_lengths, all_histories, output_dir)

        print(f"✓ Visualizations saved to: {output_dir}")

    # Cleanup
    eval_env.close()

    return episode_rewards, episode_lengths, all_histories


def plot_episode(history, episode_idx, reward, output_dir, title_prefix="Episode"):
    """
    Plot detailed visualization for a single episode.

    Args:
        history: Dictionary containing episode history
        episode_idx: Episode index
        reward: Total episode reward
        output_dir: Directory to save plots
        title_prefix: Prefix for plot title
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig)

    time = np.array(history['time'])
    position = np.array(history['position'])
    reference = np.array(history['reference'])
    error = np.array(history['error'])
    control = np.array(history['control'])
    kp = np.array(history['kp'])
    ki = np.array(history['ki'])
    kd = np.array(history['kd'])
    velocity = np.array(history['velocity'])

    # Plot 1: Position Tracking
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, position, 'b-', label='Position', linewidth=2)
    ax1.plot(time, reference, 'r--', label='Reference', linewidth=2)
    ax1.set_ylabel('Position', fontsize=12)
    ax1.set_title(f'{title_prefix} Episode {episode_idx + 1} - Total Reward: {reward:.2f}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Tracking Error
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, error, 'g-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Tracking Error', fontsize=12)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Control Input
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time, control, 'purple', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_ylabel('Control Input (u)', fontsize=12)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Plot 4: PID Gains Evolution
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(time, kp, 'r-', label='Kp', linewidth=2)
    ax4.plot(time, ki, 'g-', label='Ki', linewidth=2)
    ax4.plot(time, kd, 'b-', label='Kd', linewidth=2)
    ax4.set_ylabel('PID Gains', fontsize=12)
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    filename = f"{title_prefix.lower()}_episode_{episode_idx + 1}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {filepath}")


def plot_summary(episode_rewards, episode_lengths, all_histories, output_dir):
    """
    Plot summary statistics across all episodes.

    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        all_histories: List of episode histories
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Episode Rewards
    axes[0, 0].bar(range(1, len(episode_rewards) + 1), episode_rewards, color='steelblue')
    axes[0, 0].axhline(y=np.mean(episode_rewards), color='r', linestyle='--',
                       label=f'Mean: {np.mean(episode_rewards):.2f}')
    axes[0, 0].set_xlabel('Episode', fontsize=12)
    axes[0, 0].set_ylabel('Total Reward', fontsize=12)
    axes[0, 0].set_title('Episode Rewards', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Episode Lengths
    axes[0, 1].bar(range(1, len(episode_lengths) + 1), episode_lengths, color='coral')
    axes[0, 1].axhline(y=np.mean(episode_lengths), color='r', linestyle='--',
                       label=f'Mean: {np.mean(episode_lengths):.1f}')
    axes[0, 1].set_xlabel('Episode', fontsize=12)
    axes[0, 1].set_ylabel('Steps', fontsize=12)
    axes[0, 1].set_title('Episode Lengths', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Mean Absolute Error Distribution
    mean_errors = [np.mean(np.abs(h['error'])) for h in all_histories if h['error']]
    axes[1, 0].hist(mean_errors, bins=10, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=np.mean(mean_errors), color='r', linestyle='--',
                       label=f'Mean: {np.mean(mean_errors):.4f}')
    axes[1, 0].set_xlabel('Mean Absolute Error', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Tracking Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Final PID Gains Distribution
    final_kp = [h['kp'][-1] for h in all_histories if h['kp']]
    final_ki = [h['ki'][-1] for h in all_histories if h['ki']]
    final_kd = [h['kd'][-1] for h in all_histories if h['kd']]

    x_pos = np.arange(3)
    means = [np.mean(final_kp), np.mean(final_ki), np.mean(final_kd)]
    stds = [np.std(final_kp), np.std(final_ki), np.std(final_kd)]

    axes[1, 1].bar(x_pos, means, yerr=stds, color=['red', 'green', 'blue'],
                   alpha=0.7, capsize=5, edgecolor='black')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(['Kp', 'Ki', 'Kd'])
    axes[1, 1].set_ylabel('Gain Value', fontsize=12)
    axes[1, 1].set_title('Final PID Gains (Mean ± Std)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    filepath = os.path.join(output_dir, 'evaluation_summary.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {filepath}")


def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO agent for PID parameter tuning"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file (.zip)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='Number of evaluation episodes (default: 5)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return

    # Run evaluation
    evaluate_model(
        model_path=args.model,
        config_path=args.config,
        n_episodes=args.episodes,
        save_plots=not args.no_plots
    )


if __name__ == "__main__":
    main()

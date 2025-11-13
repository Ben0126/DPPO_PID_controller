"""
Evaluation Script for DPPO PID Controller

This script evaluates a trained PPO agent and visualizes its performance
in controlling a 2nd-order system with adaptive PID gains.
"""

import os
import argparse
import yaml
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dppo_pid_env import make_env
from utils import plot_episode, plot_summary, plot_gains_vs_error


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
    # Create a single environment instance that we can access later
    actual_env = make_env(config_path)
    
    # Wrap in DummyVecEnv for compatibility (use the same instance)
    eval_env = DummyVecEnv([lambda: actual_env])

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
        last_info = None

        while not done:
            # Get action from trained policy
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            last_info = info  # Save last info for fallback

            if done:
                break

        # Store results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Get history from environment (unwrap if needed)
        # VecNormalize wraps DummyVecEnv, which wraps the actual env
        unwrapped_env = eval_env
        while hasattr(unwrapped_env, 'venv') or hasattr(unwrapped_env, 'envs'):
            if hasattr(unwrapped_env, 'venv'):
                unwrapped_env = unwrapped_env.venv
            elif hasattr(unwrapped_env, 'envs'):
                # DummyVecEnv has envs attribute (list of environments)
                unwrapped_env = unwrapped_env.envs[0]
            else:
                break
        
        # Get history from the actual environment
        if hasattr(unwrapped_env, 'get_history'):
            history = unwrapped_env.get_history()
        elif hasattr(actual_env, 'get_history'):
            history = actual_env.get_history()
        else:
            history = {}
        
        # Debug: Check if history is empty
        if not history or (history.get('time') and len(history['time']) == 0):
            # Try to get from the wrapped environment directly
            if hasattr(eval_env, 'venv') and hasattr(eval_env.venv, 'envs'):
                try:
                    direct_env = eval_env.venv.envs[0]
                    if hasattr(direct_env, 'get_history'):
                        history = direct_env.get_history()
                except:
                    pass
        
        all_histories.append(history)

        # Print episode summary with safe access to history
        final_error = abs(history['error'][-1]) if history.get('error') and len(history['error']) > 0 else float('nan')
        mean_abs_error = np.mean(np.abs(history['error'])) if history.get('error') and len(history['error']) > 0 else float('nan')
        
        # Safe access to final gains
        if history.get('kp') and len(history['kp']) > 0:
            final_kp = history['kp'][-1]
            final_ki = history['ki'][-1] if history.get('ki') and len(history['ki']) > 0 else float('nan')
            final_kd = history['kd'][-1] if history.get('kd') and len(history['kd']) > 0 else float('nan')
            gains_str = f"Kp={final_kp:.3f}, Ki={final_ki:.3f}, Kd={final_kd:.3f}"
        elif last_info and len(last_info) > 0 and isinstance(last_info[0], dict):
            # Fallback to info dictionary if history is empty
            info_dict = last_info[0]
            gains_str = f"Kp={info_dict.get('Kp', 'N/A'):.3f}, Ki={info_dict.get('Ki', 'N/A'):.3f}, Kd={info_dict.get('Kd', 'N/A'):.3f}"
        else:
            gains_str = "N/A"

        print(f"Episode {episode + 1}/{n_episodes}:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length} steps")
        print(f"  Final Error: {final_error:.4f}")
        print(f"  Mean Abs Error: {mean_abs_error:.4f}")
        print(f"  Final Gains: {gains_str}")
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

        # Plot best/worst episode gains vs error (AirPilot style)
        plot_gains_vs_error(
            all_histories[best_episode_idx],
            output_dir,
            best_episode_idx,
            title_prefix="Best"
        )
        plot_gains_vs_error(
            all_histories[worst_episode_idx],
            output_dir,
            worst_episode_idx,
            title_prefix="Worst"
        )

        print(f"[OK] Visualizations saved to: {output_dir}")
        print("\nAll evaluation plots have been displayed.")

    # Cleanup
    eval_env.close()

    return episode_rewards, episode_lengths, all_histories


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

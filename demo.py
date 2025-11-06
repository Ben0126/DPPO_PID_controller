"""
Demo Script for DPPO PID Controller

This script demonstrates the environment with random actions
to verify the implementation before training.
"""

import numpy as np
import matplotlib.pyplot as plt
from dppo_pid_env import make_env


def run_demo_episode(config_path: str = "config.yaml", n_steps: int = 200):
    """
    Run a demo episode with random actions to test the environment.

    Args:
        config_path: Path to configuration file
        n_steps: Number of steps to run
    """
    print("=" * 60)
    print("DPPO PID Controller - Demo Script")
    print("=" * 60)
    print("Running demo episode with random actions...")
    print("-" * 60)

    # Create environment
    env = make_env(config_path)

    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print("-" * 60)

    # Run episode
    total_reward = 0
    for step in range(n_steps):
        # Sample random action (random PID gains)
        action = env.action_space.sample()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Print progress every 50 steps
        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{n_steps}:")
            print(f"  Position: {info['position']:.4f}")
            print(f"  Reference: {info['reference']:.4f}")
            print(f"  Error: {info['error']:.4f}")
            print(f"  Control: {info['control']:.4f}")
            print(f"  PID Gains: Kp={info['Kp']:.3f}, Ki={info['Ki']:.3f}, Kd={info['Kd']:.3f}")
            print(f"  Cumulative Reward: {total_reward:.2f}")
            print("-" * 60)

        # Check if episode ended
        if terminated or truncated:
            reason = "terminated" if terminated else "truncated"
            print(f"Episode ended at step {step + 1} ({reason})")
            break

    # Get history and plot
    history = env.get_history()

    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    print(f"Total Steps: {step + 1}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Final Error: {abs(info['error']):.4f}")
    print("=" * 60)

    # Plot results
    plot_demo_results(history)

    env.close()


def plot_demo_results(history):
    """
    Plot the results from the demo episode.

    Args:
        history: Dictionary containing episode history
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    time = np.array(history['time'])
    position = np.array(history['position'])
    reference = np.array(history['reference'])
    error = np.array(history['error'])
    control = np.array(history['control'])
    kp = np.array(history['kp'])
    ki = np.array(history['ki'])
    kd = np.array(history['kd'])

    # Plot 1: Position Tracking
    axes[0].plot(time, position, 'b-', label='Position', linewidth=2)
    axes[0].plot(time, reference, 'r--', label='Reference', linewidth=2)
    axes[0].set_ylabel('Position', fontsize=12)
    axes[0].set_title('Demo Episode - Random PID Gains', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Tracking Error and Control
    ax2_twin = axes[1].twinx()
    axes[1].plot(time, error, 'g-', label='Error', linewidth=2)
    ax2_twin.plot(time, control, 'purple', label='Control', linewidth=2, alpha=0.7)
    axes[1].set_ylabel('Tracking Error', fontsize=12, color='g')
    ax2_twin.set_ylabel('Control Input (u)', fontsize=12, color='purple')
    axes[1].tick_params(axis='y', labelcolor='g')
    ax2_twin.tick_params(axis='y', labelcolor='purple')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: PID Gains
    axes[2].plot(time, kp, 'r-', label='Kp', linewidth=2)
    axes[2].plot(time, ki, 'g-', label='Ki', linewidth=2)
    axes[2].plot(time, kd, 'b-', label='Kd', linewidth=2)
    axes[2].set_ylabel('PID Gains', fontsize=12)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    plt.savefig('demo_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ Demo plot saved to: demo_results.png")

    plt.show()


def test_environment_api():
    """
    Test basic environment API functionality.
    """
    print("\n" + "=" * 60)
    print("Testing Environment API")
    print("=" * 60)

    env = make_env("config.yaml")

    # Test reset
    print("Testing reset()...")
    obs, info = env.reset()
    assert obs.shape == (9,), f"Expected obs shape (9,), got {obs.shape}"
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0), "Observations should be normalized to [-1, 1]"
    print("✓ reset() works correctly")

    # Test step
    print("Testing step()...")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (9,), f"Expected obs shape (9,), got {obs.shape}"
    assert isinstance(reward, (float, np.floating)), f"Expected float reward, got {type(reward)}"
    assert isinstance(terminated, bool), f"Expected bool terminated, got {type(terminated)}"
    assert isinstance(truncated, bool), f"Expected bool truncated, got {type(truncated)}"
    assert isinstance(info, dict), f"Expected dict info, got {type(info)}"
    print("✓ step() works correctly")

    # Test multiple steps
    print("Testing multiple steps...")
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    print("✓ Multiple steps work correctly")

    # Test action space bounds
    print("Testing action space bounds...")
    action = np.array([0.0, 0.0, 0.0])
    obs, reward, terminated, truncated, info = env.step(action)
    print("✓ Lower bound actions work")

    action = np.array([10.0, 10.0, 10.0])
    obs, reward, terminated, truncated, info = env.step(action)
    print("✓ Upper bound actions work")

    env.close()

    print("\n" + "=" * 60)
    print("✓ All API tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    # Test environment API
    test_environment_api()

    # Run demo episode
    run_demo_episode(n_steps=200)

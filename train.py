"""
Training Script for DPPO PID Controller

This script trains a PPO agent to learn optimal PID gain adjustments
for a 2nd-order system using the custom DPPOPIDEnv environment.
"""

import os
import argparse
import yaml
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from dppo_pid_env import make_env
from utils import TrainingMetricsTracker, plot_airpilot_style_metrics


def plot_training_results(log_dir: str, save_path: str = None, show_plot: bool = False):
    """
    Read training logs and plot training process visualizations.
    
    Args:
        log_dir: Training log directory path
        save_path: Path to save the plot (if None, saves to log_dir/training_visualization.png)
        show_plot: Whether to display the plot interactively (default: False)
    """
    monitor_file = os.path.join(log_dir, 'monitor.csv')
    
    if not os.path.exists(monitor_file):
        print(f"Warning: Monitor file not found at {monitor_file}")
        return
    
    # Read monitor.csv (skip first line JSON metadata)
    try:
        df = pd.read_csv(monitor_file, skiprows=1)
    except Exception as e:
        print(f"Error reading monitor.csv: {e}")
        return
    
    if df.empty:
        print("Warning: Monitor file is empty")
        return
    
    # Calculate cumulative steps
    df['cumulative_steps'] = df['l'].cumsum()
    
    # Calculate moving average (for smoothing)
    window_size = min(50, len(df) // 10)  # Use 10% of data as window size, minimum 50 points
    if window_size < 1:
        window_size = 1
    
    df['reward_smooth'] = df['r'].rolling(window=window_size, center=True).mean()
    
    # Create plots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Reward vs Cumulative Steps (raw data + smoothed curve)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(df['cumulative_steps'], df['r'], alpha=0.3, s=10, color='lightblue', label='Raw Data')
    ax1.plot(df['cumulative_steps'], df['reward_smooth'], 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
    ax1.set_xlabel('Cumulative Steps', fontsize=12)
    ax1.set_ylabel('Episode Reward', fontsize=12)
    ax1.set_title('Training Process: Reward vs Steps', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Episode Reward Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(df['r'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(df['r'].mean(), color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {df["r"].mean():.2f}')
    ax2.axvline(df['r'].median(), color='g', linestyle='--', linewidth=2, 
                label=f'Median: {df["r"].median():.2f}')
    ax2.set_xlabel('Episode Reward', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Episode Reward Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Episode Length Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(df['l'], bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax3.axvline(df['l'].mean(), color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {df["l"].mean():.1f}')
    ax3.axvline(df['l'].median(), color='g', linestyle='--', linewidth=2, 
                label=f'Median: {df["l"].median():.1f}')
    ax3.set_xlabel('Episode Length (steps)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Episode Length Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Reward Trend (by episode order)
    ax4 = fig.add_subplot(gs[2, 0])
    episode_nums = range(1, len(df) + 1)
    ax4.plot(episode_nums, df['r'], 'b-', alpha=0.5, linewidth=1, label='Raw Data')
    ax4.plot(episode_nums, df['reward_smooth'], 'r-', linewidth=2, label='Moving Average')
    ax4.set_xlabel('Episode Number', fontsize=12)
    ax4.set_ylabel('Episode Reward', fontsize=12)
    ax4.set_title('Reward Trend (by Episode)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Statistics Summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Calculate statistics
    total_episodes = len(df)
    total_steps = df['cumulative_steps'].iloc[-1]
    mean_reward = df['r'].mean()
    std_reward = df['r'].std()
    min_reward = df['r'].min()
    max_reward = df['r'].max()
    mean_length = df['l'].mean()
    std_length = df['l'].std()
    
    # Calculate average reward for recent 100 episodes (if available)
    recent_episodes = min(100, total_episodes)
    recent_mean_reward = df['r'].tail(recent_episodes).mean()
    
    stats_text = f"""
    Training Statistics Summary
    
    Total Episodes: {total_episodes:,}
    Total Steps: {total_steps:,}
    
    Reward Statistics:
    Mean: {mean_reward:.2f} ± {std_reward:.2f}
    Min: {min_reward:.2f}
    Max: {max_reward:.2f}
    Recent {recent_episodes} Episodes Avg: {recent_mean_reward:.2f}
    
    Episode Length Statistics:
    Mean: {mean_length:.1f} ± {std_length:.1f}
    Min: {df['l'].min():.0f}
    Max: {df['l'].max():.0f}
    """
    
    ax5.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('DPPO PID Controller - Training Process Visualization', fontsize=16, fontweight='bold', y=0.995)
    
    # Save plot
    if save_path is None:
        save_path = os.path.join(log_dir, 'training_visualization.png')
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training visualization saved to: {save_path}")
    
    # Optionally display plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_directories(config):
    """Create necessary directories for logging and model saving."""
    os.makedirs(config['logging']['tensorboard_log'], exist_ok=True)
    os.makedirs(config['logging']['save_path'], exist_ok=True)
    os.makedirs('./eval_logs/', exist_ok=True)


def make_monitored_env(config_path: str, log_dir: str = None):
    """
    Create a monitored environment for training.

    Args:
        config_path: Path to configuration file
        log_dir: Directory for monitoring logs

    Returns:
        Monitored environment instance
    """
    env = make_env(config_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
    return env


def train(config_path: str = "config.yaml", resume: bool = False, model_path: str = None):
    """
    Train the PPO agent for PID parameter tuning.

    Args:
        config_path: Path to the configuration file
        resume: Whether to resume training from a saved model
        model_path: Path to the model to resume from
    """
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Create directories
    create_directories(config)

    print("=" * 60)
    print("DPPO PID Controller - Training Script")
    print("=" * 60)
    print(f"Configuration: {config_path}")
    print()
    # Check for quick test mode (AirPilot style)
    quick_mode = config['training'].get('quick_test_mode', False)
    if quick_mode:
        total_timesteps = config['training']['quick_test_timesteps']
        net_arch = config['training']['quick_test_net_arch']
        print("⚠️  快速訓練模式啟用（AirPilot 風格）")
        print(f"   訓練步數: {total_timesteps:,}")
        print(f"   網路架構: {net_arch}")
        print("=" * 60)
    else:
        total_timesteps = config['training']['total_timesteps']
        net_arch = config['training'].get('policy_net_arch', [128, 128])

    print("PPO Hyperparameters (Recommended Settings):")
    print(f"  Policy Network:    {net_arch}")
    print(f"  Value Network:     {net_arch}")
    print(f"  Learning Rate:     {config['training']['learning_rate']} (3×10⁻⁴)")
    print(f"  n_steps:           {config['training']['n_steps']}")
    print(f"  Batch Size:        {config['training']['batch_size']}")
    print(f"  Gamma (γ):         {config['training']['gamma']}")
    print(f"  GAE Lambda (λ):    {config['training']['gae_lambda']}")
    print(f"  Total Steps:       {total_timesteps:,}")
    print(f"  VecNormalize:      {'Enabled' if config['training'].get('use_vec_normalize', True) else 'Disabled'}")
    print("=" * 60)

    # Create training environment (vectorized for SB3 compatibility)
    print("\n[1/5] Creating training environment...")
    train_env = DummyVecEnv([lambda: make_monitored_env(config_path, './train_logs/')])

    # Optionally wrap with VecNormalize for observation/reward normalization
    # This can improve learning stability
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )

    # Create evaluation environment
    print("[2/5] Creating evaluation environment...")
    eval_env = DummyVecEnv([lambda: make_monitored_env(config_path, './eval_logs/')])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize rewards during evaluation
        clip_obs=10.0,
        training=False  # Don't update normalization stats during eval
    )

    # Create or load PPO model
    if resume and model_path and os.path.exists(model_path):
        print(f"[3/5] Loading existing model from: {model_path}")
        model = PPO.load(
            model_path,
            env=train_env,
            tensorboard_log=config['logging']['tensorboard_log']
        )
        # Load normalization statistics if they exist
        stats_path = model_path.replace('.zip', '_vec_normalize.pkl')
        if os.path.exists(stats_path):
            train_env = VecNormalize.load(stats_path, train_env)
            print(f"Loaded normalization statistics from: {stats_path}")
    else:
        print("[3/5] Creating new PPO model...")

        # Use net_arch determined earlier (from quick mode check or default)
        print(f"Policy network architecture: {net_arch}")
        print(f"Value network architecture: {net_arch}")

        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=config['training']['learning_rate'],
            n_steps=config['training']['n_steps'],
            batch_size=config['training']['batch_size'],
            n_epochs=config['training']['n_epochs'],
            gamma=config['training']['gamma'],
            gae_lambda=config['training']['gae_lambda'],
            clip_range=config['training']['clip_range'],
            ent_coef=config['training']['ent_coef'],
            vf_coef=config['training']['vf_coef'],
            max_grad_norm=config['training']['max_grad_norm'],
            verbose=1,
            tensorboard_log=config['logging']['tensorboard_log'],
            policy_kwargs=dict(
                net_arch=[dict(pi=net_arch, vf=net_arch)]
            )
        )

    # Setup callbacks
    print("[4/5] Setting up training callbacks...")

    # Create training metrics tracker (AirPilot style)
    metrics_tracker = TrainingMetricsTracker()

    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=config['logging']['checkpoint_freq'],
        save_path=config['logging']['save_path'],
        name_prefix='dppo_pid_checkpoint',
        save_vecnormalize=True
    )

    # Custom evaluation callback with metrics tracking
    class MetricsEvalCallback(EvalCallback):
        """
        自定義評估回調，記錄訓練指標（AirPilot 風格）
        """
        def __init__(self, *args, metrics_tracker=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.metrics_tracker = metrics_tracker
        
        def _on_step(self) -> bool:
            result = super()._on_step()
            
            # 在評估時記錄指標
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                # 獲取評估環境的歷史
                try:
                    # 解包環境以獲取實際環境實例
                    if hasattr(self.eval_env, 'venv'):
                        env = self.eval_env.venv.envs[0]
                    elif hasattr(self.eval_env, 'envs'):
                        env = self.eval_env.envs[0]
                    else:
                        env = None
                    
                    if env and hasattr(env, 'get_history'):
                        history = env.get_history()
                        if history and history.get('position'):
                            self.metrics_tracker.log_episode(
                                timestep=self.num_timesteps,
                                episode_history=history
                            )
                except Exception as e:
                    # 如果獲取歷史失敗，不影響訓練
                    pass
            
            return result

    # Evaluation callback - evaluate periodically and save best model
    eval_callback = MetricsEvalCallback(
        eval_env,
        metrics_tracker=metrics_tracker,
        best_model_save_path=config['logging']['save_path'],
        log_path='./eval_logs/',
        eval_freq=10000,  # Evaluate every 10k steps
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    callbacks = [checkpoint_callback, eval_callback]

    # Train the model
    print("[5/5] Starting training...")
    print(f"TensorBoard logs: {config['logging']['tensorboard_log']}")
    print(f"Model checkpoints: {config['logging']['save_path']}")
    print(f"\nTo monitor training progress, run:")
    print(f"  tensorboard --logdir {config['logging']['tensorboard_log']}")
    print("=" * 60)

    # Check if progress bar dependencies are available
    try:
        import tqdm
        import rich
        use_progress_bar = True
    except ImportError:
        use_progress_bar = False
        print("Warning: tqdm/rich not installed. Progress bar disabled.")
        print("Install with: pip install tqdm rich")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=use_progress_bar
        )

        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = os.path.join(
            config['logging']['save_path'],
            f'dppo_pid_final_{timestamp}.zip'
        )
        model.save(final_model_path)
        print(f"\n✓ Training completed! Final model saved to: {final_model_path}")

        # Save normalization statistics
        stats_path = final_model_path.replace('.zip', '_vec_normalize.pkl')
        train_env.save(stats_path)
        print(f"✓ Normalization statistics saved to: {stats_path}")

        # Save training metrics and plot AirPilot style charts
        print("\n" + "=" * 60)
        print("Saving training metrics...")
        print("=" * 60)
        metrics_tracker.save()
        try:
            plot_airpilot_style_metrics(metrics_tracker)
            print("✓ Training metrics visualization completed!")
        except Exception as e:
            print(f"⚠ Warning: Could not generate training metrics visualization: {e}")

    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user!")
        interrupt_model_path = os.path.join(
            config['logging']['save_path'],
            'dppo_pid_interrupted.zip'
        )
        model.save(interrupt_model_path)
        print(f"✓ Model saved to: {interrupt_model_path}")

        # Save normalization statistics
        stats_path = interrupt_model_path.replace('.zip', '_vec_normalize.pkl')
        train_env.save(stats_path)
        print(f"✓ Normalization statistics saved to: {stats_path}")

        # Save training metrics even if interrupted
        try:
            metrics_tracker.save()
        except:
            pass

    finally:
        # Cleanup
        train_env.close()
        eval_env.close()

    # Visualize training results
    print("\n" + "=" * 60)
    print("Generating training visualizations...")
    print("=" * 60)
    try:
        plot_training_results('./train_logs/')
        print("✓ Training visualizations displayed successfully!")
    except Exception as e:
        print(f"⚠ Warning: Could not generate training visualizations: {e}")
        print("  Training data may not be available yet.")

    print("=" * 60)
    print("Training session ended.")
    print("=" * 60)


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Train a PPO agent for PID parameter tuning"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from a saved model'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model file to resume from'
    )

    args = parser.parse_args()

    # Start training
    train(
        config_path=args.config,
        resume=args.resume,
        model_path=args.model
    )


if __name__ == "__main__":
    main()

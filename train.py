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

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from dppo_pid_env import make_env


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
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create directories
    create_directories(config)

    print("=" * 60)
    print("DPPO PID Controller - Training Script")
    print("=" * 60)
    print(f"Configuration loaded from: {config_path}")
    print(f"Total training steps: {config['training']['total_timesteps']:,}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Batch size: {config['training']['batch_size']}")
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
                net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Larger network for complex control
            )
        )

    # Setup callbacks
    print("[4/5] Setting up training callbacks...")

    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=config['logging']['checkpoint_freq'],
        save_path=config['logging']['save_path'],
        name_prefix='dppo_pid_checkpoint',
        save_vecnormalize=True
    )

    # Evaluation callback - evaluate periodically and save best model
    eval_callback = EvalCallback(
        eval_env,
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

    try:
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callbacks,
            progress_bar=True
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

    finally:
        # Cleanup
        train_env.close()
        eval_env.close()

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

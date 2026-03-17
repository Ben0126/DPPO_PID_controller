"""
Phase 1: Train State-Based PPO Expert

Trains a PPO agent to control quadrotor motors directly from 15D state observations.
The trained expert serves as the data source for Diffusion Policy imitation learning.

Usage:
    python -m scripts.train_ppo_expert
    python -m scripts.train_ppo_expert --quick  # quick test mode
"""

import os
import sys
import argparse
import numpy as np
import yaml
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env import QuadrotorEnv
from models.ppo_expert import PPOExpert, RunningMeanStd


def train(args):
    # Load configs
    with open(args.quadrotor_config, 'r', encoding='utf-8') as f:
        quad_config = yaml.safe_load(f)
    with open(args.ppo_config, 'r', encoding='utf-8') as f:
        ppo_config = yaml.safe_load(f)

    # Check quick test mode
    if args.quick:
        ppo_config['training']['total_timesteps'] = ppo_config['quick_test']['timesteps']
        ppo_config['network']['hidden_dim'] = ppo_config['quick_test']['hidden_dim']
        print("[Quick Test Mode] Reduced timesteps and network size")

    train_cfg = ppo_config['training']
    net_cfg = ppo_config['network']
    log_cfg = ppo_config['logging']

    # Create environment
    env = QuadrotorEnv(config_path=args.quadrotor_config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Create agent
    agent = PPOExpert(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=net_cfg['hidden_dim'],
        lr=train_cfg['learning_rate'],
        gamma=train_cfg['gamma'],
        gae_lambda=train_cfg['gae_lambda'],
        clip_range=train_cfg['clip_range'],
        ent_coef=train_cfg['ent_coef'],
        vf_coef=train_cfg['vf_coef'],
        max_grad_norm=train_cfg['max_grad_norm'],
        batch_size=train_cfg['batch_size'],
        n_epochs=train_cfg['n_epochs'],
    )

    # Observation normalization
    obs_rms = RunningMeanStd(shape=(state_dim,))

    # Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_cfg['tensorboard_log'], timestamp)
    save_dir = os.path.join(log_cfg['save_path'], timestamp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Training loop
    total_timesteps = train_cfg['total_timesteps']
    n_steps = train_cfg['n_steps']
    checkpoint_freq = log_cfg['checkpoint_freq']
    eval_freq = log_cfg['eval_freq']
    eval_episodes = log_cfg['eval_episodes']

    timestep = 0
    episode = 0
    best_eval_reward = -float('inf')

    # Episode tracking
    ep_reward = 0.0
    ep_length = 0
    recent_rewards = []

    print(f"\n{'='*60}")
    print(f"Training PPO Expert - Direct Motor Control")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Rollout length: {n_steps}")
    print(f"Hidden dim: {net_cfg['hidden_dim']}")
    print(f"Device: {next(agent.actor.parameters()).device}")
    print(f"{'='*60}\n")

    state, _ = env.reset()
    obs_rms.update(state)
    state_norm = obs_rms.normalize(state)

    while timestep < total_timesteps:
        # Collect rollout
        memory = {
            'states': [], 'actions': [], 'log_probs': [],
            'rewards': [], 'dones': [], 'values': [],
        }

        for _ in range(n_steps):
            action_tanh, log_prob, value = agent.get_action(state_norm)

            next_state, reward, terminated, truncated, info = env.step(action_tanh)
            done = terminated or truncated

            memory['states'].append(state_norm)
            memory['actions'].append(action_tanh)
            memory['log_probs'].append(log_prob)
            memory['rewards'].append(reward)
            memory['dones'].append(float(done))
            memory['values'].append(value)

            ep_reward += reward
            ep_length += 1
            timestep += 1

            if done:
                recent_rewards.append(ep_reward)
                writer.add_scalar('rollout/ep_reward', ep_reward, timestep)
                writer.add_scalar('rollout/ep_length', ep_length, timestep)
                episode += 1
                ep_reward = 0.0
                ep_length = 0
                next_state, _ = env.reset()

            obs_rms.update(next_state)
            state_norm = obs_rms.normalize(next_state)

        # PPO update
        metrics = agent.update(memory)

        writer.add_scalar('train/policy_loss', metrics['policy_loss'], timestep)
        writer.add_scalar('train/value_loss', metrics['value_loss'], timestep)
        writer.add_scalar('train/entropy', metrics['entropy'], timestep)

        # Progress logging
        if len(recent_rewards) > 0:
            mean_reward = np.mean(recent_rewards[-100:])
            print(f"Step {timestep:>8,} | Ep {episode:>5} | "
                  f"Mean Reward (100): {mean_reward:>8.2f} | "
                  f"Policy Loss: {metrics['policy_loss']:.4f} | "
                  f"Value Loss: {metrics['value_loss']:.4f}")

        # Checkpoint
        if timestep % checkpoint_freq < n_steps:
            ckpt_path = os.path.join(save_dir, f"ppo_expert_{timestep}.pt")
            agent.save(ckpt_path)
            # Save normalization stats
            norm_path = os.path.join(save_dir, f"obs_rms_{timestep}.npz")
            np.savez(norm_path, **obs_rms.state_dict())

        # Evaluation
        if timestep % eval_freq < n_steps:
            eval_reward = evaluate(env, agent, obs_rms, eval_episodes)
            writer.add_scalar('eval/mean_reward', eval_reward, timestep)
            print(f"  [Eval] Mean reward over {eval_episodes} episodes: {eval_reward:.2f}")

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_path = os.path.join(save_dir, "best_model.pt")
                agent.save(best_path)
                best_norm_path = os.path.join(save_dir, "best_obs_rms.npz")
                np.savez(best_norm_path, **obs_rms.state_dict())
                print(f"  [Eval] New best model saved! Reward: {eval_reward:.2f}")

    # Save final model
    final_path = os.path.join(save_dir, "final_model.pt")
    agent.save(final_path)
    final_norm_path = os.path.join(save_dir, "final_obs_rms.npz")
    np.savez(final_norm_path, **obs_rms.state_dict())

    writer.close()
    print(f"\nTraining complete! Models saved to: {save_dir}")
    print(f"Best eval reward: {best_eval_reward:.2f}")


def evaluate(env: QuadrotorEnv, agent: PPOExpert,
             obs_rms: RunningMeanStd, n_episodes: int) -> float:
    """Run evaluation episodes with deterministic policy."""
    total_rewards = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        state_norm = obs_rms.normalize(state)
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.get_action_deterministic(state_norm)
            state, reward, terminated, truncated, _ = env.step(action)
            state_norm = obs_rms.normalize(state)
            ep_reward += reward
            done = terminated or truncated

        total_rewards.append(ep_reward)

    return np.mean(total_rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Expert for Quadrotor Control")
    parser.add_argument('--quadrotor-config', type=str, default='configs/quadrotor.yaml')
    parser.add_argument('--ppo-config', type=str, default='configs/ppo_expert.yaml')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    args = parser.parse_args()
    train(args)

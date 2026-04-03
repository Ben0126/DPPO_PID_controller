"""
Phase 1: Train State-Based PPO Expert

Trains a PPO agent to control quadrotor motors directly from 15D state observations.
The trained expert serves as the data source for Diffusion Policy imitation learning.

Usage:
    python -m scripts.train_ppo_expert
    python -m scripts.train_ppo_expert --n-envs 16   # more parallel envs
    python -m scripts.train_ppo_expert --quick        # quick test mode
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


# ---------------------------------------------------------------------------
# Picklable env factory (required for AsyncVectorEnv on Windows which uses
# the 'spawn' multiprocessing start method — closures / lambdas are not
# picklable, so we use a callable class instead).
# ---------------------------------------------------------------------------

class _EnvFactory:
    """Picklable factory that creates a QuadrotorEnv with a fixed config."""

    def __init__(self, config_path: str):
        # Resolve to absolute path so worker processes find the file
        # regardless of their working directory.
        self.config_path = os.path.abspath(config_path)

    def __call__(self) -> QuadrotorEnv:
        return QuadrotorEnv(config_path=self.config_path)


def _make_vec_env(config_path: str, n_envs: int):
    """
    Create a vectorized environment.

    Tries AsyncVectorEnv first (true CPU parallelism via multiprocessing).
    Falls back to SyncVectorEnv if async fails (e.g., pickling issues).
    """
    from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

    env_fns = [_EnvFactory(config_path) for _ in range(n_envs)]
    try:
        vec_env = AsyncVectorEnv(env_fns)
        print(f"[VecEnv] AsyncVectorEnv x{n_envs} (multiprocessing)")
    except Exception as exc:
        print(f"[VecEnv] AsyncVectorEnv failed ({exc}), falling back to SyncVectorEnv")
        vec_env = SyncVectorEnv(env_fns)
    return vec_env


def train(args):
    # Load configs
    with open(args.quadrotor_config, 'r', encoding='utf-8') as f:
        quad_config = yaml.safe_load(f)
    with open(args.ppo_config, 'r', encoding='utf-8') as f:
        ppo_config = yaml.safe_load(f)

    if args.quick:
        ppo_config['training']['total_timesteps'] = ppo_config['quick_test']['timesteps']
        ppo_config['network']['hidden_dim'] = ppo_config['quick_test']['hidden_dim']
        print("[Quick Test Mode] Reduced timesteps and network size")

    train_cfg = ppo_config['training']
    net_cfg    = ppo_config['network']
    log_cfg    = ppo_config['logging']

    n_envs = args.n_envs

    # -----------------------------------------------------------------------
    # Environments
    # -----------------------------------------------------------------------
    vec_env  = _make_vec_env(args.quadrotor_config, n_envs)
    eval_env = QuadrotorEnv(config_path=args.quadrotor_config)

    state_dim  = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # -----------------------------------------------------------------------
    # Agent
    # -----------------------------------------------------------------------
    agent = PPOExpert(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=net_cfg['hidden_dim'],
        critic_hidden_dim=net_cfg.get('critic_hidden_dim', net_cfg['hidden_dim']),
        lr=train_cfg['learning_rate'],
        critic_lr_multiplier=train_cfg.get('critic_lr_multiplier', 1.0),
        gamma=train_cfg['gamma'],
        gae_lambda=train_cfg['gae_lambda'],
        clip_range=train_cfg['clip_range'],
        ent_coef=train_cfg['ent_coef'],
        vf_coef=train_cfg['vf_coef'],
        max_grad_norm=train_cfg['max_grad_norm'],
        batch_size=train_cfg['batch_size'],
        n_epochs=train_cfg['n_epochs'],
    )

    obs_rms = RunningMeanStd(shape=(state_dim,))

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir  = os.path.join(log_cfg['tensorboard_log'], timestamp)
    save_dir = os.path.join(log_cfg['save_path'], timestamp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # -----------------------------------------------------------------------
    # Training hyper-parameters
    # -----------------------------------------------------------------------
    total_timesteps = train_cfg['total_timesteps']
    n_steps         = train_cfg['n_steps']          # steps per env per rollout
    checkpoint_freq = log_cfg['checkpoint_freq']
    eval_freq       = log_cfg['eval_freq']
    eval_episodes   = log_cfg['eval_episodes']
    target_kl       = train_cfg.get('target_kl', None)

    # Each PPO update consumes n_steps * n_envs transitions
    transitions_per_update = n_steps * n_envs

    device = next(agent.actor.parameters()).device

    print(f"\n{'='*60}")
    print(f"Training PPO Expert - Direct Motor Control (Vectorized)")
    print(f"Total timesteps:      {total_timesteps:,}")
    print(f"Parallel envs:        {n_envs}")
    print(f"Rollout length/env:   {n_steps}")
    print(f"Transitions/update:   {transitions_per_update:,}")
    print(f"Hidden dim:           {net_cfg['hidden_dim']}")
    print(f"Device:               {device}")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # Initial reset
    # -----------------------------------------------------------------------
    obs, _ = vec_env.reset()                          # (n_envs, state_dim)
    obs_rms.update(obs)
    obs_norm = obs_rms.normalize(obs).astype(np.float32)

    timestep       = 0
    episode        = 0
    best_eval_reward = -float('inf')
    recent_rewards = []

    ep_rewards = np.zeros(n_envs, dtype=np.float32)
    ep_lengths = np.zeros(n_envs, dtype=np.int32)

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------
    while timestep < total_timesteps:

        # -------------------------------------------------------------------
        # Rollout buffers  (n_steps, n_envs, ...)
        # -------------------------------------------------------------------
        buf_states    = np.zeros((n_steps, n_envs, state_dim),  dtype=np.float32)
        buf_actions   = np.zeros((n_steps, n_envs, action_dim), dtype=np.float32)
        buf_log_probs = np.zeros((n_steps, n_envs),             dtype=np.float32)
        buf_rewards   = np.zeros((n_steps, n_envs),             dtype=np.float32)
        buf_dones     = np.zeros((n_steps, n_envs),             dtype=np.float32)
        buf_values    = np.zeros((n_steps, n_envs),             dtype=np.float32)

        # -------------------------------------------------------------------
        # Collect rollout
        # -------------------------------------------------------------------
        for t in range(n_steps):
            # Batch GPU inference: (n_envs, ...) → GPU → back to numpy
            actions, log_probs, values = agent.get_action_vec(obs_norm)

            next_obs, rewards, terminated, truncated, _ = vec_env.step(actions)
            dones = (terminated | truncated).astype(np.float32)

            buf_states[t]    = obs_norm
            buf_actions[t]   = actions
            buf_log_probs[t] = log_probs
            buf_rewards[t]   = rewards
            buf_dones[t]     = dones
            buf_values[t]    = values

            ep_rewards += rewards
            ep_lengths += 1
            timestep   += n_envs

            # Log finished episodes
            finished = np.where(dones > 0.5)[0]
            for i in finished:
                recent_rewards.append(float(ep_rewards[i]))
                writer.add_scalar('rollout/ep_reward', ep_rewards[i], timestep)
                writer.add_scalar('rollout/ep_length', ep_lengths[i], timestep)
                ep_rewards[i] = 0.0
                ep_lengths[i] = 0
                episode += 1

            # Gymnasium vector envs auto-reset on done; next_obs[i] is already
            # the first obs of the new episode when dones[i]=True.
            obs_rms.update(next_obs)
            obs_norm = obs_rms.normalize(next_obs).astype(np.float32)

        # -------------------------------------------------------------------
        # Bootstrap values for the last observation of each env
        # -------------------------------------------------------------------
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_norm).to(device)
            bootstrap_vals = agent.critic(obs_t).squeeze(-1).cpu().numpy()  # (n_envs,)

        # Zero bootstrap for envs that ended on the last step (terminal)
        last_dones     = buf_dones[-1]                                       # (n_envs,)
        bootstrap_vals = np.where(last_dones > 0.5, 0.0, bootstrap_vals)

        # -------------------------------------------------------------------
        # GAE per env, then flatten to (n_steps * n_envs,)
        # -------------------------------------------------------------------
        advantages_arr = np.zeros((n_steps, n_envs), dtype=np.float32)
        returns_arr    = np.zeros((n_steps, n_envs), dtype=np.float32)

        for i in range(n_envs):
            adv_i, ret_i = agent.compute_gae(
                buf_rewards[:, i].tolist(),
                buf_values[:, i].tolist(),
                buf_dones[:, i].tolist(),
                last_value=float(bootstrap_vals[i]),
            )
            advantages_arr[:, i] = adv_i
            returns_arr[:, i]    = ret_i

        flat_adv     = advantages_arr.reshape(-1)
        flat_ret     = returns_arr.reshape(-1)
        flat_memory  = {
            'states':    buf_states.reshape(-1, state_dim),
            'actions':   buf_actions.reshape(-1, action_dim),
            'log_probs': buf_log_probs.reshape(-1),
            # rewards/dones/values not used by update when advantages are pre-computed
            'rewards':   buf_rewards.reshape(-1),
            'dones':     buf_dones.reshape(-1),
            'values':    buf_values.reshape(-1),
        }

        # -------------------------------------------------------------------
        # Learning-rate and clip-range annealing
        # -------------------------------------------------------------------
        frac   = max(1.0 - timestep / total_timesteps, 0.0)
        lr_now = train_cfg['learning_rate'] * frac
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = lr_now

        clip_start       = train_cfg['clip_range']
        clip_end         = train_cfg.get('clip_range_end', clip_start)
        agent.clip_range = clip_start * frac + clip_end * (1.0 - frac)

        # -------------------------------------------------------------------
        # PPO update (GAE already computed, pass pre-computed values)
        # -------------------------------------------------------------------
        metrics = agent.update(
            flat_memory,
            precomputed_advantages=flat_adv,
            precomputed_returns=flat_ret,
            target_kl=target_kl,
        )

        writer.add_scalar('train/policy_loss',  metrics['policy_loss'],       timestep)
        writer.add_scalar('train/value_loss',   metrics['value_loss'],        timestep)
        writer.add_scalar('train/entropy',      metrics['entropy'],           timestep)
        writer.add_scalar('train/learning_rate', lr_now,                      timestep)
        writer.add_scalar('train/clip_range',   agent.clip_range,             timestep)
        writer.add_scalar('train/kl_early_stop', int(metrics['kl_early_stop']), timestep)

        # Progress logging
        if len(recent_rewards) > 0:
            mean_reward = np.mean(recent_rewards[-100:])
            print(f"Step {timestep:>8,} | Ep {episode:>5} | "
                  f"Mean Reward (100): {mean_reward:>8.2f} | "
                  f"Policy Loss: {metrics['policy_loss']:.4f} | "
                  f"Value Loss: {metrics['value_loss']:.4f}")

        # -------------------------------------------------------------------
        # Checkpoint
        # -------------------------------------------------------------------
        if timestep % checkpoint_freq < transitions_per_update:
            ckpt_path = os.path.join(save_dir, f"ppo_expert_{timestep}.pt")
            agent.save(ckpt_path)
            norm_path = os.path.join(save_dir, f"obs_rms_{timestep}.npz")
            np.savez(norm_path, **obs_rms.state_dict())

        # -------------------------------------------------------------------
        # Evaluation
        # -------------------------------------------------------------------
        if timestep % eval_freq < transitions_per_update:
            eval_reward = evaluate(eval_env, agent, obs_rms, eval_episodes)
            writer.add_scalar('eval/mean_reward', eval_reward, timestep)
            print(f"  [Eval] Mean reward over {eval_episodes} episodes: {eval_reward:.2f}")

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_path = os.path.join(save_dir, "best_model.pt")
                agent.save(best_path)
                best_norm_path = os.path.join(save_dir, "best_obs_rms.npz")
                np.savez(best_norm_path, **obs_rms.state_dict())
                print(f"  [Eval] New best model saved! Reward: {eval_reward:.2f}")

    # -----------------------------------------------------------------------
    # Save final model
    # -----------------------------------------------------------------------
    final_path = os.path.join(save_dir, "final_model.pt")
    agent.save(final_path)
    np.savez(os.path.join(save_dir, "final_obs_rms.npz"), **obs_rms.state_dict())

    vec_env.close()
    eval_env.close()
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

    return float(np.mean(total_rewards))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Expert for Quadrotor Control")
    parser.add_argument('--quadrotor-config', type=str, default='configs/quadrotor.yaml')
    parser.add_argument('--ppo-config',       type=str, default='configs/ppo_expert.yaml')
    parser.add_argument('--n-envs',           type=int, default=8,
                        help='Number of parallel environments (default: 8)')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    args = parser.parse_args()
    train(args)

"""
Phase 1 v4.0: Train State-Based PPO Expert (CTBR + INDI)

Trains a PPO agent on the QuadrotorEnvV4 environment which uses:
  - CTBR action space: [F_c_norm, wx_norm, wy_norm, wz_norm] in [-1, 1]
  - INDI inner-loop rate controller at 200Hz

Actor bias is initialised to the hover point:
  F_c_norm_hover = (mg / F_c_max) * 2 - 1 = (4.905 / 16.0) * 2 - 1 ≈ -0.387
  omega_cmd_hover = [0, 0, 0]

Usage:
    python -m scripts.train_ppo_expert_v4
    python -m scripts.train_ppo_expert_v4 --n-envs 16
    python -m scripts.train_ppo_expert_v4 --quick
"""

import os
import sys
import argparse
import numpy as np
import yaml
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env_v4 import QuadrotorEnvV4
from models.ppo_expert import PPOExpert, RunningMeanStd


class _EnvFactory:
    def __init__(self, config_path: str):
        self.config_path = os.path.abspath(config_path)

    def __call__(self) -> QuadrotorEnvV4:
        return QuadrotorEnvV4(config_path=self.config_path)


def _make_vec_env(config_path: str, n_envs: int):
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

    vec_env  = _make_vec_env(args.quadrotor_config, n_envs)
    eval_env = QuadrotorEnvV4(config_path=args.quadrotor_config)

    state_dim  = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

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

    # Initialise actor output at hover: F_c_norm ≈ -0.387, omega_cmd = [0,0,0]
    with torch.no_grad():
        agent.actor.mean_layer.bias.copy_(
            torch.tensor([-0.387, 0.0, 0.0, 0.0])
        )

    obs_rms = RunningMeanStd(shape=(state_dim,))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir  = os.path.join(log_cfg['tensorboard_log'], timestamp)
    save_dir = os.path.join(log_cfg['save_path'], timestamp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    total_timesteps = train_cfg['total_timesteps']
    n_steps         = train_cfg['n_steps']
    checkpoint_freq = log_cfg['checkpoint_freq']
    eval_freq       = log_cfg['eval_freq']
    eval_episodes   = log_cfg['eval_episodes']
    target_kl       = train_cfg.get('target_kl', None)

    transitions_per_update = n_steps * n_envs

    device = next(agent.actor.parameters()).device

    print(f"\n{'='*60}")
    print(f"Training PPO Expert v4.0 — CTBR + INDI (Vectorized)")
    print(f"Total timesteps:      {total_timesteps:,}")
    print(f"Parallel envs:        {n_envs}")
    print(f"Rollout length/env:   {n_steps}")
    print(f"Transitions/update:   {transitions_per_update:,}")
    print(f"Hidden dim:           {net_cfg['hidden_dim']}")
    print(f"Device:               {device}")
    print(f"{'='*60}\n")

    obs, _ = vec_env.reset()
    obs_rms.update(obs)
    obs_norm = obs_rms.normalize(obs).astype(np.float32)

    timestep       = 0
    episode        = 0
    best_eval_rmse = float('inf')
    best_eval_reward = -float('inf')
    recent_rewards = []

    ep_rewards = np.zeros(n_envs, dtype=np.float32)
    ep_lengths = np.zeros(n_envs, dtype=np.int32)

    while timestep < total_timesteps:

        buf_states    = np.zeros((n_steps, n_envs, state_dim),  dtype=np.float32)
        buf_actions   = np.zeros((n_steps, n_envs, action_dim), dtype=np.float32)
        buf_log_probs = np.zeros((n_steps, n_envs),             dtype=np.float32)
        buf_rewards   = np.zeros((n_steps, n_envs),             dtype=np.float32)
        buf_dones     = np.zeros((n_steps, n_envs),             dtype=np.float32)
        buf_values    = np.zeros((n_steps, n_envs),             dtype=np.float32)

        for t in range(n_steps):
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

            finished = np.where(dones > 0.5)[0]
            for i in finished:
                recent_rewards.append(float(ep_rewards[i]))
                writer.add_scalar('rollout/ep_reward', ep_rewards[i], timestep)
                writer.add_scalar('rollout/ep_length', ep_lengths[i], timestep)
                ep_rewards[i] = 0.0
                ep_lengths[i] = 0
                episode += 1

            obs_rms.update(next_obs)
            obs_norm = obs_rms.normalize(next_obs).astype(np.float32)

        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_norm).to(device)
            bootstrap_vals = agent.critic(obs_t).squeeze(-1).cpu().numpy()

        last_dones     = buf_dones[-1]
        bootstrap_vals = np.where(last_dones > 0.5, 0.0, bootstrap_vals)

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

        flat_adv    = advantages_arr.reshape(-1)
        flat_ret    = returns_arr.reshape(-1)
        flat_memory = {
            'states':    buf_states.reshape(-1, state_dim),
            'actions':   buf_actions.reshape(-1, action_dim),
            'log_probs': buf_log_probs.reshape(-1),
            'rewards':   buf_rewards.reshape(-1),
            'dones':     buf_dones.reshape(-1),
            'values':    buf_values.reshape(-1),
        }

        frac   = max(1.0 - timestep / total_timesteps, 0.0)
        lr_now = train_cfg['learning_rate'] * frac
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = lr_now

        clip_start       = train_cfg['clip_range']
        clip_end         = train_cfg.get('clip_range_end', clip_start)
        agent.clip_range = clip_start * frac + clip_end * (1.0 - frac)

        metrics = agent.update(
            flat_memory,
            precomputed_advantages=flat_adv,
            precomputed_returns=flat_ret,
            target_kl=target_kl,
        )

        writer.add_scalar('train/policy_loss',   metrics['policy_loss'],        timestep)
        writer.add_scalar('train/value_loss',    metrics['value_loss'],         timestep)
        writer.add_scalar('train/entropy',       metrics['entropy'],            timestep)
        writer.add_scalar('train/learning_rate', lr_now,                        timestep)
        writer.add_scalar('train/clip_range',    agent.clip_range,              timestep)
        writer.add_scalar('train/kl_early_stop', int(metrics['kl_early_stop']), timestep)

        if len(recent_rewards) > 0:
            mean_reward = np.mean(recent_rewards[-100:])
            print(f"Step {timestep:>8,} | Ep {episode:>5} | "
                  f"Mean Reward (100): {mean_reward:>8.2f} | "
                  f"Policy Loss: {metrics['policy_loss']:.4f} | "
                  f"Value Loss: {metrics['value_loss']:.4f}")

        if timestep % checkpoint_freq < transitions_per_update:
            ckpt_path = os.path.join(save_dir, f"ppo_expert_v4_{timestep}.pt")
            agent.save(ckpt_path)
            np.savez(os.path.join(save_dir, f"obs_rms_{timestep}.npz"), **obs_rms.state_dict())

        if timestep % eval_freq < transitions_per_update:
            eval_reward, eval_rmse = evaluate(eval_env, agent, obs_rms, eval_episodes)
            writer.add_scalar('eval/mean_reward', eval_reward, timestep)
            writer.add_scalar('eval/rmse_m',      eval_rmse,   timestep)
            print(f"  [Eval] Reward: {eval_reward:.2f} | RMSE: {eval_rmse:.4f}m")

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_eval_rmse   = eval_rmse
                agent.save(os.path.join(save_dir, "best_model.pt"))
                np.savez(os.path.join(save_dir, "best_obs_rms.npz"), **obs_rms.state_dict())
                print(f"  [Eval] New best saved! Reward: {eval_reward:.2f} | RMSE: {eval_rmse:.4f}m")

    agent.save(os.path.join(save_dir, "final_model.pt"))
    np.savez(os.path.join(save_dir, "final_obs_rms.npz"), **obs_rms.state_dict())

    vec_env.close()
    eval_env.close()
    writer.close()
    print(f"\nTraining complete! Models saved to: {save_dir}")
    print(f"Best eval reward: {best_eval_reward:.2f} | Best RMSE: {best_eval_rmse:.4f}m")


def evaluate(env: QuadrotorEnvV4, agent: PPOExpert,
             obs_rms: RunningMeanStd, n_episodes: int):
    """Run evaluation episodes; returns (mean_reward, mean_rmse)."""
    total_rewards = []
    total_rmses   = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        state_norm = obs_rms.normalize(state)
        ep_reward = 0.0
        sq_errs   = []
        done = False

        while not done:
            action = agent.get_action_deterministic(state_norm)
            state, reward, terminated, truncated, info = env.step(action)
            state_norm = obs_rms.normalize(state)
            ep_reward += reward

            pos = info.get('position', np.zeros(3))
            tgt = info.get('target',   np.zeros(3))
            sq_errs.append(float(np.sum((pos - tgt) ** 2)))

            done = terminated or truncated

        total_rewards.append(ep_reward)
        total_rmses.append(float(np.sqrt(np.mean(sq_errs))) if sq_errs else 0.0)

    return float(np.mean(total_rewards)), float(np.mean(total_rmses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Expert v4.0 — CTBR + INDI")
    parser.add_argument('--quadrotor-config', type=str, default='configs/quadrotor_v4.yaml')
    parser.add_argument('--ppo-config',       type=str, default='configs/ppo_expert_v4.yaml')
    parser.add_argument('--n-envs',           type=int, default=8,
                        help='Number of parallel environments (default: 8)')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    args = parser.parse_args()
    train(args)

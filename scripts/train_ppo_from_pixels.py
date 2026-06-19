"""
P1 baseline: PPO-from-pixels (end-to-end RL from the 6x64x64 FPV stack).

Trains a CNN actor-critic (``models.ppo_pixel.PPOPixel``) on
``QuadrotorVisualEnv(QuadrotorEnvV4)`` with the SAME CTBR + INDI dynamics, DR
renderer, and 50 Hz outer loop the flow policies were evaluated on. Mirrors the
state-based ``train_ppo_expert_v4`` loop, with two pixel-specific additions:
  * T_obs=2 frame stacking maintained across the vectorised envs (auto-reset aware);
  * a uint8 rollout buffer (float32-on-GPU would need ~13 GB).

Prior: vision RL from scratch collapses (27 ReinFlow runs); this documents how far
naive pixel-PPO gets, it is not expected to be competitive.

Usage:
  dppo/Scripts/python.exe -m scripts.train_ppo_from_pixels                 # full
  dppo/Scripts/python.exe -m scripts.train_ppo_from_pixels --quick         # smoke
  dppo/Scripts/python.exe -m scripts.train_ppo_from_pixels --n-envs 8
"""
import os
import sys
import time
import argparse
import numpy as np
import yaml
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env_v4 import QuadrotorEnvV4
from envs.quadrotor_visual_env import QuadrotorVisualEnv
from models.ppo_pixel import PPOPixel


class _VisualEnvFactory:
    """Picklable factory: QuadrotorVisualEnv(QuadrotorEnvV4) for vector envs."""

    def __init__(self, config_path: str, image_size: int = 64):
        self.config_path = os.path.abspath(config_path)
        self.image_size = image_size

    def __call__(self):
        base = QuadrotorEnvV4(config_path=self.config_path)
        return QuadrotorVisualEnv(base, image_size=self.image_size)


def _make_vec_env(config_path: str, image_size: int, n_envs: int):
    from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
    env_fns = [_VisualEnvFactory(config_path, image_size) for _ in range(n_envs)]
    try:
        vec_env = AsyncVectorEnv(env_fns)
        print(f"[VecEnv] AsyncVectorEnv x{n_envs} (multiprocessing)")
    except Exception as exc:
        print(f"[VecEnv] AsyncVectorEnv failed ({exc}); SyncVectorEnv fallback")
        vec_env = SyncVectorEnv(env_fns)
    return vec_env


def _stack(prev_u8: np.ndarray, cur_u8: np.ndarray) -> np.ndarray:
    """(n,3,H,W) + (n,3,H,W) -> (n,6,H,W) uint8 contiguous."""
    return np.concatenate([prev_u8, cur_u8], axis=1)


@torch.no_grad()
def evaluate_survival(config_path, image_size, agent, T_obs, n_episodes,
                      survive_threshold):
    """Deterministic closed-loop rollout; returns (mean_reward, mean_len, survival)."""
    env = QuadrotorVisualEnv(QuadrotorEnvV4(config_path=config_path),
                             image_size=image_size)
    max_steps = env.env.max_episode_steps
    thr = min(survive_threshold, max_steps)
    rewards, lengths, survived = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        buf = [obs['image']] * T_obs
        done = False
        ep_r = 0.0
        ep_l = 0
        while not done:
            stacked = np.concatenate(buf[-T_obs:], axis=0)[None]   # (1,6,H,W)
            action = agent.get_action_deterministic(stacked)[0]
            obs, r, term, trunc, _ = env.step(action)
            buf.append(obs['image'])
            ep_r += r
            ep_l += 1
            done = term or trunc
        rewards.append(ep_r)
        lengths.append(ep_l)
        survived.append(float(ep_l >= thr))
    env.close()
    return float(np.mean(rewards)), float(np.mean(lengths)), float(np.mean(survived))


def train(args):
    with open(args.ppo_config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    train_cfg, net_cfg, log_cfg = cfg['training'], cfg['network'], cfg['logging']

    if args.quick:
        train_cfg['total_timesteps'] = cfg['quick_test']['timesteps']
        log_cfg['eval_freq'] = train_cfg['total_timesteps']     # one eval at the end
        log_cfg['checkpoint_freq'] = train_cfg['total_timesteps']
        print("[Quick] reduced timesteps")

    n_envs = args.n_envs
    image_size = 64
    T_obs = net_cfg['T_obs']

    vec_env = _make_vec_env(args.quadrotor_config, image_size, n_envs)

    agent = PPOPixel(
        in_channels=T_obs * 3, feature_dim=net_cfg['feature_dim'],
        action_dim=net_cfg['action_dim'], hidden_dim=net_cfg['hidden_dim'],
        T_pred=8, lr=train_cfg['learning_rate'],
        critic_lr_multiplier=train_cfg.get('critic_lr_multiplier', 1.0),
        gamma=train_cfg['gamma'], gae_lambda=train_cfg['gae_lambda'],
        clip_range=train_cfg['clip_range'], ent_coef=train_cfg['ent_coef'],
        vf_coef=train_cfg['vf_coef'], max_grad_norm=train_cfg['max_grad_norm'],
        batch_size=train_cfg['batch_size'], n_epochs=train_cfg['n_epochs'],
    )
    device = next(agent.actor.parameters()).device
    action_dim = net_cfg['action_dim']
    C = T_obs * 3

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or timestamp
    log_dir = os.path.join(log_cfg['tensorboard_log'], tag)
    save_dir = os.path.join(log_cfg['save_path'], tag)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    total_timesteps = train_cfg['total_timesteps']
    n_steps = train_cfg['n_steps']
    checkpoint_freq = log_cfg['checkpoint_freq']
    eval_freq = log_cfg['eval_freq']
    eval_episodes = log_cfg['eval_episodes']
    target_kl = train_cfg.get('target_kl', None)
    transitions_per_update = n_steps * n_envs

    n_params = (sum(p.numel() for p in agent.actor.parameters())
                + sum(p.numel() for p in agent.critic.parameters()))
    print(f"\n{'='*60}\nPPO-from-pixels (CNN actor-critic, CTBR)\n"
          f"Total timesteps: {total_timesteps:,} | envs: {n_envs} | "
          f"rollout/env: {n_steps} | transitions/update: {transitions_per_update:,}\n"
          f"Params: {n_params:,} | Device: {device}\n{'='*60}\n", flush=True)

    obs, _ = vec_env.reset(seed=args.seed)
    cur_img = obs['image']                      # (n_envs, 3, H, W) uint8
    prev_img = cur_img.copy()                   # first stack duplicates frame 0

    timestep = 0
    episode = 0
    best_metric = -float('inf')                 # best by mean ep_length
    recent_rewards, recent_lengths = [], []
    ep_rewards = np.zeros(n_envs, dtype=np.float32)
    ep_lengths = np.zeros(n_envs, dtype=np.int32)

    while timestep < total_timesteps:
        buf_images = np.zeros((n_steps, n_envs, C, image_size, image_size), dtype=np.uint8)
        buf_actions = np.zeros((n_steps, n_envs, action_dim), dtype=np.float32)
        buf_log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        buf_rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        buf_dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        buf_values = np.zeros((n_steps, n_envs), dtype=np.float32)

        t0 = time.time()
        for t in range(n_steps):
            stacked = _stack(prev_img, cur_img)              # (n_envs, 6, H, W) uint8
            actions, log_probs, values = agent.get_action_vec(stacked)

            next_obs, rewards, terminated, truncated, _ = vec_env.step(actions)
            dones = (terminated | truncated).astype(np.float32)
            next_img = next_obs['image']

            buf_images[t] = stacked
            buf_actions[t] = actions
            buf_log_probs[t] = log_probs
            buf_rewards[t] = rewards
            buf_dones[t] = dones
            buf_values[t] = values

            ep_rewards += rewards
            ep_lengths += 1
            timestep += n_envs

            for i in np.where(dones > 0.5)[0]:
                recent_rewards.append(float(ep_rewards[i]))
                recent_lengths.append(int(ep_lengths[i]))
                writer.add_scalar('rollout/ep_reward', ep_rewards[i], timestep)
                writer.add_scalar('rollout/ep_length', ep_lengths[i], timestep)
                ep_rewards[i] = 0.0
                ep_lengths[i] = 0
                episode += 1

            # frame-stack bookkeeping (auto-reset aware):
            #   done  -> new episode's first frame duplicated (prev = cur = next_img)
            #   alive -> shift window (prev = old cur, cur = next_img)
            mask = (dones > 0.5).reshape(n_envs, 1, 1, 1)
            prev_img = np.where(mask, next_img, cur_img)
            cur_img = next_img

        # bootstrap value for the post-rollout stacked obs
        with torch.no_grad():
            boot_stacked = _stack(prev_img, cur_img)
            boot_x = torch.from_numpy(boot_stacked).to(device).float() / 255.0
            bootstrap_vals = agent.critic(boot_x).squeeze(-1).cpu().numpy()
        bootstrap_vals = np.where(buf_dones[-1] > 0.5, 0.0, bootstrap_vals)

        advantages_arr = np.zeros((n_steps, n_envs), dtype=np.float32)
        returns_arr = np.zeros((n_steps, n_envs), dtype=np.float32)
        for i in range(n_envs):
            adv_i, ret_i = agent.compute_gae(
                buf_rewards[:, i].tolist(), buf_values[:, i].tolist(),
                buf_dones[:, i].tolist(), last_value=float(bootstrap_vals[i]))
            advantages_arr[:, i] = adv_i
            returns_arr[:, i] = ret_i

        flat_memory = {
            'states': buf_images.reshape(-1, C, image_size, image_size),  # uint8
            'actions': buf_actions.reshape(-1, action_dim),
            'log_probs': buf_log_probs.reshape(-1),
        }

        frac = max(1.0 - timestep / total_timesteps, 0.0)
        lr_now = train_cfg['learning_rate'] * frac
        for pg in agent.optimizer.param_groups:
            pg['lr'] = lr_now
        clip_start = train_cfg['clip_range']
        clip_end = train_cfg.get('clip_range_end', clip_start)
        agent.clip_range = clip_start * frac + clip_end * (1.0 - frac)

        metrics = agent.update(flat_memory, advantages_arr.reshape(-1),
                               returns_arr.reshape(-1), target_kl=target_kl)

        writer.add_scalar('train/policy_loss', metrics['policy_loss'], timestep)
        writer.add_scalar('train/value_loss', metrics['value_loss'], timestep)
        writer.add_scalar('train/entropy', metrics['entropy'], timestep)
        writer.add_scalar('train/learning_rate', lr_now, timestep)
        writer.add_scalar('perf/rollout_sps', transitions_per_update / (time.time() - t0), timestep)

        if recent_rewards:
            mr = np.mean(recent_rewards[-100:])
            ml = np.mean(recent_lengths[-100:])
            print(f"Step {timestep:>9,} | Ep {episode:>5} | R(100) {mr:>8.2f} | "
                  f"len(100) {ml:>6.1f} | PL {metrics['policy_loss']:.4f} | "
                  f"VL {metrics['value_loss']:.3f} | {time.time()-t0:.1f}s", flush=True)

        if timestep % checkpoint_freq < transitions_per_update:
            agent.save(os.path.join(save_dir, f"ppo_pixels_{timestep}.pt"))

        if timestep % eval_freq < transitions_per_update:
            er, el, sr = evaluate_survival(args.quadrotor_config, image_size, agent,
                                           T_obs, eval_episodes, args.survive_threshold)
            writer.add_scalar('eval/mean_reward', er, timestep)
            writer.add_scalar('eval/mean_length', el, timestep)
            writer.add_scalar('eval/survival', sr, timestep)
            print(f"  [Eval] reward {er:.2f} | mean_len {el:.1f} | survive {sr*100:.0f}%",
                  flush=True)
            if el > best_metric:
                best_metric = el
                agent.save(os.path.join(save_dir, "best_model.pt"))
                print(f"  [Eval] new best (mean_len {el:.1f}) saved", flush=True)

    agent.save(os.path.join(save_dir, "final_model.pt"))
    vec_env.close()
    writer.close()
    print(f"\nDone. best mean_len={best_metric:.1f}  saved to {save_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P1 baseline: PPO-from-pixels")
    parser.add_argument('--quadrotor-config', type=str, default='configs/quadrotor_v4.yaml')
    parser.add_argument('--ppo-config', type=str, default='configs/ppo_from_pixels.yaml')
    parser.add_argument('--n-envs', type=int, default=8)
    parser.add_argument('--survive-threshold', type=int, default=250)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    train(args)

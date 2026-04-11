"""
Phase 3b: DPPO Fine-Tuning

Fine-tunes the pre-trained Vision Diffusion Policy using advantage-weighted
diffusion loss in a closed-loop RL setting.

L = E[ exp(beta * A_normalized) * ||eps_theta(a_t, t, s) - eps||^2 ]

Usage:
    python -m scripts.train_dppo --pretrained checkpoints/diffusion_policy/.../best_model.pt
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_visual_env import make_visual_env
from models.diffusion_policy import VisionDiffusionPolicy
from models.ppo_expert import RunningMeanStd


class ValueNetwork(nn.Module):
    """Value function for DPPO advantage estimation."""

    def __init__(self, feature_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def compute_gae(rewards: List[float], values: List[float],
                dones: List[bool], gamma: float, gae_lambda: float):
    """Compute GAE advantages and returns."""
    advantages = []
    returns = []
    gae = 0.0
    values_ext = values + [0.0]

    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values_ext[i + 1] * (1 - dones[i]) - values_ext[i]
        gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values_ext[i])

    return advantages, returns


def collect_rollout(env, policy, value_net, vision_encoder,
                    n_steps: int, T_obs: int, T_action: int,
                    device: torch.device):
    """
    Collect trajectory using RHC (Receding Horizon Control).

    Returns:
        rollout: dict with image_stacks, actions, rewards, dones, values
    """
    rollout = {
        'image_stacks': [], 'action_seqs': [],
        'rewards': [], 'dones': [], 'values': [],
    }

    obs, _ = env.reset()
    image_buffer = [obs['image']] * T_obs  # initialize with repeated first frame

    steps_collected = 0

    while steps_collected < n_steps:
        # Build image stack from buffer
        img_stack = np.concatenate(image_buffer[-T_obs:], axis=0)  # (T_obs*C, H, W)
        img_tensor = torch.from_numpy(img_stack).float().unsqueeze(0).to(device)

        # Get visual features for value estimation
        with torch.no_grad():
            vis_features = vision_encoder(img_tensor)
            value = value_net(vis_features).item()

        # Predict action sequence
        with torch.no_grad():
            action_seq = policy.predict_action(img_tensor)  # (1, T_pred, action_dim)
            action_seq = action_seq.squeeze(0).cpu().numpy()

        # Execute T_action steps (RHC)
        for a_idx in range(min(T_action, len(action_seq))):
            action = action_seq[a_idx]

            rollout['image_stacks'].append(img_stack.copy())
            rollout['action_seqs'].append(action_seq.copy())
            rollout['values'].append(value)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rollout['rewards'].append(reward)
            rollout['dones'].append(float(done))

            image_buffer.append(obs['image'])
            steps_collected += 1

            if done:
                obs, _ = env.reset()
                image_buffer = [obs['image']] * T_obs
                break

            if steps_collected >= n_steps:
                break

    return rollout


def train(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vision_cfg = config['vision']
    action_cfg = config['action']
    dppo_cfg = config['dppo']
    log_cfg = config['logging']

    # Create environment
    env = make_visual_env(
        config_path=args.quadrotor_config,
        image_size=vision_cfg['image_size'],
    )

    # Load pre-trained diffusion policy
    policy = VisionDiffusionPolicy(
        action_dim=action_cfg['action_dim'],
        T_obs=vision_cfg['T_obs'],
        T_pred=action_cfg['T_pred'],
        image_channels=vision_cfg['channels'],
        image_size=vision_cfg['image_size'],
        feature_dim=vision_cfg['feature_dim'],
        time_embed_dim=config['unet']['time_embed_dim'],
        down_dims=tuple(config['unet']['down_dims']),
        num_diffusion_steps=config['diffusion']['num_timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
        ddim_steps=config['diffusion']['ddim_steps'],
    ).to(device)

    if args.pretrained:
        policy.load(args.pretrained)
        print(f"Loaded pretrained policy from: {args.pretrained}")

    # Value network (shares visual encoder features)
    value_net = ValueNetwork(
        feature_dim=vision_cfg['feature_dim'],
        hidden_dim=dppo_cfg['value_hidden_dim'],
    ).to(device)
    if args.pretrained_value:
        value_net.load_state_dict(
            torch.load(args.pretrained_value, map_location=device, weights_only=True)
        )
        print(f"Loaded pretrained value net from: {args.pretrained_value}")

    # Optimizers
    policy_optimizer = torch.optim.AdamW(
        policy.parameters(), lr=dppo_cfg['learning_rate']
    )
    value_optimizer = torch.optim.Adam(
        value_net.parameters(), lr=dppo_cfg['value_lr']
    )

    # Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_cfg['tensorboard_log'], f"dppo_{timestamp}")
    save_dir = os.path.join(log_cfg['save_path'], f"dppo_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Training parameters
    n_rollout_steps      = dppo_cfg['n_rollout_steps']
    n_epochs             = dppo_cfg['n_epochs']
    beta                 = dppo_cfg['advantage_beta']
    gamma                = dppo_cfg['gamma']
    gae_lambda           = dppo_cfg['gae_lambda']
    value_warmup_updates = dppo_cfg.get('value_warmup_updates', 0)
    vloss_best_threshold = dppo_cfg.get('vloss_best_threshold', float('inf'))
    total_updates        = args.total_updates

    MINI_BATCH = 256   # safe for 24 GB VRAM

    print(f"\n{'='*60}")
    print(f"DPPO Fine-Tuning")
    print(f"Total updates: {total_updates} | Rollout steps: {n_rollout_steps}")
    print(f"Advantage beta: {beta}")
    print(f"Value warm-up: {value_warmup_updates} updates | VLoss best-ckpt threshold: {vloss_best_threshold}")
    print(f"{'='*60}\n")

    best_reward = -float('inf')

    for update in range(total_updates):
        in_warmup = update < value_warmup_updates

        # Collect rollout
        rollout = collect_rollout(
            env, policy, value_net, policy.vision_encoder,
            n_steps=n_rollout_steps,
            T_obs=vision_cfg['T_obs'],
            T_action=action_cfg['T_action'],
            device=device,
        )

        # Compute GAE
        advantages, returns = compute_gae(
            rollout['rewards'], rollout['values'],
            rollout['dones'], gamma, gae_lambda,
        )

        # Keep rollout data on CPU; slice mini-batches onto GPU to avoid OOM.
        img_stacks_cpu  = torch.FloatTensor(np.array(rollout['image_stacks']))
        action_seqs_cpu = torch.FloatTensor(np.array(rollout['action_seqs']))
        advantages_t    = torch.FloatTensor(advantages)
        returns_t       = torch.FloatTensor(returns)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        N = len(advantages_t)

        # Value network update — always runs (warm-up or joint)
        value_loss = torch.tensor(0.0)
        for _ in range(n_epochs):
            idx = torch.randperm(N)
            for start in range(0, N, MINI_BATCH):
                mb = idx[start:start + MINI_BATCH]
                with torch.no_grad():
                    vis_feat = policy.vision_encoder(
                        img_stacks_cpu[mb].to(device)
                    )
                vp = value_net(vis_feat).squeeze()
                vl = nn.functional.mse_loss(vp, returns_t[mb].to(device))
                value_optimizer.zero_grad()
                vl.backward()
                value_optimizer.step()
                value_loss = vl.detach()

        # Policy update — skipped during warm-up
        loss = torch.tensor(0.0)
        if not in_warmup:
            policy.train()
            for _ in range(n_epochs):
                idx = torch.randperm(N)
                for start in range(0, N, MINI_BATCH):
                    mb = idx[start:start + MINI_BATCH]
                    mb_loss = policy.compute_weighted_loss(
                        img_stacks_cpu[mb].to(device),
                        action_seqs_cpu[mb].to(device),
                        advantages_t[mb].to(device),
                        beta=beta,
                    )
                    policy_optimizer.zero_grad()
                    mb_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    policy_optimizer.step()
                    loss = mb_loss.detach()

        # Logging
        mean_reward = np.mean(rollout['rewards'])
        writer.add_scalar('dppo/mean_reward', mean_reward,       update)
        writer.add_scalar('dppo/policy_loss', loss.item(),       update)
        writer.add_scalar('dppo/value_loss',  value_loss.item(), update)
        writer.add_scalar('dppo/warmup',      int(in_warmup),    update)

        warmup_tag = " [WARMUP]" if in_warmup else ""
        print(f"Update {update+1:>4}/{total_updates}{warmup_tag} | "
              f"Reward: {mean_reward:.4f} | "
              f"Loss: {loss.item():.6f} | "
              f"VLoss: {value_loss.item():.6f}")

        # Save best ckpt only after VLoss is below threshold (advantage estimates reliable)
        if (not in_warmup
                and value_loss.item() < vloss_best_threshold
                and mean_reward > best_reward):
            best_reward = mean_reward
            policy.save(os.path.join(save_dir, "best_dppo_model.pt"))
            torch.save(value_net.state_dict(),
                       os.path.join(save_dir, "best_value_net.pt"))

    policy.save(os.path.join(save_dir, "final_dppo_model.pt"))
    torch.save(value_net.state_dict(),
               os.path.join(save_dir, "final_value_net.pt"))
    writer.close()
    print(f"\nDPPO training complete! Best reward: {best_reward:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPPO Fine-Tuning")
    parser.add_argument('--config', type=str, default='configs/diffusion_policy.yaml')
    parser.add_argument('--quadrotor-config', type=str, default='configs/quadrotor.yaml')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained diffusion policy')
    parser.add_argument('--pretrained-value', type=str, default=None,
                        help='Path to pretrained value net (best_value_net.pt from prior run)')
    parser.add_argument('--total-updates', type=int, default=500)
    args = parser.parse_args()
    train(args)

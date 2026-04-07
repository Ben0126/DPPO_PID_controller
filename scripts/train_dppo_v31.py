"""
Phase 3c v3.1: DPPO Closed-Loop RL Fine-tuning (IMU Late Fusion + Depth Aux)

Fine-tunes a VisionDPPOv31 checkpoint with advantage-weighted diffusion loss
in a closed-loop RL setting.

Loss per update:
    L = E[ exp(β × A_norm) × ||ε_θ - ε||² ]
      + λ_disp  × L_dispersive(vision_feat)
      + λ_depth × MSE(depth_pred, depth_gt_rollout)

Usage:
    python -m scripts.train_dppo_v31 \
        --pretrained checkpoints/diffusion_policy/v31_<timestamp>/best_model.pt
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
from models.vision_dppo_v31 import VisionDPPOv31
from models.ppo_expert import RunningMeanStd


class ValueNetworkV31(nn.Module):
    """Value function conditioned on 288D global_cond (vision + IMU)."""

    def __init__(self, global_cond_dim: int = 288, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_cond: torch.Tensor) -> torch.Tensor:
        return self.net(global_cond)


def compute_gae(rewards: List[float], values: List[float],
                dones: List[bool], gamma: float, gae_lambda: float):
    """Generalised Advantage Estimation."""
    advantages  = []
    returns     = []
    gae         = 0.0
    values_ext  = values + [0.0]

    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values_ext[i + 1] * (1 - dones[i]) - values_ext[i]
        gae   = delta + gamma * gae_lambda * (1 - dones[i]) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values_ext[i])

    return advantages, returns


def _get_imu(obs_state: np.ndarray, prev_v_body: np.ndarray,
             dt: float) -> np.ndarray:
    """
    Compute 6D IMU vector from state observation.
    ω  = state[12:15]  (angular velocity, body frame)
    a  = finite difference of body-frame linear velocity
    """
    omega  = obs_state[12:15]
    v_body = obs_state[9:12]
    if prev_v_body is None:
        accel = np.zeros(3, dtype=np.float32)
    else:
        accel = ((v_body - prev_v_body) / dt).astype(np.float32)
    return np.concatenate([omega, accel]).astype(np.float32)   # (6,)


def collect_rollout(env, policy, value_net,
                    n_steps: int, T_obs: int, T_action: int,
                    lambda_depth: float, device: torch.device):
    """
    Collect trajectory using RHC (Receding Horizon Control).
    Captures image stacks, IMU, depth (if λ_depth > 0), actions, rewards.
    """
    dt = 1.0 / 50.0   # 50 Hz control loop

    rollout = {
        'image_stacks': [], 'action_seqs': [],
        'imu_data':     [], 'depth_maps':  [],
        'rewards':      [], 'dones':       [], 'values': [],
    }

    obs, _       = env.reset()
    image_buffer = [obs['image']] * T_obs
    prev_v_body  = None

    steps_collected = 0

    while steps_collected < n_steps:
        # Build image stack
        img_stack  = np.concatenate(image_buffer[-T_obs:], axis=0)   # (T_obs*C,H,W)
        img_tensor = torch.from_numpy(img_stack).float().unsqueeze(0).to(device)

        # IMU
        imu_vec    = _get_imu(obs['state'], prev_v_body, dt)
        imu_tensor = torch.from_numpy(imu_vec).float().unsqueeze(0).to(device)
        prev_v_body = obs['state'][9:12].copy()

        # Depth (for λ_depth loss; call renderer on current env state)
        if lambda_depth > 0:
            depth_frame = env._render_depth()   # (1, H, W) uint8
        else:
            depth_frame = np.zeros((1, env.image_size, env.image_size), dtype=np.uint8)

        # Value estimate using fused features
        with torch.no_grad():
            global_cond, _ = policy._encode(img_tensor, imu_tensor)
            value = value_net(global_cond).item()

        # Predict action sequence
        with torch.no_grad():
            action_seq = policy.predict_action(img_tensor, imu_tensor)
            action_seq = action_seq.squeeze(0).cpu().numpy()   # (T_pred, 4)

        # Execute T_action steps (RHC)
        for a_idx in range(min(T_action, len(action_seq))):
            action = action_seq[a_idx]

            rollout['image_stacks'].append(img_stack.copy())
            rollout['action_seqs'].append(action_seq.copy())
            rollout['imu_data'].append(imu_vec.copy())
            rollout['depth_maps'].append(depth_frame.copy())
            rollout['values'].append(value)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rollout['rewards'].append(reward)
            rollout['dones'].append(float(done))

            image_buffer.append(obs['image'])
            steps_collected += 1

            if done:
                obs, _      = env.reset()
                image_buffer = [obs['image']] * T_obs
                prev_v_body  = None
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
    dppo_cfg   = config['dppo']
    log_cfg    = config['logging']
    v31_cfg    = config.get('v31', {})

    lambda_dispersive = v31_cfg.get('lambda_dispersive', 0.1)
    lambda_depth      = v31_cfg.get('lambda_depth', 0.1)

    # Environment
    env = make_visual_env(
        config_path=args.quadrotor_config,
        image_size=vision_cfg['image_size'],
    )

    # Policy
    policy = VisionDPPOv31(
        action_dim=action_cfg['action_dim'],
        T_obs=vision_cfg['T_obs'],
        T_pred=action_cfg['T_pred'],
        image_channels=vision_cfg['channels'],
        image_size=vision_cfg['image_size'],
        time_embed_dim=config['unet']['time_embed_dim'],
        down_dims=tuple(config['unet']['down_dims']),
        num_diffusion_steps=config['diffusion']['num_timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
        ddim_steps=config['diffusion']['ddim_steps'],
        use_depth_decoder=(lambda_depth > 0),
    ).to(device)

    if args.pretrained:
        policy.load(args.pretrained)
        print(f"Loaded pretrained policy: {args.pretrained}")

    # Value network — takes 288D global_cond as input
    global_cond_dim = VisionDPPOv31.GLOBAL_COND_DIM   # 288
    value_net = ValueNetworkV31(
        global_cond_dim=global_cond_dim,
        hidden_dim=dppo_cfg['value_hidden_dim'],
    ).to(device)
    if args.pretrained_value:
        value_net.load_state_dict(
            torch.load(args.pretrained_value, map_location=device, weights_only=True)
        )
        print(f"Loaded pretrained value net: {args.pretrained_value}")

    # Optimizers
    policy_optimizer = torch.optim.AdamW(
        policy.parameters(), lr=dppo_cfg['learning_rate']
    )
    value_optimizer = torch.optim.Adam(
        value_net.parameters(), lr=dppo_cfg['value_lr']
    )

    # Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag   = f"dppo_v31_{timestamp}"
    log_dir   = os.path.join(log_cfg['tensorboard_log'], run_tag)
    save_dir  = os.path.join(log_cfg['save_path'],       run_tag)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    n_rollout_steps = dppo_cfg['n_rollout_steps']
    n_epochs        = dppo_cfg['n_epochs']
    beta            = dppo_cfg['advantage_beta']
    gamma           = dppo_cfg['gamma']
    gae_lambda      = dppo_cfg['gae_lambda']
    total_updates   = args.total_updates

    print(f"\n{'='*60}")
    print(f"DPPO v3.1 Fine-Tuning")
    print(f"Total updates: {total_updates} | Rollout steps: {n_rollout_steps}")
    print(f"β={beta} | λ_disp={lambda_dispersive} | λ_depth={lambda_depth}")
    print(f"{'='*60}\n")

    best_reward = -float('inf')

    for update in range(total_updates):
        # Collect rollout
        rollout = collect_rollout(
            env, policy, value_net,
            n_steps=n_rollout_steps,
            T_obs=vision_cfg['T_obs'],
            T_action=action_cfg['T_action'],
            lambda_depth=lambda_depth,
            device=device,
        )

        # GAE
        advantages, returns = compute_gae(
            rollout['rewards'], rollout['values'],
            rollout['dones'], gamma, gae_lambda,
        )

        # Keep rollout data on CPU; slice mini-batches onto GPU to avoid OOM.
        img_stacks_cpu   = torch.FloatTensor(np.array(rollout['image_stacks']))
        action_seqs_cpu  = torch.FloatTensor(np.array(rollout['action_seqs']))
        imu_data_cpu     = torch.FloatTensor(np.array(rollout['imu_data']))
        depth_gt_cpu     = torch.FloatTensor(np.array(rollout['depth_maps']))
        advantages_t     = torch.FloatTensor(advantages)
        returns_t        = torch.FloatTensor(returns)

        # Normalise advantages (on CPU, then keep)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        N          = len(advantages_t)
        MINI_BATCH = 256   # safe for 24 GB VRAM

        # Policy update — mini-batch SGD over n_epochs
        policy.train()
        loss    = torch.tensor(0.0)
        metrics = {'loss_diffusion': 0.0, 'loss_dispersive': 0.0, 'loss_depth': 0.0}
        for _ in range(n_epochs):
            idx = torch.randperm(N)
            for start in range(0, N, MINI_BATCH):
                mb = idx[start:start + MINI_BATCH]
                mb_loss, mb_m = policy.compute_weighted_loss(
                    img_stacks_cpu[mb].to(device),
                    action_seqs_cpu[mb].to(device),
                    imu_data_cpu[mb].to(device),
                    advantages=advantages_t[mb].to(device),
                    beta=beta,
                    depth_gt=depth_gt_cpu[mb].to(device) if lambda_depth > 0 else None,
                    lambda_dispersive=lambda_dispersive,
                    lambda_depth=lambda_depth,
                )
                policy_optimizer.zero_grad()
                mb_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                policy_optimizer.step()
                loss    = mb_loss.detach()
                metrics = mb_m

        # Value network update — mini-batch
        value_loss = torch.tensor(0.0)
        for _ in range(n_epochs):
            idx = torch.randperm(N)
            for start in range(0, N, MINI_BATCH):
                mb = idx[start:start + MINI_BATCH]
                with torch.no_grad():
                    gc, _ = policy._encode(
                        img_stacks_cpu[mb].to(device),
                        imu_data_cpu[mb].to(device),
                    )
                vp   = value_net(gc).squeeze()
                vl   = nn.functional.mse_loss(vp, returns_t[mb].to(device))
                value_optimizer.zero_grad()
                vl.backward()
                value_optimizer.step()
                value_loss = vl.detach()

        # Logging
        mean_reward = np.mean(rollout['rewards'])
        writer.add_scalar('dppo_v31/mean_reward',        mean_reward,                update)
        writer.add_scalar('dppo_v31/policy_loss',        loss.item(),                update)
        writer.add_scalar('dppo_v31/value_loss',         value_loss.item(),          update)
        writer.add_scalar('dppo_v31/loss_diffusion',     metrics['loss_diffusion'],  update)
        writer.add_scalar('dppo_v31/loss_dispersive',    metrics['loss_dispersive'], update)
        writer.add_scalar('dppo_v31/loss_depth',         metrics['loss_depth'],      update)

        print(f"Update {update+1:>4}/{total_updates} | "
              f"Reward: {mean_reward:.4f} | "
              f"Loss: {loss.item():.6f} "
              f"(diff={metrics['loss_diffusion']:.4f} "
              f"disp={metrics['loss_dispersive']:.4f} "
              f"depth={metrics['loss_depth']:.4f}) | "
              f"VLoss: {value_loss.item():.6f}")

        if mean_reward > best_reward:
            best_reward = mean_reward
            policy.save(os.path.join(save_dir, "best_dppo_v31_model.pt"))
            torch.save(value_net.state_dict(),
                       os.path.join(save_dir, "best_value_net_v31.pt"))

    policy.save(os.path.join(save_dir, "final_dppo_v31_model.pt"))
    policy.save_deployable(os.path.join(save_dir, "deploy_model.pt"))
    torch.save(value_net.state_dict(),
               os.path.join(save_dir, "final_value_net_v31.pt"))
    writer.close()
    print(f"\nDPPO v3.1 training complete! Best reward: {best_reward:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPPO v3.1 Fine-Tuning")
    parser.add_argument('--config',           type=str, default='configs/diffusion_policy.yaml')
    parser.add_argument('--quadrotor-config', type=str, default='configs/quadrotor.yaml')
    parser.add_argument('--pretrained',       type=str, default=None,
                        help='Path to pretrained VisionDPPOv31 checkpoint')
    parser.add_argument('--pretrained-value', type=str, default=None,
                        help='Path to pretrained value net checkpoint')
    parser.add_argument('--total-updates',    type=int, default=500)
    args = parser.parse_args()
    train(args)

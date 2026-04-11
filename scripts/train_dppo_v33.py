"""
Phase 3c v3.3: DPPO Closed-Loop RL Fine-tuning (physics-based IMU)

Identical training loop to train_dppo_v31.py — same VisionDPPOv31 model,
same ValueNetworkV31, same advantage-weighted diffusion loss, same
value-warm-up + vloss-gated best-checkpoint logic.

The only substantive change: the 6-D IMU vector is pulled directly from
env.unwrapped.get_imu() (body-frame gyro + specific force) instead of
being estimated by finite-differencing v_body. This removes the
supervised→RL covariate shift documented in docs/dev_log_phase2_3.md §13.4.

Usage:
    python -m scripts.train_dppo_v33 \
        --pretrained checkpoints/diffusion_policy/v33_<timestamp>/best_model.pt
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
from scripts.train_dppo_v31 import ValueNetworkV31, compute_gae


def collect_rollout(env, policy, value_net,
                    n_steps: int, T_obs: int, T_action: int,
                    lambda_depth: float, device: torch.device):
    """
    RHC rollout for v3.3: IMU comes from env.unwrapped.get_imu() directly.
    No more finite-difference; no more prev_v_body bookkeeping.
    """
    rollout = {
        'image_stacks': [], 'action_seqs': [],
        'imu_data':     [], 'depth_maps':  [],
        'rewards':      [], 'dones':       [], 'values': [],
    }

    obs, _       = env.reset()
    image_buffer = [obs['image']] * T_obs
    base_env     = env.unwrapped   # QuadrotorEnv with get_imu()

    steps_collected = 0

    while steps_collected < n_steps:
        img_stack  = np.concatenate(image_buffer[-T_obs:], axis=0)
        img_tensor = torch.from_numpy(img_stack).float().unsqueeze(0).to(device)

        # v3.3: physics-based IMU
        imu_vec    = base_env.get_imu()
        imu_tensor = torch.from_numpy(imu_vec).float().unsqueeze(0).to(device)

        if lambda_depth > 0:
            depth_frame = env._render_depth()
        else:
            depth_frame = np.zeros((1, env.image_size, env.image_size), dtype=np.uint8)

        with torch.no_grad():
            global_cond, _ = policy._encode(img_tensor, imu_tensor)
            value = value_net(global_cond).item()

        with torch.no_grad():
            action_seq = policy.predict_action(img_tensor, imu_tensor)
            action_seq = action_seq.squeeze(0).cpu().numpy()

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
                obs, _       = env.reset()
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
    dppo_cfg   = config['dppo']
    log_cfg    = config['logging']
    v31_cfg    = config.get('v31', {})

    lambda_dispersive = v31_cfg.get('lambda_dispersive', 0.1)
    lambda_depth      = v31_cfg.get('lambda_depth', 0.1)

    env = make_visual_env(
        config_path=args.quadrotor_config,
        image_size=vision_cfg['image_size'],
    )

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

    policy_optimizer = torch.optim.AdamW(
        policy.parameters(), lr=dppo_cfg['learning_rate']
    )
    value_optimizer = torch.optim.Adam(
        value_net.parameters(), lr=dppo_cfg['value_lr']
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag   = f"dppo_v33_{timestamp}"
    log_dir   = os.path.join(log_cfg['tensorboard_log'], run_tag)
    save_dir  = os.path.join(log_cfg['save_path'],       run_tag)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    n_rollout_steps       = dppo_cfg['n_rollout_steps']
    n_epochs              = dppo_cfg['n_epochs']
    beta                  = dppo_cfg['advantage_beta']
    gamma                 = dppo_cfg['gamma']
    gae_lambda            = dppo_cfg['gae_lambda']
    value_warmup_updates  = dppo_cfg.get('value_warmup_updates', 0)
    vloss_best_threshold  = dppo_cfg.get('vloss_best_threshold', float('inf'))
    total_updates         = args.total_updates

    print(f"\n{'='*60}")
    print(f"DPPO v3.3 Fine-Tuning (physics-based IMU)")
    print(f"Total updates: {total_updates} | Rollout steps: {n_rollout_steps}")
    print(f"β={beta} | λ_disp={lambda_dispersive} | λ_depth={lambda_depth}")
    print(f"Value warm-up: {value_warmup_updates} updates | VLoss best-ckpt threshold: {vloss_best_threshold}")
    print(f"{'='*60}\n")

    best_reward = -float('inf')

    for update in range(total_updates):
        in_warmup = update < value_warmup_updates

        rollout = collect_rollout(
            env, policy, value_net,
            n_steps=n_rollout_steps,
            T_obs=vision_cfg['T_obs'],
            T_action=action_cfg['T_action'],
            lambda_depth=lambda_depth,
            device=device,
        )

        advantages, returns = compute_gae(
            rollout['rewards'], rollout['values'],
            rollout['dones'], gamma, gae_lambda,
        )

        img_stacks_cpu   = torch.FloatTensor(np.array(rollout['image_stacks']))
        action_seqs_cpu  = torch.FloatTensor(np.array(rollout['action_seqs']))
        imu_data_cpu     = torch.FloatTensor(np.array(rollout['imu_data']))
        depth_gt_cpu     = torch.FloatTensor(np.array(rollout['depth_maps']))
        advantages_t     = torch.FloatTensor(advantages)
        returns_t        = torch.FloatTensor(returns)

        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        N          = len(advantages_t)
        MINI_BATCH = 256   # safe for 24 GB VRAM

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
                vp = value_net(gc).squeeze()
                vl = nn.functional.mse_loss(vp, returns_t[mb].to(device))
                value_optimizer.zero_grad()
                vl.backward()
                value_optimizer.step()
                value_loss = vl.detach()

        loss    = torch.tensor(0.0)
        metrics = {'loss_diffusion': 0.0, 'loss_dispersive': 0.0, 'loss_depth': 0.0}
        if not in_warmup:
            policy.train()
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

        mean_reward = np.mean(rollout['rewards'])
        writer.add_scalar('dppo_v33/mean_reward',        mean_reward,                update)
        writer.add_scalar('dppo_v33/policy_loss',        loss.item(),                update)
        writer.add_scalar('dppo_v33/value_loss',         value_loss.item(),          update)
        writer.add_scalar('dppo_v33/loss_diffusion',     metrics['loss_diffusion'],  update)
        writer.add_scalar('dppo_v33/loss_dispersive',    metrics['loss_dispersive'], update)
        writer.add_scalar('dppo_v33/loss_depth',         metrics['loss_depth'],      update)
        writer.add_scalar('dppo_v33/warmup',             int(in_warmup),             update)

        warmup_tag = " [WARMUP]" if in_warmup else ""
        print(f"Update {update+1:>4}/{total_updates}{warmup_tag} | "
              f"Reward: {mean_reward:.4f} | "
              f"Loss: {loss.item():.6f} "
              f"(diff={metrics['loss_diffusion']:.4f} "
              f"disp={metrics['loss_dispersive']:.4f} "
              f"depth={metrics['loss_depth']:.4f}) | "
              f"VLoss: {value_loss.item():.6f}")

        if (not in_warmup
                and value_loss.item() < vloss_best_threshold
                and mean_reward > best_reward):
            best_reward = mean_reward
            policy.save(os.path.join(save_dir, "best_dppo_v33_model.pt"))
            torch.save(value_net.state_dict(),
                       os.path.join(save_dir, "best_value_net_v33.pt"))

    policy.save(os.path.join(save_dir, "final_dppo_v33_model.pt"))
    policy.save_deployable(os.path.join(save_dir, "deploy_model.pt"))
    torch.save(value_net.state_dict(),
               os.path.join(save_dir, "final_value_net_v33.pt"))
    writer.close()
    print(f"\nDPPO v3.3 training complete! Best reward: {best_reward:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPPO v3.3 Fine-Tuning")
    parser.add_argument('--config',           type=str, default='configs/diffusion_policy.yaml')
    parser.add_argument('--quadrotor-config', type=str, default='configs/quadrotor.yaml')
    parser.add_argument('--pretrained',       type=str, default=None,
                        help='Path to pretrained VisionDPPOv31 checkpoint (trained on v3.3 data)')
    parser.add_argument('--pretrained-value', type=str, default=None,
                        help='Path to pretrained value net checkpoint')
    parser.add_argument('--total-updates',    type=int, default=500)
    args = parser.parse_args()
    train(args)

"""
Phase 3b v4.0: ReinFlow — Advantage-Weighted Flow Matching RL Fine-Tuning
Run 10: Curriculum learning — hover first, then linearly ramp initial_pos_range/vel_range.

Curriculum stages (relative to gate-open update):
  Stage 1: hover  (gate+0  … gate+n_hover)  pos=pos_start, vel=vel_start
  Stage 2: ramp   (gate+n_hover … gate+n_hover+n_ramp)  linear interpolation
  Stage 3: OOD    (gate+n_hover+n_ramp …)   pos=pos_end,   vel=vel_end

Loss:
  L_total = exp(β × A_norm) × ||v_θ − v*||² + lambda_bc × L_bc_expert

VLoss gate: one-way — once VLoss < vloss_gate, policy updates start and never revert.

Usage:
    python -m scripts.train_reinflow_v4 \
        --pretrained checkpoints/flow_policy_v4/20260420_034314/best_model.pt
"""

import os
import sys
import argparse
import yaml
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env_v4 import QuadrotorEnvV4
from envs.quadrotor_visual_env import QuadrotorVisualEnv
from models.flow_policy_v4 import FlowMatchingPolicyV4


# ---------------------------------------------------------------------------
# Value network
# ---------------------------------------------------------------------------

class ValueNetworkV4(nn.Module):
    """MLP value function conditioned on 288-D global_cond."""

    def __init__(self, global_cond_dim: int = 288, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_cond_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_cond: torch.Tensor) -> torch.Tensor:
        return self.net(global_cond).squeeze(-1)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(rewards: List[float], values: List[float],
                dones: List[float], gamma: float,
                gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
    N = len(rewards)
    advantages = np.zeros(N, dtype=np.float32)
    last_gae   = 0.0
    for t in reversed(range(N)):
        next_val = values[t + 1] if t + 1 < N else 0.0
        delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
        last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + np.array(values[:N], dtype=np.float32)
    return advantages, returns


# ---------------------------------------------------------------------------
# Demo dataset loader (for BC regularization)
# ---------------------------------------------------------------------------

def load_demo_subset(h5_path: str, n_episodes: int,
                     T_obs: int, T_pred: int):
    """Load n_episodes from expert demo h5 into RAM for BC loss."""
    img_buf, imu_buf, act_buf = [], [], []
    with h5py.File(h5_path, 'r') as f:
        keys = sorted([k for k in f.keys() if k.startswith('episode_')])[:n_episodes]
        for key in keys:
            imgs = f[key]['images'][:]      # (T, 3, H, W) uint8
            acts = f[key]['actions'][:]     # (T, 4) float32
            imus = f[key]['imu_data'][:]    # (T, 6) float32
            T = acts.shape[0]
            for start in range(T_obs - 1, T - T_pred):
                frames = imgs[start - T_obs + 1 : start + 1]
                img_buf.append(np.concatenate(frames, axis=0))  # (T_obs*3, H, W)
                imu_buf.append(imus[start])                     # (6,)
                act_buf.append(acts[start + 1 : start + 1 + T_pred].T)  # (4, T_pred)
    img_cpu = torch.from_numpy(np.stack(img_buf))   # uint8
    imu_cpu = torch.FloatTensor(np.stack(imu_buf))
    act_cpu = torch.FloatTensor(np.stack(act_buf))
    n = len(img_cpu)
    print(f"  Demo subset: {n_episodes} eps → {n:,} samples | "
          f"img={img_cpu.nbytes/1e6:.0f}MB")
    return img_cpu, imu_cpu, act_cpu


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollout(env: QuadrotorVisualEnv,
                    base_env: QuadrotorEnvV4,
                    policy: FlowMatchingPolicyV4,
                    value_net: ValueNetworkV4,
                    n_steps: int, T_obs: int, T_action: int,
                    device: torch.device) -> Dict:
    rollout = {
        'image_stacks': [],   # (N,) × (T_obs*3, H, W) uint8
        'action_seqs':  [],   # (N,) × (action_dim, T_pred) float32
        'noise_seqs':   [],   # (N,) × (action_dim, T_pred) float32 — x1 used in predict
        'imu_data':     [],   # (N,) × (6,) float32
        'rewards':      [],
        'dones':        [],
        'values':       [],
    }

    obs, _ = env.reset()
    image_buffer = [obs['image']] * T_obs

    steps_collected = 0

    while steps_collected < n_steps:
        img_stack = np.concatenate(image_buffer[-T_obs:], axis=0)

        img_tensor = (torch.from_numpy(img_stack).float()
                      .unsqueeze(0).to(device) / 255.0)
        imu_vec    = base_env.get_imu()
        imu_tensor = torch.from_numpy(imu_vec).float().unsqueeze(0).to(device)

        with torch.no_grad():
            global_cond = policy._encode(img_tensor, imu_tensor)  # (1, 288)
            value       = value_net(global_cond).item()
            # Sample noise explicitly so we can store x1 for stable updates
            x1 = torch.randn(1, policy.action_dim, policy.T_pred, device=device)
            action_seq  = policy.predict_action(img_tensor, imu_tensor,
                                                _fixed_x1=x1)
            # (1, action_dim, T_pred)

        action_seq_np = action_seq.squeeze(0).cpu().numpy()  # (action_dim, T_pred)
        # T_pred steps, executed column-by-column: (action_dim, T_pred)
        actions_to_exec = action_seq_np.T  # (T_pred, action_dim)

        noise_np = x1.squeeze(0).cpu().numpy()  # (action_dim, T_pred)

        for a_idx in range(min(T_action, actions_to_exec.shape[0])):
            action = actions_to_exec[a_idx]

            rollout['image_stacks'].append(img_stack.copy())
            rollout['action_seqs'].append(action_seq_np.copy())  # (action_dim, T_pred)
            rollout['noise_seqs'].append(noise_np.copy())        # (action_dim, T_pred)
            rollout['imu_data'].append(imu_vec.copy())
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    with open(args.flow_config, 'r', encoding='utf-8') as f:
        flow_cfg = yaml.safe_load(f)
    with open(args.rl_config, 'r', encoding='utf-8') as f:
        rl_cfg = yaml.safe_load(f)['rl']
    with open(args.rl_config, 'r', encoding='utf-8') as f:
        log_cfg = yaml.safe_load(f)['logging']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vis_cfg  = flow_cfg['vision']
    act_cfg  = flow_cfg['action']
    flow_params = flow_cfg['flow']
    unet_cfg = flow_cfg['unet']
    imu_cfg  = flow_cfg['imu']

    # Env — use separate RL config for rollout (wider disturbances / init vel)
    rl_quad_cfg = getattr(args, 'rl_quadrotor_config', None) or args.quadrotor_config
    print(f"RL rollout env config: {rl_quad_cfg}")
    base_env   = QuadrotorEnvV4(config_path=rl_quad_cfg)
    visual_env = QuadrotorVisualEnv(base_env, image_size=vis_cfg['image_size'])

    # Policy
    policy = FlowMatchingPolicyV4(
        vision_feature_dim = vis_cfg['feature_dim'],
        imu_feature_dim    = imu_cfg['feature_dim'],
        time_embed_dim     = unet_cfg['time_embed_dim'],
        down_dims          = tuple(unet_cfg['down_dims']),
        T_obs              = vis_cfg['T_obs'],
        T_pred             = act_cfg['T_pred'],
        action_dim         = act_cfg['action_dim'],
        n_inference_steps  = flow_params['n_inference_steps'],
        t_embed_scale      = flow_params['t_embed_scale'],
    ).to(device)

    if args.pretrained:
        policy.load(args.pretrained)
        print(f"Loaded pretrained policy: {args.pretrained}")

    # Value network
    global_cond_dim = vis_cfg['feature_dim'] + imu_cfg['feature_dim']   # 288
    value_net = ValueNetworkV4(
        global_cond_dim=global_cond_dim,
        hidden_dim=rl_cfg['value_hidden_dim'],
    ).to(device)

    if args.pretrained_value:
        value_net.load_state_dict(torch.load(args.pretrained_value, map_location=device))
        print(f"Loaded pretrained value net: {args.pretrained_value}")

    policy_opt = torch.optim.AdamW(policy.parameters(), lr=rl_cfg['learning_rate'])
    value_opt  = torch.optim.Adam(value_net.parameters(), lr=rl_cfg['value_lr'])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag   = f"reinflow_v4_{timestamp}"
    log_dir   = os.path.join(log_cfg['tensorboard_log'], run_tag)
    save_dir  = os.path.join(log_cfg['save_path'],       run_tag)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    beta                 = rl_cfg['advantage_beta']
    gamma                = rl_cfg['gamma']
    gae_lambda           = rl_cfg['gae_lambda']
    n_rollout            = rl_cfg['n_rollout_steps']
    n_epochs             = rl_cfg['n_epochs']
    mini_batch           = rl_cfg['mini_batch']
    grad_clip            = rl_cfg['grad_clip']
    value_warmup         = rl_cfg['value_warmup_updates']
    vloss_thresh         = rl_cfg['vloss_best_threshold']
    vloss_gate           = rl_cfg.get('vloss_gate', 10.0)
    lambda_bc            = rl_cfg.get('lambda_bc', 0.0)
    total_updates        = rl_cfg['total_updates']

    # Curriculum config
    with open(args.rl_config, 'r', encoding='utf-8') as f:
        full_cfg = yaml.safe_load(f)
    cur_cfg = full_cfg.get('curriculum', {})
    curriculum_enabled = cur_cfg.get('enabled', False)
    cur_n_hover  = cur_cfg.get('n_hover_updates', 50)
    cur_n_ramp   = cur_cfg.get('n_ramp_updates', 200)
    cur_pos_start = cur_cfg.get('pos_start', base_env.initial_pos_range)
    cur_vel_start = cur_cfg.get('vel_start', base_env.initial_vel_range)
    cur_pos_end  = cur_cfg.get('pos_end',   cur_pos_start)
    cur_vel_end  = cur_cfg.get('vel_end',   cur_vel_start)
    cur_anchor_prob = cur_cfg.get('hover_anchor_prob', 0.0)

    # Softened crash penalty during RL rollout (Run 12: prevents shock gradient
    # destroying hover behaviour when drone crashes during OOD ramp)
    crash_penalty_rl = rl_cfg.get('crash_penalty_rl', None)
    if crash_penalty_rl is not None:
        original_crash_penalty = base_env.crash_penalty
        base_env.crash_penalty = crash_penalty_rl
        print(f"Crash penalty: RL rollout={crash_penalty_rl} (eval env retains {original_crash_penalty})")

    # Load expert demo subset for BC regularization
    demo_img_cpu = demo_imu_cpu = demo_act_cpu = None
    if lambda_bc > 0:
        demo_path = rl_cfg.get('demo_path', 'data/expert_demos_v4.h5')
        demo_eps  = rl_cfg.get('demo_episodes', 100)
        print(f"Loading demo subset ({demo_eps} eps) for BC reg (lambda_bc={lambda_bc})...")
        demo_img_cpu, demo_imu_cpu, demo_act_cpu = load_demo_subset(
            demo_path, demo_eps, vis_cfg['T_obs'], act_cfg['T_pred'])
    N_demo = len(demo_img_cpu) if demo_img_cpu is not None else 0

    if curriculum_enabled:
        print(f"Curriculum: hover({cur_pos_start}m/{cur_vel_start}m/s) "
              f"→ ramp {cur_n_ramp} upd → OOD({cur_pos_end}m/{cur_vel_end}m/s)")
        if cur_anchor_prob > 0:
            print(f"Hover anchor probability: {cur_anchor_prob} (resets forced to pos<=0.1m)")
        base_env.disturbance_enabled = False   # no disturbances during curriculum
        base_env.initial_pos_range = cur_pos_start
        base_env.initial_vel_range = cur_vel_start
        base_env.hover_anchor_prob = 0.0       # only enable post-gate

    print(f"\n{'='*60}")
    print(f"ReinFlow v4.0 — Curriculum (CTBR + INDI + Flow Matching RL + BC Reg)")
    print(f"Total updates: {total_updates} | Rollout steps: {n_rollout}")
    print(f"beta={beta} | LR_policy={rl_cfg['learning_rate']:.1e} | LR_value={rl_cfg['value_lr']:.1e}")
    print(f"lambda_bc={lambda_bc} | Demo samples: {N_demo:,}")
    print(f"Value warmup: {value_warmup} updates (one-way VLoss gate <{vloss_gate}) | Save threshold: {vloss_thresh}")
    print(f"Save: {save_dir}")
    print(f"{'='*60}\n")

    best_reward       = -float('inf')
    value_loss_t      = torch.tensor(float('inf'))
    vloss_gate_passed = False   # one-way gate: once passed, never re-enter warmup
    updates_since_gate = 0      # curriculum stage counter

    for update in range(total_updates):
        # One-way VLoss gate: warmup until BOTH time AND VLoss conditions met
        if not vloss_gate_passed:
            in_warmup = (update < value_warmup) or (value_loss_t.item() > vloss_gate)
            if not in_warmup:
                vloss_gate_passed = True
                print(f"  [Gate OPEN] VLoss={value_loss_t.item():.2f} < {vloss_gate} — policy updates start")
        else:
            in_warmup = False
            updates_since_gate += 1

        # ---- Curriculum: update env params ----
        if curriculum_enabled and vloss_gate_passed:
            if updates_since_gate <= cur_n_hover:
                # Stage 1: hover stabilisation (no anchor needed yet)
                cur_pos = cur_pos_start
                cur_vel = cur_vel_start
                base_env.hover_anchor_prob = 0.0
            elif updates_since_gate <= cur_n_hover + cur_n_ramp:
                # Stage 2: linear ramp + anchor (prevents catastrophic forgetting)
                t = (updates_since_gate - cur_n_hover) / cur_n_ramp
                cur_pos = cur_pos_start + t * (cur_pos_end - cur_pos_start)
                cur_vel = cur_vel_start + t * (cur_vel_end - cur_vel_start)
                base_env.hover_anchor_prob = cur_anchor_prob
            else:
                # Stage 3: full OOD + anchor
                cur_pos = cur_pos_end
                cur_vel = cur_vel_end
                base_env.hover_anchor_prob = cur_anchor_prob
            base_env.initial_pos_range = cur_pos
            base_env.initial_vel_range = cur_vel

        # ---- Rollout ----
        policy.eval()
        value_net.eval()
        rollout = collect_rollout(
            visual_env, base_env, policy, value_net,
            n_steps=n_rollout,
            T_obs=vis_cfg['T_obs'],
            T_action=act_cfg['T_action'],
            device=device,
        )

        advantages, returns = compute_gae(
            rollout['rewards'], rollout['values'],
            rollout['dones'], gamma, gae_lambda,
        )

        # Build CPU tensors (images stored uint8 → [0,255])
        img_cpu   = torch.from_numpy(np.array(rollout['image_stacks'], dtype=np.uint8))
        act_cpu   = torch.FloatTensor(np.array(rollout['action_seqs']))
        noise_cpu = torch.FloatTensor(np.array(rollout['noise_seqs']))
        imu_cpu   = torch.FloatTensor(np.array(rollout['imu_data']))
        adv_t     = torch.FloatTensor(advantages)
        ret_t     = torch.FloatTensor(returns)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        adv_t = torch.clamp(adv_t, -3.0, 3.0)

        N = len(adv_t)

        # ---- Value update ----
        value_net.train()
        for _ in range(n_epochs):
            idx = torch.randperm(N)
            for start in range(0, N, mini_batch):
                mb = idx[start:start + mini_batch]
                imgs_gpu = img_cpu[mb].to(device=device, dtype=torch.float32) / 255.0
                imu_gpu  = imu_cpu[mb].to(device)
                with torch.no_grad():
                    gc = policy._encode(imgs_gpu, imu_gpu)
                vp = value_net(gc)
                vl = nn.functional.mse_loss(vp, ret_t[mb].to(device))
                value_opt.zero_grad()
                vl.backward()
                value_opt.step()
                value_loss_t = vl.detach()

        # ---- Policy update (skip during warmup) ----
        policy_loss_t = torch.tensor(0.0)
        frac_pos = (advantages > 0).mean()
        if not in_warmup:
            policy.train()
            demo_perm = torch.randperm(N_demo) if N_demo > 0 else None
            demo_ptr  = 0
            for _ in range(n_epochs):
                idx = torch.randperm(N)
                for start in range(0, N, mini_batch):
                    mb = idx[start:start + mini_batch]
                    imgs_gpu = img_cpu[mb].to(device=device, dtype=torch.float32) / 255.0
                    act_gpu  = act_cpu[mb].to(device)
                    imu_gpu  = imu_cpu[mb].to(device)
                    adv_gpu  = adv_t[mb].to(device)

                    rl_loss = policy.compute_weighted_loss(
                        imgs_gpu, imu_gpu, act_gpu, adv_gpu, beta)

                    # BC regularization: anchor policy to expert demos
                    if lambda_bc > 0 and N_demo > 0:
                        if demo_ptr + mini_batch > N_demo:
                            demo_perm = torch.randperm(N_demo)
                            demo_ptr  = 0
                        dm = demo_perm[demo_ptr : demo_ptr + mini_batch]
                        demo_ptr += mini_batch
                        di = demo_img_cpu[dm].to(device=device, dtype=torch.float32) / 255.0
                        da = demo_act_cpu[dm].to(device)
                        du = demo_imu_cpu[dm].to(device)
                        bc_loss = policy.compute_loss(di, du, da)
                        pl = rl_loss + lambda_bc * bc_loss
                    else:
                        pl = rl_loss

                    if pl.requires_grad:
                        policy_opt.zero_grad()
                        pl.backward()
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
                        policy_opt.step()
                    policy_loss_t = rl_loss.detach()

        # ---- Logging ----
        mean_reward = np.mean(rollout['rewards'])
        writer.add_scalar('reinflow/mean_reward',  mean_reward,          update)
        writer.add_scalar('reinflow/policy_loss',  policy_loss_t.item(), update)
        writer.add_scalar('reinflow/value_loss',   value_loss_t.item(),  update)
        writer.add_scalar('reinflow/frac_pos_adv', float(frac_pos),      update)
        writer.add_scalar('reinflow/warmup',       int(in_warmup),       update)
        if curriculum_enabled:
            writer.add_scalar('curriculum/pos_range', base_env.initial_pos_range, update)
            writer.add_scalar('curriculum/vel_range', base_env.initial_vel_range, update)
            writer.add_scalar('curriculum/anchor_prob', getattr(base_env, 'hover_anchor_prob', 0.0), update)

        if curriculum_enabled and vloss_gate_passed:
            if updates_since_gate <= cur_n_hover:
                cur_stage_tag = f" [CUR S1 hover pos={base_env.initial_pos_range:.2f}m]"
            elif updates_since_gate <= cur_n_hover + cur_n_ramp:
                cur_stage_tag = f" [CUR S2 ramp pos={base_env.initial_pos_range:.2f}m]"
            else:
                cur_stage_tag = f" [CUR S3 OOD pos={base_env.initial_pos_range:.2f}m]"
        else:
            cur_stage_tag = ""
        warmup_tag = " [WARMUP]" if in_warmup else ""
        print(f"Update {update+1:>4}/{total_updates}{warmup_tag}{cur_stage_tag} | "
              f"Reward: {mean_reward:.4f} | "
              f"RLLoss: {policy_loss_t.item():.6f} | "
              f"VLoss: {value_loss_t.item():.6f}")

        # ---- Checkpoint ----
        if (not in_warmup
                and value_loss_t.item() < vloss_thresh
                and mean_reward > best_reward):
            best_reward = mean_reward
            policy.save(os.path.join(save_dir, "best_reinflow_model.pt"))
            torch.save(value_net.state_dict(),
                       os.path.join(save_dir, "best_value_net.pt"))
            print(f"  --> New best checkpoint (reward={best_reward:.4f})")

        if (update + 1) % log_cfg['save_freq'] == 0:
            policy.save(os.path.join(save_dir, f"update_{update+1}.pt"))

    policy.save(os.path.join(save_dir, "final_reinflow_model.pt"))
    torch.save(value_net.state_dict(),
               os.path.join(save_dir, "final_value_net.pt"))
    writer.close()
    print(f"\nReinFlow training complete!  Best reward: {best_reward:.4f}")
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReinFlow v4.0 RL Fine-tuning")
    parser.add_argument('--flow-config',          type=str, default='configs/flow_policy_v4.yaml')
    parser.add_argument('--rl-config',            type=str, default='configs/reinflow_v4.yaml')
    parser.add_argument('--quadrotor-config',     type=str, default='configs/quadrotor_v4.yaml',
                        help='Env config for evaluation (standard)')
    parser.add_argument('--rl-quadrotor-config',  type=str, default=None,
                        help='Env config for RL rollout (wider disturbances). '
                             'Defaults to --quadrotor-config if not set.')
    parser.add_argument('--pretrained',           type=str, default=None,
                        help='Path to pretrained FlowMatchingPolicyV4 checkpoint')
    parser.add_argument('--pretrained-value',     type=str, default=None)
    args = parser.parse_args()
    train(args)

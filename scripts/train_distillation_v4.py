"""
Phase 3e v4.0: Online Policy Distillation from PPO Expert (Teacher-Student DAgger)

Replaces ReinFlow / AWR completely. The flow-matching student rolls out
on-policy (DAgger-style state distribution); every step a state-based PPO
expert provides the "correct" 4D CTBR action target. The student's flow
matching loss is computed against a sliding-window (4, T_pred=8) sequence of
future teacher actions, so the model learns dense, low-variance action
supervision — not noisy scalar rewards.

Loss:
  target[t] = [a_teacher[t], a_teacher[t+1], ..., a_teacher[t+T_pred-1]]
              with zero-order-hold padding across episode boundaries
  L         = flow_matching_MSE(v_pred, eps - target[t])

No critic, no GAE, no advantage normalisation, no scalar reward in the loss.

Usage:
    python -m scripts.train_distillation_v4 \
        --pretrained checkpoints/flow_policy_v4/20260514_175219/best_model.pt
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env_v4 import QuadrotorEnvV4
from envs.quadrotor_visual_env import QuadrotorVisualEnv
from models.flow_policy_v4 import FlowMatchingPolicyV4
from models.ppo_expert import PPOExpert, RunningMeanStd


# ---------------------------------------------------------------------------
# Rollout: student acts; teacher provides per-step target action.
# ---------------------------------------------------------------------------

def collect_rollout(env: QuadrotorVisualEnv,
                    base_env: QuadrotorEnvV4,
                    policy: FlowMatchingPolicyV4,
                    teacher: PPOExpert,
                    obs_rms: RunningMeanStd,
                    n_steps: int, T_obs: int, T_action: int,
                    device: torch.device,
                    sde_noise_std: float = 0.05) -> Dict:
    """
    On-policy rollout with teacher labelling.

    Returns dict with:
      image_stacks    (N,) × (T_obs*3, H, W) uint8
      imu_data        (N,) × (6,) float32
      teacher_actions (N, 4)  float32 — deterministic PPO expert per-step action
      episode_ids     (N,)    int     — used for sliding-window boundary padding
      rewards         (N,)    float32 — logging only
      dones           (N,)    float32 — for crash-rate computation
      crashed         list of bool per finished episode
      ep_lengths      list of int  per finished episode
    """
    rollout = {
        'image_stacks':    [],
        'imu_data':        [],
        'teacher_actions': [],
        'episode_ids':     [],
        'rewards':         [],
        'dones':           [],
        'crashed':         [],
        'ep_lengths':      [],
    }

    obs, _ = env.reset()
    image_buffer = [obs['image']] * T_obs

    ep_id        = 0
    ep_len_cnt   = 0
    steps_collected = 0

    while steps_collected < n_steps:
        img_stack = np.concatenate(image_buffer[-T_obs:], axis=0)
        imu_vec   = base_env.get_imu()

        # ---- Student action (SDE-noisy 1-step Euler from x1=N(0,I)) ----
        img_tensor = (torch.from_numpy(img_stack).float()
                      .unsqueeze(0).to(device) / 255.0)
        imu_tensor = torch.from_numpy(imu_vec).float().unsqueeze(0).to(device)
        with torch.no_grad():
            global_cond = policy._encode(img_tensor, imu_tensor)
            x1 = torch.randn(1, policy.action_dim, policy.T_pred, device=device)
            t1 = policy._t_to_int(torch.ones(1, device=device))
            v  = policy.flow_net(x1, t1, global_cond)
            mu = x1 - v
            action_seq = mu + sde_noise_std * torch.randn_like(mu)
        actions_to_exec = action_seq.squeeze(0).cpu().numpy().T  # (T_pred, 4)

        for a_idx in range(min(T_action, actions_to_exec.shape[0])):
            action = actions_to_exec[a_idx]

            # ---- Teacher target for the CURRENT state (before env.step) ----
            state_15d  = obs['state']                  # 15D from QuadrotorVisualEnv Dict obs
            state_norm = obs_rms.normalize(state_15d)
            a_teacher  = teacher.get_action_deterministic(state_norm.astype(np.float32))  # (4,)

            rollout['image_stacks'].append(img_stack.copy())
            rollout['imu_data'].append(imu_vec.copy())
            rollout['teacher_actions'].append(a_teacher.copy())
            rollout['episode_ids'].append(ep_id)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rollout['rewards'].append(reward)
            rollout['dones'].append(float(done))

            image_buffer.append(obs['image'])
            steps_collected += 1
            ep_len_cnt += 1

            if done:
                # crashed = ended before max_episode_steps (matches eval convention)
                max_steps = getattr(base_env, 'max_episode_steps', 500)
                rollout['crashed'].append(bool(ep_len_cnt < max_steps))
                rollout['ep_lengths'].append(ep_len_cnt)
                obs, _ = env.reset()
                image_buffer = [obs['image']] * T_obs
                ep_id      += 1
                ep_len_cnt  = 0
                break

            if steps_collected >= n_steps:
                break

    return rollout


# ---------------------------------------------------------------------------
# Build sliding-window teacher action target sequence with zero-order-hold
# padding at episode boundaries.
# ---------------------------------------------------------------------------

def build_target_sequence(teacher_actions: np.ndarray,
                          episode_ids: np.ndarray,
                          T_pred: int) -> np.ndarray:
    """
    Args:
        teacher_actions: (N, action_dim)
        episode_ids:     (N,)
    Returns:
        target: (N, action_dim, T_pred)
    """
    N, A = teacher_actions.shape
    target = np.zeros((N, A, T_pred), dtype=np.float32)
    for t in range(N):
        last_valid = teacher_actions[t]
        for k in range(T_pred):
            j = t + k
            if j < N and episode_ids[j] == episode_ids[t]:
                last_valid = teacher_actions[j]
            target[t, :, k] = last_valid
    return target


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    with open(args.flow_config, 'r', encoding='utf-8') as f:
        flow_cfg = yaml.safe_load(f)
    with open(args.rl_config, 'r', encoding='utf-8') as f:
        full_cfg = yaml.safe_load(f)
    distill_cfg = full_cfg['distillation']
    teacher_cfg = full_cfg['teacher']
    log_cfg     = full_cfg['logging']
    cur_cfg     = full_cfg.get('curriculum', {})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    vis_cfg     = flow_cfg['vision']
    act_cfg     = flow_cfg['action']
    flow_params = flow_cfg['flow']
    unet_cfg    = flow_cfg['unet']
    imu_cfg     = flow_cfg['imu']

    # ------------------------------------------------------------------
    # Env
    # ------------------------------------------------------------------
    rl_quad_cfg = getattr(args, 'rl_quadrotor_config', None) or args.quadrotor_config
    print(f"RL rollout env config: {rl_quad_cfg}")
    base_env   = QuadrotorEnvV4(config_path=rl_quad_cfg)
    visual_env = QuadrotorVisualEnv(base_env, image_size=vis_cfg['image_size'])

    # ------------------------------------------------------------------
    # Student policy
    # ------------------------------------------------------------------
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
        print(f"Loaded pretrained student: {args.pretrained}")

    policy_opt = torch.optim.AdamW(policy.parameters(),
                                    lr=distill_cfg['learning_rate'])

    # ------------------------------------------------------------------
    # Teacher (state-based PPO expert) + obs normalisation
    # ------------------------------------------------------------------
    teacher = PPOExpert(state_dim=15, action_dim=4,
                        hidden_dim=teacher_cfg.get('hidden_dim', 256))
    teacher.load(teacher_cfg['model_path'])
    teacher.actor.eval()
    obs_rms = RunningMeanStd(shape=(15,))
    norm_data = np.load(teacher_cfg['obs_rms_path'])
    obs_rms.load_state_dict({
        'mean':  norm_data['mean'],
        'var':   norm_data['var'],
        'count': float(norm_data['count']),
    })
    print(f"Loaded teacher: {teacher_cfg['model_path']}")
    print(f"Loaded obs_rms: {teacher_cfg['obs_rms_path']}")

    # ------------------------------------------------------------------
    # Curriculum (re-uses the QuadrotorEnvV4 swift / anchor knobs)
    # ------------------------------------------------------------------
    curriculum_enabled = cur_cfg.get('enabled', False)
    cur_n_hover   = cur_cfg.get('n_hover_updates', 30)
    cur_n_ramp    = cur_cfg.get('n_ramp_updates', 50)
    cur_pos_start = cur_cfg.get('pos_start', base_env.initial_pos_range)
    cur_vel_start = cur_cfg.get('vel_start', base_env.initial_vel_range)
    cur_pos_end   = cur_cfg.get('pos_end',   cur_pos_start)
    cur_vel_end   = cur_cfg.get('vel_end',   cur_vel_start)
    cur_anchor_prob = cur_cfg.get('hover_anchor_prob', 0.0)
    cur_swift_prob  = cur_cfg.get('swift_perturbation_prob', 0.0)
    cur_swift_tilt  = cur_cfg.get('swift_perturb_tilt_deg', 25.0)
    cur_swift_vel   = cur_cfg.get('swift_perturb_vel',      1.0)
    cur_swift_term  = cur_cfg.get('swift_max_tilt_deg',     80.0)

    if curriculum_enabled:
        print(f"Curriculum: hover({cur_pos_start}m/{cur_vel_start}m/s) → "
              f"OOD({cur_pos_end}m/{cur_vel_end}m/s) "
              f"over {cur_n_hover}+{cur_n_ramp} updates")
        base_env.initial_pos_range = cur_pos_start
        base_env.initial_vel_range = cur_vel_start
        base_env.hover_anchor_prob       = 0.0
        base_env.swift_perturbation_prob = 0.0
        base_env.swift_perturb_tilt_deg  = cur_swift_tilt
        base_env.swift_perturb_vel       = cur_swift_vel
        base_env.swift_max_tilt_deg      = cur_swift_term

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag   = f"distillation_v4_{timestamp}"
    log_dir   = os.path.join(log_cfg['tensorboard_log'], run_tag)
    save_dir  = os.path.join(log_cfg['save_path'],       run_tag)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    total_updates = distill_cfg['total_updates']
    n_rollout     = distill_cfg['n_rollout_steps']
    n_epochs      = distill_cfg['n_epochs']
    mini_batch    = distill_cfg['mini_batch']
    grad_clip     = distill_cfg['grad_clip']
    sde_noise_std = distill_cfg.get('sde_noise_std', 0.05)
    T_pred        = act_cfg['T_pred']

    print(f"\n{'='*60}")
    print(f"Distillation v4.0 — Teacher: PPO Expert (state-based, 0 crash)")
    print(f"Total updates: {total_updates} | Rollout steps: {n_rollout}")
    print(f"LR: {distill_cfg['learning_rate']:.1e} | "
          f"epochs/upd: {n_epochs} | minibatch: {mini_batch}")
    print(f"SDE noise: {sde_noise_std} | T_pred: {T_pred}")
    print(f"Save: {save_dir}")
    print(f"{'='*60}\n")

    best_crash_rate = float('inf')

    for update in range(total_updates):
        # ---- Curriculum ----
        if curriculum_enabled:
            if update < cur_n_hover:
                cur_pos = cur_pos_start
                cur_vel = cur_vel_start
                base_env.hover_anchor_prob       = 0.0
                base_env.swift_perturbation_prob = 0.0
            elif update < cur_n_hover + cur_n_ramp:
                t = (update - cur_n_hover) / cur_n_ramp
                cur_pos = cur_pos_start + t * (cur_pos_end - cur_pos_start)
                cur_vel = cur_vel_start + t * (cur_vel_end - cur_vel_start)
                base_env.hover_anchor_prob       = cur_anchor_prob
                base_env.swift_perturbation_prob = cur_swift_prob
            else:
                cur_pos = cur_pos_end
                cur_vel = cur_vel_end
                base_env.hover_anchor_prob       = cur_anchor_prob
                base_env.swift_perturbation_prob = cur_swift_prob
            base_env.initial_pos_range = cur_pos
            base_env.initial_vel_range = cur_vel

        # ---- Rollout (student on-policy, teacher labels per step) ----
        policy.eval()
        rollout = collect_rollout(
            visual_env, base_env, policy, teacher, obs_rms,
            n_steps=n_rollout,
            T_obs=vis_cfg['T_obs'],
            T_action=act_cfg['T_action'],
            device=device,
            sde_noise_std=sde_noise_std,
        )

        # ---- Build teacher target sequence (N, 4, T_pred) ----
        teacher_actions = np.stack(rollout['teacher_actions']).astype(np.float32)  # (N, 4)
        episode_ids     = np.array(rollout['episode_ids'], dtype=np.int64)
        target_actions  = build_target_sequence(teacher_actions, episode_ids, T_pred)

        # ---- CPU tensors ----
        img_cpu = torch.from_numpy(np.array(rollout['image_stacks'], dtype=np.uint8))
        imu_cpu = torch.FloatTensor(np.array(rollout['imu_data']))
        tgt_cpu = torch.from_numpy(target_actions)
        N = img_cpu.shape[0]

        # ---- Distillation update (standard flow matching loss vs teacher target) ----
        policy.train()
        epoch_losses = []
        epoch_ts_mses = []
        for _ in range(n_epochs):
            idx = torch.randperm(N)
            for start in range(0, N, mini_batch):
                mb = idx[start:start + mini_batch]
                if mb.numel() < 2:
                    continue
                imgs_gpu = img_cpu[mb].to(device=device,
                                          dtype=torch.float32,
                                          non_blocking=True) / 255.0
                imu_gpu  = imu_cpu[mb].to(device, non_blocking=True)
                tgt_gpu  = tgt_cpu[mb].to(device, non_blocking=True)

                distill_loss = policy.compute_loss(imgs_gpu, imu_gpu, tgt_gpu)

                policy_opt.zero_grad()
                distill_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
                policy_opt.step()

                epoch_losses.append(distill_loss.item())

        # ---- Logging-only metric: student vs teacher action MSE (1 minibatch) ----
        policy.eval()
        with torch.no_grad():
            mb = torch.randperm(N)[:min(mini_batch, N)]
            imgs_gpu = img_cpu[mb].to(device=device, dtype=torch.float32) / 255.0
            imu_gpu  = imu_cpu[mb].to(device)
            tgt_gpu  = tgt_cpu[mb].to(device)
            student_pred = policy.predict_action(imgs_gpu, imu_gpu, n_steps=2)  # (B, 4, T_pred)
            ts_mse = F.mse_loss(student_pred, tgt_gpu).item()

        # ---- Rollout stats ----
        mean_loss   = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
        mean_reward = float(np.mean(rollout['rewards']))
        n_finished_eps = len(rollout['ep_lengths'])
        if n_finished_eps > 0:
            mean_steps = float(np.mean(rollout['ep_lengths']))
            crash_rate = float(np.mean(rollout['crashed']))
        else:
            mean_steps = float(n_rollout)  # no episode finished in this rollout
            crash_rate = 0.0

        # ---- TensorBoard ----
        writer.add_scalar('distillation/loss',                       mean_loss,   update)
        writer.add_scalar('distillation/teacher_student_action_mse', ts_mse,      update)
        writer.add_scalar('rollout/mean_reward',                     mean_reward, update)
        writer.add_scalar('rollout/mean_episode_steps',              mean_steps,  update)
        writer.add_scalar('rollout/crash_rate',                      crash_rate,  update)
        writer.add_scalar('rollout/n_finished_episodes',             n_finished_eps, update)
        if curriculum_enabled:
            writer.add_scalar('curriculum/pos_range',
                              base_env.initial_pos_range, update)
            writer.add_scalar('curriculum/vel_range',
                              base_env.initial_vel_range, update)
            writer.add_scalar('curriculum/swift_prob',
                              getattr(base_env, 'swift_perturbation_prob', 0.0), update)

        print(f"Update {update+1:>4}/{total_updates} | "
              f"DistillLoss: {mean_loss:.6f} | TS-MSE: {ts_mse:.6f} | "
              f"Reward: {mean_reward:+.4f} | "
              f"Steps: {mean_steps:.1f} | Crash: {crash_rate:.2%} "
              f"({n_finished_eps} eps)")

        # ---- Checkpoint best-by-crash-rate ----
        if n_finished_eps >= 5 and crash_rate < best_crash_rate:
            best_crash_rate = crash_rate
            policy.save(os.path.join(save_dir, 'best_distillation_model.pt'))
            print(f"  --> New best checkpoint (crash_rate={best_crash_rate:.2%})")

        if (update + 1) % log_cfg['save_freq'] == 0:
            policy.save(os.path.join(save_dir, f'update_{update+1}.pt'))

    policy.save(os.path.join(save_dir, 'final_distillation_model.pt'))
    writer.close()
    print(f"\nDistillation training complete!")
    print(f"Best crash rate: {best_crash_rate:.2%}")
    print(f"Saved to: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Online Policy Distillation v4.0 (PPO Teacher → Flow Matching Student)')
    parser.add_argument('--flow-config',         type=str, default='configs/flow_policy_v4.yaml')
    parser.add_argument('--rl-config',           type=str, default='configs/distillation_v4.yaml')
    parser.add_argument('--quadrotor-config',    type=str, default='configs/quadrotor_v4.yaml',
                        help='Env config for evaluation (standard)')
    parser.add_argument('--rl-quadrotor-config', type=str, default=None,
                        help='Env config for RL rollout (wider disturbances). '
                             'Defaults to --quadrotor-config if not set.')
    parser.add_argument('--pretrained',          type=str, default=None,
                        help='Path to pretrained FlowMatchingPolicyV4 checkpoint (H4 BC SOTA)')
    args = parser.parse_args()
    train(args)

"""
Phase 3e v5.0: Online Policy Distillation with cross-attention + state aux loss.

Differences from train_distillation_v4.py:
  * Loads FlowMatchingPolicyV5 (cross-attn + state predictor)
  * collect_rollout additionally stores normalised 15D state per step
  * compute_loss is called with states_gt and lambda_state so the vision
    encoder continues to be grounded on physics during distillation
  * TensorBoard logs distillation/state_loss alongside flow + ts MSE

Usage:
    dppo/Scripts/python.exe -m scripts.train_distillation_v5 \
        --pretrained checkpoints/flow_policy_v5/<bc_ts>/best_model.pt \
        --flow-config configs/flow_policy_v5.yaml \
        --rl-config   configs/distillation_v5.yaml
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
from models.flow_policy_v5 import FlowMatchingPolicyV5
from models.ppo_expert import PPOExpert, RunningMeanStd


# ---------------------------------------------------------------------------
# Rollout: student acts; teacher gives per-step action target; we also record
# the normalised state so the student's state predictor can be supervised.
# ---------------------------------------------------------------------------

def _normalise_state(state_15d: np.ndarray, obs_rms: RunningMeanStd,
                     mode: str) -> np.ndarray:
    """
    Build the state-aux target.
    State layout: [pos_error(0:3), rot_6d(3:9), vel(9:12), omega(12:15)]
    rot_6d 在 hover-only obs_rms 下 std≈0.04，任何 OOD tilt 都會被放大成 ±5σ
    outlier；'partial' 模式只 normalize pos/vel/omega，保留 rot_6d 原值。
    """
    if mode == 'raw':
        return state_15d.astype(np.float32)
    full = obs_rms.normalize(state_15d).astype(np.float32)
    if mode == 'full':
        return full
    if mode == 'partial':
        full[3:9] = state_15d[3:9].astype(np.float32)
        return full
    raise ValueError(f"unknown state_target_norm: {mode}")


def collect_rollout(env: QuadrotorVisualEnv,
                    base_env: QuadrotorEnvV4,
                    policy: FlowMatchingPolicyV5,
                    teacher: PPOExpert,
                    obs_rms: RunningMeanStd,
                    n_steps: int, T_obs: int, T_action: int,
                    device: torch.device,
                    sde_noise_std: float = 0.05,
                    state_target_norm: str = 'partial') -> Dict:
    rollout = {
        'image_stacks':    [],
        'imu_data':        [],
        'states_norm':     [],   # NEW: normalised 15D state for aux loss
        'teacher_actions': [],
        'episode_ids':     [],
        'rewards':         [],
        'dones':           [],
        'crashed':         [],
        'ep_lengths':      [],
    }

    obs, _ = env.reset()
    image_buffer = [obs['image']] * T_obs

    ep_id           = 0
    ep_len_cnt      = 0
    steps_collected = 0

    while steps_collected < n_steps:
        img_stack = np.concatenate(image_buffer[-T_obs:], axis=0)
        imu_vec   = base_env.get_imu()

        # Student action (SDE-noisy 1-step Euler)
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

            state_15d  = obs['state']
            # Teacher always sees fully-normalised state (its own training space).
            state_for_teacher = obs_rms.normalize(state_15d).astype(np.float32)
            a_teacher  = teacher.get_action_deterministic(state_for_teacher)
            # Aux target may use a different normalisation (Stage A fix).
            state_aux  = _normalise_state(state_15d, obs_rms, state_target_norm)

            rollout['image_stacks'].append(img_stack.copy())
            rollout['imu_data'].append(imu_vec.copy())
            rollout['states_norm'].append(state_aux.copy())
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
                max_steps = getattr(base_env, 'max_episode_steps', 500)
                rollout['crashed'].append(bool(ep_len_cnt < max_steps))
                rollout['ep_lengths'].append(ep_len_cnt)
                obs, _ = env.reset()
                image_buffer = [obs['image']] * T_obs
                ep_id     += 1
                ep_len_cnt = 0
                break

            if steps_collected >= n_steps:
                break

    return rollout


def build_target_sequence(teacher_actions: np.ndarray,
                          episode_ids: np.ndarray,
                          T_pred: int) -> np.ndarray:
    """Zero-order-hold padding at episode boundaries."""
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
    xattn_cfg   = flow_cfg.get('cross_attn', {})
    sp_cfg      = flow_cfg.get('state_predictor', {})

    # Env
    rl_quad_cfg = getattr(args, 'rl_quadrotor_config', None) or args.quadrotor_config
    print(f"RL rollout env config: {rl_quad_cfg}")
    base_env   = QuadrotorEnvV4(config_path=rl_quad_cfg)
    visual_env = QuadrotorVisualEnv(base_env, image_size=vis_cfg['image_size'])

    # Student
    policy = FlowMatchingPolicyV5(
        vision_feature_dim     = vis_cfg['feature_dim'],
        imu_feature_dim        = imu_cfg['feature_dim'],
        time_embed_dim         = unet_cfg['time_embed_dim'],
        down_dims              = tuple(unet_cfg['down_dims']),
        T_obs                  = vis_cfg['T_obs'],
        T_pred                 = act_cfg['T_pred'],
        action_dim             = act_cfg['action_dim'],
        n_inference_steps      = flow_params['n_inference_steps'],
        t_embed_scale          = flow_params['t_embed_scale'],
        cross_attn_heads       = xattn_cfg.get('n_heads', 8),
        state_predictor_hidden = sp_cfg.get('hidden_dim', 256),
        state_dim              = sp_cfg.get('state_dim', 15),
    ).to(device)

    if args.pretrained:
        policy.load(args.pretrained)
        print(f"Loaded pretrained v5 student: {args.pretrained}")
    elif args.transfer_from_h4:
        policy.transfer_from_h4(args.transfer_from_h4)
        print(f"Transferred from H4: {args.transfer_from_h4}")

    # Stage C: optionally freeze vision_encoder to protect OOB pretrained weights.
    freeze_vision = distill_cfg.get('freeze_vision_encoder', False)
    if freeze_vision:
        for p in policy.vision_encoder.parameters():
            p.requires_grad = False
        frozen_n = sum(p.numel() for p in policy.vision_encoder.parameters())
        print(f"[Stage C] vision_encoder FROZEN ({frozen_n:,} params)")

    policy_opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=distill_cfg['learning_rate']
    )

    # Teacher
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

    # Curriculum
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
        print(f"Curriculum: hover({cur_pos_start}m/{cur_vel_start}m/s) -> "
              f"OOD({cur_pos_end}m/{cur_vel_end}m/s) "
              f"over {cur_n_hover}+{cur_n_ramp} updates")
        base_env.initial_pos_range = cur_pos_start
        base_env.initial_vel_range = cur_vel_start
        base_env.hover_anchor_prob       = 0.0
        base_env.swift_perturbation_prob = 0.0
        base_env.swift_perturb_tilt_deg  = cur_swift_tilt
        base_env.swift_perturb_vel       = cur_swift_vel
        base_env.swift_max_tilt_deg      = cur_swift_term

    # Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag   = f"distillation_v5_{timestamp}"
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
    lambda_state  = distill_cfg.get('lambda_state', 0.1)
    state_target_norm = distill_cfg.get('state_target_norm', 'full')
    state_loss_type   = distill_cfg.get('state_loss_type',   'mse')
    T_pred        = act_cfg['T_pred']

    print(f"\n{'='*60}")
    print(f"Distillation v5.0 — cross-attn + state aux (lambda={lambda_state})")
    print(f"Total updates: {total_updates} | Rollout steps: {n_rollout}")
    print(f"LR: {distill_cfg['learning_rate']:.1e} | "
          f"epochs/upd: {n_epochs} | minibatch: {mini_batch}")
    print(f"SDE noise: {sde_noise_std} | T_pred: {T_pred}")
    print(f"State target norm: {state_target_norm} | state loss: {state_loss_type}")
    print(f"Freeze vision_encoder: {freeze_vision} | lambda_state: {lambda_state}")
    print(f"Save: {save_dir}")
    print(f"{'='*60}\n")

    best_crash_rate = float('inf')

    for update in range(total_updates):
        # Curriculum
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

        # Rollout
        policy.eval()
        rollout = collect_rollout(
            visual_env, base_env, policy, teacher, obs_rms,
            n_steps=n_rollout,
            T_obs=vis_cfg['T_obs'],
            T_action=act_cfg['T_action'],
            device=device,
            sde_noise_std=sde_noise_std,
            state_target_norm=state_target_norm,
        )

        # Build teacher target sequence (N, 4, T_pred)
        teacher_actions = np.stack(rollout['teacher_actions']).astype(np.float32)
        episode_ids     = np.array(rollout['episode_ids'], dtype=np.int64)
        target_actions  = build_target_sequence(teacher_actions, episode_ids, T_pred)
        states_arr      = np.stack(rollout['states_norm']).astype(np.float32)

        img_cpu   = torch.from_numpy(np.array(rollout['image_stacks'], dtype=np.uint8))
        imu_cpu   = torch.FloatTensor(np.array(rollout['imu_data']))
        tgt_cpu   = torch.from_numpy(target_actions)
        state_cpu = torch.from_numpy(states_arr)
        N = img_cpu.shape[0]

        # Distillation update
        policy.train()
        ep_total, ep_flow, ep_state = [], [], []
        for _ in range(n_epochs):
            idx = torch.randperm(N)
            for start in range(0, N, mini_batch):
                mb = idx[start:start + mini_batch]
                if mb.numel() < 2:
                    continue
                imgs_gpu  = img_cpu[mb].to(device=device, dtype=torch.float32,
                                           non_blocking=True) / 255.0
                imu_gpu   = imu_cpu[mb].to(device, non_blocking=True)
                tgt_gpu   = tgt_cpu[mb].to(device, non_blocking=True)
                state_gpu = state_cpu[mb].to(device, non_blocking=True)

                total_loss, comp = policy.compute_loss(
                    imgs_gpu, imu_gpu, tgt_gpu,
                    states_gt=state_gpu, lambda_state=lambda_state,
                    state_loss_type=state_loss_type,
                    return_components=True,
                )

                policy_opt.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
                policy_opt.step()

                ep_total.append(total_loss.item())
                ep_flow.append(comp['flow_loss'].item())
                ep_state.append(comp['state_loss'].item())

        # Logging-only metric: student vs teacher action MSE
        policy.eval()
        with torch.no_grad():
            mb = torch.randperm(N)[:min(mini_batch, N)]
            imgs_gpu = img_cpu[mb].to(device=device, dtype=torch.float32) / 255.0
            imu_gpu  = imu_cpu[mb].to(device)
            tgt_gpu  = tgt_cpu[mb].to(device)
            student_pred = policy.predict_action(imgs_gpu, imu_gpu, n_steps=2)
            ts_mse = F.mse_loss(student_pred, tgt_gpu).item()

        mean_total = float(np.mean(ep_total)) if ep_total else float('nan')
        mean_flow  = float(np.mean(ep_flow))  if ep_flow  else float('nan')
        mean_state = float(np.mean(ep_state)) if ep_state else float('nan')
        mean_reward = float(np.mean(rollout['rewards']))
        n_finished_eps = len(rollout['ep_lengths'])
        if n_finished_eps > 0:
            mean_steps = float(np.mean(rollout['ep_lengths']))
            crash_rate = float(np.mean(rollout['crashed']))
        else:
            mean_steps = float(n_rollout)
            crash_rate = 0.0

        # TensorBoard
        writer.add_scalar('distillation/total_loss',                mean_total, update)
        writer.add_scalar('distillation/flow_loss',                 mean_flow,  update)
        writer.add_scalar('distillation/state_loss',                mean_state, update)
        writer.add_scalar('distillation/teacher_student_action_mse', ts_mse,    update)
        writer.add_scalar('rollout/mean_reward',                    mean_reward, update)
        writer.add_scalar('rollout/mean_episode_steps',             mean_steps, update)
        writer.add_scalar('rollout/crash_rate',                     crash_rate, update)
        writer.add_scalar('rollout/n_finished_episodes',            n_finished_eps, update)
        if curriculum_enabled:
            writer.add_scalar('curriculum/pos_range',
                              base_env.initial_pos_range, update)
            writer.add_scalar('curriculum/vel_range',
                              base_env.initial_vel_range, update)
            writer.add_scalar('curriculum/swift_prob',
                              getattr(base_env, 'swift_perturbation_prob', 0.0), update)

        print(f"Update {update+1:>4}/{total_updates} | "
              f"flow={mean_flow:.5f} state={mean_state:.5f} | "
              f"TS-MSE: {ts_mse:.5f} | "
              f"R:{mean_reward:+.3f} Steps:{mean_steps:.1f} Crash:{crash_rate:.2%} "
              f"({n_finished_eps} eps)")

        if n_finished_eps >= 5 and crash_rate < best_crash_rate:
            best_crash_rate = crash_rate
            policy.save(os.path.join(save_dir, 'best_distillation_model.pt'))
            print(f"  --> New best checkpoint (crash_rate={best_crash_rate:.2%})")

        if (update + 1) % log_cfg['save_freq'] == 0:
            policy.save(os.path.join(save_dir, f'update_{update+1}.pt'))

    policy.save(os.path.join(save_dir, 'final_distillation_model.pt'))
    writer.close()
    print(f"\nDistillation v5 complete!")
    print(f"Best crash rate: {best_crash_rate:.2%}")
    print(f"Saved to: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Online Policy Distillation v5.0 (cross-attn + state aux)')
    parser.add_argument('--flow-config',         type=str, default='configs/flow_policy_v5.yaml')
    parser.add_argument('--rl-config',           type=str, default='configs/distillation_v5.yaml')
    parser.add_argument('--quadrotor-config',    type=str, default='configs/quadrotor_v4.yaml')
    parser.add_argument('--rl-quadrotor-config', type=str, default=None,
                        help='Env config for RL rollout (defaults to --quadrotor-config)')
    parser.add_argument('--pretrained',          type=str, default=None,
                        help='Path to a v5 BC checkpoint')
    parser.add_argument('--transfer-from-h4',    type=str, default=None,
                        help='Alternative: partial-load weights from an H4 v4 ckpt')
    args = parser.parse_args()
    train(args)

"""
Gate gamma (RESEARCH_PLAN_v9 Phase 0) — is the compounding divergence ARRESTABLE by feedback?

Gate alpha found the hover specialist T0O1 DIVERGES from a perfect (zero-offset) start
(second-half slope +1.1 m/100 steps, frac_plateau ~0) -> a b1 compounding signature, not a
steady-state bias. Before committing GPU-days to a closed-loop fine-tune (b1), this gate
cheaply checks that the divergence is *fixable* by a corrective signal the BC policy lacks:
blend the vision policy's CTBR with the state-based PID-CTBR teacher's corrective CTBR at a
small weight w and see whether the drift is arrested (slope -> 0, equilibrium drops).

  action = (1 - w) * policy_action + w * teacher_action,   w in {0, 0.1, 0.25, 0.5, 1.0}

  * a small w (0.1-0.25) arrests the drift (slope->0, cond-IAE drops sharply)
        -> the corrective behaviour is a small feedback the policy can be taught
           -> b1 closed-loop fine-tune is VIABLE.
  * even large w barely helps / w=1 (pure teacher) needed
        -> the limit is deeper than a learnable correction.

Faithfully mirrors scripts.evaluate_hierarchical.rollout_episode for the vision-policy action
(T_obs frame stack, /255, dynamic task_cond) and only adds the teacher blend + per-step
position-error logging (reuses measure_equilibrium.episode_equilibrium_stats).

Usage:
  dppo/Scripts/python.exe -m scripts.measure_feedback_blend \
      --ckpt checkpoints/flow_policy_v5/p2to_T0O1_s0/best_model.pt \
      --render perspective --n-episodes 20 --weights 0 0.1 0.25 0.5 1.0
"""
import os
import sys
import json
import argparse
import numpy as np
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env_v4 import QuadrotorEnvV4
from envs.quadrotor_visual_env import QuadrotorVisualEnv
from envs.quadrotor_dynamics import get_tilt_angle
from controllers.pid_controller import CascadePIDController
from scripts.evaluate_hierarchical import build_policy
from scripts.measure_equilibrium import episode_equilibrium_stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def policy_action(policy, arch, base_env, image_buffer, T_obs, n_steps, device):
    """Vision-policy CTBR for one step — faithful to rollout_episode."""
    img_stack = np.concatenate(image_buffer[-T_obs:], axis=0)
    img_t = torch.from_numpy(img_stack).float().unsqueeze(0).to(device) / 255.0
    imu_t = torch.from_numpy(base_env.get_imu()).float().unsqueeze(0).to(device)
    extra = {}
    if arch.get('task_dim', 0) > 0:
        R = base_env.dynamics.get_rotation_matrix()
        tilt_deg = get_tilt_angle(R)
        pe = np.linalg.norm(base_env.target_position - base_env.dynamics.position)
        av = np.linalg.norm(base_env.dynamics.ang_velocity)
        is_rec = float(pe > 1.0 or tilt_deg > 15.0 or av > 2.0)
        extra['task_cond'] = torch.tensor([[1.0 - is_rec, is_rec]], device=device)
    with torch.no_grad():
        a = policy.predict_action(img_t, imu_t, n_steps=n_steps, **extra)
    return a.squeeze(0).T.cpu().numpy()[0]      # first action in the horizon (T_action=1)


def run_blend(policy, arch, teacher, base_env, visual_env, T_obs, n_steps, device,
              w, seed, survive_threshold):
    np.random.seed(seed); torch.manual_seed(seed)
    obs, _ = visual_env.reset(seed=seed)
    image_buffer = [obs['image']] * T_obs
    positions, targets = [], []
    done = False
    while not done:
        a_pol = policy_action(policy, arch, base_env, image_buffer, T_obs, n_steps, device)
        if w > 0.0:
            a_tea = teacher.compute_ctbr_action(base_env.dynamics.state,
                                                base_env.target_position,
                                                base_env.F_c_max, base_env.omega_max)
            a = (1.0 - w) * a_pol + w * a_tea
        else:
            a = a_pol
        obs, _, terminated, truncated, info = visual_env.step(a)
        image_buffer.append(obs['image'])
        positions.append(info['position'].copy())
        targets.append(info['target'].copy())
        if terminated or truncated:
            done = True
    return positions, targets


def main():
    ap = argparse.ArgumentParser(description='Gate gamma: feedback-blend divergence arrest')
    ap.add_argument('--ckpt', default='checkpoints/flow_policy_v5/p2to_T0O1_s0/best_model.pt')
    ap.add_argument('--render', default='perspective', choices=['crosshair', 'perspective'])
    ap.add_argument('--n-episodes', type=int, default=20)
    ap.add_argument('--base-seed', type=int, default=12345)
    ap.add_argument('--survive-threshold', type=int, default=250)
    ap.add_argument('--weights', nargs='+', type=float, default=[0.0, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument('--quadrotor-config', default='configs/quadrotor_v4.yaml')
    ap.add_argument('--flow-config', default='configs/flow_policy_v4.yaml')
    ap.add_argument('--n-inference-steps', type=int, default=2)
    ap.add_argument('--out', default='evaluation_results/p2cl_gate_gamma_feedback.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(ROOT, args.flow_config), 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    T_obs = cfg['vision']['T_obs']
    print(f"Device: {device}  render: {args.render}  weights: {args.weights}")

    policy, arch = build_policy(os.path.join(ROOT, args.ckpt.lstrip('./')),
                                cfg, args.n_inference_steps, device)
    base_env = QuadrotorEnvV4(config_path=args.quadrotor_config)
    visual_env = QuadrotorVisualEnv(base_env, image_size=cfg['vision']['image_size'],
                                    target_render=args.render)
    # gentle recovery-tuned teacher (v7 Gate-A gains: vel_max=1.0, Kp_pos=0.8, omega_max=6.0)
    teacher = CascadePIDController(base_env.dynamics.params, omega_max=6.0,
                                   dt=base_env.dt_outer, vel_max=1.0, Kp_pos=0.8)
    thr = min(args.survive_threshold, base_env.max_episode_steps)

    by_w = {}
    for w in args.weights:
        eqs, n_surv = [], 0
        for ep in range(args.n_episodes):
            seed = args.base_seed + ep
            pos, tgt = run_blend(policy, arch, teacher, base_env, visual_env,
                                 T_obs, args.n_inference_steps, device, w, seed, thr)
            s = episode_equilibrium_stats(pos, tgt, thr)
            if s:
                eqs.append(s); n_surv += 1
        if eqs:
            eq = np.array([e['equilibrium'] for e in eqs])
            sl = np.array([e['slope_per100'] for e in eqs])
            a = {'n_cond': len(eqs), 'survive_frac': round(n_surv / args.n_episodes, 3),
                 'cond_iae': round(float(eq.mean()), 4), 'cond_iae_std': round(float(eq.std()), 4),
                 'slope_per100': round(float(sl.mean()), 4)}
        else:
            a = {'n_cond': 0, 'survive_frac': round(n_surv / args.n_episodes, 3),
                 'cond_iae': float('nan'), 'slope_per100': float('nan')}
        by_w[f'{w:.2f}'] = a
        print(f"  w={w:.2f}: n_cond={a['n_cond']:>2} survive={a['survive_frac']*100:>4.0f}% "
              f"cond-IAE={a['cond_iae']}m slope/100={a['slope_per100']}")

    # verdict: does a small w arrest the drift?
    base = by_w.get('0.00', {})
    small_w = next((by_w[k] for k in ['0.10', '0.25'] if k in by_w and by_w[k]['n_cond'] > 0), None)
    verdict = {}
    if base.get('cond_iae') and small_w and not np.isnan(small_w['cond_iae']):
        drop = base['cond_iae'] - small_w['cond_iae']
        slope_arrested = abs(small_w['slope_per100']) < 0.3 and base['slope_per100'] >= 0.6
        verdict = {
            'base_cond_iae': base['cond_iae'], 'base_slope': base['slope_per100'],
            'smallw_cond_iae': small_w['cond_iae'], 'smallw_slope': small_w['slope_per100'],
            'iae_drop_at_smallw': round(drop, 4), 'slope_arrested': bool(slope_arrested),
            'b1_viable': bool(slope_arrested and drop > 0.5),
        }

    out = {'gate': 'gamma — feedback-blend divergence arrest', 'ckpt': args.ckpt,
           'render': args.render, 'weights': args.weights, 'by_weight': by_w, 'verdict': verdict}
    op = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 72)
    print("Gate gamma — feedback-blend (policy (1-w) + teacher w)")
    print("=" * 72)
    print(f"{'w':>5s}{'survive':>9s}{'cond-IAE':>12s}{'slope/100':>11s}")
    for k, a in by_w.items():
        print(f"{k:>5s}{a['survive_frac']*100:>7.0f}%{a['cond_iae']:>10}m{a['slope_per100']:>11}")
    if verdict:
        print(f"\n  small-w (0.1-0.25): cond-IAE {verdict['base_cond_iae']:.2f}->"
              f"{verdict['smallw_cond_iae']:.2f}m (drop {verdict['iae_drop_at_smallw']:.2f}), "
              f"slope {verdict['base_slope']:+.2f}->{verdict['smallw_slope']:+.2f}")
        print(f"  => b1 closed-loop fine-tune {'VIABLE' if verdict['b1_viable'] else 'NOT clearly viable'} "
              f"(small corrective feedback {'arrests' if verdict['slope_arrested'] else 'does NOT arrest'} the drift)")
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()

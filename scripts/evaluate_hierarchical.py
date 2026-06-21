"""
Hierarchical evaluation for vision-based flow matching policy.

Implements 飛 → 穩 → 準 hierarchy:
  Tier 1 (Flying):   survival_rate = ep_length / max_episode_steps
  Tier 2 (Stability): IAE_steady = mean(|e_t|) for t ∈ [T/2, T]  (skip transient)
                      omega_iae   = mean(|omega_t|) for t ∈ [T/2, T]
  Tier 3 (Accuracy):  terminal_err = mean(|e_t|) for t ∈ [0.9T, T]

Composite score:
  score = survival_rate * (0.6 * stability_score + 0.4 * accuracy_score)
  where stability_score = exp(-IAE_steady / sigma)
        accuracy_score  = exp(-terminal_err / sigma)
  Here, sigma is the physical tolerance scale factor (e.g. 2.0m).
  The Tier 1 hard gate is removed to ensure a continuous and smooth scoring function.

Auto-detects H3a (imu_feature_dim=128) vs H4 (imu_feature_dim=512) from checkpoint.
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import yaml

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.quadrotor_env_v4 import QuadrotorEnvV4
from envs.quadrotor_visual_env import QuadrotorVisualEnv
from models.flow_policy_v4 import FlowMatchingPolicyV4
from models.flow_policy_v5 import FlowMatchingPolicyV5


def detect_arch(ckpt_path: str) -> dict:
    """Auto-detect architecture from checkpoint state_dict shapes."""
    state = torch.load(ckpt_path, map_location='cpu')
    w0 = state['imu_encoder.0.weight'].shape    # (hidden, 6)
    w2 = state['imu_encoder.2.weight'].shape    # (feature_dim, hidden)
    hidden_dim = w0[0]
    feature_dim = w2[0]
    has_tilt = 'tilt_head.weight' in state
    # v5 = H4 + cross-attention + state predictor head
    is_v5 = ('cross_attn.q_proj.weight' in state
             and 'state_predictor.net.0.weight' in state)
    task_dim = 0
    if is_v5:
        era = 'V5'; activation = 'ReLU'
        # Check flow_net.encoder_blocks.0.film.net.1.weight to find cond_dim
        cond_dim = state['flow_net.encoder_blocks.0.film.net.1.weight'].shape[1]
        task_dim = cond_dim - (256 + feature_dim + 128)
    elif hidden_dim == 64:
        era = 'Original'; activation = 'Mish'
    elif hidden_dim == 256:
        era = 'H3a'; activation = 'ReLU'
    else:
        era = 'H4'; activation = 'ReLU'
    return {'imu_hidden': hidden_dim, 'imu_feature_dim': feature_dim,
            'has_tilt': has_tilt, 'era': era, 'activation': activation,
            'is_v5': is_v5, 'task_dim': task_dim}


def rebuild_policy_for_arch(policy, arch, device):
    """Dynamically rebuild imu_encoder and tilt_head to match detected architecture.

    Default model is H4 (hidden=1024, feature=512, ReLU, with tilt_head).
    Rebuild only the parts that differ.
    """
    import torch.nn as nn
    hidden = arch['imu_hidden']
    feat = arch['imu_feature_dim']
    if arch['activation'] == 'Mish':
        # Original era: Linear→Mish→Linear (no trailing ReLU)
        policy.imu_encoder = nn.Sequential(
            nn.Linear(6, hidden),
            nn.Mish(),
            nn.Linear(hidden, feat),
        ).to(device)
    else:
        # H3a / H4: Linear→ReLU→Linear→ReLU
        policy.imu_encoder = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feat),
            nn.ReLU(),
        ).to(device)
    # tilt_head dim depends on imu_feature_dim
    policy.tilt_head = nn.Linear(feat, 1).to(device)
    # Need to rebuild flow_net since cond_dim depends on global_cond_dim = 256 + feat
    from models.conditional_unet1d import ConditionalUnet1d
    policy.flow_net = ConditionalUnet1d(
        action_dim=policy.action_dim,
        feature_dim=256 + feat,   # global_cond_dim
        time_embed_dim=128,
        down_dims=(256, 512),
        kernel_size=5,
        n_groups=8,
    ).to(device)
    return policy


def compute_hierarchical_metrics(positions: np.ndarray, targets: np.ndarray,
                                  angular_vels: np.ndarray, ep_length: int,
                                  max_steps: int = 500) -> dict:
    """Compute Tier 1/2/3 metrics for a single episode.

    Args:
      positions: (T, 3) actual positions
      targets:   (T, 3) target positions
      angular_vels: (T, 3) angular velocity ω
      ep_length: actual steps survived
      max_steps: max episode length
    """
    T = ep_length
    errs = np.linalg.norm(positions - targets, axis=1)    # (T,) scalar error per step
    omegas = np.linalg.norm(angular_vels, axis=1)         # (T,) scalar |ω|

    # Tier 1: Survival
    survival_rate = T / max_steps
    crashed = T < max_steps

    # Tier 2: Stability (steady state = second half)
    half = max(1, T // 2)
    iae_steady = float(np.mean(errs[half:])) if T > half else float(np.mean(errs))
    omega_steady = float(np.mean(omegas[half:])) if T > half else float(np.mean(omegas))

    # Tier 3: Accuracy (terminal = last 10%)
    tail = max(1, T // 10)
    terminal_err = float(np.mean(errs[-tail:]))

    # Also compute IAE (whole episode, control engineering definition)
    iae_full = float(np.sum(errs))    # ∫|e|dt with dt=1
    iae_normalized = float(iae_full / T) if T > 0 else 0.0

    return {
        'ep_length': T,
        'crashed': bool(crashed),
        'survival_rate': float(survival_rate),
        'iae_steady': iae_steady,           # Tier 2 main metric
        'iae_full': iae_full,
        'iae_normalized': iae_normalized,    # IAE / T (alive period avg)
        'omega_steady': omega_steady,        # Tier 2 secondary (oscillation indicator)
        'terminal_err': terminal_err,        # Tier 3
        'rmse_legacy': float(np.sqrt(np.mean(errs**2))),    # for backward compat
    }


def composite_score(metrics: dict, sigma: float = 2.0) -> dict:
    """Score = survival × (0.6×stability + 0.4×accuracy) without Tier 1 gate.
    Uses exponential decay with physical tolerance scale factor sigma.
    """
    sr = metrics['survival_rate']
    stability_score = float(np.exp(-metrics['iae_steady'] / sigma))
    accuracy_score  = float(np.exp(-metrics['terminal_err'] / sigma))
    score = sr * (0.6 * stability_score + 0.4 * accuracy_score)
    return {'tier1_pass': bool(sr >= 0.5),
            'stability_score': stability_score,
            'accuracy_score': accuracy_score,
            'score': score}


def build_policy(ckpt_path: str, cfg: dict, n_inference_steps: int, device) -> tuple:
    """Construct a flow policy matching the checkpoint's architecture and load weights.

    Returns (policy, arch). Extracted from evaluate_one so other scripts (e.g. the
    frozen P0 protocol) can reuse identical model construction.
    """
    vis_cfg = cfg['vision']
    act_cfg = cfg['action']
    flow_cfg = cfg['flow']

    arch = detect_arch(ckpt_path)
    imu_feature_dim = arch['imu_feature_dim']

    if arch['is_v5']:
        # v5 has its own class with cross-attention + state predictor head.
        policy = FlowMatchingPolicyV5(
            vision_feature_dim=vis_cfg['feature_dim'],
            imu_feature_dim=imu_feature_dim,
            time_embed_dim=cfg['unet']['time_embed_dim'],
            down_dims=tuple(cfg['unet']['down_dims']),
            T_obs=vis_cfg['T_obs'],
            T_pred=act_cfg['T_pred'],
            action_dim=act_cfg['action_dim'],
            n_inference_steps=n_inference_steps,
            t_embed_scale=flow_cfg['t_embed_scale'],
            task_dim=arch.get('task_dim', 0),
        ).to(device)
    else:
        policy = FlowMatchingPolicyV4(
            vision_feature_dim=vis_cfg['feature_dim'],
            imu_feature_dim=imu_feature_dim,
            time_embed_dim=cfg['unet']['time_embed_dim'],
            down_dims=tuple(cfg['unet']['down_dims']),
            T_obs=vis_cfg['T_obs'],
            T_pred=act_cfg['T_pred'],
            action_dim=act_cfg['action_dim'],
            n_inference_steps=n_inference_steps,
            t_embed_scale=flow_cfg['t_embed_scale'],
        ).to(device)
        # Rebuild architecture-dependent layers to match v4 checkpoint variants
        rebuild_policy_for_arch(policy, arch, device)
    # Load checkpoint, allowing strict=False for Original era (no tilt_head)
    state = torch.load(ckpt_path, map_location=device)
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if unexpected:
        print(f"  (note: unexpected keys ignored: {unexpected[:3]}...)")
    if missing:
        # Original era has no tilt_head — that's expected; flag others
        non_tilt = [k for k in missing if 'tilt_head' not in k]
        if non_tilt:
            print(f"  (warning: missing keys: {non_tilt[:3]}...)")
    policy.eval()
    return policy, arch


def rollout_episode(policy, base_env, visual_env, arch, T_obs, T_action,
                    n_inference_steps, device, seed=None,
                    cue_scale=3.0, cue_noise=0.0) -> dict:
    """Run one closed-loop RHC episode and return raw per-step trajectories.

    If ``seed`` is given, the episode is made fully reproducible: the gym env
    init (via ``reset(seed=)``), the *global* numpy RNG used by the visual
    domain-randomisation / per-frame noise, and the torch RNG used for flow
    inference noise are all seeded. This is what the frozen P0 protocol uses to
    guarantee every model sees identical initial conditions (paired comparison).
    When ``seed`` is None the behaviour is identical to the original eval.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        obs, _ = visual_env.reset(seed=seed)
    else:
        obs, _ = visual_env.reset()

    image_buffer = [obs['image']] * T_obs
    positions, targets, omegas = [], [], []
    done = False
    ep_length = 0
    ep_reward = 0.0

    while not done:
        img_stack = np.concatenate(image_buffer[-T_obs:], axis=0)
        img_tensor = torch.from_numpy(img_stack).float().unsqueeze(0).to(device) / 255.0
        imu_tensor = torch.from_numpy(base_env.get_imu()).float().unsqueeze(0).to(device)

        # Only v5-with-task accepts a task_cond kwarg; v4 predict_action does not.
        extra_kwargs = {}
        task_dim = arch.get('task_dim', 0)
        if task_dim > 0:
            R = base_env.dynamics.get_rotation_matrix()
            from envs.quadrotor_dynamics import get_tilt_angle
            tilt_deg = get_tilt_angle(R)
            pos_err_world = base_env.target_position - base_env.dynamics.position
            pos_err = np.linalg.norm(pos_err_world)
            ang_vel = np.linalg.norm(base_env.dynamics.ang_velocity)
            is_recovery = float(pos_err > 1.0 or tilt_deg > 15.0 or ang_vel > 2.0)
            is_hover = 1.0 - is_recovery
            tc = [is_hover, is_recovery]
            # Phase 3b range-cue arms (task_dim > 2): fold in the metric position
            # error the FPV cannot encode, matching train_flow_v5's cue exactly
            # (raw metres / cue_scale, body frame for pos3d; + sensor noise).
            cue_dim = task_dim - 2
            if cue_dim == 1:        # scalar ||pos_err|| (frame-invariant)
                cue = np.array([pos_err], dtype=np.float32)
            elif cue_dim == 3:      # pos_err_body = R.T @ (target - pos)
                cue = (R.T @ pos_err_world).astype(np.float32)
            else:
                cue = np.zeros(cue_dim, dtype=np.float32)
            cue = cue / cue_scale
            if cue_noise > 0:
                cue = cue + np.random.randn(cue_dim).astype(np.float32) * (cue_noise / cue_scale)
            tc = tc + cue.tolist()
            extra_kwargs['task_cond'] = torch.tensor([tc], device=device)

        with torch.no_grad():
            action_seq = policy.predict_action(
                img_tensor, imu_tensor, n_steps=n_inference_steps, **extra_kwargs)
        action_seq = action_seq.squeeze(0).T.cpu().numpy()  # (T_pred, action_dim)

        for a_idx in range(min(T_action, action_seq.shape[0])):
            action = action_seq[a_idx]
            obs, reward, terminated, truncated, info = visual_env.step(action)
            image_buffer.append(obs['image'])
            ep_reward += reward
            ep_length += 1

            positions.append(info['position'].copy())
            targets.append(info['target'].copy())
            omegas.append(base_env.dynamics.ang_velocity.copy())

            if terminated or truncated:
                done = True
                break

    return {
        'positions': np.array(positions),
        'targets': np.array(targets),
        'omegas': np.array(omegas),
        'ep_length': ep_length,
        'ep_reward': ep_reward,
    }


def evaluate_one(ckpt_path: str, n_episodes: int = 30,
                 quadrotor_config: str = 'configs/quadrotor_v4.yaml',
                 flow_config: str = 'configs/flow_policy_v4.yaml',
                 n_inference_steps: int = 2,
                 device_str: str = 'cuda',
                 sigma: float = 2.0) -> dict:
    """Evaluate one checkpoint with hierarchical metrics."""
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    with open(flow_config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    vis_cfg = cfg['vision']
    act_cfg = cfg['action']

    policy, arch = build_policy(ckpt_path, cfg, n_inference_steps, device)

    base_env = QuadrotorEnvV4(config_path=quadrotor_config)
    visual_env = QuadrotorVisualEnv(base_env, image_size=vis_cfg['image_size'])

    T_obs = vis_cfg['T_obs']
    T_action = act_cfg['T_action']

    all_metrics = []

    for ep in range(n_episodes):
        roll = rollout_episode(policy, base_env, visual_env, arch,
                               T_obs, T_action, n_inference_steps, device)
        metrics = compute_hierarchical_metrics(
            roll['positions'], roll['targets'], roll['omegas'],
            roll['ep_length'], base_env.max_episode_steps)
        score = composite_score(metrics, sigma=sigma)
        metrics.update(score)
        metrics['ep_reward'] = roll['ep_reward']
        all_metrics.append(metrics)

        print(f"  Ep {ep+1:>2}/{n_episodes} | steps={roll['ep_length']:>3} | "
              f"survive={metrics['survival_rate']*100:>4.1f}% | "
              f"IAE_st={metrics['iae_steady']:.3f}m | "
              f"term={metrics['terminal_err']:.3f}m | "
              f"score={metrics['score']:.3f}")

    # Aggregate
    agg = {
        'n_episodes': n_episodes,
        'arch': arch,
        'mean_ep_length': float(np.mean([m['ep_length'] for m in all_metrics])),
        'mean_survival_rate': float(np.mean([m['survival_rate'] for m in all_metrics])),
        'tier1_pass_rate': float(np.mean([m['tier1_pass'] for m in all_metrics])),
        'mean_iae_steady': float(np.mean([m['iae_steady'] for m in all_metrics])),
        'mean_omega_steady': float(np.mean([m['omega_steady'] for m in all_metrics])),
        'mean_terminal_err': float(np.mean([m['terminal_err'] for m in all_metrics])),
        'mean_score': float(np.mean([m['score'] for m in all_metrics])),
        'mean_rmse_legacy': float(np.mean([m['rmse_legacy'] for m in all_metrics])),
    }
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpts', nargs='+', required=True,
                        help='List of "label:path" pairs')
    parser.add_argument('--n-episodes', type=int, default=30)
    parser.add_argument('--quadrotor-config', default='configs/quadrotor_v4.yaml')
    parser.add_argument('--flow-config', default='configs/flow_policy_v4.yaml')
    parser.add_argument('--n-inference-steps', type=int, default=2)
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='Physical tolerance scale factor in meters (default: 2.0)')
    parser.add_argument('--output', default='evaluation_results/hierarchical_comparison.json')
    args = parser.parse_args()

    all_results = {}
    for entry in args.ckpts:
        label, path = entry.split(':', 1)
        print(f"\n{'='*70}\n=== {label}  ({path})\n{'='*70}")
        if not os.path.exists(path):
            print(f"  SKIP: not found"); continue
        try:
            agg = evaluate_one(path, args.n_episodes, args.quadrotor_config,
                               args.flow_config, args.n_inference_steps, sigma=args.sigma)
            agg['label'] = label
            agg['path'] = path
            all_results[label] = agg
            print(f"\n  >> {label}: score={agg['mean_score']:.3f}  "
                  f"survive={agg['mean_survival_rate']*100:.1f}%  "
                  f"IAE_st={agg['mean_iae_steady']:.3f}m  "
                  f"term={agg['mean_terminal_err']:.3f}m")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    # Final summary
    print("\n" + "="*90)
    print(f"{'Label':<30} {'Arch':>6} {'Score':>6} {'Survive':>8} {'IAE_st':>8} {'Term':>7} {'RMSE':>7}")
    print("-"*90)
    sorted_results = sorted(all_results.values(), key=lambda r: -r['mean_score'])
    for r in sorted_results:
        arch_tag = r['arch'].get('era', '?')
        print(f"{r['label']:<30} {arch_tag:>9} {r['mean_score']:>6.3f} "
              f"{r['mean_survival_rate']*100:>7.1f}% "
              f"{r['mean_iae_steady']:>7.3f}m "
              f"{r['mean_terminal_err']:>6.3f}m "
              f"{r['mean_rmse_legacy']:>6.3f}m")
    print("="*90)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

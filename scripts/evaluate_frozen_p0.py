"""
P0 Frozen Evaluation Protocol — clean, reproducible leaderboard.

Motivation (see RESEARCH_PLAN re-plan, Path A / Phase 0):
The project changed its eval metric three times (RMSE -> linear-clip hierarchical
-> exp-decay hierarchical), so cross-run numbers are not comparable, and the
short-survival artifact keeps inflating IAE/precision for policies that simply
crash early. This script *freezes* one protocol and reports the metrics that are
robust to that artifact.

What is frozen here (do NOT change between runs once published):
  1. Fixed seed list. Episode i uses seed = base_seed + i, and we seed the gym
     env (init pose), the GLOBAL numpy RNG (visual domain-randomisation + per-frame
     noise live on np.random, not env.np_random) and torch (flow inference noise).
     => every checkpoint sees IDENTICAL initial conditions  =>  paired comparison.
  2. sigma = 2.0 m exponential-decay composite score (same formula as
     evaluate_hierarchical.composite_score), reported with a bootstrap 95% CI.
  3. Conditional-IAE / conditional-terminal: stability & accuracy measured ONLY
     over episodes that survived >= --survive-threshold steps. This is the direct
     fix for the short-survival artifact (Finding #1 / Phase D warning): a policy
     that crashes at step 24 is NOT credited with a "0.65 m precise hover".
  4. Oracle-normalised score column (score / oracle_score), so the gap to the
     state-based PPO oracle is visible at a glance.

Reuses build_policy / rollout_episode / compute_hierarchical_metrics /
composite_score from evaluate_hierarchical (single source of truth for the model
construction and the RHC rollout) — this script only adds the frozen sampling and
the artifact-robust aggregation.

Example:
  dppo/Scripts/python.exe -m scripts.evaluate_frozen_p0 \
      --n-episodes 30 --base-seed 12345 --survive-threshold 250 \
      --oracle-ckpt checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
      --oracle-norm checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz \
      --ckpts \
        "H4_BC:checkpoints/flow_policy_v4/20260514_175219/best_model.pt" \
        "Joint_E2E_v5:checkpoints/flow_policy_v5/20260603_171316/best_model.pt" \
        "v5_BC:checkpoints/flow_policy_v5/20260604_141454/best_model.pt" \
        "v5_RL_best:checkpoints/reinflow_v5/reinflow_v5_20260604_193923/best_reinflow_model.pt"

  # Oracle-only (fast): just measure the %Oracle normalisation constant.
  dppo/Scripts/python.exe -m scripts.evaluate_frozen_p0 --n-episodes 30 \
      --oracle-ckpt checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
      --oracle-norm checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz \
      --output evaluation_results/frozen_p0_oracle.json
"""
import os, sys, json, argparse
import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.quadrotor_env_v4 import QuadrotorEnvV4
from envs.quadrotor_visual_env import QuadrotorVisualEnv
from models.ppo_expert import PPOExpert, RunningMeanStd
from scripts.evaluate_hierarchical import (
    build_policy, rollout_episode, compute_hierarchical_metrics, composite_score,
)


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    """Bootstrap (percentile) CI for the mean of `values`."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float('nan'), float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (float(values.mean()), lo, hi)


def _aggregate_frozen(per_ep, n_episodes, base_seed, survive_threshold, arch_tag):
    """Build the leaderboard aggregate from a list of per-episode metric dicts.

    Single source of truth shared by the vision-policy path (evaluate_frozen) and
    the state-based oracle path (evaluate_oracle_frozen) so both report identical
    columns (score CI, survival, tier1, conditional precision, ...).
    """
    scores = [m['score'] for m in per_ep]
    survivals = [m['survival_rate'] for m in per_ep]
    cond = [m for m in per_ep if m['survived_threshold']]

    score_mean, score_lo, score_hi = bootstrap_ci(scores)
    surv_mean, surv_lo, surv_hi = bootstrap_ci(survivals)

    return {
        'arch': arch_tag,
        'n_episodes': n_episodes,
        'base_seed': base_seed,
        'survive_threshold': survive_threshold,
        # --- artifact-robust headline (all episodes) ---
        'score_mean': score_mean, 'score_ci95': [score_lo, score_hi],
        'survival_mean': surv_mean, 'survival_ci95': [surv_lo, surv_hi],
        'tier1_pass_rate': float(np.mean([m['tier1_pass'] for m in per_ep])),
        'mean_ep_length': float(np.mean([m['ep_length'] for m in per_ep])),
        # --- naive (all-episode) precision — KEPT but flagged as artifact-prone ---
        'iae_steady_all': float(np.mean([m['iae_steady'] for m in per_ep])),
        'terminal_err_all': float(np.mean([m['terminal_err'] for m in per_ep])),
        # --- CONDITIONAL precision (only episodes that actually flew) ---
        'n_conditional': len(cond),
        'cond_frac': float(len(cond) / n_episodes),
        'iae_steady_cond': float(np.mean([m['iae_steady'] for m in cond])) if cond else float('nan'),
        'terminal_err_cond': float(np.mean([m['terminal_err'] for m in cond])) if cond else float('nan'),
        'omega_steady_cond': float(np.mean([m['omega_steady'] for m in cond])) if cond else float('nan'),
        'rmse_legacy_all': float(np.mean([m['rmse_legacy'] for m in per_ep])),
        'per_episode': per_ep,
    }


def evaluate_frozen(ckpt_path, n_episodes, base_seed, survive_threshold,
                    quadrotor_config, flow_config, n_inference_steps,
                    sigma, device_str='cuda', cue_scale=3.0, cue_noise=0.0):
    """Evaluate one checkpoint under the frozen protocol; returns aggregate + per-ep."""
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
    max_steps = base_env.max_episode_steps
    survive_threshold = min(survive_threshold, max_steps)

    per_ep = []
    for ep in range(n_episodes):
        seed = base_seed + ep
        roll = rollout_episode(policy, base_env, visual_env, arch,
                               T_obs, T_action, n_inference_steps, device, seed=seed,
                               cue_scale=cue_scale, cue_noise=cue_noise)
        m = compute_hierarchical_metrics(roll['positions'], roll['targets'],
                                         roll['omegas'], roll['ep_length'], max_steps)
        m.update(composite_score(m, sigma=sigma))
        m['ep_reward'] = roll['ep_reward']
        m['seed'] = seed
        m['survived_threshold'] = bool(roll['ep_length'] >= survive_threshold)
        per_ep.append(m)
        print(f"  Ep {ep+1:>2}/{n_episodes} seed={seed} | steps={roll['ep_length']:>3} | "
              f"survive={m['survival_rate']*100:>5.1f}% | IAE_st={m['iae_steady']:.3f}m | "
              f"term={m['terminal_err']:.3f}m | score={m['score']:.3f}"
              f"{'  [cond]' if m['survived_threshold'] else ''}")

    return _aggregate_frozen(per_ep, n_episodes, base_seed, survive_threshold,
                             arch.get('era', '?'))


def rollout_oracle_episode(agent, obs_rms, base_env, seed):
    """One closed-loop episode of the state-based PPO oracle under frozen seeding.

    The oracle reads the 15D privileged state, so no visual env / rendering is
    needed. We still seed np.random + torch + env.reset(seed) so the *initial pose*
    is byte-identical to the one the vision policies saw for this episode index —
    this makes %Oracle a genuine paired upper bound on the exact same episodes.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    obs, _ = base_env.reset(seed=seed)
    positions, targets, omegas = [], [], []
    done = False
    ep_length = 0
    ep_reward = 0.0
    while not done:
        obs_n = obs_rms.normalize(obs)
        action = agent.get_action_deterministic(obs_n)
        obs, reward, terminated, truncated, info = base_env.step(action)
        ep_reward += reward
        ep_length += 1
        positions.append(info['position'].copy())
        targets.append(info['target'].copy())
        omegas.append(base_env.dynamics.ang_velocity.copy())
        if terminated or truncated:
            done = True
    return {
        'positions': np.array(positions),
        'targets': np.array(targets),
        'omegas': np.array(omegas),
        'ep_length': ep_length,
        'ep_reward': ep_reward,
    }


def evaluate_oracle_frozen(ckpt_path, norm_path, n_episodes, base_seed,
                           survive_threshold, quadrotor_config, sigma,
                           hidden_dim=256, critic_hidden_dim=256):
    """Evaluate the state-based PPO oracle under the SAME frozen protocol & env
    config as the vision policies. Returns an aggregate shaped identically to
    evaluate_frozen() (so it slots into the same leaderboard); its score_mean is
    the measured %Oracle normalisation constant — replacing the hard-coded 0.85.
    """
    base_env = QuadrotorEnvV4(config_path=quadrotor_config)
    state_dim = base_env.observation_space.shape[0]
    action_dim = base_env.action_space.shape[0]

    agent = PPOExpert(state_dim=state_dim, action_dim=action_dim,
                      hidden_dim=hidden_dim, critic_hidden_dim=critic_hidden_dim)
    agent.load(ckpt_path)

    obs_rms = RunningMeanStd(shape=(state_dim,))
    nd = np.load(norm_path)
    obs_rms.load_state_dict({'mean': nd['mean'], 'var': nd['var'],
                             'count': float(nd['count'])})

    max_steps = base_env.max_episode_steps
    survive_threshold = min(survive_threshold, max_steps)

    per_ep = []
    for ep in range(n_episodes):
        seed = base_seed + ep
        roll = rollout_oracle_episode(agent, obs_rms, base_env, seed)
        m = compute_hierarchical_metrics(roll['positions'], roll['targets'],
                                         roll['omegas'], roll['ep_length'], max_steps)
        m.update(composite_score(m, sigma=sigma))
        m['ep_reward'] = roll['ep_reward']
        m['seed'] = seed
        m['survived_threshold'] = bool(roll['ep_length'] >= survive_threshold)
        per_ep.append(m)
        print(f"  [oracle] Ep {ep+1:>2}/{n_episodes} seed={seed} | steps={roll['ep_length']:>3} | "
              f"survive={m['survival_rate']*100:>5.1f}% | IAE_st={m['iae_steady']:.3f}m | "
              f"term={m['terminal_err']:.3f}m | score={m['score']:.3f}")

    return _aggregate_frozen(per_ep, n_episodes, base_seed, survive_threshold, 'state')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpts', nargs='*', default=[],
                        help='List of "label:path" pairs (vision flow policies)')
    parser.add_argument('--n-episodes', type=int, default=30)
    parser.add_argument('--base-seed', type=int, default=12345,
                        help='Episode i uses seed = base_seed + i (frozen across all models)')
    parser.add_argument('--survive-threshold', type=int, default=250,
                        help='Min steps for an episode to count toward conditional-IAE/term '
                             '(default 250 = half of max 500)')
    parser.add_argument('--quadrotor-config', default='configs/quadrotor_v4.yaml')
    parser.add_argument('--flow-config', default='configs/flow_policy_v4.yaml')
    parser.add_argument('--n-inference-steps', type=int, default=2)
    parser.add_argument('--sigma', type=float, default=2.0)
    parser.add_argument('--oracle-ckpt', default=None,
                        help='State-based PPO oracle checkpoint (best_model.pt). When given, '
                             'the oracle is rolled out under the SAME frozen protocol/env and its '
                             'MEASURED composite score becomes the %Oracle normaliser — the rigorous '
                             'replacement for the hard-coded 0.85.')
    parser.add_argument('--oracle-norm', default=None,
                        help='State-based PPO oracle obs normalisation (best_obs_rms.npz). '
                             'Required when --oracle-ckpt is given.')
    parser.add_argument('--oracle-hidden-dim', type=int, default=256)
    parser.add_argument('--oracle-critic-hidden-dim', type=int, default=256)
    parser.add_argument('--oracle-score', type=float, default=0.85,
                        help='FALLBACK oracle composite score used only when --oracle-ckpt is NOT '
                             'given (legacy ~0.85 ref). Set <=0 to disable the %Oracle column.')
    parser.add_argument('--cue-scale', type=float, default=3.0,
                        help='Phase 3b range-cue: metric cue divisor (must match training; default 3.0)')
    parser.add_argument('--cue-noise', type=float, default=0.0,
                        help='Phase 3b range-cue: sensor noise std (metres) added to the cue at eval '
                             '(match the noised training arm)')
    parser.add_argument('--output', default='evaluation_results/frozen_p0_leaderboard.json')
    args = parser.parse_args()

    if not args.ckpts and not args.oracle_ckpt:
        parser.error('nothing to evaluate: pass --ckpts and/or --oracle-ckpt')
    if args.oracle_ckpt and not args.oracle_norm:
        parser.error('--oracle-ckpt requires --oracle-norm (best_obs_rms.npz)')

    print(f"\n{'#'*92}")
    print(f"# P0 FROZEN PROTOCOL | n_ep={args.n_episodes} base_seed={args.base_seed} "
          f"sigma={args.sigma} survive_threshold={args.survive_threshold} "
          f"n_inf_steps={args.n_inference_steps}")
    print(f"# Conditional precision = episodes surviving >= {args.survive_threshold} steps only "
          f"(short-survival artifact fix)")
    print(f"{'#'*92}")

    results = {}
    for entry in args.ckpts:
        label, path = entry.split(':', 1)
        print(f"\n{'='*72}\n=== {label}  ({path})\n{'='*72}")
        if not os.path.exists(path):
            print("  SKIP: not found"); continue
        try:
            agg = evaluate_frozen(path, args.n_episodes, args.base_seed,
                                  args.survive_threshold, args.quadrotor_config,
                                  args.flow_config, args.n_inference_steps, args.sigma,
                                  cue_scale=args.cue_scale, cue_noise=args.cue_noise)
            agg['label'] = label
            agg['path'] = path
            results[label] = agg
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    # ---- State-based PPO oracle (measured %Oracle normaliser) ----
    measured_oracle = None
    if args.oracle_ckpt:
        print(f"\n{'='*72}\n=== PPO_Oracle  ({args.oracle_ckpt})\n{'='*72}")
        if not os.path.exists(args.oracle_ckpt):
            print("  SKIP: oracle ckpt not found")
        else:
            try:
                oagg = evaluate_oracle_frozen(
                    args.oracle_ckpt, args.oracle_norm, args.n_episodes,
                    args.base_seed, args.survive_threshold, args.quadrotor_config,
                    args.sigma, args.oracle_hidden_dim, args.oracle_critic_hidden_dim)
                oagg['label'] = 'PPO_Oracle'
                oagg['path'] = args.oracle_ckpt
                results['PPO_Oracle'] = oagg
                measured_oracle = oagg['score_mean']
                print(f"\n  >> measured oracle composite score = {measured_oracle:.4f} "
                      f"(replaces hard-coded {args.oracle_score:.2f})")
            except Exception as e:
                print(f"  ORACLE FAILED: {e}")
                import traceback; traceback.print_exc()

    # ---- Leaderboard ----
    # Prefer the measured oracle score; fall back to the legacy constant only when
    # no oracle checkpoint was rolled out this run.
    oracle_norm = measured_oracle if measured_oracle is not None else args.oracle_score
    oracle_src = 'measured' if measured_oracle is not None else 'legacy-const'
    use_oracle = oracle_norm and oracle_norm > 0
    print("\n" + "=" * 118)
    hdr = (f"{'Label':<22}{'Arch':>6}{'Score(95% CI)':>22}{'Survive%':>10}"
           f"{'Tier1%':>8}{'cond-IAE':>10}{'cond-Term':>11}{'n_cond':>7}{'all-IAE':>9}")
    if use_oracle:
        hdr += f"{'%Oracle':>9}"
    print(hdr)
    print("-" * 118)
    for r in sorted(results.values(), key=lambda x: -x['score_mean']):
        ci = r['score_ci95']
        line = (f"{r['label']:<22}{r['arch']:>6}"
                f"{r['score_mean']:>8.3f} [{ci[0]:.3f},{ci[1]:.3f}]"
                f"{r['survival_mean']*100:>9.1f}%"
                f"{r['tier1_pass_rate']*100:>7.1f}%"
                f"{r['iae_steady_cond']:>9.3f}m"
                f"{r['terminal_err_cond']:>10.3f}m"
                f"{r['n_conditional']:>4}/{r['n_episodes']:<2}"
                f"{r['iae_steady_all']:>8.3f}m")
        if use_oracle:
            line += f"{r['score_mean']/oracle_norm*100:>8.1f}%"
        print(line)
    print("=" * 118)
    print("Reading guide:")
    print("  - Score(95% CI): headline composite, sigma-exp metric, bootstrap CI over episodes.")
    print("  - cond-IAE / cond-Term: precision over episodes that survived the threshold ONLY.")
    print("    If n_cond is tiny (e.g. 1/30) the cond columns are unreliable -> policy doesn't fly.")
    print("  - all-IAE: naive all-episode IAE; LOW value + LOW n_cond = short-survival ARTIFACT,")
    print("    not precision. Compare all-IAE vs cond-IAE to expose it.")
    if use_oracle:
        print(f"  - %Oracle: score / {oracle_norm:.4f} ({oracle_src} state-based PPO oracle ref).")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults (incl. per-episode raw) saved to: {args.output}")


if __name__ == '__main__':
    main()

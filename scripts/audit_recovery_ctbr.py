"""
Phase 0 / Gate A — far-range setpoint-offset RECOVERY AUDIT (teacher selection).

RESEARCH_PLAN_v7's decisive Teacher × Observation 2×2 needs a competent FAR-RANGE
teacher to generate 1-3 m recovery coverage (the band where closed-loop hover drifts
to ~2.8 m and BC has <0.2% of its mass). Two candidate teachers are state-based and
CTBR-native for the v4 (INDI) pipeline:

  * PID-CTBR : CascadePIDController.compute_ctbr_action  (cascade Levels 1-3 → CTBR,
               bypassing the per-motor mixer; no training needed)
  * PPO      : the state-based PPO expert (known to crash beyond ~2 m offset)

This script pins the drone at a near-origin hover, shifts the hover target by a
horizontal offset of {1,2,3,4} m (random azimuth, frozen seed → PID & PPO see the
SAME offset for a paired comparison), and rolls each teacher closed-loop on the base
QuadrotorEnvV4 (no rendering — both teachers read privileged state). Per (teacher,
distance) it reports survival%, conditional steady-IAE (episodes surviving
>= --survive-threshold steps), Tier1% and a bootstrap 95% CI, reusing the frozen-P0
metric machinery so numbers slot into the same leaderboard convention.

GATE DECISION (pre-registered, aligned with the v7 T1O1 <=1.5 m target):
  a teacher "passes" the far-range band if, over dist in {1,2,3} m, it keeps
  survival >= ~80% AND conditional steady-IAE <= ~1.5 m (it actually flies back and
  converges, not merely survives while drifting).
    * PID-CTBR passes & beats PPO's >2 m collapse  -> use PID-CTBR as the teacher
      (no retrain).
    * PID-CTBR also collapses past 2 m             -> retrain a PPO with wide
      setpoint-offset init (Phase 1).

Usage:
  dppo/Scripts/python.exe -m scripts.audit_recovery_ctbr \
      --ppo-model checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
      --ppo-norm  checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz \
      --distances 1 2 3 4 --n-trials 20
"""
import os
import sys
import json
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.quadrotor_env_v4 import QuadrotorEnvV4
from controllers.pid_controller import CascadePIDController
from models.ppo_expert import PPOExpert, RunningMeanStd
from scripts.evaluate_hierarchical import (
    compute_hierarchical_metrics, composite_score,
)
from scripts.evaluate_frozen_p0 import _aggregate_frozen


def _trial_seed(base_seed, dist_idx, trial):
    """Deterministic per-(distance, trial) seed so PID & PPO see identical offsets."""
    return base_seed + dist_idx * 1000 + trial


def _offset_for(base_seed, dist_idx, trial, dist):
    """Horizontal offset of magnitude `dist` at a frozen random azimuth (z=0, so the
    target altitude matches the near-origin hover start)."""
    rng = np.random.default_rng(_trial_seed(base_seed, dist_idx, trial))
    phi = rng.uniform(-np.pi, np.pi)
    return dist * np.array([np.cos(phi), np.sin(phi), 0.0])


def _metrics_from_rollout(positions, targets, omegas, ep_length, max_steps,
                          survive_threshold, sigma, seed):
    m = compute_hierarchical_metrics(np.array(positions), np.array(targets),
                                     np.array(omegas), ep_length, max_steps)
    m.update(composite_score(m, sigma=sigma))
    m['seed'] = seed
    m['survived_threshold'] = bool(ep_length >= survive_threshold)
    return m


def rollout_pid_ctbr(ctrl, base_env, seed, offset):
    """One closed-loop episode of the PID-CTBR teacher from a setpoint offset."""
    ctrl.reset()
    base_env.reset(seed=seed, options={'setpoint_offset': offset})
    positions, targets, omegas = [], [], []
    done, ep_length = False, 0
    while not done:
        action = ctrl.compute_ctbr_action(
            base_env.dynamics.state, base_env.target_position,
            base_env.F_c_max, base_env.omega_max)
        _, _, terminated, truncated, info = base_env.step(action)
        ep_length += 1
        positions.append(info['position'].copy())
        targets.append(info['target'].copy())
        omegas.append(base_env.dynamics.ang_velocity.copy())
        done = terminated or truncated
    return positions, targets, omegas, ep_length


def rollout_ppo(agent, obs_rms, base_env, seed, offset):
    """One closed-loop episode of the state-based PPO expert from a setpoint offset."""
    obs, _ = base_env.reset(seed=seed, options={'setpoint_offset': offset})
    positions, targets, omegas = [], [], []
    done, ep_length = False, 0
    while not done:
        action = agent.get_action_deterministic(obs_rms.normalize(obs))
        obs, _, terminated, truncated, info = base_env.step(action)
        ep_length += 1
        positions.append(info['position'].copy())
        targets.append(info['target'].copy())
        omegas.append(base_env.dynamics.ang_velocity.copy())
        done = terminated or truncated
    return positions, targets, omegas, ep_length


def audit_teacher(name, rollout_fn, base_env, distances, n_trials, base_seed,
                  survive_threshold, sigma):
    """Run rollout_fn over all distances × trials; return {dist: frozen-aggregate}."""
    max_steps = base_env.max_episode_steps
    survive_threshold = min(survive_threshold, max_steps)
    out = {}
    for di, dist in enumerate(distances):
        per_ep = []
        for trial in range(n_trials):
            seed = _trial_seed(base_seed, di, trial)
            offset = _offset_for(base_seed, di, trial, dist)
            positions, targets, omegas, ep_length = rollout_fn(base_env, seed, offset)
            m = _metrics_from_rollout(positions, targets, omegas, ep_length,
                                      max_steps, survive_threshold, sigma, seed)
            per_ep.append(m)
        agg = _aggregate_frozen(per_ep, n_trials, base_seed, survive_threshold, name)
        out[f'{dist:g}m'] = agg
        print(f"  [{name:<8s}] {dist:g}m : survive={agg['survival_mean']*100:5.1f}% "
              f"tier1={agg['tier1_pass_rate']*100:5.1f}% "
              f"cond-IAE={agg['iae_steady_cond']:.3f}m (n={agg['n_conditional']}) "
              f"all-IAE={agg['iae_steady_all']:.3f}m")
    return out


def _gate_verdict(teacher_results, band=(1.0, 2.0, 3.0),
                  surv_min=0.80, iae_max=1.5):
    """Pre-registered pass test over the 1-3 m band."""
    verdict = {}
    for name, res in teacher_results.items():
        ok = True
        detail = []
        for d in band:
            agg = res.get(f'{d:g}m')
            if agg is None:
                continue
            surv = agg['survival_mean']
            iae = agg['iae_steady_cond']
            passed = (surv >= surv_min) and np.isfinite(iae) and (iae <= iae_max)
            ok = ok and passed
            detail.append({'dist_m': d, 'survival': surv, 'cond_iae': iae,
                           'pass': bool(passed)})
        verdict[name] = {'passes_band': bool(ok), 'band_m': list(band),
                         'surv_min': surv_min, 'iae_max': iae_max, 'per_dist': detail}
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ppo-model', default='checkpoints/ppo_expert_v4/20260419_142245/best_model.pt')
    ap.add_argument('--ppo-norm',  default='checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz')
    ap.add_argument('--distances', nargs='+', type=float, default=[1, 2, 3, 4])
    ap.add_argument('--n-trials', type=int, default=20)
    ap.add_argument('--base-seed', type=int, default=12345)
    ap.add_argument('--survive-threshold', type=int, default=250)
    ap.add_argument('--sigma', type=float, default=2.0)
    ap.add_argument('--quadrotor-config', default='configs/quadrotor_v4.yaml')
    # PID-CTBR teacher gains. Defaults are the GENTLE recovery tune: the validated
    # PID-baseline gains (vel_max=2.0, Kp_pos=1.5) approach too aggressively and trip
    # the 60-deg tilt termination on diagonal 2 m offsets (a "death valley"); easing
    # vel_max/Kp_pos keeps the approach inside the tilt envelope and gives 100%
    # survival + ~0.15 m steady-IAE across the whole 1-4 m band.
    ap.add_argument('--pid-vel-max', type=float, default=1.0)
    ap.add_argument('--pid-kp-pos', type=float, default=0.8)
    ap.add_argument('--out', default='evaluation_results/p0_recovery_audit_ctbr.json')
    args = ap.parse_args()

    base_env = QuadrotorEnvV4(config_path=os.path.join(ROOT, args.quadrotor_config))
    # Pin init near origin (anchor mode) so the only variable is the setpoint offset
    # and a 4 m horizontal target stays inside the |pos|<5 m position_bound.
    base_env.hover_anchor_prob = 1.0

    print(f"Recovery audit | distances={args.distances} m | n_trials={args.n_trials} "
          f"| survive>={args.survive_threshold} steps | seed={args.base_seed}")

    # --- Teacher 1: PID-CTBR (omega_max relaxed to 6 rad/s so Level 3 isn't clipped
    #     before the env's per-axis CTBR normalisation [6,6,3] does the capping;
    #     vel_max/Kp_pos eased to the gentle recovery tune) ---
    ctrl = CascadePIDController(base_env.dynamics.params,
                               omega_max=6.0, dt=base_env.dt_outer,
                               vel_max=args.pid_vel_max, Kp_pos=args.pid_kp_pos)
    print(f"PID-CTBR teacher gains: vel_max={args.pid_vel_max} Kp_pos={args.pid_kp_pos}")
    pid_results = audit_teacher(
        'PID_CTBR', lambda env, s, o: rollout_pid_ctbr(ctrl, env, s, o),
        base_env, args.distances, args.n_trials, args.base_seed,
        args.survive_threshold, args.sigma)

    # --- Teacher 2: state-based PPO expert (CTBR-native control group) ---
    state_dim = base_env.observation_space.shape[0]
    action_dim = base_env.action_space.shape[0]
    agent = PPOExpert(state_dim=state_dim, action_dim=action_dim,
                      hidden_dim=256, critic_hidden_dim=256)
    agent.load(os.path.join(ROOT, args.ppo_model))
    obs_rms = RunningMeanStd(shape=(state_dim,))
    nd = np.load(os.path.join(ROOT, args.ppo_norm))
    obs_rms.load_state_dict({'mean': nd['mean'], 'var': nd['var'],
                             'count': float(nd['count'])})
    ppo_results = audit_teacher(
        'PPO', lambda env, s, o: rollout_ppo(agent, obs_rms, env, s, o),
        base_env, args.distances, args.n_trials, args.base_seed,
        args.survive_threshold, args.sigma)

    teacher_results = {'PID_CTBR': pid_results, 'PPO': ppo_results}
    verdict = _gate_verdict(teacher_results)

    out = {
        'distances_m': args.distances,
        'n_trials': args.n_trials,
        'base_seed': args.base_seed,
        'survive_threshold': args.survive_threshold,
        'sigma': args.sigma,
        'results': teacher_results,
        'gate_verdict': verdict,
    }
    os.makedirs(os.path.join(ROOT, os.path.dirname(args.out)), exist_ok=True)
    with open(os.path.join(ROOT, args.out), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 78)
    print("GATE A SUMMARY — far-range recovery (cond-IAE is the deciding precision axis)")
    print("=" * 78)
    print(f"{'teacher/dist':<18s}{'survive':>10s}{'tier1':>9s}{'cond-IAE(n)':>16s}{'all-IAE':>10s}")
    for name, res in teacher_results.items():
        for dkey, agg in res.items():
            print(f"{name+'/'+dkey:<18s}{agg['survival_mean']*100:>9.1f}%"
                  f"{agg['tier1_pass_rate']*100:>8.1f}%"
                  f"{agg['iae_steady_cond']:>10.3f}m(n{agg['n_conditional']:>2d})"
                  f"{agg['iae_steady_all']:>9.3f}m")
    print("\nPre-registered band-pass (1-3 m, survive>=80% & cond-IAE<=1.5 m):")
    for name, v in verdict.items():
        print(f"  {name:<8s}: {'PASS' if v['passes_band'] else 'FAIL'}")
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()

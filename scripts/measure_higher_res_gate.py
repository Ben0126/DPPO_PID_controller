"""
Phase 3b higher-res sensing GATE (free, no training).

§6.3 of the diagnosis argues precision is information-gated: the 64x64 FPV image
does not encode metric range past ~2 m (R^2_far = 0.12, byte-identical renders >=2 m;
see docs/experiment_report_image_distance_info.md). Before paying to RE-COLLECT data at
higher resolution and retrain, this gate asks — cheaply, with no training — whether a
more capable renderer would even CARRY the far-range information:

  resolution  in {64, 128, 256}
  target      in {production crosshair (saturating), perspective AA disk (non-saturating)}

For each (resolution, target) it re-runs the same image -> distance recoverability
measurement as scripts.measure_image_distance_info (adjacent d-prime + ridge R^2,
stratified NEAR <1 m vs FAR >=1.5 m). This separates two confounds:
  * does DE-SATURATING the target alone (same 64 px) restore far-range R^2?  -> the loss
    was a renderer artifact (the quantised size formula), not the pixel count;
  * does adding RESOLUTION help on top?                                       -> pixel
    count is the binding limit.

GATE DECISION: only if R^2_far rises substantially (>= ~0.5) under some achievable
(resolution, target) is the higher-res RE-COLLECT + retrain justified. If R^2_far stays
~0.12 everywhere, even a richer monocular FPV cannot encode 2-3 m range -> the
information-gated finding hardens and higher-res is NOT worth the compute.

Usage:
  dppo/Scripts/python.exe -m scripts.measure_higher_res_gate
  dppo/Scripts/python.exe -m scripts.measure_higher_res_gate --no-dr   # noiseless ceiling
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
from envs.quadrotor_visual_env import QuadrotorVisualEnv
from scripts.measure_image_distance_info import dprime


def ridge_r2(X, y, alpha=1.0, seed=0):
    """Ridge decode y from X, R^2 on a 30% holdout. Uses the DUAL (kernel) form
    so cost is O(n^2) in samples, not O(d^2) in pixels — primal would build a
    d x d matrix (d=H*W*3 = 196608 at 256px -> 154 GB). Mathematically identical
    to the primal ridge in scripts.measure_image_distance_info.ridge_r2."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_te = max(1, int(0.3 * n))
    te, tr = idx[:n_te], idx[n_te:]
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
    Xtr = (X[tr] - mu) / sd
    Xte = (X[te] - mu) / sd
    ytr = y[tr].astype(np.float64); yte = y[te].astype(np.float64)
    ybar = ytr.mean()
    K = Xtr @ Xtr.T                                  # (n_tr, n_tr) Gram matrix
    a = np.linalg.solve(K + alpha * np.eye(K.shape[0]), ytr - ybar)
    pred = (Xte @ Xtr.T) @ a + ybar                  # dual prediction on holdout
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum()) + 1e-9
    return 1.0 - ss_res / ss_tot


# NOTE: the perspective AA-disk target now lives in the production env
# (QuadrotorVisualEnv(target_render="perspective"), the _draw_target_perspective
# branch) — single source of truth. This script just toggles the flag below.


def render_set(visual_env, base_env, dist, n_samples):
    """Render n_samples FPV images of a level drone with the target `dist` m straight
    ahead (same pose protocol as measure_image_distance_info.render_at_distance)."""
    imgs = []
    base_env.reset(seed=0)
    base_env.dynamics.position = np.array([0.0, 0.0, -2.0])
    base_env.dynamics.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    unit = np.array([0.985, 0.174, 0.0])
    unit = unit / np.linalg.norm(unit)
    base_env.target_position = base_env.dynamics.position + dist * unit
    for _ in range(n_samples):
        if visual_env.dr_enabled:
            visual_env._randomize_episode()
        img = visual_env._render_fpv()                 # (3, R, R) uint8
        imgs.append(img.astype(np.float32).reshape(-1))
    return np.stack(imgs)


def measure(visual_env, base_env, distances, n_samples):
    per = {d: render_set(visual_env, base_env, d, n_samples) for d in distances}
    # adjacent d-prime (focus far pairs)
    adj = {}
    for i in range(len(distances) - 1):
        adj[f'{distances[i]}->{distances[i+1]}'] = round(
            dprime(per[distances[i]], per[distances[i + 1]]), 4)
    # ridge decode image -> distance, overall + near/far
    def r2_for(ds):
        if len(ds) < 2:
            return None
        X = np.concatenate([per[d] for d in ds], 0)
        y = np.concatenate([np.full(len(per[d]), d) for d in ds])
        return round(ridge_r2(X, y), 4)
    return {
        'r2_overall': r2_for(distances),
        'r2_near_<1m': r2_for([d for d in distances if d < 1.0]),
        'r2_far_>=1.5m': r2_for([d for d in distances if d >= 1.5]),
        'adjacent_dprime': adj,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resolutions', nargs='+', type=int, default=[64, 128, 256])
    ap.add_argument('--distances', nargs='+', type=float,
                    default=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument('--n-samples', type=int, default=200)
    ap.add_argument('--physical-size', type=float, default=0.5)
    ap.add_argument('--quadrotor-config', default='configs/quadrotor_v4.yaml')
    ap.add_argument('--no-dr', action='store_true')
    ap.add_argument('--out', default='evaluation_results/p3b_higher_res_gate.json')
    args = ap.parse_args()

    dr = not args.no_dr
    results = {}
    print(f"Higher-res sensing gate  (DR={'OFF' if args.no_dr else 'ON'}, "
          f"n={args.n_samples}/dist, S={args.physical_size} m)")
    for res in args.resolutions:
        base_env = QuadrotorEnvV4(config_path=os.path.join(ROOT, args.quadrotor_config))
        for target_kind, target_render in [('crosshair_prod', 'crosshair'),
                                           ('perspective_aa', 'perspective')]:
            venv = QuadrotorVisualEnv(base_env, image_size=res, dr_enabled=dr,
                                      target_render=target_render,
                                      physical_size=args.physical_size)
            m = measure(venv, base_env, args.distances, args.n_samples)
            results[f'{res}px/{target_kind}'] = m
            print(f"  {res:>3}px  {target_kind:<14s}  "
                  f"R2 overall={m['r2_overall']}  near={m['r2_near_<1m']}  "
                  f"far={m['r2_far_>=1.5m']}")

    out = {
        'dr_enabled': dr, 'n_samples': args.n_samples,
        'physical_size_m': args.physical_size, 'distances': args.distances,
        'results': results,
        'baseline_64px_far_r2': 0.12,   # docs/experiment_report_image_distance_info.md
    }
    os.makedirs(os.path.join(ROOT, os.path.dirname(args.out)), exist_ok=True)
    with open(os.path.join(ROOT, args.out), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 72)
    print("GATE SUMMARY  —  R^2(image -> distance), FAR (>=1.5 m) is the deciding cell")
    print("=" * 72)
    print(f"{'config':<24s}{'far R2':>10s}{'near R2':>10s}{'overall':>10s}")
    for k, m in results.items():
        print(f"{k:<24s}{str(m['r2_far_>=1.5m']):>10s}"
              f"{str(m['r2_near_<1m']):>10s}{str(m['r2_overall']):>10s}")
    print(f"\nbaseline (64px production, prior report): far R2 = 0.12")
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()

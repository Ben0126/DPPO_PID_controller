"""
Phase 3b (negative-result diagnosis) — does the 64x64 FPV image even CARRY the
metric distance the policy would need to null a 1-3 m position error?

Motivation: the OOD-coverage probe (`measure_ood_coverage.py`) showed precision is
NOT limited by survival but the policy parks at ~2.8 m where BC has ~no labels. Two
hypotheses: (i) DATA-gated (fill coverage → precision improves) vs (ii)
INFORMATION-gated (the FPV observation simply doesn't encode metric distance, so no
data can fix it). Reading the renderer (`quadrotor_visual_env._render_fpv`) shows the
ONLY distance-dependent feature is the target crosshair SIZE:
    size = max(2, min(6, int(6/(dist+0.5)) + dr)),  dr ~ U{-2..+2} per episode
which is ~4 quantised levels, saturates at 2 px beyond ~2 m, and is corrupted by the
DR size jitter (+-2 px) and per-frame Gaussian pixel noise (sigma=5). The crosshair
POSITION encodes only normalised direction (distance-invariant), and the crosshair is
drawn only when the target is in front (forward>0).

This script QUANTIFIES the distance information cheaply (no training): with the drone
held level at fixed altitude and the target straight ahead, it renders many DR samples
per target distance and measures how recoverable distance is from the image:

  * crosshair-size distribution per distance (the literal signal channel);
  * d-prime between adjacent distances = ||mean_img(d2) - mean_img(d1)|| / within-d noise;
  * Ridge decode image -> distance, R^2 on a held-out split, stratified NEAR (<1 m)
    vs FAR (>=1.5 m).

If FAR d-prime ~ 0 and FAR R^2 ~ 0 while NEAR is high, precision at 1-3 m is
INFORMATION-gated by the observation model — a wider-init retrain cannot help.

Usage:
  dppo/Scripts/python.exe -m scripts.measure_image_distance_info
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


def render_at_distance(visual_env, base_env, dist, n_samples, rng):
    """Render n_samples FPV images of a level drone with the target `dist` m straight
    ahead (body-forward). Each render re-samples per-episode DR + per-frame noise."""
    imgs = []
    sizes = []
    # Level hover pose at a fixed altitude; target straight ahead (+ tiny lateral so
    # the crosshair is unambiguously on-screen but direction is ~constant across dist).
    base_env.reset(seed=0)
    base_env.dynamics.position = np.array([0.0, 0.0, -2.0])
    # force identity (level) attitude
    base_env.dynamics.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    unit = np.array([0.985, 0.174, 0.0])  # mostly forward, ~10deg right (forward>0)
    unit = unit / np.linalg.norm(unit)
    base_env.target_position = base_env.dynamics.position + dist * unit
    for _ in range(n_samples):
        if visual_env.dr_enabled:
            visual_env._randomize_episode()         # per-episode DR (incl crosshair jitter)
        img = visual_env._render_fpv()              # (3,64,64) uint8 (per-frame noise inside)
        imgs.append(img.astype(np.float32).reshape(-1))
        sizes.append(_crosshair_size(base_env, visual_env, dist))
    return np.stack(imgs), np.array(sizes)


def _crosshair_size(base_env, visual_env, dist):
    """The literal size channel (pre-noise), for reporting the quantisation."""
    dr = visual_env._dr_crosshair_d
    return max(2, min(6, int(6 / (dist + 0.5)) + dr))


def dprime(a, b):
    """Separability of two image sample sets: ||mean_a - mean_b|| / pooled within-std."""
    ma, mb = a.mean(0), b.mean(0)
    signal = np.linalg.norm(ma - mb)
    noise = 0.5 * (a.std(0).mean() + b.std(0).mean()) * np.sqrt(a.shape[1])
    return float(signal / (noise + 1e-9))


def ridge_r2(X, y, alpha=1.0, seed=0):
    """Ridge regression decode y from X, R^2 on a held-out 30% split."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_te = max(1, int(0.3 * n))
    te, tr = idx[:n_te], idx[n_te:]
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    # closed-form ridge
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + alpha * np.eye(d)
    w = np.linalg.solve(A, Xtr.T @ (ytr - ytr.mean()))
    pred = Xte @ w + ytr.mean()
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum()) + 1e-9
    return 1.0 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distances', nargs='+', type=float,
                    default=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument('--n-samples', type=int, default=300)
    ap.add_argument('--quadrotor-config', default='configs/quadrotor_v4.yaml')
    ap.add_argument('--no-dr', action='store_true', help='disable domain randomisation (clean signal)')
    ap.add_argument('--out', default='evaluation_results/p3b_image_distance_info.json')
    args = ap.parse_args()

    base_env = QuadrotorEnvV4(config_path=os.path.join(ROOT, args.quadrotor_config))
    visual_env = QuadrotorVisualEnv(base_env, image_size=64, dr_enabled=not args.no_dr)

    rng = np.random.default_rng(12345)
    per_dist_imgs = {}
    print(f"Rendering {args.n_samples} samples/distance, DR={'OFF' if args.no_dr else 'ON'} ...")
    size_table = {}
    for d in args.distances:
        imgs, sizes = render_at_distance(visual_env, base_env, d, args.n_samples, rng)
        per_dist_imgs[d] = imgs
        size_table[d] = {'min': int(sizes.min()), 'max': int(sizes.max()),
                         'mean': round(float(sizes.mean()), 2)}
        print(f"  d={d:>4.2f}m  crosshair_size min/mean/max = "
              f"{size_table[d]['min']}/{size_table[d]['mean']}/{size_table[d]['max']} px")

    # --- adjacent d-prime ---
    ds = args.distances
    adj = {}
    for i in range(len(ds) - 1):
        dp = dprime(per_dist_imgs[ds[i]], per_dist_imgs[ds[i + 1]])
        adj[f'{ds[i]}->{ds[i+1]}'] = round(dp, 4)

    # --- ridge decode image -> distance, overall + near/far ---
    Xall = np.concatenate([per_dist_imgs[d] for d in ds], 0)
    yall = np.concatenate([np.full(len(per_dist_imgs[d]), d) for d in ds])
    r2_all = ridge_r2(Xall, yall)

    near_ds = [d for d in ds if d < 1.0]
    far_ds = [d for d in ds if d >= 1.5]
    r2_near = r2_far = None
    if len(near_ds) >= 2:
        Xn = np.concatenate([per_dist_imgs[d] for d in near_ds], 0)
        yn = np.concatenate([np.full(len(per_dist_imgs[d]), d) for d in near_ds])
        r2_near = ridge_r2(Xn, yn)
    if len(far_ds) >= 2:
        Xf = np.concatenate([per_dist_imgs[d] for d in far_ds], 0)
        yf = np.concatenate([np.full(len(per_dist_imgs[d]), d) for d in far_ds])
        r2_far = ridge_r2(Xf, yf)

    out = {
        'dr_enabled': not args.no_dr,
        'n_samples': args.n_samples,
        'distances': ds,
        'crosshair_size_by_distance': size_table,
        'adjacent_dprime': adj,
        'ridge_r2_image_to_distance': {
            'overall': round(r2_all, 4),
            'near_<1m': round(r2_near, 4) if r2_near is not None else None,
            'far_>=1.5m': round(r2_far, 4) if r2_far is not None else None,
        },
    }
    os.makedirs(os.path.join(ROOT, os.path.dirname(args.out)), exist_ok=True)
    with open(os.path.join(ROOT, args.out), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 76)
    print(f"FPV IMAGE DISTANCE INFORMATION  (DR={'OFF' if args.no_dr else 'ON'})")
    print("=" * 76)
    print("crosshair size (px) by distance:")
    for d in ds:
        s = size_table[d]
        print(f"  {d:>4.2f}m : {s['min']}-{s['max']} px (mean {s['mean']})")
    print("\nadjacent-distance d-prime (image separability / noise):")
    for k, v in adj.items():
        print(f"  {k:>14s} m : d'={v}")
    print(f"\nRidge decode  image -> distance  R^2:")
    print(f"  overall      : {out['ridge_r2_image_to_distance']['overall']}")
    print(f"  near (<1 m)  : {out['ridge_r2_image_to_distance']['near_<1m']}")
    print(f"  far (>=1.5m) : {out['ridge_r2_image_to_distance']['far_>=1.5m']}")
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()

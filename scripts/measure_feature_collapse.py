"""
Phase 3 (negative-result diagnosis) — measure the feature collapse that the
Dispersive Loss is *supposed* to fix, directly on the P2 ablation checkpoints.

Research question (RESEARCH_PLAN_v6.md Phase 3a):
  Dispersive Loss repels the vision encoder's pooled features
  (`flow_policy_v5._dispersive_loss(vis_pooled)`). It claims to prevent
  "representation collapse". P2 showed it gives NO survival/Tier1 gain. This
  script asks the mechanistic question one level down: *does the dispersive term
  even change the geometry of `vis_pooled`, and is there collapse to fix?*

We load each (cell, seed) checkpoint from the P2 manifest, push a fixed,
seeded batch of images (hover + recovery mix) through the vision encoder, pull
out `vis_pooled` (N, 256), and measure collapse / spread:

  effective_rank      exp(entropy of normalised covariance eigenvalues) [Roy-Vetterli].
                      The canonical collapse metric: low => features live on a
                      low-dim subspace (collapsed); ~D => full spread.
  participation_ratio (Sum l)^2 / Sum l^2  — same idea, "soft" rank.
  stable_rank         Sum l / l_max.
  total_variance      trace of covariance (overall feature energy).
  n_eff_dims_99       # eigenvalues to reach 99% cumulative variance.
  mean_pairwise_dist  mean_{i!=j} ||x_i - x_j||  — EXACTLY what dispersive pushes up.
  mean_pairwise_cos   mean cosine similarity     — collapse => near 1.
  dispersive_loss     -mean_{i!=j} log(||x_i-x_j||+eps)  — the quantity the
                      dispersive term *minimises*. The decisive number: if D1E1
                      (Disp ON) does not beat D0E1 (Disp OFF) here, the mechanism
                      did not even act; if it does beat it but survival is flat,
                      the mechanism acts but is irrelevant.

Decisive comparisons:
  D1E0 vs D0E0  (frozen encoder)  -> must be IDENTICAL (no-op confirmation, complements MD5).
  D1E1 vs D0E1  (trainable encoder) -> Dispersive's only chance to act.

Usage:
  dppo/Scripts/python.exe -m scripts.measure_feature_collapse \
      --manifest evaluation_results/p2_ablation_manifest.json \
      --n-samples 4000
"""
import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import h5py
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.flow_policy_v5 import FlowMatchingPolicyV5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Fixed, seeded image batch (hover + recovery mix) — identical for every model.
# ---------------------------------------------------------------------------

def build_image_batch(hover_h5, recovery_h5, n_samples, T_obs, seed=12345):
    """Return (N, T_obs*3, 64, 64) uint8 sampled deterministically across episodes.

    Half hover, half recovery (matches the BC training mix). For each chosen
    episode we take stacked T_obs-frame windows at seeded random start indices.
    """
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    stacks = []

    for h5_path, n_take in [(hover_h5, half), (recovery_h5, n_samples - half)]:
        with h5py.File(h5_path, 'r') as f:
            n_ep = int(f.attrs['n_episodes'])
            # deterministic episode pool, then sample windows until we have n_take
            ep_pool = rng.permutation(n_ep)
            per_ep = max(1, n_take // min(n_ep, 120))
            got = 0
            for ep_idx in ep_pool:
                if got >= n_take:
                    break
                key = f'episode_{ep_idx}'
                if key not in f:
                    continue
                imgs = f[key]['images'][:]          # (T, 3, 64, 64) uint8
                T = imgs.shape[0]
                if T < T_obs:
                    continue
                starts = rng.integers(T_obs - 1, T, size=min(per_ep, n_take - got))
                for s in starts:
                    frames = imgs[s - T_obs + 1: s + 1]            # (T_obs, 3, 64, 64)
                    stacks.append(np.concatenate(frames, axis=0))  # (T_obs*3, 64, 64)
                    got += 1
                    if got >= n_take:
                        break
    arr = np.stack(stacks).astype(np.uint8)
    return arr


# ---------------------------------------------------------------------------
# Collapse / spread metrics on a (N, D) feature matrix.
# ---------------------------------------------------------------------------

def collapse_metrics(X: torch.Tensor, pair_subset=1024, eps=1e-6):
    """X: (N, D) float32 on device. Returns dict of scalar metrics."""
    X = X.float()
    N, D = X.shape

    # --- covariance spectrum (centered) ---
    Xc = X - X.mean(dim=0, keepdim=True)
    cov = (Xc.T @ Xc) / (N - 1)                       # (D, D)
    eig = torch.linalg.eigvalsh(cov)                  # ascending, real
    eig = torch.clamp(eig, min=0.0)
    eig = torch.flip(eig, dims=[0])                   # descending
    total = eig.sum()
    p = eig / (total + eps)
    # effective rank = exp(Shannon entropy of normalised eigenvalues)
    nz = p[p > 0]
    entropy = -(nz * torch.log(nz)).sum()
    eff_rank = torch.exp(entropy).item()
    participation = (eig.sum() ** 2 / (eig.pow(2).sum() + eps)).item()
    stable_rank = (eig.sum() / (eig.max() + eps)).item()
    total_var = total.item()
    cum = torch.cumsum(p, dim=0)
    n_eff_99 = int((cum < 0.99).sum().item()) + 1
    top1_var_ratio = p[0].item()
    top2_var_ratio = (p[0] + p[1]).item()

    # --- raw scale (to expose dispersive's norm-inflation "objective gaming") ---
    mean_feat_norm = X.norm(dim=-1).mean().item()

    # --- pairwise geometry (the quantity dispersive directly touches) ---
    M = min(pair_subset, N)
    Xs = X[:M]
    diff = Xs.unsqueeze(1) - Xs.unsqueeze(0)          # (M, M, D)
    dist = torch.norm(diff, dim=-1)                   # (M, M)
    mask = 1.0 - torch.eye(M, device=X.device)
    n_pairs = M * (M - 1)
    mean_dist = (dist * mask).sum().item() / n_pairs
    disp_loss = (-torch.log(dist + eps) * mask).sum().item() / n_pairs

    Xn = torch.nn.functional.normalize(Xs, dim=-1)
    cos = Xn @ Xn.T
    mean_cos = ((cos * mask).sum() / n_pairs).item()

    return {
        'effective_rank': round(eff_rank, 3),
        'rank_ratio': round(eff_rank / D, 4),
        'participation_ratio': round(participation, 3),
        'stable_rank': round(stable_rank, 3),
        'n_eff_dims_99': n_eff_99,
        'top1_var_ratio': round(top1_var_ratio, 4),
        'top2_var_ratio': round(top2_var_ratio, 4),
        'mean_feat_norm': round(mean_feat_norm, 4),
        'total_variance': round(total_var, 5),
        'mean_pairwise_dist': round(mean_dist, 5),
        'mean_pairwise_cos': round(mean_cos, 5),
        'dispersive_loss': round(disp_loss, 5),
        'feature_dim': D,
        'n_samples': N,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model(cfg, device):
    vis_cfg = cfg['vision']; imu_cfg = cfg['imu']; unet_cfg = cfg['unet']
    act_cfg = cfg['action']; flow_cfg = cfg['flow']
    xattn_cfg = cfg.get('cross_attn', {}); sp_cfg = cfg.get('state_predictor', {})
    model = FlowMatchingPolicyV5(
        vision_feature_dim=vis_cfg['feature_dim'],
        imu_feature_dim=imu_cfg['feature_dim'],
        time_embed_dim=unet_cfg['time_embed_dim'],
        down_dims=tuple(unet_cfg['down_dims']),
        T_obs=vis_cfg['T_obs'],
        T_pred=act_cfg['T_pred'],
        action_dim=act_cfg['action_dim'],
        n_inference_steps=flow_cfg['n_inference_steps'],
        t_embed_scale=flow_cfg['t_embed_scale'],
        cross_attn_heads=xattn_cfg.get('n_heads', 8),
        state_predictor_hidden=sp_cfg.get('hidden_dim', 256),
        state_dim=sp_cfg.get('state_dim', 15),
        task_dim=2,
    ).to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_vis_pooled(model, images_u8, device, batch=512):
    """images_u8: (N, C, H, W) uint8 numpy -> vis_pooled (N, D) on CPU."""
    outs = []
    for i in range(0, len(images_u8), batch):
        chunk = torch.from_numpy(images_u8[i:i + batch]).to(device).float()  # encoder /255 internally
        pooled, _ = model.vision_encoder(chunk, return_spatial=True)
        outs.append(pooled.cpu())
    return torch.cat(outs, dim=0)


def main():
    ap = argparse.ArgumentParser(description='Measure vis_pooled feature collapse across P2 cells')
    ap.add_argument('--manifest', default='evaluation_results/p2_ablation_manifest.json')
    ap.add_argument('--config', default='configs/flow_policy_v5.yaml')
    ap.add_argument('--hover-h5', default='data/expert_demos_v4.h5')
    ap.add_argument('--recovery-h5', default='data/expert_demos_v4_recovery.h5')
    ap.add_argument('--n-samples', type=int, default=4000)
    ap.add_argument('--pair-subset', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--out', default='evaluation_results/p2_feature_collapse.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(ROOT, args.config), 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    T_obs = cfg['vision']['T_obs']

    with open(os.path.join(ROOT, args.manifest), 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # ---- fixed image batch, identical for every checkpoint ----
    print(f"Building fixed image batch: n={args.n_samples}, seed={args.seed} (hover+recovery) ...")
    images_u8 = build_image_batch(
        os.path.join(ROOT, args.hover_h5),
        os.path.join(ROOT, args.recovery_h5),
        args.n_samples, T_obs, seed=args.seed)
    print(f"  image batch: {images_u8.shape} {images_u8.dtype}")

    model = build_model(cfg, device)

    per_run = {}
    by_cell = defaultdict(list)
    runs = manifest['runs']
    for tag in sorted(runs.keys()):
        info = runs[tag]
        cell, seed = info['cell'], info['seed']
        ckpt = os.path.join(ROOT, info['ckpt'])
        if not os.path.exists(ckpt):
            print(f"  [skip] {tag}: ckpt missing {ckpt}")
            continue
        sd = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(sd)
        model.eval()
        feats = extract_vis_pooled(model, images_u8, device).to(device)
        m = collapse_metrics(feats, pair_subset=args.pair_subset)
        m['cell'] = cell
        m['seed'] = seed
        per_run[tag] = m
        by_cell[cell].append(m)
        print(f"  {tag:14s} cell={cell} seed={seed} | "
              f"eff_rank={m['effective_rank']:.2f} ({m['rank_ratio']*100:.1f}%) "
              f"PR={m['participation_ratio']:.2f} "
              f"pair_dist={m['mean_pairwise_dist']:.3f} "
              f"disp_loss={m['dispersive_loss']:.4f} cos={m['mean_pairwise_cos']:.4f}")

    # ---- aggregate by cell (mean +- std over seeds) ----
    agg = {}
    metric_keys = ['effective_rank', 'rank_ratio', 'participation_ratio', 'stable_rank',
                   'n_eff_dims_99', 'top1_var_ratio', 'top2_var_ratio', 'mean_feat_norm',
                   'total_variance', 'mean_pairwise_dist',
                   'mean_pairwise_cos', 'dispersive_loss']
    for cell, lst in by_cell.items():
        agg[cell] = {'n_seeds': len(lst)}
        for k in metric_keys:
            vals = np.array([d[k] for d in lst], dtype=float)
            agg[cell][k] = {'mean': round(float(vals.mean()), 5),
                            'std': round(float(vals.std()), 5)}

    out = {
        'config': args.config,
        'n_samples': args.n_samples,
        'pair_subset': args.pair_subset,
        'seed': args.seed,
        'image_source': f'{args.hover_h5} + {args.recovery_h5} (50/50)',
        'cells': {c: manifest.get('cells', {}).get(c) for c in by_cell},
        'per_run': per_run,
        'by_cell': agg,
    }
    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    # ---- summary table + decisive comparisons ----
    print("\n" + "=" * 92)
    print("Feature-collapse by cell  (vis_pooled, D=256, mean +- std over seeds)")
    print("=" * 92)
    hdr = (f"{'Cell':5s} {'seeds':5s} {'eff_rank':>13s} {'PR':>11s} "
           f"{'feat_norm':>12s} {'pair_dist':>13s} {'disp_loss':>13s} {'cos':>9s}")
    print(hdr)
    for cell in ['D0E0', 'D1E0', 'D0E1', 'D1E1']:
        if cell not in agg:
            continue
        a = agg[cell]
        print(f"{cell:5s} {a['n_seeds']:>5d} "
              f"{a['effective_rank']['mean']:>6.2f}±{a['effective_rank']['std']:<5.2f} "
              f"{a['participation_ratio']['mean']:>5.2f}±{a['participation_ratio']['std']:<4.2f} "
              f"{a['mean_feat_norm']['mean']:>7.2f}±{a['mean_feat_norm']['std']:<4.2f} "
              f"{a['mean_pairwise_dist']['mean']:>7.2f}±{a['mean_pairwise_dist']['std']:<5.2f} "
              f"{a['dispersive_loss']['mean']:>7.3f}±{a['dispersive_loss']['std']:<4.3f} "
              f"{a['mean_pairwise_cos']['mean']:>6.4f}")

    def cmp(c1, c0, key):
        if c1 in agg and c0 in agg:
            return agg[c1][key]['mean'] - agg[c0][key]['mean']
        return float('nan')

    print("\nDecisive comparisons:")
    print(f"  D1E0 vs D0E0 (frozen, must be ~0): "
          f"Δeff_rank={cmp('D1E0','D0E0','effective_rank'):+.4f}  "
          f"Δdisp_loss={cmp('D1E0','D0E0','dispersive_loss'):+.6f}  "
          f"Δpair_dist={cmp('D1E0','D0E0','mean_pairwise_dist'):+.6f}")
    print(f"  D1E1 vs D0E1 (Disp's only chance): "
          f"Δeff_rank={cmp('D1E1','D0E1','effective_rank'):+.4f}  "
          f"Δdisp_loss={cmp('D1E1','D0E1','dispersive_loss'):+.6f}  "
          f"Δpair_dist={cmp('D1E1','D0E1','mean_pairwise_dist'):+.6f}")
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()

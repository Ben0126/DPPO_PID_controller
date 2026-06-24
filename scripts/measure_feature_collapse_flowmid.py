"""
Phase 3a (faithful-placement addendum) — measure feature geometry at the FAITHFUL
Dispersive placement: the `flow_net` mid-block, exactly where the official-code
InfoNCE-L2 term acts (models/flow_policy_v5._dispersive_loss_infonce on
flow_net(..., return_mid=True)).

Companion to `measure_feature_collapse.py`, which measures the LEGACY off-path
`vis_pooled` placement (§6.1 of the negative-result paper). That probe could not
say whether the *faithful* mid-features also game the objective / decouple from
survival, because they were never measured. This script closes that gap.

Unlike vis_pooled (a pure function of the image), the flow_net mid-features depend
on (image+IMU -> global_cond, the noised action x_t, and the timestep t). To make
the geometry comparable and paired across all 12 p2f checkpoints, we fix ONE
seeded batch of (images, imu, actions, task_cond) and ONE seeded draw of (t, eps),
identical for every checkpoint, and extract
    mid = flow_net( x_t, t, global_cond, return_mid=True ).flatten(1)
exactly as the training-time dispersive term saw it (t ~ U(0,1) per sample,
x_t = (1-t) a + t eps, eps ~ N(0,1)).

Metrics (same spectrum as the vis_pooled probe, plus the faithful objective):
  effective_rank / n_eff_dims_99 / mean_feat_norm / mean_pairwise_cos
  disp_infonce  = log E_{i,j}[exp(-D/tau)],  D = ||zi-zj||^2 / d    (the term D1* MINIMISES)
  disp_logdist  = -mean_{i!=j} log(||zi-zj||+eps)                    (legacy, for comparability)

Decisive comparison: D1E1 vs D0E1 on disp_infonce (did the faithful term act?) and
whether any drop comes from norm inflation (mean_feat_norm) and a WORSE intrinsic
rank — the "objective gaming" signature, or is genuinely benign.

Usage:
  dppo/Scripts/python.exe -m scripts.measure_feature_collapse_flowmid \
      --manifest evaluation_results/p2f_ablation_manifest.json \
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
# Fixed, seeded (images, imu, actions, task) batch — identical for every model.
# Mirrors scripts.train_flow_v5.FlowDatasetV5 windowing so the inputs are the
# same distribution the dispersive term was trained on.
# ---------------------------------------------------------------------------

def build_input_batch(hover_h5, recovery_h5, n_samples, T_obs, T_pred, seed=12345):
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    img_buf, imu_buf, act_buf, task_buf = [], [], [], []

    for h5_path, n_take, is_rec in [(hover_h5, half, 0.0),
                                    (recovery_h5, n_samples - half, 1.0)]:
        task_label = (np.array([0.0, 1.0], dtype=np.float32) if is_rec
                      else np.array([1.0, 0.0], dtype=np.float32))
        with h5py.File(h5_path, 'r') as f:
            n_ep = int(f.attrs['n_episodes'])
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
                acts = f[key]['actions'][:]          # (T, action_dim)
                imus = f[key]['imu_data'][:]         # (T, 6)
                T = acts.shape[0]
                hi = T - T_pred                      # need T_pred future actions
                if hi <= T_obs - 1:
                    continue
                starts = rng.integers(T_obs - 1, hi, size=min(per_ep, n_take - got))
                for s in starts:
                    frames = imgs[s - T_obs + 1: s + 1]                 # (T_obs, 3, 64, 64)
                    img_buf.append(np.concatenate(frames, axis=0))      # (T_obs*3, 64, 64)
                    imu_buf.append(imus[s])                             # (6,)
                    act_buf.append(acts[s + 1: s + 1 + T_pred].T)        # (action_dim, T_pred)
                    task_buf.append(task_label)
                    got += 1
                    if got >= n_take:
                        break

    images = np.stack(img_buf).astype(np.uint8)
    imu    = np.stack(imu_buf).astype(np.float32)
    actions = np.stack(act_buf).astype(np.float32)
    task   = np.stack(task_buf).astype(np.float32)
    return images, imu, actions, task


# ---------------------------------------------------------------------------
# Geometry metrics on a (N, D) feature matrix — memory-safe (cdist, no (M,M,D)).
# ---------------------------------------------------------------------------

def collapse_metrics(X: torch.Tensor, tau=0.5, pair_subset=2048, eps=1e-6):
    X = X.float()
    N, D = X.shape

    # --- covariance spectrum (centered) ---
    Xc = X - X.mean(dim=0, keepdim=True)
    cov = (Xc.T @ Xc) / (N - 1)                       # (D, D)
    eigv = torch.linalg.eigvalsh(cov)
    eigv = torch.flip(torch.clamp(eigv, min=0.0), dims=[0])
    total = eigv.sum()
    p = eigv / (total + eps)
    nz = p[p > 0]
    eff_rank = torch.exp(-(nz * torch.log(nz)).sum()).item()
    participation = (eigv.sum() ** 2 / (eigv.pow(2).sum() + eps)).item()
    cum = torch.cumsum(p, dim=0)
    n_eff_99 = int((cum < 0.99).sum().item()) + 1
    top2 = (p[0] + p[1]).item() if D >= 2 else p[0].item()
    mean_feat_norm = X.norm(dim=-1).mean().item()

    # --- pairwise geometry on a subset (cdist => no (M,M,D) tensor) ---
    M = min(pair_subset, N)
    Xs = X[:M]
    dist = torch.cdist(Xs, Xs, p=2)                   # (M, M)
    off = ~torch.eye(M, dtype=torch.bool, device=X.device)
    n_pairs = M * (M - 1)
    mean_dist = dist[off].mean().item()
    disp_logdist = (-torch.log(dist[off] + eps)).mean().item()

    Xn = torch.nn.functional.normalize(Xs, dim=-1)
    cos = Xn @ Xn.T
    mean_cos = cos[off].mean().item()

    # --- faithful InfoNCE-L2 objective (the quantity D1* minimises), full BxB
    #     incl. diagonal, matching flow_policy_v5._dispersive_loss_infonce ---
    Dnorm = (torch.cdist(Xs, Xs, p=2) ** 2) / D
    disp_infonce = torch.log(torch.exp(-Dnorm / tau).mean()).item()

    return {
        'effective_rank': round(eff_rank, 3),
        'rank_ratio': round(eff_rank / D, 5),
        'participation_ratio': round(participation, 3),
        'n_eff_dims_99': n_eff_99,
        'top2_var_ratio': round(top2, 4),
        'mean_feat_norm': round(mean_feat_norm, 4),
        'mean_pairwise_dist': round(mean_dist, 5),
        'mean_pairwise_cos': round(mean_cos, 5),
        'disp_infonce': round(disp_infonce, 5),
        'disp_logdist': round(disp_logdist, 5),
        'feature_dim': D,
        'n_samples': N,
    }


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
def extract_flow_mid(model, images_u8, imu, actions, task, t, eps, device, batch=256):
    outs = []
    for i in range(0, len(images_u8), batch):
        img = torch.from_numpy(images_u8[i:i + batch]).to(device).float()
        im  = torch.from_numpy(imu[i:i + batch]).to(device).float()
        act = torch.from_numpy(actions[i:i + batch]).to(device).float()
        tc  = torch.from_numpy(task[i:i + batch]).to(device).float()
        tt  = t[i:i + batch].to(device)
        ep  = eps[i:i + batch].to(device)
        global_cond = model._encode(img, im, task_cond=tc)
        te = tt[:, None, None]
        x_t = (1.0 - te) * act + te * ep
        _, mid = model.flow_net(x_t, model._t_to_int(tt), global_cond, return_mid=True)
        outs.append(mid.flatten(1).cpu())
    return torch.cat(outs, dim=0)


def main():
    ap = argparse.ArgumentParser(description='Faithful flow_mid feature geometry across p2f cells')
    ap.add_argument('--manifest', default='evaluation_results/p2f_ablation_manifest.json')
    ap.add_argument('--config', default='configs/flow_policy_v5.yaml')
    ap.add_argument('--hover-h5', default='data/expert_demos_v4.h5')
    ap.add_argument('--recovery-h5', default='data/expert_demos_v4_recovery.h5')
    ap.add_argument('--n-samples', type=int, default=4000)
    ap.add_argument('--pair-subset', type=int, default=2048)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--out', default='evaluation_results/p2f_feature_collapse_flowmid.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(ROOT, args.config), 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    T_obs = cfg['vision']['T_obs']; T_pred = cfg['action']['T_pred']
    action_dim = cfg['action']['action_dim']

    with open(os.path.join(ROOT, args.manifest), 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    print(f"Building fixed (img,imu,act,task) batch: n={args.n_samples}, seed={args.seed} ...")
    images, imu, actions, task = build_input_batch(
        os.path.join(ROOT, args.hover_h5), os.path.join(ROOT, args.recovery_h5),
        args.n_samples, T_obs, T_pred, seed=args.seed)
    N = len(images)
    print(f"  images={images.shape} imu={imu.shape} actions={actions.shape} task={task.shape}")

    # fixed (t, eps), identical for every checkpoint (paired)
    g = torch.Generator().manual_seed(args.seed)
    t = torch.rand(N, generator=g)
    eps = torch.randn(N, action_dim, T_pred, generator=g)

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
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        model.eval()
        feats = extract_flow_mid(model, images, imu, actions, task, t, eps, device).to(device)
        m = collapse_metrics(feats, tau=args.tau, pair_subset=args.pair_subset)
        m['cell'] = cell; m['seed'] = seed
        per_run[tag] = m
        by_cell[cell].append(m)
        print(f"  {tag:14s} cell={cell} seed={seed} | d={m['feature_dim']} "
              f"eff_rank={m['effective_rank']:.2f} ({m['rank_ratio']*100:.2f}%) "
              f"feat_norm={m['mean_feat_norm']:.2f} "
              f"disp_infonce={m['disp_infonce']:.4f} cos={m['mean_pairwise_cos']:.4f}")

    metric_keys = ['effective_rank', 'rank_ratio', 'participation_ratio', 'n_eff_dims_99',
                   'top2_var_ratio', 'mean_feat_norm', 'mean_pairwise_dist',
                   'mean_pairwise_cos', 'disp_infonce', 'disp_logdist']
    agg = {}
    for cell, lst in by_cell.items():
        agg[cell] = {'n_seeds': len(lst)}
        for k in metric_keys:
            vals = np.array([d[k] for d in lst], dtype=float)
            agg[cell][k] = {'mean': round(float(vals.mean()), 5),
                            'std': round(float(vals.std()), 5)}

    out = {
        'placement': 'flow_net mid-block (faithful, return_mid=True, InfoNCE-L2 tau=%.2f)' % args.tau,
        'config': args.config, 'n_samples': args.n_samples, 'pair_subset': args.pair_subset,
        'tau': args.tau, 'seed': args.seed,
        'image_source': f'{args.hover_h5} + {args.recovery_h5} (50/50)',
        'per_run': per_run, 'by_cell': agg,
    }
    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 96)
    print("Faithful flow_mid feature geometry by cell  (mean +- std over seeds)")
    print("=" * 96)
    print(f"{'Cell':5s} {'seeds':5s} {'eff_rank':>13s} {'n99':>6s} {'feat_norm':>14s} "
          f"{'disp_infonce':>15s} {'cos':>9s}")
    for cell in ['D0E0', 'D1E0', 'D0E1', 'D1E1']:
        if cell not in agg:
            continue
        a = agg[cell]
        print(f"{cell:5s} {a['n_seeds']:>5d} "
              f"{a['effective_rank']['mean']:>7.2f}±{a['effective_rank']['std']:<5.2f} "
              f"{a['n_eff_dims_99']['mean']:>5.0f} "
              f"{a['mean_feat_norm']['mean']:>8.2f}±{a['mean_feat_norm']['std']:<5.2f} "
              f"{a['disp_infonce']['mean']:>8.4f}±{a['disp_infonce']['std']:<5.4f} "
              f"{a['mean_pairwise_cos']['mean']:>6.4f}")

    def cmp(c1, c0, key):
        if c1 in agg and c0 in agg:
            return agg[c1][key]['mean'] - agg[c0][key]['mean']
        return float('nan')

    print("\nDecisive comparisons (faithful placement):")
    for c1, c0, label in [('D1E1', 'D0E1', 'trainable-encoder'),
                          ('D1E0', 'D0E0', 'frozen-encoder')]:
        print(f"  D1 vs D0 [{label}]: "
              f"Δeff_rank={cmp(c1, c0, 'effective_rank'):+.3f}  "
              f"Δfeat_norm={cmp(c1, c0, 'mean_feat_norm'):+.3f}  "
              f"Δdisp_infonce={cmp(c1, c0, 'disp_infonce'):+.5f}  "
              f"Δcos={cmp(c1, c0, 'mean_pairwise_cos'):+.5f}")
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()

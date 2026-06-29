"""
Gate A (RESEARCH_PLAN_v8 Phase 0) — Open-loop fitting interference vs closed-loop
compounding error.

The v7 T×O 2×2 localized the precision floor to a *Robustness–Precision Capacity
Conflict*: adding far-range recovery labels (T0O1 → T1O1) DEGRADES hover precision
(cond-IAE 2.48 → 2.93 m) while buying survival. "Capacity" is the leading-but-UNTESTED
explanation. Before spending GPU-days scaling the model, this gate asks the cheap,
decisive question on the EXISTING 12 p2to checkpoints (inference only):

    Does the generalist (T1O1, trained on hover + far) fit the SAME hover near-target
    distribution WORSE, open-loop, than the hover-specialist (T0O1, hover-only)?

  * generalist hover-fit  ~=  specialist  (Δ within seed pooled-std)
        -> the model HAS capacity to represent precise hover open-loop; the floor is a
           CLOSED-LOOP / partial-observability property -> scaling predicted INERT
           -> Stage 1 = minimal scaling confirm + reframe toward recurrence/memory.
  * generalist hover-fit  >>  specialist  (Δ > pooled-std, and gen fits FAR better)
        -> FITTING INTERFERENCE is real -> capacity / parameter-sharing implicated
           -> Stage 1 = capacity scaling ladder.

Measured paired across models on ONE fixed, seeded batch (same (t, eps) draw for the
flow-matching loss, same _fixed_x1 inference noise for the open-loop action error),
mirroring the FlowDatasetV5 windowing the policies trained on. We report fit on the
hover near-target batch (the precision-relevant distribution) AND on the far batch (to
confirm the tradeoff direction: the generalist should fit far BETTER). The O1 (perspective)
pair is decisive — it is the render on which the floor is measured; the O0 (crosshair)
pair is replication.

Usage:
  dppo/Scripts/python.exe -m scripts.measure_capacity_fit \
      --manifest evaluation_results/p2to_ablation_manifest.json \
      --n-samples 3000
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

# Windows consoles / piped output default to cp950, which cannot encode some of the
# symbols below (e.g. U+2212 minus). Force UTF-8 so the summary never crashes mid-print.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from scripts.evaluate_hierarchical import build_policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Fixed, seeded windowed batch from ONE h5 (hover or far), mirroring
# FlowDatasetV5 windowing (T_obs stacked frames, T_pred future CTBR actions).
# ---------------------------------------------------------------------------

def build_fixed_batch(h5_path, n_samples, T_obs, T_pred, seed=12345):
    rng = np.random.default_rng(seed)
    img_buf, imu_buf, act_buf = [], [], []
    with h5py.File(h5_path, 'r') as f:
        n_ep = int(f.attrs['n_episodes'])
        ep_pool = rng.permutation(n_ep)
        per_ep = max(1, n_samples // min(n_ep, 200))
        got = 0
        for ep_idx in ep_pool:
            if got >= n_samples:
                break
            key = f'episode_{ep_idx}'
            if key not in f:
                continue
            imgs = f[key]['images'][:]            # (T, 3, 64, 64) uint8
            acts = f[key]['actions'][:]            # (T, action_dim)
            imus = f[key]['imu_data'][:]           # (T, 6)
            T = acts.shape[0]
            hi = T - T_pred
            if hi <= T_obs - 1:
                continue
            starts = rng.integers(T_obs - 1, hi, size=min(per_ep, n_samples - got))
            for s in starts:
                frames = imgs[s - T_obs + 1: s + 1]                 # (T_obs, 3, 64, 64)
                img_buf.append(np.concatenate(frames, axis=0))      # (T_obs*3, 64, 64)
                imu_buf.append(imus[s])
                act_buf.append(acts[s + 1: s + 1 + T_pred].T)        # (action_dim, T_pred)
                got += 1
                if got >= n_samples:
                    break
    images  = np.stack(img_buf).astype(np.uint8)
    imu     = np.stack(imu_buf).astype(np.float32)
    actions = np.stack(act_buf).astype(np.float32)
    return images, imu, actions


# ---------------------------------------------------------------------------
# Paired open-loop fit on a fixed batch: flow-matching loss (fixed t, eps) and
# open-loop action-prediction error (fixed _fixed_x1, 2-step inference).
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_fit(model, images_u8, imu, actions, task_label, t, eps, x1,
                device, n_steps, batch=256):
    N = len(images_u8)
    tc_row = torch.tensor(task_label, dtype=torch.float32, device=device)
    flow_se = 0.0      # summed squared error for flow loss
    act_se = 0.0       # summed squared error for open-loop action
    act_ae = 0.0       # summed abs error for open-loop action
    n_elem_flow = 0
    n_elem_act = 0
    for i in range(0, N, batch):
        j = min(i + batch, N)
        img = torch.from_numpy(images_u8[i:j]).to(device).float() / 255.0
        im  = torch.from_numpy(imu[i:j]).to(device).float()
        act = torch.from_numpy(actions[i:j]).to(device).float()
        tc  = tc_row.unsqueeze(0).expand(j - i, -1)
        tt  = t[i:j].to(device)
        ep  = eps[i:j].to(device)
        xx1 = x1[i:j].to(device)

        # --- flow-matching loss with the SAME (t, eps) for every model (paired) ---
        global_cond = model._encode(img, im, task_cond=tc)
        te = tt[:, None, None]
        x_t = (1.0 - te) * act + te * ep
        v_target = ep - act
        v_pred = model.flow_net(x_t, model._t_to_int(tt), global_cond)
        flow_se += ((v_pred - v_target) ** 2).sum().item()
        n_elem_flow += v_pred.numel()

        # --- open-loop action prediction (same fixed x1 noise, 2-step inference) ---
        pred = model.predict_action(img, im, n_steps=n_steps,
                                    _fixed_x1=xx1, task_cond=tc)
        act_se += ((pred - act) ** 2).sum().item()
        act_ae += (pred - act).abs().sum().item()
        n_elem_act += pred.numel()

    return {
        'flow_mse': flow_se / n_elem_flow,
        'act_mse':  act_se / n_elem_act,
        'act_l1':   act_ae / n_elem_act,
    }


def pooled_std(s_gen, s_spec):
    return float(np.sqrt((s_gen ** 2 + s_spec ** 2) / 2.0))


def main():
    ap = argparse.ArgumentParser(description='Gate A: open-loop fitting interference vs closed-loop')
    ap.add_argument('--manifest', default='evaluation_results/p2to_ablation_manifest.json')
    ap.add_argument('--config', default='configs/flow_policy_v5.yaml')
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--n-samples', type=int, default=3000)
    ap.add_argument('--n-steps', type=int, default=2,
                    help='open-loop inference steps (match frozen-P0 eval = 2)')
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--out', default='evaluation_results/p2cap_gateA_fit.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(ROOT, args.config), 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    T_obs = cfg['vision']['T_obs']; T_pred = cfg['action']['T_pred']
    action_dim = cfg['action']['action_dim']

    with open(os.path.join(ROOT, args.manifest), 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # Fixed batches per render: hover near-target (precision-relevant) + far (tradeoff dir).
    render_files = {
        'crosshair':   {'hover': 'expert_demos_v7_hover_crosshair.h5',
                        'far':   'expert_demos_v7_far_crosshair.h5'},
        'perspective': {'hover': 'expert_demos_v7_hover_persp.h5',
                        'far':   'expert_demos_v7_far_persp.h5'},
    }
    batches = {}
    for render, files in render_files.items():
        for kind, base in files.items():
            path = os.path.join(ROOT, args.data_dir, base)
            print(f"Building fixed {render}/{kind} batch from {base} (n={args.n_samples}, seed={args.seed}) ...")
            imgs, imu, acts = build_fixed_batch(path, args.n_samples, T_obs, T_pred, seed=args.seed)
            batches[(render, kind)] = (imgs, imu, acts)
            print(f"  -> images={imgs.shape}")

    # Paired noise draws, identical for every checkpoint AND every (render,kind) of the
    # same size N (so flow-loss / action-error are paired across models).
    N = args.n_samples
    g = torch.Generator().manual_seed(args.seed)
    t   = torch.rand(N, generator=g)
    eps = torch.randn(N, action_dim, T_pred, generator=g)
    x1  = torch.randn(N, action_dim, T_pred, generator=g)

    # hover task = [1,0]; far(recovery) task = [0,1]  (matches FlowDatasetV5 labels)
    TASK = {'hover': [1.0, 0.0], 'far': [0.0, 1.0]}

    per_run = {}
    by_cell = defaultdict(list)
    runs = manifest['runs']
    for tag in sorted(runs.keys()):
        info = runs[tag]
        cell, seed, render = info['cell'], info['seed'], info['render']
        ckpt = os.path.join(ROOT, info['ckpt'].lstrip('./'))
        if not os.path.exists(ckpt):
            print(f"  [skip] {tag}: ckpt missing {ckpt}")
            continue
        policy, arch = build_policy(ckpt, cfg, args.n_steps, device)
        n_params = sum(p.numel() for p in policy.parameters())

        rec = {'cell': cell, 'seed': seed, 'render': render,
               'task_dim': arch.get('task_dim', 0), 'n_params': n_params}
        for kind in ('hover', 'far'):
            imgs, imu, acts = batches[(render, kind)]
            m = measure_fit(policy, imgs, imu, acts, TASK[kind], t, eps, x1,
                            device, args.n_steps)
            rec[kind] = m
        per_run[tag] = rec
        by_cell[cell].append(rec)
        print(f"  {tag:14s} cell={cell} seed={seed} render={render} | "
              f"hover act_mse={rec['hover']['act_mse']:.5f} flow_mse={rec['hover']['flow_mse']:.5f} | "
              f"far act_mse={rec['far']['act_mse']:.5f}")

    # ---- aggregate per cell (mean +- std over seeds) ----
    def agg_metric(lst, kind, key):
        vals = np.array([d[kind][key] for d in lst], dtype=float)
        return float(vals.mean()), float(vals.std())

    cell_agg = {}
    for cell, lst in by_cell.items():
        a = {'n_seeds': len(lst), 'render': lst[0]['render']}
        for kind in ('hover', 'far'):
            for key in ('flow_mse', 'act_mse', 'act_l1'):
                mu, sd = agg_metric(lst, kind, key)
                a[f'{kind}_{key}_mean'] = mu
                a[f'{kind}_{key}_std'] = sd
        cell_agg[cell] = a

    # ---- decisive verdict per render pair (generalist T1 vs specialist T0) ----
    pairs = [('perspective', 'T0O1', 'T1O1'), ('crosshair', 'T0O0', 'T1O0')]
    verdicts = {}
    for render, spec_cell, gen_cell in pairs:
        if spec_cell not in cell_agg or gen_cell not in cell_agg:
            continue
        spec, gen = cell_agg[spec_cell], cell_agg[gen_cell]
        v = {}
        for key in ('act_mse', 'flow_mse', 'act_l1'):
            d_hover = gen[f'hover_{key}_mean'] - spec[f'hover_{key}_mean']
            ps = pooled_std(gen[f'hover_{key}_std'], spec[f'hover_{key}_std'])
            rel = d_hover / spec[f'hover_{key}_mean'] if spec[f'hover_{key}_mean'] else float('nan')
            d_far = gen[f'far_{key}_mean'] - spec[f'far_{key}_mean']
            v[key] = {'d_hover': d_hover, 'pooled_std': ps, 'rel_hover': rel, 'd_far': d_far}
        # decision on the interpretable open-loop action MSE.
        #   d_hover = generalist(T1) - specialist(T0) on the hover near-target batch.
        #   POSITIVE d_hover => the generalist fits precise hover WORSE (interference).
        am = v['act_mse']
        sig = am['d_hover'] > am['pooled_std']            # resolvable above seed noise
        if sig and am['rel_hover'] > 0.05:
            branch = 'FITTING_INTERFERENCE'    # strong: > pooled_std AND > 5% relative
        elif sig:
            branch = 'WEAK_INTERFERENCE'       # real (> pooled_std) but small (<= 5% relative)
        elif abs(am['d_hover']) <= am['pooled_std']:
            branch = 'FIT_PRESERVED'           # within seed noise -> closed-loop story
        else:
            branch = 'GENERALIST_FITS_BETTER'  # generalist fits hover BETTER (d_hover < -std)
        v['branch'] = branch
        v['generalist_fits_far_better'] = bool(am['d_far'] < 0)
        verdicts[render] = v

    out = {
        'gate': 'A — open-loop fitting interference vs closed-loop compounding',
        'n_samples': args.n_samples, 'n_steps': args.n_steps, 'seed': args.seed,
        'config': args.config, 'per_run': per_run, 'cell_agg': cell_agg,
        'verdicts': verdicts,
    }
    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    # ---- console summary ----
    print("\n" + "=" * 96)
    print("Gate A - open-loop fit by cell  (mean +/- std over seeds)")
    print("=" * 96)
    print(f"{'Cell':5s} {'render':12s} {'hover_act_mse':>16s} {'hover_flow_mse':>16s} {'far_act_mse':>14s}")
    for cell in ['T0O0', 'T1O0', 'T0O1', 'T1O1']:
        if cell not in cell_agg:
            continue
        a = cell_agg[cell]
        print(f"{cell:5s} {a['render']:12s} "
              f"{a['hover_act_mse_mean']:>9.5f}+/-{a['hover_act_mse_std']:<7.5f} "
              f"{a['hover_flow_mse_mean']:>9.5f}+/-{a['hover_flow_mse_std']:<7.5f} "
              f"{a['far_act_mse_mean']:>8.5f}+/-{a['far_act_mse_std']:<7.5f}")

    print("\nDecisive comparisons (generalist T1 - specialist T0 on the HOVER batch):")
    for render in ('perspective', 'crosshair'):
        if render not in verdicts:
            continue
        v = verdicts[render]; am = v['act_mse']
        tag = 'DECISIVE' if render == 'perspective' else 'replication'
        print(f"  [{render:11s} {tag:11s}] act_mse d_hover={am['d_hover']:+.6f} "
              f"(pooled_std {am['pooled_std']:.6f}, rel {am['rel_hover']*100:+.1f}%)  "
              f"d_far={am['d_far']:+.6f}  far_better={v['generalist_fits_far_better']}  "
              f"-> {v['branch']}")

    if 'perspective' in verdicts:
        b = verdicts['perspective']['branch']
        am = verdicts['perspective']['act_mse']
        print("\n" + "-" * 96)
        if b in ('FITTING_INTERFERENCE', 'WEAK_INTERFERENCE'):
            strength = 'STRONG' if b == 'FITTING_INTERFERENCE' else 'WEAK'
            print(f"BRANCH: open-loop fitting interference is REAL but {strength} "
                  f"(d_hover {am['rel_hover']*100:+.1f}% rel, {am['d_hover']/am['pooled_std']:.1f}x seed std).")
            print("        Capacity/sharing IS implicated open-loop -> Phase 1 = capacity SCALING")
            print("        LADDER on the T1O1 recipe. NOTE the magnitude gap: the open-loop hover")
            print("        cost is tiny in absolute terms while the closed-loop cond-IAE regression")
            print("        is large -> the decisive test is whether scaling erases BOTH. If scaling")
            print("        removes the open-loop interference but the closed-loop floor HOLDS, the")
            print("        floor is closed-loop, not capacity (strongest negative result).")
        else:
            print(f"BRANCH: hover open-loop fit PRESERVED ({b}) -> the floor is CLOSED-LOOP /")
            print("        partial-observability, NOT open-loop fitting capacity.")
            print("        Phase 1 = MINIMAL scaling confirm (predict floor holds) + reframe Stage 2")
            print("        toward recurrence/memory.")
        print("-" * 96)
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()

"""
P2CAP capacity-ladder sweep driver — RESEARCH_PLAN_v8 Phase 1.

Tests H_v8: the cond-IAE ~2.8 m precision floor (and the negative coverage x precision
interaction) is caused by insufficient usable capacity / parameter sharing in the
13.5 M-param policy. Holds the T1O1 frontier recipe fixed (the cell where the conflict
bites) and varies ONLY flow_net capacity via --down-dims, 3 seeds/rung, frozen-P0
(perspective eval render):

  rung  down_dims        flow_net   total   note
  S     (256,512)        12.1 M     13.5 M  baseline == existing p2to_T1O1 (REUSED, no retrain)
  M     (256,512,768)    33.6 M     35.0 M  deepen one level (first 2 levels H4-transfer-friendly)
  XL    (512,1024)       42.6 M     44.0 M  widen (flow_net from scratch; watch val_flow)

Fixed T1O1 recipe for every rung (identical to scripts.run_p2to_ablation cell T1O1):
  * --hover-h5 expert_demos_v7_hover_persp.h5   --hover-episodes 500
  * --recovery-h5 expert_demos_v7_far_persp.h5  --recovery-episodes 500
  * --lambda-disp 0.0   (Dispersive OFF)        --transfer-from-h4 <H4>  (E2E, no freeze)
  * task-cond (default)                         eval render = perspective

S is NOT retrained: the (256,512) rung is exactly the p2to T1O1 checkpoint, so the
manifest points S at checkpoints/flow_policy_v5/p2to_T1O1_s{seed}/best_model.pt with
status 'reused'. Only M and XL launch training (6 runs).

Runs SEQUENTIALLY (Known Failure Mode #7). Writes a manifest mapping (rung, seed) ->
{ckpt, down_dims, render, hover_h5, recovery_h5} for scripts.evaluate_p2cap_ablation.

Usage (launch in background with run_in_background=true):
  dppo/Scripts/python.exe -m scripts.run_p2cap_ablation \
      --h4-ckpt checkpoints/flow_policy_v4/20260514_175219/best_model.pt \
      --rungs M XL --seeds 0 1 2

  # smoke test wiring (5 epochs, tiny pool, 1 seed, M only):
  dppo/Scripts/python.exe -m scripts.run_p2cap_ablation --quick --rungs M --seeds 0
  # print commands without launching:
  dppo/Scripts/python.exe -m scripts.run_p2cap_ablation --dry-run
"""
import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# rung -> down_dims override string (None == S baseline, reuse existing p2to_T1O1)
RUNGS = {
    'S':  None,
    'M':  '256,512,768',
    'XL': '512,1024',
}
# The T1O1 frontier recipe (perspective observation, far-recovery mix-in).
HOVER_H5 = 'expert_demos_v7_hover_persp.h5'
REC_H5   = 'expert_demos_v7_far_persp.h5'
RENDER   = 'perspective'
S_CKPT_TMPL = './checkpoints/flow_policy_v5/p2to_T1O1_s{seed}/best_model.pt'


def build_cmd(python, rung, seed, args):
    """Build the train_flow_v5 command for one (rung, seed). Returns (tag, cmd, meta)."""
    hover_h5 = os.path.join(args.data_dir, HOVER_H5).replace('\\', '/')
    rec_h5   = os.path.join(args.data_dir, REC_H5).replace('\\', '/')
    tag = f"p2cap_{rung}_s{seed}"

    he = args.quick_episodes if args.quick else args.hover_episodes
    re = args.quick_episodes if args.quick else args.recovery_episodes

    cmd = [python, '-m', 'scripts.train_flow_v5',
           '--config', args.config,
           '--hover-h5', hover_h5,
           '--hover-episodes', str(he),
           '--recovery-h5', rec_h5,
           '--recovery-episodes', str(re),
           '--lambda-disp', '0.0',            # T1O1 frontier recipe: Dispersive OFF
           '--seed', str(seed),
           '--tag', tag]
    if RUNGS[rung]:
        cmd += ['--down-dims', RUNGS[rung]]    # capacity lever (S has no override)
    if args.h4_ckpt:
        cmd += ['--transfer-from-h4', args.h4_ckpt]   # E2E: NO --freeze-vision
    if args.quick:
        cmd += ['--quick']

    meta = {'rung': rung, 'seed': seed, 'render': RENDER, 'down_dims': RUNGS[rung],
            'hover_h5': hover_h5, 'recovery_h5': rec_h5}
    return tag, cmd, meta


def ckpt_path_for(save_path, tag):
    return os.path.join(save_path, tag, 'best_model.pt').replace('\\', '/')


def load_manifest(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'runs': {}}


def save_manifest(path, manifest):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='P2CAP capacity-ladder sweep (RESEARCH_PLAN_v8 Phase 1)')
    parser.add_argument('--rungs', nargs='+', default=['S', 'M', 'XL'],
                        choices=list(RUNGS.keys()),
                        help='Capacity rungs to include (S is reused from p2to_T1O1, not retrained)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--config', default='configs/flow_policy_v5.yaml')
    parser.add_argument('--h4-ckpt', default='checkpoints/flow_policy_v4/20260514_175219/best_model.pt')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--hover-episodes', type=int, default=500)
    parser.add_argument('--recovery-episodes', type=int, default=500)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--quick-episodes', type=int, default=20)
    parser.add_argument('--skip-done', action='store_true', default=True)
    parser.add_argument('--no-skip-done', dest='skip_done', action='store_false')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--manifest', default='evaluation_results/p2cap_ablation_manifest.json')
    args = parser.parse_args()

    with open(os.path.join(ROOT, args.config), 'r', encoding='utf-8') as f:
        save_path = yaml.safe_load(f)['logging']['save_path']

    manifest = load_manifest(os.path.join(ROOT, args.manifest))
    manifest.setdefault('runs', {})
    manifest['rungs'] = {r: {'down_dims': RUNGS[r]} for r in RUNGS}
    manifest['recipe'] = {'cell': 'T1O1', 'lambda_disp': 0.0, 'freeze_vision': False,
                          'transfer_from_h4': args.h4_ckpt, 'task_cond': True,
                          'hover_h5': HOVER_H5, 'recovery_h5': REC_H5, 'render': RENDER}
    manifest['config'] = args.config
    manifest['quick'] = args.quick

    # seeds outer, rungs inner -> a full ladder completes per seed as early as possible
    jobs = [(seed, rung) for seed in args.seeds for rung in args.rungs]
    print(f"P2CAP sweep: {len(jobs)} (rung,seed) jobs  rungs={args.rungs}  seeds={args.seeds}  "
          f"quick={args.quick}  skip_done={args.skip_done}")

    for i, (seed, rung) in enumerate(jobs, 1):
        tag, cmd, meta = build_cmd(args.python, rung, seed, args)

        # S rung == existing p2to_T1O1 checkpoint: reuse, never retrain.
        if rung == 'S':
            s_ckpt = S_CKPT_TMPL.format(seed=seed)
            exists = os.path.exists(os.path.join(ROOT, s_ckpt.lstrip('./')))
            print(f"\n[{i}/{len(jobs)}] rung=S seed={seed} -> REUSE {s_ckpt} (exists={exists})")
            manifest['runs'][tag] = {**meta, 'ckpt': s_ckpt,
                                     'status': 'reused' if exists else 'missing',
                                     'returncode': 0}
            save_manifest(os.path.join(ROOT, args.manifest), manifest)
            continue

        ckpt = ckpt_path_for(save_path, tag)
        print(f"\n{'='*84}\n[{i}/{len(jobs)}] rung={rung} seed={seed} tag={tag} "
              f"down_dims={RUNGS[rung]} render={RENDER}\n  {' '.join(cmd)}\n{'='*84}")

        if args.skip_done and os.path.exists(os.path.join(ROOT, ckpt)):
            print(f"  SKIP: {ckpt} already exists")
            manifest['runs'][tag] = {**meta, 'ckpt': ckpt,
                                     'status': 'skipped_existing', 'returncode': 0}
            save_manifest(os.path.join(ROOT, args.manifest), manifest)
            continue

        if args.dry_run:
            manifest['runs'][tag] = {**meta, 'ckpt': ckpt, 'status': 'dry_run'}
            continue

        t0 = time.time()
        manifest['runs'][tag] = {**meta, 'ckpt': ckpt, 'status': 'running',
                                 'started': datetime.now().isoformat()}
        save_manifest(os.path.join(ROOT, args.manifest), manifest)

        ret = subprocess.run(cmd, cwd=ROOT).returncode

        manifest['runs'][tag].update({
            'status': 'done' if ret == 0 else 'failed',
            'returncode': ret,
            'seconds': round(time.time() - t0, 1),
            'ckpt_exists': os.path.exists(os.path.join(ROOT, ckpt)),
            'finished': datetime.now().isoformat(),
        })
        save_manifest(os.path.join(ROOT, args.manifest), manifest)
        print(f"  -> {manifest['runs'][tag]['status']} in "
              f"{manifest['runs'][tag]['seconds']}s  ckpt_exists={manifest['runs'][tag]['ckpt_exists']}")

    if args.dry_run:
        save_manifest(os.path.join(ROOT, args.manifest), manifest)
    print(f"\nManifest: {args.manifest}")
    print("Next: dppo/Scripts/python.exe -m scripts.evaluate_p2cap_ablation "
          f"--manifest {args.manifest} "
          "--oracle-ckpt checkpoints/ppo_expert_v4/20260419_142245/best_model.pt "
          "--oracle-norm checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz")


if __name__ == '__main__':
    main()

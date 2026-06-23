"""
P2 core ablation sweep driver — Dispersive {OFF,ON} x E2E {OFF,ON}, >=3 seeds.

Primary axis = Tier1% / survival under the P0 frozen protocol (NOT the composite
score, which the P0 re-plan proved is artifact-prone). Each cell is one call to
``scripts.train_flow_v5`` with two flags varied:

  Dispersive : --lambda-disp 0.05  (ON)   vs  --lambda-disp 0.0  (OFF)
  E2E        : vision encoder trainable    vs  --freeze-vision   (OFF)

Mechanistic note: the dispersive loss acts on the (vision) ``vis_pooled`` features,
so with a frozen encoder it is a no-op -> D1E0 is expected to ~= D0E0. That collapse
is itself the result (Dispersive needs a trainable encoder to act); the full 2x2
makes the Dispersive x E2E interaction explicit. See RESEARCH_PLAN_v6.md Phase 2.

All cells share identical init (--transfer-from-h4), data (hover+recovery mix), and
epochs/LR/batch (configs/flow_policy_v5.yaml). Only the two factors vary.

Runs SEQUENTIALLY (one training at a time) to avoid GPU contention (Known Failure
Mode #7). Writes a manifest mapping (cell, seed) -> checkpoint so
``scripts.evaluate_p2_ablation`` can aggregate by cell with mean+-std across seeds.

Usage (launch in background with run_in_background=true):
  dppo/Scripts/python.exe -m scripts.run_p2_ablation \
      --h4-ckpt checkpoints/flow_policy_v4/20260514_175219/best_model.pt \
      --recovery-h5 data/expert_demos_v4_recovery.h5 \
      --recovery-episodes 500 --hover-episodes 500 \
      --cells D0E0 D0E1 D1E0 D1E1 --seeds 0 1 2

  # one cell only (e.g. the headline D1E1, seed 0):
  dppo/Scripts/python.exe -m scripts.run_p2_ablation --cells D1E1 --seeds 0 \
      --h4-ckpt checkpoints/flow_policy_v4/20260514_175219/best_model.pt \
      --recovery-h5 data/expert_demos_v4_recovery.h5

  # print the commands without launching:
  dppo/Scripts/python.exe -m scripts.run_p2_ablation --dry-run
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

# cell -> (lambda_disp, freeze_vision)
CELLS = {
    'D0E0': (0.0,  True),    # Dispersive OFF, E2E OFF (frozen encoder)
    'D0E1': (0.0,  False),   # Dispersive OFF, E2E ON  (encoder trainable)
    'D1E0': (0.05, True),    # Dispersive ON,  E2E OFF  (-> expected == D0E0)
    'D1E1': (0.05, False),   # Dispersive ON,  E2E ON   (the decisive cell)
}


def build_cmd(python, cell, seed, args):
    """Build the train_flow_v5 command for one (cell, seed). Returns (tag, cmd)."""
    lam, freeze = CELLS[cell]
    prefix = 'p2f' if args.faithful else 'p2'
    tag = f"{prefix}_{cell}_s{seed}"
    if args.faithful and lam > 0.0:
        lam = 0.5  # faithful Dispersive weight per [13]/[14] (vs legacy 0.05)
    cmd = [python, '-m', 'scripts.train_flow_v5',
           '--config', args.config,
           '--recovery-h5', args.recovery_h5,
           '--recovery-episodes', str(args.recovery_episodes),
           '--hover-episodes', str(args.hover_episodes),
           '--lambda-disp', str(lam),
           '--seed', str(seed),
           '--tag', tag]
    if args.faithful:
        cmd += ['--dispersive-target', 'flow_mid', '--dispersive-tau', '0.5']
    if args.h4_ckpt:
        cmd += ['--transfer-from-h4', args.h4_ckpt]
    if freeze:
        cmd += ['--freeze-vision']
    if args.quick:
        cmd += ['--quick']
    return tag, cmd


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
    parser = argparse.ArgumentParser(description='P2 Dispersive x E2E ablation sweep')
    parser.add_argument('--cells', nargs='+', default=list(CELLS.keys()),
                        choices=list(CELLS.keys()),
                        help='Which 2x2 cells to run (default: all four)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                        help='Seeds per cell (>=3 for the paper)')
    parser.add_argument('--config', default='configs/flow_policy_v5.yaml')
    parser.add_argument('--h4-ckpt', default='checkpoints/flow_policy_v4/20260514_175219/best_model.pt',
                        help='Shared init for every cell (--transfer-from-h4)')
    parser.add_argument('--recovery-h5', default='data/expert_demos_v4_recovery.h5')
    parser.add_argument('--recovery-episodes', type=int, default=500)
    parser.add_argument('--hover-episodes', type=int, default=500)
    parser.add_argument('--quick', action='store_true', help='5-epoch smoke test per cell')
    parser.add_argument('--faithful', action='store_true',
                        help='Faithful Dispersive re-run: ON cells use lambda=0.5 and InfoNCE-L2 '
                             'on the flow_net mid-block (--dispersive-target flow_mid). Tags are '
                             'prefixed p2f_ and a separate manifest is used so legacy P2 runs are '
                             'untouched. NOTE: under this placement flow_net is always trainable, '
                             'so D1E0 != D0E0 (the legacy byte-identical no-op no longer holds).')
    parser.add_argument('--skip-done', action='store_true', default=True,
                        help='Skip a (cell,seed) whose best_model.pt already exists')
    parser.add_argument('--no-skip-done', dest='skip_done', action='store_false')
    parser.add_argument('--dry-run', action='store_true', help='Print commands, do not launch')
    parser.add_argument('--python', default=sys.executable, help='Interpreter for subprocesses')
    parser.add_argument('--manifest', default='evaluation_results/p2_ablation_manifest.json')
    args = parser.parse_args()

    # Faithful re-run writes a separate manifest so the legacy P2 artifact is preserved.
    if args.faithful and args.manifest == 'evaluation_results/p2_ablation_manifest.json':
        args.manifest = 'evaluation_results/p2f_ablation_manifest.json'

    with open(os.path.join(ROOT, args.config), 'r', encoding='utf-8') as f:
        save_path = yaml.safe_load(f)['logging']['save_path']

    manifest = load_manifest(os.path.join(ROOT, args.manifest))
    manifest.setdefault('runs', {})
    manifest['cells'] = {c: {'lambda_disp': CELLS[c][0], 'freeze_vision': CELLS[c][1]}
                         for c in CELLS}
    manifest['config'] = args.config
    manifest['h4_ckpt'] = args.h4_ckpt
    manifest['quick'] = args.quick

    # seeds outer, cells inner -> a full 2x2 completes per seed as early as possible
    jobs = [(seed, cell) for seed in args.seeds for cell in args.cells]
    print(f"P2 sweep: {len(jobs)} runs  cells={args.cells}  seeds={args.seeds}  "
          f"quick={args.quick}  skip_done={args.skip_done}")

    for i, (seed, cell) in enumerate(jobs, 1):
        tag, cmd = build_cmd(args.python, cell, seed, args)
        ckpt = ckpt_path_for(save_path, tag)
        print(f"\n{'='*84}\n[{i}/{len(jobs)}] cell={cell} seed={seed} tag={tag}\n  {' '.join(cmd)}\n{'='*84}")

        if args.skip_done and os.path.exists(os.path.join(ROOT, ckpt)):
            print(f"  SKIP: {ckpt} already exists")
            manifest['runs'][tag] = {'cell': cell, 'seed': seed, 'ckpt': ckpt,
                                     'status': 'skipped_existing', 'returncode': 0}
            save_manifest(os.path.join(ROOT, args.manifest), manifest)
            continue

        if args.dry_run:
            manifest['runs'][tag] = {'cell': cell, 'seed': seed, 'ckpt': ckpt,
                                     'status': 'dry_run'}
            continue

        t0 = time.time()
        manifest['runs'][tag] = {'cell': cell, 'seed': seed, 'ckpt': ckpt,
                                 'status': 'running',
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
    print("Next: dppo/Scripts/python.exe -m scripts.evaluate_p2_ablation "
          f"--manifest {args.manifest} "
          "--oracle-ckpt checkpoints/ppo_expert_v4/20260419_142245/best_model.pt "
          "--oracle-norm checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz")


if __name__ == '__main__':
    main()

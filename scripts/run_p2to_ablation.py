"""
P2TO decisive sweep driver — Teacher {OFF,ON} x Observation {crosshair,perspective}, >=3 seeds.

RESEARCH_PLAN_v7 Phase 4. The pre-registered hypothesis H_v7 is that the
cond-IAE ~2.8 m precision floor only breaks when BOTH:
  (T1) BC has competent 1-3 m recovery labels   (recovery mix-in), AND
  (O1) the observation can encode far range      (perspective target render).
The paper already showed each factor ALONE is inert, so the decisive design is a
Teacher x Observation 2x2, 3 seeds/cell, evaluated under the P0 frozen protocol.

Every cell shares the D0E1 frontier recipe (NOT a Dispersive sweep):
  * --lambda-disp 0.0          (Dispersive OFF)
  * vision encoder E2E         (NO --freeze-vision)
  * --transfer-from-h4 <H4>    (identical warm start)
  * task-cond                  (default; episode_type drives [hover]/[recovery])
  * --hover-episodes 500 --recovery-episodes 500

Only the two factors vary, and they vary by which Phase-3 h5 the data comes from
(the crosshair/perspective observation is rendered INTO the h5 at collection time;
training has no render flag):

  cell  hover pool (--hover-h5)              recovery (--recovery-h5)            eval render
  T0O0  expert_demos_v7_hover_crosshair      (none, hover-only)                  crosshair
  T0O1  expert_demos_v7_hover_persp          (none, hover-only)                  perspective
  T1O0  expert_demos_v7_hover_crosshair      expert_demos_v7_far_crosshair       crosshair
  T1O1  expert_demos_v7_hover_persp          expert_demos_v7_far_persp           perspective

DESIGN NOTE (data-volume asymmetry): T0 = hover-only (500 ep), T1 = hover+far
(1000 ep). This is intrinsic to "add far-range labels". If a reviewer objects, the
fallback is a size-matched near-recovery control (setpoint_offset ~= 0); the default
here uses the Phase-3 hover-only T0 files.

Runs SEQUENTIALLY (one training at a time) to avoid GPU contention (Known Failure
Mode #7). Writes a manifest mapping (cell, seed) -> {ckpt, render, hover_h5,
recovery_h5} so scripts.evaluate_p2to_ablation can aggregate by cell with mean+-std
across seeds AND pass the correct --target-render per cell.

Usage (launch in background with run_in_background=true):
  dppo/Scripts/python.exe -m scripts.run_p2to_ablation \
      --h4-ckpt checkpoints/flow_policy_v4/20260514_175219/best_model.pt \
      --cells T0O0 T0O1 T1O0 T1O1 --seeds 0 1 2

  # smoke test wiring (5 epochs, tiny episode pool, 1 seed):
  dppo/Scripts/python.exe -m scripts.run_p2to_ablation --quick --seeds 0

  # print the commands without launching:
  dppo/Scripts/python.exe -m scripts.run_p2to_ablation --dry-run
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

# cell -> (hover_basename, recovery_basename_or_None, eval_render)
# basenames are joined to --data-dir; recovery None == hover-only (T0).
CELLS = {
    'T0O0': ('expert_demos_v7_hover_crosshair.h5', None,                              'crosshair'),
    'T0O1': ('expert_demos_v7_hover_persp.h5',     None,                              'perspective'),
    'T1O0': ('expert_demos_v7_hover_crosshair.h5', 'expert_demos_v7_far_crosshair.h5', 'crosshair'),
    'T1O1': ('expert_demos_v7_hover_persp.h5',     'expert_demos_v7_far_persp.h5',     'perspective'),
}


def build_cmd(python, cell, seed, args):
    """Build the train_flow_v5 command for one (cell, seed). Returns (tag, cmd, meta)."""
    hover_base, rec_base, render = CELLS[cell]
    hover_h5 = os.path.join(args.data_dir, hover_base).replace('\\', '/')
    rec_h5 = os.path.join(args.data_dir, rec_base).replace('\\', '/') if rec_base else None
    tag = f"p2to_{cell}_s{seed}"

    # Quick smoke: shrink the episode pool so wiring is verified in minutes.
    # (train_flow_v5 --quick caps epochs/batch but --hover-episodes overrides its
    #  _max_episodes, so we must pass small counts here too.)
    he = args.quick_episodes if args.quick else args.hover_episodes
    re = args.quick_episodes if args.quick else args.recovery_episodes

    cmd = [python, '-m', 'scripts.train_flow_v5',
           '--config', args.config,
           '--hover-h5', hover_h5,
           '--hover-episodes', str(he),
           '--lambda-disp', '0.0',          # D0E1 frontier recipe: Dispersive OFF
           '--seed', str(seed),
           '--tag', tag]
    if rec_h5:
        cmd += ['--recovery-h5', rec_h5, '--recovery-episodes', str(re)]
    if args.h4_ckpt:
        cmd += ['--transfer-from-h4', args.h4_ckpt]   # E2E: NO --freeze-vision
    if args.quick:
        cmd += ['--quick']

    meta = {'cell': cell, 'seed': seed, 'render': render,
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
    parser = argparse.ArgumentParser(description='P2TO Teacher x Observation decisive 2x2 sweep')
    parser.add_argument('--cells', nargs='+', default=list(CELLS.keys()),
                        choices=list(CELLS.keys()),
                        help='Which 2x2 cells to run (default: all four)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                        help='Seeds per cell (>=3 for the paper)')
    parser.add_argument('--config', default='configs/flow_policy_v5.yaml')
    parser.add_argument('--h4-ckpt', default='checkpoints/flow_policy_v4/20260514_175219/best_model.pt',
                        help='Shared warm start for every cell (--transfer-from-h4)')
    parser.add_argument('--data-dir', default='data',
                        help='Directory holding the Phase-3 v7 h5 datasets')
    parser.add_argument('--hover-episodes', type=int, default=500)
    parser.add_argument('--recovery-episodes', type=int, default=500)
    parser.add_argument('--quick', action='store_true',
                        help='5-epoch smoke test per cell with a tiny episode pool')
    parser.add_argument('--quick-episodes', type=int, default=20,
                        help='Episodes per pool in --quick mode (keeps the smoke fast)')
    parser.add_argument('--skip-done', action='store_true', default=True,
                        help='Skip a (cell,seed) whose best_model.pt already exists')
    parser.add_argument('--no-skip-done', dest='skip_done', action='store_false')
    parser.add_argument('--dry-run', action='store_true', help='Print commands, do not launch')
    parser.add_argument('--python', default=sys.executable, help='Interpreter for subprocesses')
    parser.add_argument('--manifest', default='evaluation_results/p2to_ablation_manifest.json')
    args = parser.parse_args()

    with open(os.path.join(ROOT, args.config), 'r', encoding='utf-8') as f:
        save_path = yaml.safe_load(f)['logging']['save_path']

    manifest = load_manifest(os.path.join(ROOT, args.manifest))
    manifest.setdefault('runs', {})
    manifest['cells'] = {c: {'hover_h5': CELLS[c][0], 'recovery_h5': CELLS[c][1],
                             'render': CELLS[c][2]} for c in CELLS}
    manifest['recipe'] = {'lambda_disp': 0.0, 'freeze_vision': False,
                          'transfer_from_h4': args.h4_ckpt, 'task_cond': True}
    manifest['config'] = args.config
    manifest['quick'] = args.quick

    # seeds outer, cells inner -> a full 2x2 completes per seed as early as possible
    jobs = [(seed, cell) for seed in args.seeds for cell in args.cells]
    print(f"P2TO sweep: {len(jobs)} runs  cells={args.cells}  seeds={args.seeds}  "
          f"quick={args.quick}  skip_done={args.skip_done}")

    for i, (seed, cell) in enumerate(jobs, 1):
        tag, cmd, meta = build_cmd(args.python, cell, seed, args)
        ckpt = ckpt_path_for(save_path, tag)
        print(f"\n{'='*84}\n[{i}/{len(jobs)}] cell={cell} seed={seed} tag={tag} render={meta['render']}"
              f"\n  {' '.join(cmd)}\n{'='*84}")

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
    print("Next: dppo/Scripts/python.exe -m scripts.evaluate_p2to_ablation "
          f"--manifest {args.manifest} "
          "--oracle-ckpt checkpoints/ppo_expert_v4/20260419_142245/best_model.pt "
          "--oracle-norm checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz")


if __name__ == '__main__':
    main()

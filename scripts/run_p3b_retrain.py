"""
P3b wider-init retrain driver — test whether the OOD coverage gap (measured by
`scripts/measure_ood_coverage.py`) is what caps closed-loop precision.

Holds the D0E1 *recipe* fixed (the clean E2E frontier: Dispersive OFF, encoder
trainable, H4-transfer init, task-conditioned, 80 ep / lr 1e-4 / batch 256) and
varies ONLY the recovery-init coverage via the `--recovery-h5` dataset (collected by
`scripts.collect_data_v4_recovery` with a wider `--pos-range/--tilt-max/--perturb-vel`).

Decision (after `evaluate_p2_ablation`-style cond-IAE eval on the frozen protocol):
  * cond-IAE drops vs the ±1 m-init D0E1 (2.81 m) -> precision was DATA-gated (fixable).
  * cond-IAE unchanged despite coverage      -> precision is INFORMATION-gated
    (64x64 FPV can't resolve fine position at 2-3 m) — a stronger negative result.

Runs SEQUENTIALLY (one training at a time; Known Failure Mode #7), writes a manifest
shaped like the P2 one so `evaluate_p2_ablation` / `evaluate_frozen_p0` can score it.

Usage (launch in background with run_in_background=true):
  dppo/Scripts/python.exe -m scripts.run_p3b_retrain \
      --recovery-h5 data/expert_demos_v4_recovery_wide3.h5 \
      --tag-prefix p3b_wide3 --seeds 0 1 2
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


def build_cmd(python, seed, args):
    """D0E1 recipe (Disp OFF, E2E ON, H4-transfer) on a given recovery-h5."""
    tag = f"{args.tag_prefix}_s{seed}"
    cmd = [python, '-m', 'scripts.train_flow_v5',
           '--config', args.config,
           '--recovery-h5', args.recovery_h5,
           '--recovery-episodes', str(args.recovery_episodes),
           '--hover-episodes', str(args.hover_episodes),
           '--lambda-disp', '0.0',          # Dispersive OFF (it's dead — Phase 2/3a)
           '--seed', str(seed),
           '--tag', tag]
    if args.h4_ckpt:
        cmd += ['--transfer-from-h4', args.h4_ckpt]
    # E2E ON: encoder trainable -> do NOT pass --freeze-vision
    if args.quick:
        cmd += ['--quick']
    return tag, cmd


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
    ap = argparse.ArgumentParser(description='P3b wider-init retrain (D0E1 recipe)')
    ap.add_argument('--recovery-h5', required=True,
                    help='Wider-init recovery dataset from collect_data_v4_recovery')
    ap.add_argument('--tag-prefix', required=True,
                    help='Run-name prefix, e.g. p3b_wide3 -> p3b_wide3_s0/s1/s2')
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--config', default='configs/flow_policy_v5.yaml')
    ap.add_argument('--h4-ckpt', default='checkpoints/flow_policy_v4/20260514_175219/best_model.pt')
    ap.add_argument('--recovery-episodes', type=int, default=500)
    ap.add_argument('--hover-episodes', type=int, default=500)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--skip-done', action='store_true', default=True)
    ap.add_argument('--no-skip-done', dest='skip_done', action='store_false')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--manifest', default=None,
                    help='Default: evaluation_results/<tag_prefix>_manifest.json')
    args = ap.parse_args()

    manifest_path = args.manifest or f'evaluation_results/{args.tag_prefix}_manifest.json'

    with open(os.path.join(ROOT, args.config), 'r', encoding='utf-8') as f:
        save_path = yaml.safe_load(f)['logging']['save_path']

    manifest = load_manifest(os.path.join(ROOT, manifest_path))
    manifest.setdefault('runs', {})
    manifest['recipe'] = {'lambda_disp': 0.0, 'freeze_vision': False,
                          'transfer_from_h4': args.h4_ckpt, 'cell_equiv': 'D0E1'}
    manifest['recovery_h5'] = args.recovery_h5
    manifest['config'] = args.config

    print(f"P3b retrain: recovery={args.recovery_h5}  seeds={args.seeds}  "
          f"recipe=D0E1(DispOFF,E2E ON,H4-transfer)  quick={args.quick}")

    for i, seed in enumerate(args.seeds, 1):
        tag, cmd = build_cmd(args.python, seed, args)
        ckpt = os.path.join(save_path, tag, 'best_model.pt').replace('\\', '/')
        final_ckpt = os.path.join(save_path, tag, 'final_model.pt').replace('\\', '/')
        print(f"\n{'='*84}\n[{i}/{len(args.seeds)}] seed={seed} tag={tag}\n  {' '.join(cmd)}\n{'='*84}")

        # Completion is final_model.pt (NOT best_model.pt — a partial run leaves a
        # stale best_model.pt; lesson from the P2 reboot, see memory).
        if args.skip_done and os.path.exists(os.path.join(ROOT, final_ckpt)):
            print(f"  SKIP: {final_ckpt} already exists (run complete)")
            manifest['runs'][tag] = {'seed': seed, 'ckpt': ckpt,
                                     'status': 'skipped_existing', 'returncode': 0}
            save_manifest(os.path.join(ROOT, manifest_path), manifest)
            continue

        if args.dry_run:
            manifest['runs'][tag] = {'seed': seed, 'ckpt': ckpt, 'status': 'dry_run'}
            continue

        t0 = time.time()
        manifest['runs'][tag] = {'seed': seed, 'ckpt': ckpt, 'status': 'running',
                                 'started': datetime.now().isoformat()}
        save_manifest(os.path.join(ROOT, manifest_path), manifest)

        ret = subprocess.run(cmd, cwd=ROOT).returncode

        manifest['runs'][tag].update({
            'status': 'done' if ret == 0 else 'failed',
            'returncode': ret,
            'seconds': round(time.time() - t0, 1),
            'ckpt_exists': os.path.exists(os.path.join(ROOT, ckpt)),
            'final_exists': os.path.exists(os.path.join(ROOT, final_ckpt)),
            'finished': datetime.now().isoformat(),
        })
        save_manifest(os.path.join(ROOT, manifest_path), manifest)
        print(f"  -> {manifest['runs'][tag]['status']} in {manifest['runs'][tag]['seconds']}s")

    if args.dry_run:
        save_manifest(os.path.join(ROOT, manifest_path), manifest)
    print(f"\nManifest: {manifest_path}")
    print("Next: eval each best_model.pt through the frozen protocol "
          "(scripts.evaluate_frozen_p0 / measure_ood_coverage) and compare cond-IAE "
          "to the +-1m-init D0E1 baseline (2.81 m).")


if __name__ == '__main__':
    main()

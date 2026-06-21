"""
Phase 3b RANGE-CUE positive-control ablation (RESEARCH_PLAN_v6 Phase 3b).

Tests, by INTERVENTION, the claim of docs/experiment_report_image_distance_info.md
(§6.3): precision is limited because the 64x64 FPV cannot encode metric position
error past ~2 m. We inject that missing channel directly into the policy (folded into
the v5 task-cond slot, so the model concats it into global_cond unchanged) and re-eval
cond-IAE under the frozen P0 protocol. If precision improves, the lack of metric range
in the OBSERVATION was causally the precision bottleneck (sensing-gated, upgrades the
information argument to experimental evidence). If it does NOT, precision is gated by
something other than sensing (an even stronger negative result).

Every arm uses the SAME frontier recipe as P2 cell D0E1 (dispersive OFF, encoder E2E
ON, H4 transfer init, 500 hover + 500 recovery), differing ONLY in the injected cue:

  control       : --range-cue none                      (== p2_D0E1, REUSED, not retrained)
  scalar_clean  : --range-cue scalar  --cue-noise 0.0   (||pos_err||, oracle)
  scalar_noised : --range-cue scalar  --cue-noise 0.15  (realisable range sensor)
  pos3d_clean   : --range-cue pos3d   --cue-noise 0.0   (full body-frame pos err)
  pos3d_noised  : --range-cue pos3d   --cue-noise 0.15  (realisable pos sensor)

Scalar isolates the documented missing channel (metric range magnitude); pos3d is the
upper bound (range + direction, ~the state-oracle input). Noised arms bridge to "a real
ToF/baro/optical-expansion sensor with this accuracy would suffice".

Usage:
  # 1-seed smoke pass over the 4 trained arms (control is reused):
  dppo/Scripts/python.exe -m scripts.run_p3b_rangecue --seeds 0
  # paper pass (3 seeds):
  dppo/Scripts/python.exe -m scripts.run_p3b_rangecue --seeds 0 1 2
  # preview only:
  dppo/Scripts/python.exe -m scripts.run_p3b_rangecue --seeds 0 --dry-run
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

# arm -> (range_cue_mode, cue_noise_m). All share the D0E1 frontier recipe.
ARMS = {
    'control':       ('none',   0.0),
    'scalar_clean':  ('scalar', 0.0),
    'scalar_noised': ('scalar', 0.15),
    'pos3d_clean':   ('pos3d',  0.0),
    'pos3d_noised':  ('pos3d',  0.15),
}
# control reuses the already-trained P2 D0E1 checkpoint instead of retraining.
CONTROL_CKPT_TPL = 'checkpoints/flow_policy_v5/p2_D0E1_s{seed}/best_model.pt'


def build_cmd(python, arm, seed, args):
    mode, noise = ARMS[arm]
    tag = f"p3b_rc_{arm}_s{seed}"
    cmd = [python, '-m', 'scripts.train_flow_v5',
           '--config', args.config,
           '--recovery-h5', args.recovery_h5,
           '--recovery-episodes', str(args.recovery_episodes),
           '--hover-episodes', str(args.hover_episodes),
           '--lambda-disp', '0.0',            # D0 (dispersive OFF)
           '--range-cue', mode,
           '--cue-noise', str(noise),
           '--cue-scale', str(args.cue_scale),
           '--seed', str(seed),
           '--tag', tag]
    if args.h4_ckpt:
        cmd += ['--transfer-from-h4', args.h4_ckpt]   # E1 (encoder trainable: no --freeze-vision)
    if args.quick:
        cmd += ['--quick']
    return tag, cmd


def ckpt_path_for(save_path, tag):
    return os.path.join(save_path, tag, 'best_model.pt').replace('\\', '/')


def main():
    parser = argparse.ArgumentParser(description='P3b range-cue positive-control ablation')
    parser.add_argument('--arms', nargs='+', default=list(ARMS.keys()),
                        choices=list(ARMS.keys()))
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--config', default='configs/flow_policy_v5.yaml')
    parser.add_argument('--h4-ckpt', default='checkpoints/flow_policy_v4/20260514_175219/best_model.pt')
    parser.add_argument('--recovery-h5', default='data/expert_demos_v4_recovery.h5')
    parser.add_argument('--recovery-episodes', type=int, default=500)
    parser.add_argument('--hover-episodes', type=int, default=500)
    parser.add_argument('--cue-scale', type=float, default=3.0)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--skip-done', action='store_true', default=True)
    parser.add_argument('--no-skip-done', dest='skip_done', action='store_false')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--manifest', default='evaluation_results/p3b_rangecue_manifest.json')
    args = parser.parse_args()

    with open(os.path.join(ROOT, args.config), 'r', encoding='utf-8') as f:
        save_path = yaml.safe_load(f)['logging']['save_path']

    manifest = {'arms': {}, 'config': args.config, 'h4_ckpt': args.h4_ckpt,
                'cue_scale': args.cue_scale, 'seeds': args.seeds, 'runs': {}}

    jobs = [(seed, arm) for seed in args.seeds for arm in args.arms]
    print(f"P3b range-cue sweep: arms={args.arms} seeds={args.seeds} "
          f"quick={args.quick} skip_done={args.skip_done}")

    for i, (seed, arm) in enumerate(jobs, 1):
        mode, noise = ARMS[arm]
        manifest['arms'][arm] = {'range_cue': mode, 'cue_noise': noise}

        # control: reuse the P2 D0E1 checkpoint, do not retrain
        if arm == 'control':
            ckpt = CONTROL_CKPT_TPL.format(seed=seed)
            exists = os.path.exists(os.path.join(ROOT, ckpt))
            print(f"\n[{i}/{len(jobs)}] arm=control seed={seed}  REUSE {ckpt} "
                  f"({'found' if exists else 'MISSING'})")
            manifest['runs'][f'p3b_rc_control_s{seed}'] = {
                'arm': arm, 'seed': seed, 'ckpt': ckpt, 'range_cue': 'none',
                'cue_noise': 0.0, 'status': 'reused' if exists else 'missing'}
            continue

        tag, cmd = build_cmd(args.python, arm, seed, args)
        ckpt = ckpt_path_for(save_path, tag)
        print(f"\n{'='*84}\n[{i}/{len(jobs)}] arm={arm} seed={seed} tag={tag}\n  {' '.join(cmd)}\n{'='*84}")

        rec = {'arm': arm, 'seed': seed, 'ckpt': ckpt, 'range_cue': mode,
               'cue_noise': noise}
        if args.skip_done and os.path.exists(os.path.join(ROOT, ckpt)):
            print(f"  SKIP: {ckpt} already exists")
            rec['status'] = 'skipped_existing'; rec['returncode'] = 0
            manifest['runs'][tag] = rec
            continue
        if args.dry_run:
            rec['status'] = 'dry_run'; manifest['runs'][tag] = rec
            continue

        t0 = time.time()
        rec['status'] = 'running'; rec['started'] = datetime.now().isoformat()
        manifest['runs'][tag] = rec
        _save(args.manifest, manifest)
        ret = subprocess.run(cmd, cwd=ROOT).returncode
        rec.update({'status': 'done' if ret == 0 else 'failed',
                    'returncode': ret, 'minutes': round((time.time() - t0) / 60, 1)})
        _save(args.manifest, manifest)

    _save(args.manifest, manifest)
    print(f"\nManifest -> {args.manifest}")
    print("\nEval each arm under frozen P0 (cue-noise MUST match the trained arm):")
    for tag, r in manifest['runs'].items():
        print(f"  dppo/Scripts/python.exe -m scripts.evaluate_frozen_p0 "
              f"--ckpts \"{tag}:{r['ckpt']}\" --n-episodes 30 "
              f"--cue-scale {args.cue_scale} --cue-noise {r['cue_noise']} "
              f"--output evaluation_results/{tag}_frozen.json")


def _save(rel, manifest):
    p = os.path.join(ROOT, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)


if __name__ == '__main__':
    main()

"""
P6 scale-invariant FORM ablation driver — RESEARCH_PLAN_v7 Phase 6 / Direction 4.

Closes the last §6.1 rebuttal ("you used a scale-sensitive criterion"). The faithful
Dispersive (InfoNCE-L2 on flow_mid, /d, tau=0.5, lambda=0.5) GAMES its own objective
by norm inflation (feat_norm ~9x on flow_mid) yet still does not move closed-loop
control. This sweep holds the WHOLE P2f D1E1 base recipe fixed and varies ONLY the
dispersive FORM, so the new arms line up directly against the existing §6.1 numbers:

  form     definition (on the flow_net mid-block features)              source
  off      lambda=0                                            == P2f D0E1 (reuse, NOT re-run)
  infonce  faithful InfoNCE-L2  D=||zi-zj||^2/d, L=log E[exp(-D/t)]  == P2f D1E1 (reuse)
  cosine   unit-sphere InfoNCE: L2-normalise then ||.||^2 (no /d) -> norm-inflation impossible
  vicreg   variance hinge + off-diagonal covariance^2 (scale-invariant)

So this driver trains ONLY the two NEW scale-invariant forms x 3 seeds = 6 runs; the
off / infonce arms are reused from evaluation_results/p2f_ablation_manifest.json by
scripts.evaluate_p6_form_ablation. Every NEW run shares the P2f D1E1 recipe EXACTLY:

  --transfer-from-h4 checkpoints/flow_policy_v4/20260514_175219/best_model.pt  (E2E, no freeze)
  --recovery-h5 data/expert_demos_v4_recovery.h5 --recovery-episodes 500 --hover-episodes 500
  --lambda-disp 0.5 --dispersive-target flow_mid --dispersive-tau 0.5 --dispersive-form <form>

Runs SEQUENTIALLY (one training at a time) to avoid GPU contention (Known Failure
Mode #7). Writes a manifest mapping (form, seed) -> {ckpt} so
scripts.evaluate_p6_form_ablation can aggregate by form with mean+-std across seeds.
A NEW manifest/tag namespace (p6_*) is used so the P2f artifacts are never overwritten.

Usage (launch in background with run_in_background=true):
  dppo/Scripts/python.exe -m scripts.run_p6_form_ablation \
      --h4-ckpt checkpoints/flow_policy_v4/20260514_175219/best_model.pt \
      --recovery-h5 data/expert_demos_v4_recovery.h5 \
      --recovery-episodes 500 --hover-episodes 500 \
      --forms cosine vicreg --seeds 0 1 2

  # smoke test wiring (5 epochs, tiny episode pool, 1 seed):
  dppo/Scripts/python.exe -m scripts.run_p6_form_ablation --quick --seeds 0

  # print the commands without launching:
  dppo/Scripts/python.exe -m scripts.run_p6_form_ablation --dry-run
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

# NEW forms only (off=P2f D0E1, infonce=P2f D1E1 are reused by the evaluator).
NEW_FORMS = ['cosine', 'vicreg']

# Base recipe = P2f D1E1, with only --dispersive-form varied. Kept here so the
# driver is the single source of truth for "what D1E1 was".
LAMBDA_DISP = 0.5
DISPERSIVE_TARGET = 'flow_mid'
DISPERSIVE_TAU = 0.5


def build_cmd(python, form, seed, args):
    """Build the train_flow_v5 command for one (form, seed). Returns (tag, cmd, meta)."""
    tag = f"p6_{form}_s{seed}"
    he = args.quick_episodes if args.quick else args.hover_episodes
    re = args.quick_episodes if args.quick else args.recovery_episodes

    cmd = [python, '-m', 'scripts.train_flow_v5',
           '--config', args.config,
           '--recovery-h5', args.recovery_h5,
           '--recovery-episodes', str(re),
           '--hover-episodes', str(he),
           '--lambda-disp', str(LAMBDA_DISP),
           '--dispersive-target', DISPERSIVE_TARGET,
           '--dispersive-tau', str(DISPERSIVE_TAU),
           '--dispersive-form', form,
           '--seed', str(seed),
           '--tag', tag]
    if args.h4_ckpt:                                   # E2E: NO --freeze-vision
        cmd += ['--transfer-from-h4', args.h4_ckpt]
    if args.quick:
        cmd += ['--quick']

    meta = {'form': form, 'seed': seed, 'lambda_disp': LAMBDA_DISP,
            'dispersive_target': DISPERSIVE_TARGET, 'dispersive_tau': DISPERSIVE_TAU,
            'freeze_vision': False}
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
    parser = argparse.ArgumentParser(description='P6 scale-invariant FORM ablation sweep')
    parser.add_argument('--forms', nargs='+', default=list(NEW_FORMS),
                        choices=list(NEW_FORMS),
                        help='Which NEW scale-invariant forms to train (default: cosine vicreg). '
                             'off/infonce are reused from the P2f manifest by the evaluator.')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                        help='Seeds per form (>=3 for the paper)')
    parser.add_argument('--config', default='configs/flow_policy_v5.yaml')
    parser.add_argument('--h4-ckpt', default='checkpoints/flow_policy_v4/20260514_175219/best_model.pt',
                        help='Shared warm start for every run (--transfer-from-h4) = P2f D1E1 init')
    parser.add_argument('--recovery-h5', default='data/expert_demos_v4_recovery.h5')
    parser.add_argument('--recovery-episodes', type=int, default=500)
    parser.add_argument('--hover-episodes', type=int, default=500)
    parser.add_argument('--quick', action='store_true', help='5-epoch smoke test per run')
    parser.add_argument('--quick-episodes', type=int, default=20,
                        help='Episodes per pool in --quick mode (keeps the smoke fast)')
    parser.add_argument('--skip-done', action='store_true', default=True,
                        help='Skip a (form,seed) whose best_model.pt already exists')
    parser.add_argument('--no-skip-done', dest='skip_done', action='store_false')
    parser.add_argument('--dry-run', action='store_true', help='Print commands, do not launch')
    parser.add_argument('--python', default=sys.executable, help='Interpreter for subprocesses')
    parser.add_argument('--manifest', default='evaluation_results/p6_form_ablation_manifest.json')
    args = parser.parse_args()

    with open(os.path.join(ROOT, args.config), 'r', encoding='utf-8') as f:
        save_path = yaml.safe_load(f)['logging']['save_path']

    manifest = load_manifest(os.path.join(ROOT, args.manifest))
    manifest.setdefault('runs', {})
    manifest['base_recipe'] = {
        'description': 'P2f D1E1 (faithful flow_mid, lambda=0.5, tau=0.5, E2E, H4-transfer, '
                       'v4 recovery mix); only --dispersive-form varies',
        'lambda_disp': LAMBDA_DISP, 'dispersive_target': DISPERSIVE_TARGET,
        'dispersive_tau': DISPERSIVE_TAU, 'freeze_vision': False,
        'transfer_from_h4': args.h4_ckpt, 'recovery_h5': args.recovery_h5,
        'hover_episodes': args.hover_episodes, 'recovery_episodes': args.recovery_episodes,
    }
    # Controls reused from the P2f sweep (NOT re-run here).
    manifest['reuse_from_p2f'] = {
        'manifest': 'evaluation_results/p2f_ablation_manifest.json',
        'off': 'D0E1', 'infonce': 'D1E1',
    }
    manifest['config'] = args.config
    manifest['quick'] = args.quick

    # seeds outer, forms inner -> a full form-pair completes per seed as early as possible
    jobs = [(seed, form) for seed in args.seeds for form in args.forms]
    print(f"P6 sweep: {len(jobs)} runs  forms={args.forms}  seeds={args.seeds}  "
          f"quick={args.quick}  skip_done={args.skip_done}")

    for i, (seed, form) in enumerate(jobs, 1):
        tag, cmd, meta = build_cmd(args.python, form, seed, args)
        ckpt = ckpt_path_for(save_path, tag)
        print(f"\n{'='*84}\n[{i}/{len(jobs)}] form={form} seed={seed} tag={tag}\n  {' '.join(cmd)}\n{'='*84}")

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
    print("Next: dppo/Scripts/python.exe -m scripts.evaluate_p6_form_ablation "
          f"--manifest {args.manifest} "
          "--oracle-ckpt checkpoints/ppo_expert_v4/20260419_142245/best_model.pt "
          "--oracle-norm checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz")


if __name__ == '__main__':
    main()

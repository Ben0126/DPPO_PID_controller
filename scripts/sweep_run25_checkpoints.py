"""
Sweep Run 25 checkpoints to find policy degradation curve.

Eval at key updates: 50 (BC level), 200 (early climb), 300 (mid), 375 (peak),
500 (late hover), 650 (ramp start). Reports steps avg + RMSE per checkpoint.
"""
import sys, os, subprocess, re

PY = sys.executable  # absolute path to current python (the venv one)
CKPT_DIR = "checkpoints/reinflow_v4/reinflow_v4_20260515_023519"
CKPTS = [50, 200, 300, 350, 400, 500, 650]

def parse_output(text):
    steps = [int(s) for s in re.findall(r'steps=(\d+)', text)]
    rmse = re.search(r'Position RMSE:\s+([\d.]+)\s*m', text)
    crashes = text.count('| CRASH')
    return {
        'steps_avg': sum(steps)/len(steps) if steps else 0,
        'steps_min': min(steps) if steps else 0,
        'steps_max': max(steps) if steps else 0,
        'rmse': float(rmse.group(1)) if rmse else 0,
        'crashes': crashes,
    }

results = []
for u in CKPTS:
    ckpt = f"{CKPT_DIR}/update_{u}.pt"
    if not os.path.exists(ckpt):
        print(f"  SKIP: {ckpt} not found"); continue
    print(f"\n>>> Evaluating update_{u}.pt ...")
    cmd = [
        PY, "-m", "scripts.evaluate_rhc_v4",
        "--flow-model", ckpt,
        "--ppo-model", "checkpoints/ppo_expert_v4/20260419_142245/best_model.pt",
        "--ppo-norm",  "checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz",
        "--n-inference-steps", "2",
        "--output-dir", f"evaluation_results/run25_sweep_u{u}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED: {r.stderr[-300:]}"); continue
    res = parse_output(r.stdout)
    results.append((u, res))
    print(f"  u{u}: steps={res['steps_avg']:.1f} (min={res['steps_min']}, max={res['steps_max']})  "
          f"RMSE={res['rmse']:.3f}m  crashes={res['crashes']}/50")

print("\n" + "="*70)
print(f"{'Update':>8}  {'Steps avg':>10}  {'Steps min/max':>14}  {'RMSE (m)':>9}  {'Crashes':>8}")
print("-"*70)
for u, res in results:
    print(f"  u{u:>5}  {res['steps_avg']:>10.1f}  "
          f"{res['steps_min']:>5}/{res['steps_max']:<5}    {res['rmse']:>8.3f}  {res['crashes']:>6}/50")
print("="*70)
print("\nReference baselines:")
print(f"  H4 BC ep76:  steps=202.3  RMSE=1.152m  crashes=50/50")
print(f"  Run 25 best: steps= 85.9  RMSE=0.544m  crashes=50/50")

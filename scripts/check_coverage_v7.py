"""
Phase 3 v7.0 — coverage HARD GATE for the four T x O datasets (go/no-go before Phase 4).

The decisive H_v7 T-axis assumes the `far` datasets actually fill the 1-3 m
position-error band the paper's BC data lacks (paper's frac>1 m ~0.4%), and that
the `hover` datasets do NOT. This script reuses the validated position-error
machinery from scripts/measure_ood_coverage.py (states[:,0:3] norm == ||pos-target||,
since the 15D obs is [pos_error_body(3), ...]) to verify both, cheaply, with no
training.

Gate (pre-registered):
  * far  datasets:  frac(pos-err in [1,3] m) >= --min-far-frac  AND p99 >= ~1.5 m
  * hover datasets: frac(pos-err > 1 m)        <= --max-hover-frac (~0)

Bonus: confirms the clean O-swap design property -- the crosshair and perspective
file of the SAME mode share byte-identical trajectories (states), so the O axis is
a pure observation swap.

Usage:
  dppo/Scripts/python.exe -m scripts.check_coverage_v7
  dppo/Scripts/python.exe -m scripts.check_coverage_v7 --suffix _smoke
"""
import os
import sys
import json
import argparse

import numpy as np
import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.measure_ood_coverage import (
    pos_err_from_h5, quantiles, frac_exceeding, ascii_hist,
)


def frac_in_band(x, lo, hi):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return round(float(((x >= lo) & (x <= hi)).mean()), 5)


def states_identical(h5_a, h5_b, max_ep):
    """Verify two h5 files share byte-identical per-episode states (clean O-swap)."""
    with h5py.File(h5_a, "r") as fa, h5py.File(h5_b, "r") as fb:
        n = min(int(fa.attrs["n_episodes"]), int(fb.attrs["n_episodes"]), max_ep)
        for ep in range(n):
            k = f"episode_{ep}"
            if k not in fa or k not in fb:
                return False
            if not np.array_equal(fa[k]["states"][:], fb[k]["states"][:]):
                return False
    return True


def main():
    ap = argparse.ArgumentParser(description="Phase 3 v7 coverage hard gate")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--prefix", default="expert_demos_v7")
    ap.add_argument("--suffix", default="", help="e.g. _smoke for the smoke datasets")
    ap.add_argument("--max-episodes", type=int, default=100000)
    ap.add_argument("--min-far-frac", type=float, default=0.05,
                    help="far datasets must have >= this fraction of steps in [1,3] m")
    ap.add_argument("--max-hover-frac", type=float, default=0.01,
                    help="hover datasets must have <= this fraction of steps > 1 m")
    ap.add_argument("--out", default="evaluation_results/p3_coverage_v7.json")
    args = ap.parse_args()

    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    cells = {
        "T0O0_hover_crosshair": ("hover", f"{args.prefix}_hover_crosshair{args.suffix}.h5"),
        "T0O1_hover_persp":     ("hover", f"{args.prefix}_hover_persp{args.suffix}.h5"),
        "T1O0_far_crosshair":   ("far",   f"{args.prefix}_far_crosshair{args.suffix}.h5"),
        "T1O1_far_persp":       ("far",   f"{args.prefix}_far_persp{args.suffix}.h5"),
    }

    results = {}
    all_pass = True
    print("=" * 84)
    print("PHASE 3 v7 — coverage hard gate (position-error ||states[:,0:3]||)")
    print("=" * 84)
    for cell, (mode, fname) in cells.items():
        path = os.path.join(ROOT, args.data_dir, fname)
        if not os.path.exists(path):
            print(f"\n[{cell}] MISSING: {fname} (skip)")
            results[cell] = {"mode": mode, "file": fname, "present": False}
            continue
        err = pos_err_from_h5(path, args.max_episodes)
        q = quantiles(err)
        fe = frac_exceeding(err, thresholds)
        f13 = frac_in_band(err, 1.0, 3.0)

        if mode == "far":
            passed = (f13 >= args.min_far_frac) and (q.get("p99", 0) >= 1.5)
        else:  # hover
            passed = fe.get(">1.0m", 1.0) <= args.max_hover_frac
        all_pass = all_pass and passed

        results[cell] = {
            "mode": mode, "file": fname, "present": True,
            "n_steps": q.get("n", 0),
            "quantiles": q, "frac_exceeding": fe, "frac_in_1_3m": f13,
            "pass": bool(passed),
        }
        print(f"\n[{cell}]  ({mode})  n={q.get('n',0):,} steps")
        print(f"  p50={q.get('p50')}  p90={q.get('p90')}  p99={q.get('p99')}  max={q.get('max')} m")
        print(f"  frac>1m={fe.get('>1.0m')}  frac>2m={fe.get('>2.0m')}  frac>3m={fe.get('>3.0m')}  "
              f"frac in [1,3]m={f13}")
        print(f"  GATE: {'PASS' if passed else 'FAIL'}")
        print(ascii_hist(err, label=f"{cell} pos-err"))

    # clean O-swap cross-check: cross vs persp share identical trajectories
    print("\n" + "-" * 84)
    print("Clean O-swap check (crosshair vs perspective share identical states):")
    swap = {}
    for mode in ("hover", "far"):
        a = os.path.join(ROOT, args.data_dir, f"{args.prefix}_{mode}_crosshair{args.suffix}.h5")
        b = os.path.join(ROOT, args.data_dir, f"{args.prefix}_{mode}_persp{args.suffix}.h5")
        if os.path.exists(a) and os.path.exists(b):
            ok = states_identical(a, b, args.max_episodes)
            swap[mode] = bool(ok)
            print(f"  {mode}: states identical = {ok}")
            all_pass = all_pass and ok
        else:
            swap[mode] = None
            print(f"  {mode}: one of the files missing (skip)")

    out = {
        "min_far_frac": args.min_far_frac,
        "max_hover_frac": args.max_hover_frac,
        "cells": results,
        "clean_o_swap": swap,
        "overall_pass": bool(all_pass),
    }
    os.makedirs(os.path.join(ROOT, os.path.dirname(args.out)), exist_ok=True)
    with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 84)
    print(f"OVERALL COVERAGE GATE: {'PASS' if all_pass else 'FAIL'}")
    print(f"Wrote {args.out}")
    print("=" * 84)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

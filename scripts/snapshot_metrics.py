"""
Claude Code Stop hook — snapshot DPPO metrics to dppo_status.json.

Runs when the Claude Code session ends (Stop event).
Reads TensorBoard events from the latest DPPO run and
refreshes the dppo_status.json cache used by status_line.py.

Requires tensorboard package (available in project venv).
Run via: DPPO_PID_controller/dppo/Scripts/python scripts/snapshot_metrics.py
"""

import json
import os
import glob
from pathlib import Path

REPO = Path(__file__).parent.parent
STATUS_FILE = REPO / "dppo_status.json"
LOGDIR_GLOB = str(REPO / "logs/diffusion_policy/dppo_*")


def find_latest_dppo_logdir():
    dirs = glob.glob(LOGDIR_GLOB)
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)


def read_scalars(logdir: str) -> dict:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
        ea.Reload()
        scalars = {}
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            if events:
                scalars[tag] = [(e.step, e.value) for e in events]
        return scalars
    except Exception:
        return {}


def main():
    logdir = find_latest_dppo_logdir()
    if not logdir:
        return

    run_name = Path(logdir).name
    scalars = read_scalars(logdir)
    status = {"run": run_name}

    rewards = scalars.get("dppo/mean_reward", [])
    if rewards:
        status["latest_update"] = rewards[-1][0]
        status["latest_reward"] = rewards[-1][1]
        status["best_reward"] = max(r for _, r in rewards)
        status["best_update"] = max(rewards, key=lambda x: x[1])[0]
        if len(rewards) >= 20:
            last10 = sum(r for _, r in rewards[-10:]) / 10
            prev10 = sum(r for _, r in rewards[-20:-10]) / 10
            if last10 > prev10 + 0.01:
                status["trend"] = "improving"
            elif last10 < prev10 - 0.01:
                status["trend"] = "declining"
            else:
                status["trend"] = "stable"
        else:
            status["trend"] = "early (< 20 updates)"

    vlosses = scalars.get("dppo/value_loss", [])
    if vlosses:
        status["latest_value_loss"] = vlosses[-1][1]

    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    main()

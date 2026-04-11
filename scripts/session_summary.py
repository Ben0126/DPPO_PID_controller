"""
Claude Code SessionStart hook — DPPO run status summary.

Reads the latest TensorBoard event file from the most recent DPPO run,
outputs an additionalContext JSON payload so Claude starts each session
already knowing the run's health.

Also writes dppo_status.json (used by status_line.py).

Requires tensorboard package (available in project venv).
Run via: DPPO_PID_controller/dppo/Scripts/python scripts/session_summary.py
"""

import sys
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


def read_tfevents_scalars(logdir: str) -> dict:
    """Read scalar summaries using tensorboard EventAccumulator."""
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


def compute_status(scalars: dict) -> dict:
    s = {}
    rewards = scalars.get("dppo/mean_reward", [])
    vlosses = scalars.get("dppo/value_loss", [])

    if rewards:
        s["latest_update"] = rewards[-1][0]
        s["latest_reward"] = rewards[-1][1]
        s["best_reward"] = max(r for _, r in rewards)
        s["best_update"] = max(rewards, key=lambda x: x[1])[0]
        if len(rewards) >= 20:
            last10 = sum(r for _, r in rewards[-10:]) / 10
            prev10 = sum(r for _, r in rewards[-20:-10]) / 10
            if last10 > prev10 + 0.01:
                s["trend"] = "improving"
            elif last10 < prev10 - 0.01:
                s["trend"] = "declining"
            else:
                s["trend"] = "stable"
        else:
            s["trend"] = "early (< 20 updates)"

    if vlosses:
        s["latest_value_loss"] = vlosses[-1][1]

    return s


def health_label(s: dict) -> str:
    reward = s.get("latest_reward")
    if reward is None:
        return "No data yet"
    if reward < 0:
        return "COLLAPSE RISK"
    if reward < 0.1:
        return "Marginal"
    if reward < 0.3:
        return "Developing"
    return "Healthy"


def build_context(run_name: str, s: dict) -> str:
    lines = [f"=== DPPO Status at Session Start ==="]
    lines.append(f"Active run : {run_name}")

    if not s:
        lines.append("No scalar data found (run may not have started yet).")
        return "\n".join(lines)

    lines.append(f"Health     : {health_label(s)}")
    lines.append(f"Update     : {s.get('latest_update', '?')}")

    reward = s.get("latest_reward")
    if reward is not None:
        lines.append(f"Reward/step: {reward:+.4f}  (healthy range: +0.3 to +0.6)")

    vloss = s.get("latest_value_loss")
    if vloss is not None:
        note = " ✓ converged" if vloss < 1.0 else " ← still warming up" if vloss > 5 else ""
        lines.append(f"Value loss : {vloss:.4f}{note}")

    best_r = s.get("best_reward")
    if best_r is not None:
        lines.append(f"Best reward: {best_r:+.4f} at update {s.get('best_update', '?')}")

    lines.append(f"Trend      : {s.get('trend', '?')}")
    return "\n".join(lines)


def main():
    logdir = find_latest_dppo_logdir()
    status_data = {}

    if logdir:
        run_name = Path(logdir).name
        scalars = read_tfevents_scalars(logdir)
        s = compute_status(scalars)
        status_data = {**s, "run": run_name}
        context = build_context(run_name, s)
    else:
        context = "No DPPO training run found in logs/diffusion_policy/. Run 2 may not have started yet."

    # Update cache for status_line.py
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2)
    except Exception:
        pass

    # Output SessionStart additionalContext
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()

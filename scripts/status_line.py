"""
Claude Code statusLine script — compact DPPO run status.

Reads dppo_status.json (written by session_summary.py / snapshot_metrics.py)
and outputs a single status line for the Claude Code UI.

Only stdlib — fast and no venv required.
"""

import json
from pathlib import Path

STATUS_FILE = Path(__file__).parent.parent / "dppo_status.json"

TREND_ICON = {
    "improving": "↑",
    "stable": "→",
    "declining": "↓",
    "early (< 20 updates)": "~",
}


def main():
    if not STATUS_FILE.exists():
        print("DPPO: no status (run session_summary.py)")
        return

    try:
        with open(STATUS_FILE, encoding="utf-8") as f:
            s = json.load(f)
    except Exception:
        print("DPPO: status read error")
        return

    run = s.get("run", "?")[-8:]           # e.g. "130513"
    update = s.get("latest_update", "?")
    reward = s.get("latest_reward")
    vloss = s.get("latest_value_loss")
    trend = TREND_ICON.get(s.get("trend", ""), "?")

    reward_str = f"{reward:+.3f}" if reward is not None else "N/A"
    vloss_str = f"{vloss:.2f}" if vloss is not None else "N/A"

    # Health flag
    if reward is not None:
        if reward < 0:
            flag = " ⚠COLLAPSE"
        elif reward < 0.1:
            flag = " ⚠low"
        else:
            flag = ""
    else:
        flag = ""

    print(f"DPPO:{run} u={update} r={reward_str}{trend} vl={vloss_str}{flag}")


if __name__ == "__main__":
    main()

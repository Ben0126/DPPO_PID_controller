"""
Claude Code PreToolUse hook — protect critical project files.

Blocks any Bash command that attempts to delete or overwrite:
  - data/expert_demos.h5          (1000 episodes, 33 min to regenerate)
  - checkpoints/.../best_model.pt (supervised, 14h to regenerate)
  - checkpoints/ppo_expert/20260401_103107/  (Run 6 gold standard)
  - Any best_dppo_model.pt

Only stdlib — no venv required.
"""

import sys
import json
import re

# Patterns that indicate a destructive filesystem operation
DANGEROUS_OPS = [
    r"\brm\b",
    r"\bdel\b(?!ta)",   # del but not delta
    r"\brmdir\b",
    r"shutil\.rmtree",
    r"os\.remove",
    r"truncate\b",
]

# Patterns that match protected file paths
PROTECTED_PATTERNS = [
    (r"expert_demos\.h5",
     "expert_demos.h5 — 1000 episodes, 33 min to regenerate"),
    (r"20260402_032701[/\\]best_model\.pt",
     "supervised best_model.pt — 14h of training to regenerate"),
    (r"20260401_103107",
     "PPO Expert Run 6 checkpoint — gold standard baseline"),
    (r"best_dppo_model\.pt",
     "best_dppo_model.pt — current best D²PPO checkpoint"),
]


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return  # Malformed JSON — allow by default

    command = data.get("tool_input", {}).get("command", "")
    if not command:
        return

    # Check if this is a dangerous operation
    is_dangerous = any(re.search(pat, command, re.IGNORECASE) for pat in DANGEROUS_OPS)
    if not is_dangerous:
        return  # Safe operation — allow

    # Check if any protected file is targeted
    violations = []
    for pat, description in PROTECTED_PATTERNS:
        if re.search(pat, command, re.IGNORECASE):
            violations.append(description)

    if not violations:
        return  # Dangerous op but not on protected files — allow

    # Block and explain
    block_msg = (
        "BLOCKED by file guard hook.\n\n"
        "Attempting to delete/overwrite protected file(s):\n"
        + "\n".join(f"  • {v}" for v in violations)
        + "\n\nIf you are certain, delete manually from a terminal outside Claude Code."
    )
    print(json.dumps({"continue": False, "stopReason": block_msg}))


if __name__ == "__main__":
    main()

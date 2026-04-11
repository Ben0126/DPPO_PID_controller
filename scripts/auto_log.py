"""
Claude Code PostToolUse hook — unified dev log recorder.

處理三種 tool 事件，分別記錄：
  Bash  → 訓練/評估腳本的執行紀錄（script output）
  Edit  → config 超參數變動 / bug fix diff（old → new）
  Write → 新建檔案的摘要

只記錄有研究價值的異動，雜訊（docs/、cache、.claude/）跳過。
Only stdlib — no venv required.
"""

import sys
import json
import re
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).parent.parent
LOG_PATH = REPO / "docs" / "dev_log_phase2_3.md"

# ── Bash 監控清單 ──────────────────────────────────────────────
BASH_PATTERNS = [
    (r"check_device",        "Device Check"),
    (r"train_dppo",          "DPPO Training — Started"),
    (r"train_diffusion",     "Diffusion Training — Started"),
    (r"evaluate_rhc",        "RHC Evaluation"),
    (r"collect_data",        "Expert Data Collection"),
    (r"evaluate_ppo_expert", "PPO Expert Evaluation"),
]

# ── Edit/Write：要記錄的路徑 pattern → 分類標籤 ────────────────
FILE_CATEGORIES = [
    (r"configs[\\/].*\.yaml",      "Config / HP Change"),
    (r"scripts[\\/].*\.py",        "Script Fix"),
    (r"models[\\/].*\.py",         "Model Fix"),
    (r"envs[\\/].*\.py",           "Env Fix"),
    (r"utils[\\/].*\.py",          "Utils Fix"),
]

# 這些路徑的異動不記錄（雜訊）
SKIP_PATTERNS = [
    r"docs[\\/]",
    r"dppo_status\.json",
    r"\.claude[\\/]",
    r"__pycache__",
    r"\.pyc$",
]

MAX_DIFF_CHARS = 800   # diff 單側截斷長度
MAX_OUTPUT_CHARS = 600  # bash output 截斷長度


# ── 工具函式 ──────────────────────────────────────────────────

def rel_path(abs_path: str) -> str:
    """把絕對路徑轉成相對於 REPO 的路徑（顯示用）。"""
    try:
        return str(Path(abs_path).relative_to(REPO))
    except ValueError:
        return abs_path


def truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) > limit:
        return text[:limit] + f"\n... [truncated — {len(text)} chars total]"
    return text or "(empty)"


def append_log(entry: str):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception:
        pass  # Never crash Claude Code


def detect_lang(file_path: str) -> str:
    return "yaml" if file_path.endswith((".yaml", ".yml")) else "python"


# ── Bash handler ──────────────────────────────────────────────

def handle_bash(data: dict):
    command = data.get("tool_input", {}).get("command", "")
    if not command:
        return

    label = next((lbl for pat, lbl in BASH_PATTERNS if re.search(pat, command)), None)
    if label is None:
        return

    response = data.get("tool_response", {})
    if isinstance(response, dict):
        raw = response.get("output", "") or response.get("stderr", "") or ""
    else:
        raw = str(response)
    output = truncate(raw, MAX_OUTPUT_CHARS)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"\n---\n"
        f"<!-- auto-log {ts} bash -->\n"
        f"### [Auto-Log] {ts} — {label}\n\n"
        f"**Command:** `{command.strip()}`\n\n"
        f"**Output:**\n```\n{output}\n```\n"
    )
    append_log(entry)


# ── Edit handler ──────────────────────────────────────────────

def handle_edit(data: dict):
    inp = data.get("tool_input", {})
    file_path = inp.get("file_path", "")
    old_str = inp.get("old_string", "")
    new_str = inp.get("new_string", "")

    if not file_path:
        return
    if any(re.search(p, file_path, re.IGNORECASE) for p in SKIP_PATTERNS):
        return

    category = next(
        (lbl for pat, lbl in FILE_CATEGORIES if re.search(pat, file_path, re.IGNORECASE)),
        None,
    )
    if category is None:
        return

    lang = detect_lang(file_path)
    rel = rel_path(file_path)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    old_display = truncate(old_str, MAX_DIFF_CHARS)
    new_display = truncate(new_str, MAX_DIFF_CHARS)

    entry = (
        f"\n---\n"
        f"<!-- auto-log {ts} edit -->\n"
        f"### [Auto-Log] {ts} — {category}\n\n"
        f"**File:** `{rel}`\n\n"
        f"**Before:**\n```{lang}\n{old_display}\n```\n\n"
        f"**After:**\n```{lang}\n{new_display}\n```\n"
    )
    append_log(entry)


# ── Write handler ─────────────────────────────────────────────

def handle_write(data: dict):
    inp = data.get("tool_input", {})
    file_path = inp.get("file_path", "")
    content = inp.get("content", "")

    if not file_path:
        return
    if any(re.search(p, file_path, re.IGNORECASE) for p in SKIP_PATTERNS):
        return

    category = next(
        (lbl for pat, lbl in FILE_CATEGORIES if re.search(pat, file_path, re.IGNORECASE)),
        None,
    )
    if category is None:
        return

    lang = detect_lang(file_path)
    rel = rel_path(file_path)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snippet = truncate(content, MAX_DIFF_CHARS)

    entry = (
        f"\n---\n"
        f"<!-- auto-log {ts} write -->\n"
        f"### [Auto-Log] {ts} — New File: {category}\n\n"
        f"**File:** `{rel}`\n\n"
        f"**Content:**\n```{lang}\n{snippet}\n```\n"
    )
    append_log(entry)


# ── Entry point ───────────────────────────────────────────────

HANDLERS = {
    "Bash":  handle_bash,
    "Edit":  handle_edit,
    "Write": handle_write,
}


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return

    tool = data.get("tool_name", "")
    handler = HANDLERS.get(tool)
    if handler:
        handler(data)


if __name__ == "__main__":
    main()

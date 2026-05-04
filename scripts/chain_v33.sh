#!/usr/bin/env bash
# chain_v33.sh — Step 4→5 自動 pipeline（資料已備妥，直接從 Step 4 開始）
#
# 用法:
#   bash scripts/chain_v33.sh            # 直接跑 Step 4→5
#   bash scripts/chain_v33.sh <PID>      # 等 PID 結束後再跑（保留向後相容）
#
# 修正: 不使用 tee pipe（Windows bash 會誤報 segfault）
#       改用直接輸出重導向，chain 本身不含任何 pipe

set -euo pipefail

WAIT_PID="${1:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
source dppo/Scripts/activate

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ── 若有指定 PID，先等它結束 ─────────────────────────────────────────────────
if [[ -n "$WAIT_PID" ]]; then
    echo "[chain_v33] 等待 PID $WAIT_PID 結束 ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 60
    done
    echo "[chain_v33] PID $WAIT_PID 已結束。"
fi

# ── Step 4: train_diffusion_v33 ───────────────────────────────────────────────
DIFF_LOG="$LOG_DIR/train_diffusion_v33_$(date +%Y%m%d_%H%M%S).log"
echo "[chain_v33] ── Step 4 開始：train_diffusion_v33"
echo "[chain_v33]    log → $DIFF_LOG"

# 直接重導向，不用 tee（避免 Windows bash pipe segfault 誤報）
python -u -m scripts.train_diffusion_v33 --config configs/diffusion_policy.yaml \
    >> "$DIFF_LOG" 2>&1
STEP4_EXIT=$?

if [[ $STEP4_EXIT -ne 0 ]]; then
    echo "[chain_v33] ✗ Step 4 失敗（exit $STEP4_EXIT）。查看: $DIFF_LOG" >&2
    exit 1
fi

# 找出剛完成的 checkpoint（取最新的 v33_ run）
BEST_CKPT=$(ls -t checkpoints/diffusion_policy/v33_*/best_model.pt 2>/dev/null | head -1)
if [[ -z "$BEST_CKPT" ]]; then
    echo "[chain_v33] ✗ 找不到 v33 best_model.pt，中止。" >&2
    exit 1
fi
echo "[chain_v33] ✓ Step 4 完成。Best checkpoint: $BEST_CKPT"

# ── Step 5: train_dppo_v33 ────────────────────────────────────────────────────
DPPO_LOG="$LOG_DIR/train_dppo_v33_$(date +%Y%m%d_%H%M%S).log"
echo "[chain_v33] ── Step 5 開始：train_dppo_v33"
echo "[chain_v33]    pretrained: $BEST_CKPT"
echo "[chain_v33]    log → $DPPO_LOG"

python -u -m scripts.train_dppo_v33 \
    --pretrained "$BEST_CKPT" \
    >> "$DPPO_LOG" 2>&1
STEP5_EXIT=$?

if [[ $STEP5_EXIT -ne 0 ]]; then
    echo "[chain_v33] ✗ Step 5 失敗（exit $STEP5_EXIT）。查看: $DPPO_LOG" >&2
    exit 1
fi

echo "[chain_v33] ✓ 全部完成！Step 4→5 pipeline 結束。"

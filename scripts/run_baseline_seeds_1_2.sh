#!/usr/bin/env bash
# Sequential driver: P1 baseline seeds 1 & 2 (BC-vision-only + PPO-from-pixels).
# Mirrors the seed-0 commands in RESEARCH_PLAN_v6.md §1.1/§1.2 exactly, only
# changing --seed/--tag and routing eval output to a per-seed JSON. Oracle is the
# fixed 0.9668 from seed 0, so it is NOT re-run here. Runs strictly serialized to
# avoid GPU contention (Known Failure Mode #7).
#
# Launch with the Bash tool, run_in_background=true.
set -u
cd "$(dirname "$0")/.." || exit 1          # -> inner repo root
PY=dppo/Scripts/python.exe
LOG=logs/baseline_seeds_1_2.log
mkdir -p logs evaluation_results
echo "=== driver start $(date) ===" | tee "$LOG"

run() {   # label, command...
  local label="$1"; shift
  echo "" | tee -a "$LOG"
  echo ">>> [$label] start $(date)" | tee -a "$LOG"
  echo ">>> $*" | tee -a "$LOG"
  "$@" >> "$LOG" 2>&1
  local rc=$?
  echo ">>> [$label] done rc=$rc $(date)" | tee -a "$LOG"
  return $rc
}

# ---- BC-vision-only seed 1 ----
run "BC-train-s1" $PY -m scripts.train_bc_vision_only \
    --recovery-h5 data/expert_demos_v4_recovery.h5 \
    --recovery-episodes 500 --hover-episodes 500 \
    --tag bc_vision_only_s1 --seed 1
run "BC-eval-s1" $PY -m scripts.evaluate_baselines_frozen \
    --ckpts "BC_vis:checkpoints/bc_vision_only/bc_vision_only_s1/best_model.pt" \
    --output evaluation_results/baselines_frozen_bc_s1.json

# ---- BC-vision-only seed 2 ----
run "BC-train-s2" $PY -m scripts.train_bc_vision_only \
    --recovery-h5 data/expert_demos_v4_recovery.h5 \
    --recovery-episodes 500 --hover-episodes 500 \
    --tag bc_vision_only_s2 --seed 2
run "BC-eval-s2" $PY -m scripts.evaluate_baselines_frozen \
    --ckpts "BC_vis:checkpoints/bc_vision_only/bc_vision_only_s2/best_model.pt" \
    --output evaluation_results/baselines_frozen_bc_s2.json

# ---- PPO-from-pixels seed 1 ----
run "PPO-train-s1" $PY -m scripts.train_ppo_from_pixels --tag ppo_px_s1 --seed 1
run "PPO-eval-s1" $PY -m scripts.evaluate_baselines_frozen --kind ppo_pixels \
    --ckpts "PPO_px:checkpoints/ppo_from_pixels/ppo_px_s1/best_model.pt" \
    --output evaluation_results/baselines_frozen_ppopx_s1.json

# ---- PPO-from-pixels seed 2 ----
run "PPO-train-s2" $PY -m scripts.train_ppo_from_pixels --tag ppo_px_s2 --seed 2
run "PPO-eval-s2" $PY -m scripts.evaluate_baselines_frozen --kind ppo_pixels \
    --ckpts "PPO_px:checkpoints/ppo_from_pixels/ppo_px_s2/best_model.pt" \
    --output evaluation_results/baselines_frozen_ppopx_s2.json

echo "" | tee -a "$LOG"
echo "=== driver done $(date) ===" | tee -a "$LOG"

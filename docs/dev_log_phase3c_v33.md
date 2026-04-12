# DPPO PID Controller — Phase 3c v3.3: Physics-based IMU + Normalization

> Part of the dev log series. See [index](dev_log_phase2_3.md) for all phases.
> Covers: v3.3 architecture changes, v3.3 Phase 3a supervised pre-training, v3.3 DPPO fine-tuning.

---

## Table of Contents

*(will be filled as runs progress)*

---


---
<!-- auto-log 2026-04-12 04:18:56 bash -->
### [Auto-Log] 2026-04-12 04:18:56 — DPPO Training — Started

**Command:** `cd /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller

# Collect progress
echo "=== COLLECT V33 ==="
CLOG=$(ls -t logs/collect_v33_*.log 2>/dev/null | head -1)
tail -1 "$CLOG" 2>/dev/null | tr -d '\r'

# Chain status
echo "=== CHAIN ==="
CHALOG=$(ls -t logs/chain_v33_*.log 2>/dev/null | head -1)
tail -2 "$CHALOG" 2>/dev/null | tr -d '\r'

# DPPO Run 5
echo "=== DPPO RUN 5 ==="
DLOG=$(ls -t logs/train_dppo_run5_*.log 2>/dev/null | head -1)
tail -3 "$DLOG" 2>/dev/null | tr -d '\r'`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-12 04:19:07 bash -->
### [Auto-Log] 2026-04-12 04:19:07 — DPPO Training — Started

**Command:** `cd /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller

# Segfault details
echo "=== DIFFUSION V33 LOG (last 20) ==="
DIFFLOG=$(ls -t logs/train_diffusion_v33_*.log 2>/dev/null | head -1)
echo "Log: $DIFFLOG"
cat "$DIFFLOG" 2>/dev/null | tr -d '\r' | tail -20

# DPPO Run 5 full recent
echo ""
echo "=== DPPO RUN 5 RECENT ==="
DLOG=$(ls -t logs/train_dppo_run5_*.log 2>/dev/null | head -1)
echo "Log: $DLOG"
cat "$DLOG" | tr -d '\r' | grep "Update" | tail -10`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-12 04:20:43 bash -->
### [Auto-Log] 2026-04-12 04:20:43 — Diffusion v3.3 Training — Started

**Command:** `cd /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller

# Segfault - what happened
DIFFLOG=$(ls -t logs/train_diffusion_v33_*.log 2>/dev/null | head -1)
wc -l "$DIFFLOG"
cat "$DIFFLOG" | tr -d '\r'

# Check chain log fully
CHALOG=$(ls -t logs/chain_v33_*.log 2>/dev/null | head -1)
tail -10 "$CHALOG" | tr -d '\r'`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-12 04:21:15 bash -->
### [Auto-Log] 2026-04-12 04:21:15 — Diffusion v3.3 Training — Started

**Command:** `cd /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller

# Check first diffusion v33 log (from first chain attempt)
DIFFLOG1="logs/train_diffusion_v33_20260411_192314.log"
echo "=== First diffusion v33 log ==="
wc -l "$DIFFLOG1"
cat "$DIFFLOG1" | tr -d '\r' | head -30`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-12 04:27:48 bash -->
### [Auto-Log] 2026-04-12 04:27:48 — Diffusion v3.3 Training — Started

**Command:** `cd /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller
grep -n "resume\|checkpoint\|load" scripts/train_diffusion_v33.py | head -15`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-12 05:23:40 bash -->
### [Auto-Log] 2026-04-12 05:23:40 — Diffusion v3.3 Training — Started

**Command:** `cd /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller
cat logs/train_diffusion_v33_20260412_052326.log | tr -d '\r' | head -20`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-12 05:23:43 bash -->
### [Auto-Log] 2026-04-12 05:23:43 — Diffusion v3.3 Training — Started

**Command:** `ls -lh logs/train_diffusion_v33_20260412_052326.log && wc -l logs/train_diffusion_v33_20260412_052326.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-12 05:25:51 bash -->
### [Auto-Log] 2026-04-12 05:25:51 — Diffusion v3.3 Training — Started

**Command:** `cd /c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller
# Wait and recheck
ls -lt logs/train_diffusion_v33_*.log
# Check if python is running
tasklist 2>/dev/null | grep python || true
# Check new TB dir
ls -lt logs/diffusion_policy/ | head -3`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-12 05:25:58 bash -->
### [Auto-Log] 2026-04-12 05:25:58 — Diffusion v3.3 Training — Started

**Command:** `ls -lh logs/diffusion_policy/v33_20260412_052333/ && wc -l logs/train_diffusion_v33_20260412_052326.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-13 03:36:39 bash -->
### [Auto-Log] 2026-04-13 03:36:39 — DPPO v3.3 Training — Started

**Command:** `source dppo/Scripts/activate && BEST_CKPT="checkpoints/diffusion_policy/v33_20260412_052333/best_model.pt" && DPPO_LOG="logs/train_dppo_v33_$(date +%Y%m%d_%H%M%S).log" && echo "Launching Step 5..." && echo "pretrained: $BEST_CKPT" && echo "log: $DPPO_LOG" && nohup python -u -m scripts.train_dppo_v33 --pretrained "$BEST_CKPT" > "$DPPO_LOG" 2>&1 & echo "PID: $!"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-13 03:36:49 bash -->
### [Auto-Log] 2026-04-13 03:36:49 — DPPO v3.3 Training — Started

**Command:** `ls logs/train_dppo_v33_*.log 2>/dev/null && head -5 logs/train_dppo_v33_*.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-13 03:36:53 bash -->
### [Auto-Log] 2026-04-13 03:36:53 — DPPO v3.3 Training — Started

**Command:** `cat logs/train_dppo_v33_20260413_033639.log | tr -d '\r'`

**Output:**
```
(empty)
```

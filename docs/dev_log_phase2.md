# DPPO PID Controller — Phase 2: Expert Demonstration Collection

> Part of the dev log series. See [index](dev_log_phase2_3.md) for all phases.

---

## 1. Phase 2: Expert Demonstration Collection

### Objective

Collect (image, action, state) trajectories from the Run 6 PPO expert to serve as supervised training data for the Vision Diffusion Policy. The expert flies in `QuadrotorVisualEnv`, which wraps the base physics environment with a synthetic FPV renderer (64×64 RGB).

### Expert Chosen: Run 6

| Metric | Run 6 |
|--------|-------|
| X error | **0.0096m** |
| Y error | 0.0095m |
| Z error | 0.0684m |
| 3D RMSE | 0.0693m |
| Under 0.1m | 50/50 |
| Crashes | 0/50 |

Run 6 was selected over Runs 7–11 because it had the best X/Y accuracy and zero crashes. The Z bias (~6.8cm) is a consistent systematic offset — learnable and correctable via D²PPO fine-tuning in Phase 3b.

### Pre-Collection Environment Fixes

Before collection, two bugs were identified and fixed that would have silently corrupted the training data:

**Bug 1: `initial_z_range` fallback error**
- `_load_config()` had `self.initial_z_range = e.get('initial_z_range', self.initial_pos_range)` — wrong fallback
- When `initial_z_range` key was removed from YAML, the drone started with a ±10cm Z offset from the target
- This would introduce spurious altitude correction trajectories into the "hover" demonstration data
- **Fix:** Changed fallback to `0.0` → target_z equals init_z exactly

**Bug 2: YAML key not removed**
- `initial_z_range` key was still present in `quadrotor.yaml` from Run 10/11 experiments
- **Fix:** Removed key from config; verified `mean |Z init offset| = 0.000000m` ✓

### Collection Run

```
Episodes:     1,000
Steps/episode: 500 (10s at 50Hz)
Total steps:  500,000
Output:       data/expert_demos.h5  (90MB)
Duration:     ~33 minutes
```

**HDF5 structure per episode:**
```
episode_N/
  images   (500, 3, 64, 64)  uint8   FPV frames
  actions  (500, 4)           float32 motor thrusts ∈ [-1, 1]
  states   (500, 15)          float32 body-frame state observation
```

**Metadata:**
```
n_episodes:  1000
total_steps: 500,000
image_size:  64
state_dim:   15
action_dim:  4
```

---


---
<!-- auto-log 2026-04-11 19:11:56 edit -->
### [Auto-Log] 2026-04-11 19:11:56 — Script Fix

**File:** `scripts\collect_data.py`

**Before:**
```python
if args.v31 and args.v32:
        raise ValueError("--v31 and --v32 are mutually exclusive; pick one.")

    with_aux = args.v31 or args.v32
    if args.v31:
        print("v3.1 mode: saving imu_data (6D, finite-difference) + depth_maps")
    elif args.v32:
        print("v3.2 mode: saving imu_data (6D, physics-based) + depth_maps")
```

**After:**
```python
if args.v31 and args.v33:
        raise ValueError("--v31 and --v33 are mutually exclusive; pick one.")

    with_aux = args.v31 or args.v33
    if args.v31:
        print("v3.1 mode: saving imu_data (6D, finite-difference) + depth_maps")
    elif args.v33:
        print("v3.3 mode: saving imu_data (6D, physics-based, normalized) + depth_maps")
```

---
<!-- auto-log 2026-04-11 19:12:00 edit -->
### [Auto-Log] 2026-04-11 19:12:00 — Script Fix

**File:** `scripts\collect_data.py`

**Before:**
```python
# v3.2: physics-based IMU pulled straight from the env
                if args.v32:
```

**After:**
```python
# v3.3: physics-based normalized IMU pulled straight from the env
                if args.v33:
```

---
<!-- auto-log 2026-04-11 19:12:04 edit -->
### [Auto-Log] 2026-04-11 19:12:04 — Script Fix

**File:** `scripts\collect_data.py`

**Before:**
```python
hf.attrs['v31'] = args.v31
        hf.attrs['v32'] = args.v32
```

**After:**
```python
hf.attrs['v31'] = args.v31
        hf.attrs['v33'] = args.v33
```

---
<!-- auto-log 2026-04-11 19:12:08 edit -->
### [Auto-Log] 2026-04-11 19:12:08 — Script Fix

**File:** `scripts\collect_data.py`

**Before:**
```python
elif args.v32:
        fmt = "imu_data (physics) + depth_maps"
```

**After:**
```python
elif args.v33:
        fmt = "imu_data (physics, normalized) + depth_maps"
```

---
<!-- auto-log 2026-04-11 19:12:13 edit -->
### [Auto-Log] 2026-04-11 19:12:13 — Script Fix

**File:** `scripts\collect_data.py`

**Before:**
```python
parser.add_argument('--v32', action='store_true',
                        help='Enable v3.2 format: physics-based IMU via env.get_imu() + depth_maps')
```

**After:**
```python
parser.add_argument('--v33', action='store_true',
                        help='Enable v3.3 format: physics-based normalized IMU via env.get_imu() + depth_maps')
```

---
<!-- auto-log 2026-04-11 19:14:42 edit -->
### [Auto-Log] 2026-04-11 19:14:42 — Script Fix

**File:** `scripts\collect_data.py`

**Before:**
```python
# v3.2 collection (physics-based IMU via env.get_imu() + depth_maps):
    python -m scripts.collect_data --model checkpoints/ppo_expert/.../best_model.pt \
                                   --norm checkpoints/ppo_expert/.../best_obs_rms.npz \
                                   --output data/expert_demos_v32.h5 \
                                   --v32
```

**After:**
```python
# v3.3 collection (physics-based normalized IMU via env.get_imu() + depth_maps):
    python -m scripts.collect_data --model checkpoints/ppo_expert/.../best_model.pt \
                                   --norm checkpoints/ppo_expert/.../best_obs_rms.npz \
                                   --output data/expert_demos_v33.h5 \
                                   --v33
```

---
<!-- auto-log 2026-04-19 16:52:08 bash -->
### [Auto-Log] 2026-04-19 16:52:08 — Expert Data Collection

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && source dppo/Scripts/activate && python -m scripts.collect_data_v4 \
  --model  checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
  --norm   checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz \
  --output data/expert_demos_v4_test.h5 \
  --n-episodes 5 2>&1`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-19 16:52:31 bash -->
### [Auto-Log] 2026-04-19 16:52:31 — Expert Data Collection

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && source dppo/Scripts/activate && python -u -m scripts.collect_data_v4 \
  --model  checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
  --norm   checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz \
  --output data/expert_demos_v4.h5 \
  --n-episodes 1000 >> validation_results/collect_v4_log.txt 2>&1 &
echo "PID=$!"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-29 15:18:55 bash -->
### [Auto-Log] 2026-04-29 15:18:55 — Expert Data Collection

**Command:** `cd "/c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && grep -n "n_episodes\|max_episode_steps" scripts/collect_data.py`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-29 15:19:04 bash -->
### [Auto-Log] 2026-04-29 15:19:04 — Expert Data Collection

**Command:** `cd "/c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && grep -r "collect_data\|expert_demos" scripts/ --include="*.py" | grep -v ".pyc"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-29 15:19:32 bash -->
### [Auto-Log] 2026-04-29 15:19:32 — Expert Data Collection

**Command:** `cd "/c/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && grep -A 30 "collect_data" scripts/train_ppo_expert.py 2>/dev/null | head -40`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-29 16:03:52 bash -->
### [Auto-Log] 2026-04-29 16:03:52 — Expert Data Collection

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && source dppo/Scripts/activate && python -m scripts.collect_data_v4_approach \
    --model checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
    --norm  checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz \
    --n-episodes 9 --dry-run --deterministic-range --verbose`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-29 16:08:47 bash -->
### [Auto-Log] 2026-04-29 16:08:47 — Expert Data Collection

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && source dppo/Scripts/activate && python -m scripts.collect_data_v4_approach \
    --model checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
    --norm  checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz \
    --output data/expert_demos_v4_approach.h5 \
    --n-episodes 300 2>&1 | tee logs/collect_approach_run13.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-29 17:12:55 edit -->
### [Auto-Log] 2026-04-29 17:12:55 — Script Fix

**File:** `scripts\collect_data_v4_approach.py`

**Before:**
```python
APPROACH_RANGES = [1.0, 1.5, 2.0]


def collect_data(args):
    base_env = QuadrotorEnvV4(config_path=args.quadrotor_config)
    env = QuadrotorVisualEnv(base_env, image_size=args.image_size, dr_enabled=True)
```

**After:**
```python
APPROACH_RANGES = [1.0, 1.5, 2.0]


def _parse_csv_floats(s: str):
    return [float(x.strip()) for x in s.split(',') if x.strip()]


def collect_data(args):
    base_env = QuadrotorEnvV4(config_path=args.quadrotor_config)
    env = QuadrotorVisualEnv(base_env, image_size=args.image_size, dr_enabled=True)

    pos_ranges = _parse_csv_floats(args.pos_ranges) if args.pos_ranges else APPROACH_RANGES
    vel_ranges = _parse_csv_floats(args.vel_ranges) if args.vel_ranges else None
    print(f"pos_ranges (m): {pos_ranges}")
    print(f"vel_ranges (m/s): {vel_ranges if vel_ranges else f'config default ({base_env.initial_vel_range})'}")
```

---
<!-- auto-log 2026-04-29 17:13:03 edit -->
### [Auto-Log] 2026-04-29 17:13:03 — Script Fix

**File:** `scripts\collect_data_v4_approach.py`

**Before:**
```python
rng = np.random.default_rng(args.seed)
    total_steps = 0
    ep_lengths_by_range = {r: [] for r in APPROACH_RANGES}
    crashes_by_range = {r: 0 for r in APPROACH_RANGES}

    try:
        for ep in tqdm(range(args.n_episodes), desc="Approach episodes"):
            pos_range_now = APPROACH_RANGES[ep % len(APPROACH_RANGES)] \
                if args.deterministic_range \
                else float(rng.choice(APPROACH_RANGES))
            base_env.initial_pos_range = pos_range_now
```

**After:**
```python
rng = np.random.default_rng(args.seed)
    total_steps = 0
    ep_lengths_by_range = {r: [] for r in pos_ranges}
    crashes_by_range = {r: 0 for r in pos_ranges}

    try:
        for ep in tqdm(range(args.n_episodes), desc="Approach episodes"):
            pos_range_now = pos_ranges[ep % len(pos_ranges)] \
                if args.deterministic_range \
                else float(rng.choice(pos_ranges))
            base_env.initial_pos_range = pos_range_now
            if vel_ranges is not None:
                vel_range_now = vel_ranges[ep % len(vel_ranges)] \
                    if args.deterministic_range \
                    else float(rng.choice(vel_ranges))
                base_env.initial_vel_range = vel_range_now
```

---
<!-- auto-log 2026-04-29 17:13:09 edit -->
### [Auto-Log] 2026-04-29 17:13:09 — Script Fix

**File:** `scripts\collect_data_v4_approach.py`

**Before:**
```python
ep_grp.attrs['initial_pos_range'] = pos_range_now
```

**After:**
```python
ep_grp.attrs['initial_pos_range'] = pos_range_now
                if vel_ranges is not None:
                    ep_grp.attrs['initial_vel_range'] = base_env.initial_vel_range
```

---
<!-- auto-log 2026-04-29 17:13:15 edit -->
### [Auto-Log] 2026-04-29 17:13:15 — Script Fix

**File:** `scripts\collect_data_v4_approach.py`

**Before:**
```python
hf.attrs['action_space'] = 'ctbr'
            hf.attrs['version']      = 'v4_approach'
            hf.attrs['initial_pos_ranges'] = np.array(APPROACH_RANGES, dtype=np.float32)
```

**After:**
```python
hf.attrs['action_space'] = 'ctbr'
            hf.attrs['version']      = 'v4_approach'
            hf.attrs['initial_pos_ranges'] = np.array(pos_ranges, dtype=np.float32)
            if vel_ranges is not None:
                hf.attrs['initial_vel_ranges'] = np.array(vel_ranges, dtype=np.float32)
```

---
<!-- auto-log 2026-04-29 17:13:22 edit -->
### [Auto-Log] 2026-04-29 17:13:22 — Script Fix

**File:** `scripts\collect_data_v4_approach.py`

**Before:**
```python
parser.add_argument('--n-episodes', type=int, default=300)
```

**After:**
```python
parser.add_argument('--n-episodes', type=int, default=300)
    parser.add_argument('--pos-ranges', type=str, default=None,
                        help='Comma-separated initial_pos_range list (m). Default: 1.0,1.5,2.0')
    parser.add_argument('--vel-ranges', type=str, default=None,
                        help='Comma-separated initial_vel_range list (m/s). '
                             'If set, overrides config initial_vel_range per episode. '
                             'Use 1.0,1.5,2.0 for high-speed incoming demos.')
```

---
<!-- auto-log 2026-04-29 17:18:25 bash -->
### [Auto-Log] 2026-04-29 17:18:25 — Expert Data Collection

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && source dppo/Scripts/activate && python -m scripts.collect_data_v4_approach \
    --model checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
    --norm  checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz \
    --pos-ranges 1.0,1.5,2.0 \
    --vel-ranges 1.0,1.5,2.0 \
    --n-episodes 9 --dry-run --deterministic-range --verbose`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-29 17:21:17 bash -->
### [Auto-Log] 2026-04-29 17:21:17 — Expert Data Collection

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && source dppo/Scripts/activate && python -m scripts.collect_data_v4_approach \
    --model checkpoints/ppo_expert_v4/20260419_142245/best_model.pt \
    --norm  checkpoints/ppo_expert_v4/20260419_142245/best_obs_rms.npz \
    --pos-ranges 1.0,1.5,2.0 \
    --vel-ranges 1.0,1.5,2.0 \
    --output data/expert_demos_v4_approach.h5 \
    --n-episodes 300 2>&1 | tee logs/collect_incoming_run13.log`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-05-04 08:22:11 bash -->
### [Auto-Log] 2026-05-04 08:22:11 — Expert Data Collection

**Command:** `cd c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller && ls scripts/ | sort && echo "---" && ls scripts/collect_data_v4*.py scripts/merge*.py scripts/collect_data_v4_approach.py 2>/dev/null`

**Output:**
```
(empty)
```

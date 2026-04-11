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


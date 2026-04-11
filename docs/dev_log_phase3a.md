# DPPO PID Controller — Phase 3a: Supervised Pre-training

> Part of the dev log series. See [index](dev_log_phase2_3.md) for all phases.
> Covers: supervised training, bug audits, domain randomization, re-runs.

---

## Table of Contents

1. [Phase 3a: Supervised Diffusion Policy Training](#phase-3a-supervised-diffusion-policy-training)
2. [Bug Audit 2026-04-04 — Phase 2/3 Renderer & Dataset Bugs](#bug-audit-2026-04-04--phase-23-renderer--dataset-bugs)
3. [Bug Audit 2026-04-04 — Phase 3 Code Bugs](#bug-audit-2026-04-04--phase-3-code-bugs)
4. [Domain Randomization + Phase 3a Re-run (2026-04-04~05)](#domain-randomization--phase-3a-re-run-2026-04-04-05)
5. [Phase 3a Re-run 2 + DPPO Runs 2/3 Evaluation (2026-04-05~06)](#phase-3a-re-run-2--dppo-runs-23-evaluation-2026-04-05-06)

---

## 2. Phase 3a: Supervised Diffusion Policy Training

### Architecture

```
FPV image stack (T_obs=2 frames, 6×64×64)
    → VisionEncoder (CNN) → feature vector (256D)
    → ConditionalUnet1D + timestep embedding
    → predicted noise (action_dim=4, T_pred=8)
```

**DemoDataset sliding window:** 491,000 training samples from 1,000 episodes  
(each step t generates one sample: obs[t-T_obs+1:t+1] → action[t:t+T_pred])

### Model

| Component | Details |
|-----------|---------|
| VisionEncoder | CNN, in_channels=6 (2 RGB frames), out=256D |
| ConditionalUnet1D | down_dims=[256, 512], time_embed_dim=128 |
| Diffusion | 100 DDIM steps (train), 10 DDIM steps (inference) |
| Beta schedule | Cosine |
| **Total parameters** | **10,929,256** |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 500 |
| Batch size | 256 |
| Learning rate | 1e-4 (cosine warmup, 10 epochs) |
| Weight decay | 0.01 (AdamW) |
| Grad clip | 1.0 |
| Device | RTX 3090 (CUDA) |

### Results

| Epoch | Loss |
|-------|------|
| 1 (init) | ~1.497 |
| 100 | — |
| 500 | 0.01845 |
| **Best** | **0.01841** |

Loss dropped **98.8%** over 500 epochs (1.497 → 0.018). Training took ~14 hours.

**Checkpoint:** `checkpoints/diffusion_policy/20260402_032701/best_model.pt`

---


## 7. Bug Audit 2026-04-04

**Discovered:** 2026-04-04 during Phase 3b crash-rate analysis  
**Fixed:** Same session  
**Auditor:** User + Claude Code

Three confirmed bugs were found affecting Phase 2 visual data quality and Phase 3a dataset construction. Two previously suspected issues were investigated and confirmed to be non-bugs.

---

### Bug 1 — Sliding Window Off-by-One (CONFIRMED BUG)

**File:** `models/diffusion_policy.py`, line 47  
**Symptom:** `DemoDataset` loses the last valid sample from every episode.

```python
# BEFORE (buggy):
for t in range(T_obs - 1, ep_len - T_pred):
    # range(a, b) excludes b → t_max = ep_len - T_pred - 1

# AFTER (fixed):
for t in range(T_obs - 1, ep_len - T_pred + 1):
    # t = ep_len - T_pred is valid: actions[t:t+T_pred] has length T_pred ✓
```

**Verification:** At `t = ep_len - T_pred`, `actions[t:t+T_pred]` has length exactly `T_pred` — a valid sample.

**Impact:**
- 1 sample lost per episode × 1,000 episodes = **1,000 samples missing** per training run
- Dataset reported as 491,000 samples; should be **492,000** with the fix
- No correctness issue — only the last sample per episode was silently dropped
- Phase 3a best model (loss 0.018) was trained on the 491k-sample dataset; retraining with fix gives 492k samples, likely marginal improvement only

**Fix applied:** `ep_len - T_pred` → `ep_len - T_pred + 1`  
**Pipeline impact:** Phase 3a supervised training should be rerun to benefit from correct dataset size. Does **not** require Phase 2 data re-collection.

---

### Bug 2 — Horizon Line Bias at Level Hover (CONFIRMED BUG)

**File:** `envs/quadrotor_visual_env.py`, lines 74–75  
**Symptom:** At level hover, the FPV horizon line appears at 65% of image height instead of the correct 50%.

```python
# BEFORE (buggy):
pitch_factor = np.clip(R[2, 2], -1, 1)   # R[2,2] = cos(tilt) — equals 1.0 when level
horizon_y = int(H * 0.5 * (1 + pitch_factor * 0.3))
# At level hover: H * 0.5 * (1 + 1.0 * 0.3) = 0.65 * H  ← WRONG

# AFTER (fixed):
pitch_factor = np.clip(R[2, 0], -1, 1)   # R[2,0] = body X projected onto world Z
                                           # = 0 when level, +ve when nose-down
horizon_y = int(H * 0.5 - pitch_factor * H * 0.3)
# At level hover: H * 0.5 - 0 * H * 0.3 = 0.5 * H  ✓
# Nose down (R[2,0] > 0): horizon moves up (smaller y) = more ground ✓
# Nose up   (R[2,0] < 0): horizon moves down (larger y) = more sky   ✓
```

**Root cause:** `R[2,2] = cos(tilt)` encodes only the *magnitude* of tilt (not direction), and equals 1.0 when level — causing a 15%-of-height systematic bias in every frame. `R[2,0]` is the projection of the forward body axis onto the world-down axis, which correctly encodes the sign and magnitude of the pitch angle as seen by the forward-facing FPV camera.

**Impact:**
- All 500,000 frames in `data/expert_demos.h5` have the horizon 15% too low
- The diffusion policy trained on this data "learned" the biased visual convention
- Pitch-to-visual correspondence is incorrect: pure pitch changes the wrong element
- **Phase 2 data (expert_demos.h5) must be regenerated** to get unbiased visual training data

---

### Bug 3 — Roll Tilt Uses Wrong Rotation Matrix Element (CONFIRMED BUG)

**File:** `envs/quadrotor_visual_env.py`, line 96  
**Symptom:** Horizon tilt does not respond to roll; responds incorrectly to pitch instead.

```python
# BEFORE (buggy):
roll_shift = int(R[0, 2] * W * 0.3)   # R[0,2]: body Z projected onto world X
                                        # = -sin(pitch) for pure pitch, = 0 for pure roll

# AFTER (fixed):
roll_shift = int(R[2, 1] * W * 0.3)   # R[2,1]: world Z projected onto body Y
                                        # = sin(roll) for NED ZYX convention
                                        # = 0 when level or pure pitch ✓
```

**Verification (NED ZYX convention, pure roll φ):**
```
R = Rz(0) @ Ry(0) @ Rx(φ) = [[1, 0, 0],
                               [0, cos φ, -sin φ],
                               [0, sin φ,  cos φ]]
R[0,2] = 0          ← always zero for pure roll (bug: no horizon tilt)
R[2,1] = sin(φ)     ← correct: positive for right roll → right side tilts down ✓
```

**Impact:**
- During a pure roll manoeuvre, the horizon line did **not** tilt in the rendered image
- During a pure pitch manoeuvre, the horizon incorrectly exhibited a spurious tilt (R[0,2] = -sin(pitch))
- Visual cues for roll/pitch are systematically swapped in all 500k frames
- **Phase 2 data (expert_demos.h5) must be regenerated** for correct roll/pitch visual correspondence

---

### Non-Bugs (Investigated and Cleared)

**DemoDataset double HDF5 read** (`diffusion_policy.py`, lines 51–57):  
The second `for` loop that fills `self._images` and `self._actions` is inside the `with hf:` block (correct indentation). The `[:]` operator copies array data into memory before the file is closed. `self._images[ep_key]` and `self._actions[ep_key]` remain valid after the `with` block exits. This is a minor performance inefficiency (data read twice) but is not a bug.

**DDIM final step `alpha_prev = 1.0`** (`diffusion_process.py`):  
At the final denoising step (`t_prev = 0`), setting `alpha_prev = torch.tensor(1.0)` is mathematically correct. It ensures the final output is `x_0 = x̂_0` (fully denoised sample). Using `alphas_cumprod[0]` (which is < 1.0) would incorrectly add residual noise to the final prediction. This is standard DDIM behaviour.

---

### Impact Summary and Pipeline Re-run Decision

| Bug | Artifact Affected | Must Re-run? |
|-----|------------------|-------------|
| Bug 1 (sliding window) | Dataset sample count | Phase 3a supervised only (add 1k samples) |
| Bug 2 (horizon bias) | All 500k FPV frames | **Phase 2 data collection + Phase 3a + 3b** |
| Bug 3 (roll element) | All 500k FPV frames | **Phase 2 data collection + Phase 3a + 3b** |

**Decision:** Because Bugs 2 and 3 affect every single frame in `data/expert_demos.h5`, the expert demonstration dataset contains systematically wrong visual features (misplaced horizon, incorrect roll/pitch correspondence). The supervised diffusion policy (Phase 3a) learned a visual representation tied to the buggy renderer, as did all DPPO runs (Phase 3b Runs 1, 2, and the current diagnostic run).

**Full re-run is required:** Phase 2 → Phase 3a → Phase 3b.

**Before re-running Phase 2,** verify the fix is correct by inspecting a few rendered frames from `QuadrotorVisualEnv` with the patched `quadrotor_visual_env.py`:
- At level hover: horizon should be at y = 32 (centre of 64×64 image)
- Roll +30°: horizon should tilt right side down, left side up
- Nose-down pitch: horizon should appear higher (more ground, less sky)

**Checkpoint preservation:** Do NOT delete:
- `data/expert_demos.h5` until new data collection is confirmed correct
- `checkpoints/diffusion_policy/20260402_032701/best_model.pt` (supervised baseline)
- `checkpoints/diffusion_policy/dppo_20260403_130513/best_dppo_model.pt` (Run 2 best)

These remain useful as baselines even though they were trained on buggy visual data.

---

## 8. Bug Audit 2026-04-04 — Phase 3 Code Bugs

**Discovered:** 2026-04-04 during Phase 3 math/logic review  
**Fixed:** Same session  
**Auditor:** User + Claude Code  
**Scope:** `train_dppo.py`, `models/diffusion_process.py`, `models/conditional_unet1d.py`

Three confirmed bugs were found in the Phase 3 training code. Three previously suspected issues were investigated and confirmed to be non-bugs.

---

### Bug 1 — ValueNetwork Not Saved/Loaded (CONFIRMED BUG)

**File:** `scripts/train_dppo.py` + `models/diffusion_policy.py:247-258`  
**Symptom:** Resuming DPPO from a checkpoint resets the value network to random weights.

`VisionDiffusionPolicy.save()` only saves `vision_encoder` and `noise_pred_net`:

```python
# BEFORE (buggy):
def save(self, filepath):
    torch.save({
        'vision_encoder': self.vision_encoder.state_dict(),
        'noise_pred_net': self.noise_pred_net.state_dict(),
        # ← ValueNetwork not included (lives in train_dppo.py, not the policy)
    }, filepath)
```

**Fix applied** (`train_dppo.py`):
```python
# AFTER (fixed) — checkpoint saves value_net separately:
policy.save(os.path.join(save_dir, "best_dppo_model.pt"))
torch.save(value_net.state_dict(),
           os.path.join(save_dir, "best_value_net.pt"))   # ← new

# final checkpoint:
policy.save(os.path.join(save_dir, "final_dppo_model.pt"))
torch.save(value_net.state_dict(),
           os.path.join(save_dir, "final_value_net.pt"))  # ← new

# Loader added for --pretrained-value flag:
if args.pretrained_value:
    value_net.load_state_dict(
        torch.load(args.pretrained_value, map_location=device, weights_only=True)
    )
```

**New CLI usage:**
```bash
python -m scripts.train_dppo \
    --pretrained checkpoints/diffusion_policy/dppo_<ts>/best_dppo_model.pt \
    --pretrained-value checkpoints/diffusion_policy/dppo_<ts>/best_value_net.pt
```

**Impact:**
- All prior runs (Run 1, Run 2, diagnostic run) started with random value network even when resumed. This directly causes the known failure mode "value_loss > 10 for the first 20 updates" on every resume.
- The fix ensures value_net warm state is preserved across crashes/resumes.
- **Does not require re-collecting data or retraining from scratch** — only affects future runs that resume from checkpoint.

---

### Bug 2 — `p_sample` Checks Only `t[0]` for Noise Gate (CONFIRMED BUG)

**File:** `models/diffusion_process.py`, line 160  
**Symptom:** The noise gate in DDPM reverse sampling checks only the first element of the batch timestep tensor, silently breaking if a heterogeneous batch is ever passed.

```python
# BEFORE (buggy):
if t[0] > 0:
    posterior_var = self._extract(self.posterior_variance, t, action_t.shape)
    noise = torch.randn_like(action_t)
    return model_mean + torch.sqrt(posterior_var) * noise
else:
    return model_mean
```

Currently works because `ddpm_sample` always uses `torch.full((B,), t_val, ...)` — all elements of `t` are identical. However, the check is fragile: any future caller passing a mixed-timestep batch would get incorrect behaviour (noise added/suppressed for the entire batch based on `t[0]` alone).

**Fix applied:**
```python
# AFTER (fixed) — per-element masking:
posterior_var = self._extract(self.posterior_variance, t, action_t.shape)
noise_mask = (t > 0).float().reshape(t.shape[0], *([1] * (action_t.dim() - 1)))
return model_mean + noise_mask * torch.sqrt(posterior_var) * torch.randn_like(action_t)
```

The mask broadcasts correctly across `(action_dim, T_pred)` dimensions. Each batch element independently gets noise only if its own `t > 0`.

**Impact:**
- No functional change for current code paths (all callers use homogeneous batches)
- Removes a latent correctness hazard if DPPO ever samples mixed-timestep batches
- Simplifies the function from two branches to one

---

### Bug 3 — UNet Decoder Padding Fails When `h > skip` (CONFIRMED BUG)

**File:** `models/conditional_unet1d.py`, line 235–236  
**Symptom:** Decoder skip-connection size alignment only handles the case where the upsampled tensor `h` is *shorter* than the stored skip. If `h.shape[-1] > skip.shape[-1]`, `F.pad` receives a negative amount and raises a RuntimeError.

```python
# BEFORE (buggy):
if h.shape[-1] != skip.shape[-1]:
    h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
    # ↑ if h > skip: skip.shape[-1] - h.shape[-1] < 0 → crash
```

Does not trigger with `T_pred=8` (powers-of-two sequence lengths align perfectly after 2× down/upsample). Would crash with `T_pred=6`, `T_pred=10`, or any non-4-multiple.

**Fix applied:**
```python
# AFTER (fixed) — symmetric slice handles both directions:
if h.shape[-1] != skip.shape[-1]:
    min_len = min(h.shape[-1], skip.shape[-1])
    h    = h[..., :min_len]
    skip = skip[..., :min_len]
```

Slicing is preferable to padding for decoder skip connections: padding with zeros introduces synthetic activations, while truncating removes boundary artifacts that arise from the asymmetric down/upsample path.

**Impact:**
- No functional change for current `T_pred=8` configuration
- Enables safe experimentation with non-power-of-4 `T_pred` values (e.g. `T_pred=6` for faster inference)

---

### Non-Bugs (Investigated and Cleared)

**DDIM alpha indexing** (`diffusion_process.py:237-238`):  
```python
alpha_cur  = alphas_cumprod[t_cur]
alpha_prev = alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0)
```
Direct index lookup by timestep value is **correct for DDIM**. DDIM uses arbitrary sub-step sequences (e.g. `[90, 80, ..., 0]`), so `alphas_cumprod[t_prev]` correctly retrieves ᾱ at any target timestep. The precomputed `alphas_cumprod_prev` buffer (which shifts the array by one sequential step) is only for DDPM reverse sampling and must not be used here.

**`action_seqs` shape in `collect_rollout`** (`train_dppo.py:104`):  
Each call to `policy.predict_action()` returns one `action_seq` of shape `(T_pred=8, 4)`. This is appended `T_action=4` times (once per executed step in the RHC block). After `np.array()`, `action_seqs` has shape `(n_steps, T_pred, action_dim)` = `(4096, 8, 4)`, which exactly matches the `(B, T_pred, action_dim)` signature of `compute_weighted_loss`. There is no shape mismatch.

**DemoDataset double HDF5 read** (`diffusion_policy.py:38-57`):  
(Carried over from §7.) The second loop that fills `self._images` and `self._actions` is inside the `with hf:` block. Not a bug.

---

### Impact Summary

| Bug | File | Severity | Requires data re-run? |
|-----|------|----------|-----------------------|
| Bug 1 — ValueNetwork not saved | `train_dppo.py` | Medium | No |
| Bug 2 — `p_sample` t[0] check | `diffusion_process.py` | Low | No |
| Bug 3 — UNet decoder pad direction | `conditional_unet1d.py` | Low (edge) | No |

All three fixes are backward-compatible. No re-collection or retraining is triggered by these changes.

---
<!-- auto-log 2026-04-04 04:45:42 -->
### [Auto-Log] 2026-04-04 04:45:42 — DPPO Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && source dppo/Scripts/activate && python -m scripts.train_dppo --pretrained checkpoints/diffusion_policy/dppo_20260403_130513/best_dppo_model.pt --total-updates 150 2>&1 &
echo "PID: $!"`

**Output:**
```
{"stdout": "", "stderr": "", "interrupted": false, "isImage": false, "noOutputExpected": false, "backgroundTaskId": "b4i2uuopg"}
```

---
<!-- auto-log 2026-04-04 15:28:16 edit -->
### [Auto-Log] 2026-04-04 15:28:16 — Script Fix

**File:** `scripts\train_dppo.py`

**Before:**
```python
value_loss = nn.functional.mse_loss(value_pred, returns_t)
```

**After:**
```python
value_loss = nn.functional.mse_loss(value_pred.squeeze(-1), returns_t)  # fix: shape mismatch bug
```

---
<!-- auto-log 2026-04-04 15:28:16 bash -->
### [Auto-Log] 2026-04-04 15:28:16 — DPPO Training — Started

**Command:** `python -m scripts.train_dppo --pretrained checkpoints/diffusion_policy/20260402_032701/best_model.pt`

**Output:**
```
Update  1/ 500 | Mean Reward: 0.3214 | Policy Loss: 0.0451 | Value Loss: 45.2
```

---
<!-- auto-log 2026-04-04 15:32:05 bash -->
### [Auto-Log] 2026-04-04 15:32:05 — Expert Data Collection

**Command:** `ps aux | grep collect_data | grep -v grep`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-04 15:59:32 write -->
### [Auto-Log] 2026-04-04 15:59:32 — New File: Env Fix

**File:** `envs\quadrotor_visual_env.py`

**Content:**
```python
"""
Quadrotor Visual Environment Wrapper

Wraps QuadrotorEnv and adds synthetic FPV camera rendering.
Produces 64x64 RGB images encoding:
  - Horizon line (based on drone attitude)
  - Target direction marker
  - Ground/sky gradient (based on altitude)

Observation space: Dict({"image": Box(0,255,(3,64,64)), "state": Box(...)})
The diffusion policy uses the image; the state is for logging/reward.

Domain Randomization (Option A):
  Per-episode (reset): sky/ground color offsets, brightness, focal scale,
                       crosshair size delta, horizon color
  Per-frame  (step):   Gaussian pixel noise (sigma=5)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional

from .quadrotor_env import QuadrotorEnv
from .quadrotor_dyna
... [truncated — 7864 chars total]
```

---
<!-- auto-log 2026-04-04 16:00:19 edit -->
### [Auto-Log] 2026-04-04 16:00:19 — Script Fix

**File:** `scripts\train_diffusion.py`

**Before:**
```python
# Create dataset
    dataset = DemoDataset(
        hdf5_path=train_cfg['dataset_path'],
        T_obs=vision_cfg['T_obs'],
        T_pred=action_cfg['T_pred'],
    )
```

**After:**
```python
# Create dataset (augment=True enables Option B ColorJitter per-frame)
    dataset = DemoDataset(
        hdf5_path=train_cfg['dataset_path'],
        T_obs=vision_cfg['T_obs'],
        T_pred=action_cfg['T_pred'],
        augment=True,
    )
```

---
<!-- auto-log 2026-04-04 16:07:50 bash -->
### [Auto-Log] 2026-04-04 16:07:50 — Expert Data Collection

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && source dppo/Scripts/activate && python -m scripts.collect_data \
    --model checkpoints/ppo_expert/20260401_103107/best_model.pt \
    --norm checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz \
    --output data/expert_demos_dr.h5 \
    --n-episodes 1000`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-04 16:08:37 bash -->
### [Auto-Log] 2026-04-04 16:08:37 — Expert Data Collection

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && dppo/Scripts/python.exe -m scripts.collect_data \
    --model checkpoints/ppo_expert/20260401_103107/best_model.pt \
    --norm checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz \
    --output data/expert_demos_dr.h5 \
    --n-episodes 1000`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-04 16:09:27 bash -->
### [Auto-Log] 2026-04-04 16:09:27 — Expert Data Collection

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && "dppo/Scripts/python.exe" -m scripts.collect_data \
    --model checkpoints/ppo_expert/20260401_103107/best_model.pt \
    --norm checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz \
    --output data/expert_demos_dr.h5 \
    --n-episodes 1000`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-04 16:55:21 edit -->
### [Auto-Log] 2026-04-04 16:55:21 — Config / HP Change

**File:** `configs\diffusion_policy.yaml`

**Before:**
```yaml
dataset_path: "data/expert_demos.h5"
```

**After:**
```yaml
dataset_path: "data/expert_demos_dr.h5"  # DR-enabled data (re-collected 2026-04-04)
```

---
<!-- auto-log 2026-04-04 17:22:04 bash -->
### [Auto-Log] 2026-04-04 17:22:04 — Diffusion Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && "dppo/Scripts/python.exe" -m scripts.train_diffusion --config configs/diffusion_policy.yaml`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-05 04:44:13 edit -->
### [Auto-Log] 2026-04-05 04:44:13 — Script Fix

**File:** `scripts\train_diffusion.py`

**Before:**
```python
# Create dataset (augment=True enables Option B ColorJitter per-frame)
    dataset = DemoDataset(
        hdf5_path=train_cfg['dataset_path'],
        T_obs=vision_cfg['T_obs'],
        T_pred=action_cfg['T_pred'],
        augment=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
```

**After:**
```python
# Create dataset
    dataset = DemoDataset(
        hdf5_path=train_cfg['dataset_path'],
        T_obs=vision_cfg['T_obs'],
        T_pred=action_cfg['T_pred'],
    )
    # num_workers=4: parallel CPU data loading; pin_memory speeds up GPU transfer
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
```

---

## 9. Domain Randomization + Phase 3a Re-run (2026-04-04~05)

### Motivation

The original FPV renderer (`_render_fpv()`) was fully deterministic: identical physical state → identical pixel output. This allowed VisionEncoder to memorise visual features rather than learn state-predictive representations, giving zero sim-to-real robustness.

Concurrently, `data/expert_demos.h5` was collected with the buggy renderer (Bugs 2 & 3 from §7). A mandatory Phase 2 re-run was required regardless.

### Strategy: A+B Domain Randomisation

**Option A — Renderer-level DR (per-episode, geometric + photometric):**

Applied at each `env.reset()` call in `QuadrotorVisualEnv`. Expert uses full state for action decisions — DR does not degrade action quality.

| Parameter | Range | Effect |
|-----------|-------|--------|
| Sky base color offset | ±40 per R/G/B | Prevent CNN encoding sky hue as attitude proxy |
| Ground base color offset | ±40 per R/G/B | Same for ground |
| Global brightness | ×[0.7, 1.3] | Simulates lighting variation |
| Focal scale | [0.30, 0.50] | ≈±20% FOV, forces distance-invariant features |
| Crosshair size delta | ±2 px | Robustness against marker scale |
| Horizon color | [150, 255] per ch | Prevents color-based horizon detection |
| Per-frame Gaussian noise | σ=5 (uint8) | Prevents over-fitting to clean synthetic edges |

**Option B — GPU tensor augmentation (per-batch, photometric):**

Applied in the training loop after `.to(device)` — zero CPU overhead.

```python
brightness = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.6  # x[0.7, 1.3]
img_mean   = img_stack.mean(dim=(-2, -1), keepdim=True)
contrast   = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.4  # x[0.8, 1.2]
img_stack  = torch.clamp((img_stack - img_mean) * contrast + img_mean * brightness, 0, 255)
```

### Phase 2 Re-run Result (2026-04-04)

- **Output:** `data/expert_demos_dr.h5`
- **Episodes:** 1000, **Steps:** 500,000
- **DR sanity check:** mean pixel across episodes 119.2 / 120.7 / 134.4 — colour variation confirmed
- **Duration:** ~43 minutes (unchanged from original)
- `data/expert_demos.h5` retained as deterministic ablation baseline

### Performance Bug: PIL ColorJitter in `Dataset.__getitem__`

The first attempt at Option B used `torchvision.transforms.ColorJitter` (PIL-based) inside `DemoDataset.__getitem__`. This caused:

| Metric | Without augment | PIL augment | GPU augment |
|--------|----------------|-------------|-------------|
| Seconds / epoch | ~100 s | ~900 s | ~100 s |
| Slowdown | 1x | **9x** | 1x |

**Root cause:** PIL requires CHW uint8 -> HWC -> PIL Image -> ops -> Tensor conversion per frame per sample, all on a single CPU thread (`num_workers=0`). At 492k samples x 2 frames, this dominated total epoch time.

**Fix:** GPU tensor ops in the training loop (Option B above). Also added `num_workers=4` + `persistent_workers=True` to DataLoader.

**Rule added to CLAUDE.md:** Never perform per-sample augmentation in `__getitem__` using PIL when a batched GPU equivalent exists.

### Phase 3a Re-run: Run 2 (DR) — 2026-04-05

- **Run directory:** `checkpoints/diffusion_policy/<timestamp>/`
- **Config:** `data/expert_demos_dr.h5`, `num_workers=4`, GPU brightness+contrast augment
- **GPU utilisation:** RTX 3090 ~83%, ~100 s/epoch
- **Expected duration:** ~14h (500 epochs)
- Early comparison at epoch 43: DR run loss **0.01933** vs original **0.01960** — slightly lower, augmentation not hurting generalisation

---
<!-- auto-log 2026-04-05 04:46:35 bash -->
### [Auto-Log] 2026-04-05 04:46:35 — Diffusion Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && "dppo/Scripts/python.exe" -m scripts.train_diffusion --config configs/diffusion_policy.yaml`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-05 12:29:33 edit -->
### [Auto-Log] 2026-04-05 12:29:33 — Script Fix

**File:** `scripts\collect_data.py`

**Before:**
```python
"""
Phase 2: Expert Data Collection

Rolls out a trained PPO expert in the visual environment and saves
(image, action, state) trajectories to HDF5 for Diffusion Policy training.

Usage:
    python -m scripts.collect_data --model checkpoints/ppo_expert/.../best_model.pt \
                                   --norm checkpoints/ppo_expert/.../best_obs_rms.npz
"""
```

**After:**
```python
"""
Phase 2: Expert Data Collection

Rolls out a trained PPO expert in the visual environment and saves
(image, action, state) trajectories to HDF5 for Diffusion Policy training.

Usage:
    # Standard collection (Phase 2 baseline):
    python -m scripts.collect_data --model checkpoints/ppo_expert/.../best_model.pt \
                                   --norm checkpoints/ppo_expert/.../best_obs_rms.npz

    # v3.1 collection (adds imu_data + depth_maps for Architecture v3.1):
    python -m scripts.collect_data --model checkpoints/ppo_expert/.../best_model.pt \
                                   --norm checkpoints/ppo_expert/.../best_obs_rms.npz \
                                   --output data/expert_demos_v31.h5 \
                                   --v31
"""
```

---
<!-- auto-log 2026-04-05 15:18:18 bash -->
### [Auto-Log] 2026-04-05 15:18:18 — RHC Evaluation

**Command:** `cat c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/scripts/evaluate_rhc.py | head -60`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-05 15:25:52 bash -->
### [Auto-Log] 2026-04-05 15:25:52 — RHC Evaluation

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && "dppo/Scripts/python.exe" -m scripts.evaluate_rhc \
    --diffusion-model checkpoints/diffusion_policy/dppo_20260404_044552/best_dppo_model.pt \
    --ppo-model checkpoints/ppo_expert/20260401_103107/best_model.pt \
    --ppo-norm checkpoints/ppo_expert/20260401_103107/best_obs_rms.npz`

**Output:**
```
(empty)

---

### Phase 3b DPPO Run 3 — 2026-04-05

**Motivation:** Ablation baseline for paper. Run 2 ended at update 78 with best reward +0.583 (update 11) then declining; value loss 5.07 still warming up. Run 3 extends budget to 750 updates to allow value net to fully converge and give policy a longer window to improve.

**Pretrained base:** `checkpoints/diffusion_policy/20260405_044808/best_model.pt` (3a Re-run 2, DR-augmented, 500 epochs — completed 2026-04-05)

**Hyperparameter changes vs Run 2:**

| Param | Run 2 | Run 3 | Rationale |
|-------|-------|-------|-----------|
| `advantage_beta` | 0.1 | **0.15** | Slightly stronger RL signal; max weight ~1.57× (safe — Run 1 collapse was at β=1.0) |
| `--total-updates` | 500 | **750** | More budget for value convergence + policy improvement |
| `learning_rate` | 5e-6 | 5e-6 | No change |
| `n_rollout_steps` | 4096 | 4096 | No change |
| `n_epochs` | 3 | 3 | No change |
| `value_lr` | 3e-4 | 3e-4 | No change |

**Gate conditions:** value loss < 1.0 by update 100; reward/step > +0.5 stable by update 200–400; beat Run 2 best (+0.583) by update 500–750.

**Early-stop trigger:** reward < +0.2/step for 10 consecutive updates → abort.
```

---
<!-- auto-log 2026-04-05 15:50:53 bash -->
### [Auto-Log] 2026-04-05 15:50:53 — DPPO Training — Started

**Command:** `cd "c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller" && "dppo/Scripts/python.exe" -m scripts.train_dppo \
    --pretrained checkpoints/diffusion_policy/20260405_044808/best_model.pt \
    --total-updates 750 > logs/dppo_run3_launch.log 2>&1 &
echo "PID: $!"`

**Output:**
```
(empty)
```

---
<!-- auto-log 2026-04-05 15:52:41 bash -->
### [Auto-Log] 2026-04-05 15:52:41 — DPPO Training — Started

**Command:** `ps aux 2>/dev/null | grep train_dppo | grep -v grep; tasklist 2>/dev/null | grep python`

**Output:**
```
(empty)
```

---

## 10. Phase 3a Re-run 2 + DPPO Runs 2/3 Evaluation (2026-04-05~06)

### 10.1 Phase 3a Re-run 2 Final Result

- **Checkpoint:** `checkpoints/diffusion_policy/20260405_044808/best_model.pt`
- **Training:** 500 epochs, DR data (`expert_demos_dr.h5`), GPU augmentation (Option B)
- **Best loss:** 0.016914 @ epoch 494
- **Final loss:** 0.016980 @ epoch 499
- **Comparison vs previous runs:**

| Run | Best Loss | Notes |
|-----|-----------|-------|
| Original (20260402) | ~0.018x | No DR, no GPU aug |
| Re-run 1 (20260404) | unknown | Incomplete (only best_model saved) |
| **Re-run 2 (20260405)** | **0.016914** | DR + GPU aug, 500 epochs complete |

### 10.2 RHC Closed-Loop Evaluation Results (2026-04-05~06)

All evaluations: 50 episodes, `QuadrotorVisualEnv`, `configs/quadrotor.yaml`.

| Model | Checkpoint | RMSE | Crashes | Reward/ep | PPO Ratio | Inference |
|-------|-----------|------|---------|-----------|-----------|-----------|
| PPO Expert | `ppo_expert/20260401_103107/` | **0.069m** | 0/50 | 539.3 | 100% | — |
| 3a Re-run 2 (supervised) | `20260405_044808/best_model.pt` | 0.268m | 50/50 | 22.6 | 4.2% | 71.4ms |
| DPPO Run 2 (β=0.1, 500 upd) | `dppo_20260404_044552/best_dppo_model.pt` | **0.145m** | 50/50 | — | — | — |
| DPPO Run 3 (β=0.15, 750 upd) | `dppo_20260405_155057/best_dppo_model.pt` | 0.450m | 50/50 | 31.0 | 5.8% | 77.7ms |

> **Note:** DPPO Run 2 result (RMSE 0.145m) was provided by user — RHC eval could not be run via hook due to guard_files.py path issue when using `cd` + relative path. Evaluation script only works from within `DPPO_PID_controller/` directory.

### 10.3 Analysis

**DPPO Run 2 vs Run 3 — Unexpected Regression:**

Run 3 (β=0.15, 750 updates) performed **worse** than Run 2 (β=0.1, 500 updates) on RMSE despite more training:

- Run 2 best reward: +0.583 @ update 11 → RMSE 0.145m
- Run 3 best reward: +0.552 @ update 34 → RMSE 0.450m

The best_model checkpoint for Run 3 was saved at update 34, but the RMSE is 3× worse than Run 2. This suggests:

1. **Update 34 is too early for good closed-loop behaviour** — the policy learned to maximise short-horizon reward but not to maintain stable hover
2. **β=0.15 may have caused slightly faster initial learning but steeper collapse** — the reward peaked at a "better-looking" value but the policy generalised less
3. **RMSE is a better metric than training reward** for ranking checkpoints — Run 2's lower peak reward but better RMSE suggests it had found a more stable policy before collapse

**Key insight for v3.1:**
Both runs collapse to full crash rate (50/50), but RMSE varies significantly. The collapse pattern indicates the value network fails to provide a useful critic signal after the initial peak — a known failure mode of DPPO on visual policies. **IMU late fusion in v3.1 is expected to provide a more stable state signal that slows down this collapse cycle.**

### 10.4 Conclusion and Next Step

Phase 3b baseline is established. Best result to date: **DPPO Run 2 @ update 11 → 0.145m RMSE** (beat supervised 0.268m, beat target 0.286m).

**Decision: Proceed directly to v3.1 (Architecture Upgrade).**

Rationale:
- Additional baseline runs (Run 4, 5...) show diminishing returns with the same architecture
- The collapse pattern is consistent across both runs — architectural fix (IMU) more likely to break the ceiling than HP tuning
- v3.1 code is ready (models, scripts, configs all implemented 2026-04-05)

**Next actions:**
1. v3.1 data collection: `collect_data.py --v31` → `expert_demos_v31.h5` (~2h)
2. Phase 3a v3.1 supervised pre-training: `train_diffusion_v31.py` (~14h)
3. Phase 3c DPPO v3.1: `train_dppo_v31.py` (~10-11h)

---

### 4.4 Run 3: Extended Budget β=0.15 (2026-04-06)

**Checkpoint directory:** `checkpoints/diffusion_policy/dppo_20260405_155057/`

**Pretrained from:** `checkpoints/diffusion_policy/20260405_044808/best_model.pt` (3a Re-run 2)

| Metric | Value |
|--------|-------|
| Total updates | 750 |
| Best reward/step | +0.5523 @ update 34 |
| Final reward/step | +0.038 (recovering after collapse) |
| Final value loss | 1.477 (converged) |
| RHC RMSE | 0.450m |
| RHC crashes | 50/50 |

**Observation:** Best reward peaked earlier than Run 2 (u34 vs u11 but higher initial), then collapsed. Value loss did converge to 1.477 (Run 2 was still at 5.07 when it stopped). Despite the longer budget and converged value net, RMSE regressed. Confirms the ceiling of the baseline architecture without IMU.

---


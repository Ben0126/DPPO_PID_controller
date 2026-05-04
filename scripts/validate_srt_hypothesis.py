"""
SRT Root-Cause Validation Experiment
=====================================
Two tests in one script:

Test A -- INDI Angular Stability  (--test indi_hover)
    Runs QuadrotorEnvV4 with hardcoded hover CTBR (F_c=hover, omega=0),
    external disturbances DISABLED to isolate INDI.
    Verifies that tilt stays < 5 deg and angular rates < 0.5 rad/s.
    Note: translational drift is expected (no outer position loop).
    Phase 0 gate: required before any PPO training.

Test B -- BC-Vision-MLP RHC  (--test bc_mlp)
    Trains a single-step BC policy (VisionEncoder + MLP -> SRT action) on
    existing expert_demos_v33.h5 for N epochs.  Evaluates the trained policy
    in RHC closed-loop on QuadrotorEnv (v3.3 SRT environment).

    Hypothesis check:
      crash_rate ~= 50/50 -> covariate shift is the root cause; SRT / CTBR
                             restructure is justified by learning stability,
                             NOT by action-space physics.
      crash_rate << 50/50 -> inference latency was a key factor; CTBR
                             restructure may be less critical.

Usage:
    python -m scripts.validate_srt_hypothesis --test indi_hover
    python -m scripts.validate_srt_hypothesis --test bc_mlp --epochs 30
    python -m scripts.validate_srt_hypothesis --test both --epochs 30
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env import QuadrotorEnv
from envs.quadrotor_env_v4 import QuadrotorEnvV4
from envs.quadrotor_visual_env import QuadrotorVisualEnv
from models.vision_encoder import VisionEncoder


# ============================================================
# BC-Vision-MLP Model
# ============================================================

class IMUEncoder(nn.Module):
    def __init__(self, in_dim: int = 6, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.Mish(),
            nn.Linear(64, out_dim), nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BCVisionMLP(nn.Module):
    """
    Single-step BC policy: FPV images + IMU -> SRT motor commands.
    Inference latency ~= 1 ms (vs 74 ms for 10-step DDIM).
    """

    def __init__(self, T_obs: int = 2, feature_dim: int = 256,
                 imu_dim: int = 32, action_dim: int = 4):
        super().__init__()
        self.vision = VisionEncoder(in_channels=T_obs * 3, feature_dim=feature_dim)
        self.imu_enc = IMUEncoder(in_dim=6, out_dim=imu_dim)
        cond_dim = feature_dim + imu_dim

        self.action_head = nn.Sequential(
            nn.Linear(cond_dim, 256), nn.Mish(),
            nn.Linear(256, 128),     nn.Mish(),
            nn.Linear(128, action_dim),
            nn.Tanh(),   # SRT normalized to [-1, 1]
        )

    def forward(self, images: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        vis  = self.vision(images)
        imu_ = self.imu_enc(imu)
        return self.action_head(torch.cat([vis, imu_], dim=-1))

    @torch.no_grad()
    def predict(self, images: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(images, imu)


# ============================================================
# Dataset
# ============================================================

class ExpertDatasetV33(Dataset):
    """
    Sliding-window dataset over expert_demos_v33.h5.
    Each sample: (stacked T_obs frames, imu, action).
    Action is the expert SRT action at the LAST observation step.
    """

    def __init__(self, h5_path: str, T_obs: int = 2, max_episodes: int = None):
        self.T_obs   = T_obs
        self.samples = []

        with h5py.File(h5_path, 'r') as f:
            ep_keys = sorted(f.keys())
            if max_episodes:
                ep_keys = ep_keys[:max_episodes]

            for ek in ep_keys:
                ep      = f[ek]
                images  = ep['images'][:]    # (T, 3, H, W) uint8
                actions = ep['actions'][:]   # (T, 4)
                imu     = ep['imu_data'][:] # (T, 6)
                T       = len(images)

                for t in range(T_obs - 1, T):
                    stack = np.concatenate(images[t - T_obs + 1:t + 1], axis=0)
                    self.samples.append((stack, imu[t], actions[t]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stack, imu, action = self.samples[idx]
        return (
            torch.from_numpy(stack).float() / 255.0,
            torch.from_numpy(imu).float(),
            torch.from_numpy(action).float(),
        )


# ============================================================
# Test A: INDI Angular Stability
# ============================================================

def test_indi_hover(config_path: str = "configs/quadrotor_v4.yaml",
                    n_episodes: int = 10, max_steps: int = 500) -> dict:
    """
    Hardcode hover CTBR (F_c=hover, omega=0), disable external disturbances,
    and verify angular stability for 10 s.

    A rate-only inner-loop has no position control, so the drone will drift
    translationally due to initial velocity -- this is expected and is NOT a
    failure criterion.  What we verify is that INDI keeps the drone angularly
    stable (tilt and angular-rate magnitudes stay small).

    Phase 0 gate:
        mean_max_tilt < 5 deg  AND  mean_max_omega < 0.5 rad/s
    """
    from envs.quadrotor_dynamics import get_tilt_angle

    print("\n" + "=" * 60)
    print("TEST A: INDI Angular Stability (disturbances disabled)")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Episodes: {n_episodes}  Steps: {max_steps} (10 s @ 50 Hz)")

    env = QuadrotorEnvV4(config_path=config_path)
    env.disturbance_enabled = False   # isolate INDI from external force/torque

    F_hover  = env.dynamics.params.mass * env.dynamics.params.gravity  # 4.905 N
    F_c_norm = float((F_hover / env.F_c_max) * 2.0 - 1.0)
    hover_act = np.array([F_c_norm, 0.0, 0.0, 0.0], dtype=np.float32)

    print(f"Hover F_c = {F_hover:.3f} N  |  F_c_norm = {F_c_norm:.4f}")
    print("Disturbances: DISABLED (isolating INDI from translational perturbations)")

    results = {
        'max_tilt_deg': [], 'max_omega': [], 'tilt_failed': 0,
        'survived': 0, 'step_counts': []
    }

    TILT_LIMIT  = 5.0   # deg
    OMEGA_LIMIT = 0.5   # rad/s

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        max_tilt  = 0.0
        max_omega = 0.0
        tilt_fail = False

        for step in range(max_steps):
            obs, _, terminated, truncated, info = env.step(hover_act)
            R     = env.dynamics.get_rotation_matrix()
            tilt  = get_tilt_angle(R)
            omega = float(np.linalg.norm(env.dynamics.ang_velocity))
            max_tilt  = max(max_tilt, tilt)
            max_omega = max(max_omega, omega)

            if tilt > TILT_LIMIT:
                tilt_fail = True
                results['tilt_failed'] += 1
                results['step_counts'].append(step + 1)
                break

            if terminated:   # drift out of bounds -- note but don't fail
                results['step_counts'].append(step + 1)
                break
        else:
            results['step_counts'].append(max_steps)
            results['survived'] += 1

        results['max_tilt_deg'].append(max_tilt)
        results['max_omega'].append(max_omega)

        status = ("TILT_FAIL" if tilt_fail
                  else f"steps={results['step_counts'][-1]}")
        print(f"  Ep {ep+1:>2}/{n_episodes} | {status} | "
              f"max_tilt={max_tilt:.2f}deg | max_omega={max_omega:.3f}rad/s")

    mean_tilt  = float(np.mean(results['max_tilt_deg']))
    mean_omega = float(np.mean(results['max_omega']))
    print(f"\n  Mean max tilt:  {mean_tilt:.2f} deg  (gate < 5 deg)")
    print(f"  Mean max omega: {mean_omega:.3f} rad/s (gate < 0.5 rad/s)")

    passed = mean_tilt < 5.0 and mean_omega < 0.5
    gate   = "PASS" if passed else "FAIL"
    print(f"\n  Phase 0 Gate: {gate}")
    if not passed:
        print("  -> Increase kp_roll/kp_pitch or check mixer_inv in quadrotor_env_v4.py.")
    else:
        print("  -> INDI angularly stable. Proceed to Phase 1 (PPO Expert CTBR).")

    return results


# ============================================================
# Test B: BC-Vision-MLP (SRT hypothesis)
# ============================================================

def train_bc_mlp(h5_path: str, device: torch.device,
                 T_obs: int = 2, epochs: int = 30,
                 batch_size: int = 256, lr: float = 1e-3,
                 max_episodes: int = 200) -> BCVisionMLP:
    """Train BC-Vision-MLP on SRT expert data."""
    print("\n" + "=" * 60)
    print("TEST B: BC-Vision-MLP Training (SRT action space)")
    print("=" * 60)
    print(f"Data: {h5_path}  (max {max_episodes} episodes)")
    print(f"Epochs: {epochs}  Batch: {batch_size}  LR: {lr}")

    print("Building dataset ...", end=' ', flush=True)
    t0      = time.time()
    dataset = ExpertDatasetV33(h5_path, T_obs=T_obs, max_episodes=max_episodes)
    print(f"{len(dataset)} samples in {time.time()-t0:.1f}s")

    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=4, pin_memory=True, persistent_workers=True)
    model   = BCVisionMLP(T_obs=T_obs).to(device)
    opt     = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for images, imu, actions in loader:
            images  = images.to(device, non_blocking=True)
            imu     = imu.to(device,    non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            pred    = model(images, imu)
            loss    = loss_fn(pred, actions)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        if ep % 5 == 0 or ep == 1:
            print(f"  Epoch {ep:>3}/{epochs} | loss={np.mean(losses):.6f} | "
                  f"lr={sched.get_last_lr()[0]:.2e}")

    return model


def evaluate_bc_mlp(model: BCVisionMLP, env: QuadrotorVisualEnv,
                    base_env: QuadrotorEnv, device: torch.device,
                    n_episodes: int = 50, T_obs: int = 2) -> dict:
    """Evaluate BC-MLP in closed-loop RHC (single-step, no action sequence)."""
    print(f"\nEvaluating BC-MLP ({n_episodes} episodes, single-step inference) ...")
    model.eval()

    results = {'crashes': 0, 'rmse': [], 'lengths': [], 'inference_ms': []}

    for ep in range(n_episodes):
        obs, _       = env.reset()
        image_buf    = [obs['image']] * T_obs
        positions, targets = [], []
        done = False

        while not done:
            stack = np.concatenate(image_buf[-T_obs:], axis=0)
            img_t = torch.from_numpy(stack).float().unsqueeze(0).to(device) / 255.0
            imu_v = base_env.get_imu()
            imu_t = torch.from_numpy(imu_v).float().unsqueeze(0).to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                action = model.predict(img_t, imu_t).squeeze(0).cpu().numpy()
            results['inference_ms'].append((time.perf_counter() - t0) * 1000)

            obs, _, terminated, truncated, info = env.step(action)
            image_buf.append(obs['image'])
            positions.append(info['position'].copy())
            targets.append(info['target'].copy())
            done = terminated or truncated

        ep_len = len(positions)
        results['lengths'].append(ep_len)
        crashed = ep_len < base_env.max_episode_steps
        if crashed:
            results['crashes'] += 1

        pos_err = np.array(targets) - np.array(positions)
        rmse    = np.sqrt(np.mean(np.sum(pos_err**2, axis=1)))
        results['rmse'].append(rmse)

        print(f"  Ep {ep+1:>3}/{n_episodes} | "
              f"steps={ep_len} | RMSE={rmse:.4f} m | "
              f"{'CRASH' if crashed else 'OK'}")

    return results


def test_bc_mlp(h5_path: str, quadrotor_config: str,
                device: torch.device, epochs: int = 30,
                n_episodes: int = 50, max_episodes: int = 200) -> dict:
    """Full BC-MLP validation pipeline."""
    model      = train_bc_mlp(h5_path, device, epochs=epochs,
                               max_episodes=max_episodes)
    base_env   = QuadrotorEnv(config_path=quadrotor_config)
    visual_env = QuadrotorVisualEnv(base_env, image_size=64, dr_enabled=True)
    results    = evaluate_bc_mlp(model, visual_env, base_env, device,
                                 n_episodes=n_episodes)

    n = n_episodes
    print(f"\n--- BC-Vision-MLP Results ({n} episodes) ---")
    print(f"  Crashes:   {results['crashes']}/{n}")
    print(f"  Mean RMSE: {np.mean(results['rmse']):.4f} m")
    print(f"  Inference: {np.mean(results['inference_ms']):.2f} ms "
          f"(median {np.median(results['inference_ms']):.2f} ms)")

    crash_pct = results['crashes'] / n * 100
    print("\n--- Hypothesis Evaluation ---")
    if crash_pct >= 80:
        verdict = (
            "COVARIATE SHIFT is the dominant failure mode.\n"
            "  BC-MLP crashes at similar rate despite ~1ms inference.\n"
            "  CTBR restructure is still justified (INDI + hardware compat)\n"
            "  but won't alone fix crashes. ReinFlow RL is the critical piece."
        )
    elif crash_pct >= 30:
        verdict = (
            "MIXED: both inference latency and covariate shift contribute.\n"
            f"  BC-MLP crash rate = {crash_pct:.0f}% (diffusion = ~100%).\n"
            "  Faster inference helps but doesn't fully fix. Both CTBR + RL needed."
        )
    else:
        verdict = (
            "INFERENCE LATENCY was a major factor.\n"
            f"  BC-MLP crash rate = {crash_pct:.0f}% -- much lower than diffusion.\n"
            "  Consider Flow Matching 1-step BEFORE committing to full CTBR restructure."
        )

    print(f"  Verdict: {verdict}")
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SRT Root-Cause Validation")
    parser.add_argument('--test', choices=['indi_hover', 'bc_mlp', 'both'],
                        default='both')
    parser.add_argument('--v4-config',   default='configs/quadrotor_v4.yaml')
    parser.add_argument('--v33-config',  default='configs/quadrotor.yaml')
    parser.add_argument('--h5-path',     default='data/expert_demos_v33.h5')
    parser.add_argument('--epochs',      type=int, default=30)
    parser.add_argument('--n-episodes',  type=int, default=50)
    parser.add_argument('--max-demos',   type=int, default=200,
                        help='Max episodes from h5 (200 ep = ~99k samples)')
    parser.add_argument('--indi-eps',    type=int, default=10)
    parser.add_argument('--output-json', default='validation_results.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    all_results = {}

    if args.test in ('indi_hover', 'both'):
        all_results['indi_hover'] = test_indi_hover(
            config_path=args.v4_config,
            n_episodes=args.indi_eps,
        )

    if args.test in ('bc_mlp', 'both'):
        all_results['bc_mlp'] = test_bc_mlp(
            h5_path          = args.h5_path,
            quadrotor_config = args.v33_config,
            device           = device,
            epochs           = args.epochs,
            n_episodes       = args.n_episodes,
            max_episodes     = args.max_demos,
        )

    os.makedirs('validation_results', exist_ok=True)
    out_path = os.path.join('validation_results', args.output_json)
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {
            kk: (vv.tolist() if hasattr(vv, 'tolist') else
                 [float(x) for x in vv] if isinstance(vv, list) else vv)
            for kk, vv in v.items()
        }
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()

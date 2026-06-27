"""
Phase 3 v7.0 — PID-CTBR demonstration collection for the Teacher x Observation 2x2.

RESEARCH_PLAN_v7's decisive ablation needs four datasets matching the 2x2:

    factor T (teacher coverage): T0 = hover only (target == init)
                                 T1 = far 1-3 m recovery (target != init)
    factor O (observation):      O0 = crosshair  (production, saturating)
                                 O1 = perspective (non-saturating disk, far-range cue)

This collector builds the building blocks of that 2x2 with a SINGLE teacher
(PID-CTBR, gentle recovery gains) for BOTH hover and far recovery, so the T-axis is
a pure coverage operation with no teacher swap (user decision 2026-06-25, confirmed
by Gate A: gentle PID-CTBR recovers the full 1-4 m band at 100% survival /
cond-IAE 0.14-0.18 m -- see docs/experiment_report_p0_teacher_renderer_gates.md).

CLEAN O-SWAP: each (state, target) trajectory is DUAL-RENDERED at every step with both
the crosshair and the perspective marker from the SAME dynamics state and -- crucially --
the SAME per-frame DR noise (the global np.random state is saved/restored around the two
renders), so the O0 / O1 images differ ONLY in the target marker. The trajectory
(states, actions, imu) is byte-identical across O0 and O1.

Two h5 files per mode (rendered in one pass):

    --mode hover -> data/expert_demos_v7_hover_crosshair.h5  (T0O0)
                    data/expert_demos_v7_hover_persp.h5       (T0O1)
    --mode far   -> data/expert_demos_v7_far_crosshair.h5     (T1O0)
                    data/expert_demos_v7_far_persp.h5          (T1O1)

h5 format matches collect_data_v4_recovery.py (per-episode group: images, actions,
states, imu_data). hover episodes carry attr episode_type='hover' and NO init_tilt_deg
so FlowDatasetV5 keeps task-cond=[1,0]; far episodes carry episode_type='recovery'
(+ offset metadata) so the task-cond=[0,1] recovery flag fires (train_flow_v5.py:86-91).

Teacher (Gate A gentle tune):
    CascadePIDController(env.dynamics.params, omega_max=6.0, dt=env.dt_outer,
                         vel_max=1.0, Kp_pos=0.8) -> compute_ctbr_action

Usage (ALWAYS via dppo/Scripts/python.exe + Bash run_in_background=true):
    dppo/Scripts/python.exe -m scripts.collect_data_v7_pidctbr --mode hover --n-episodes 500
    dppo/Scripts/python.exe -m scripts.collect_data_v7_pidctbr --mode far   --n-episodes 500
    # smoke test (small, *_smoke.h5 outputs -- does not clobber the real datasets):
    dppo/Scripts/python.exe -m scripts.collect_data_v7_pidctbr --mode far --n-episodes 5 --smoke
"""

import os
import sys
import argparse
import numpy as np
import h5py
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.quadrotor_env_v4 import QuadrotorEnvV4
from envs.quadrotor_visual_env import QuadrotorVisualEnv
from envs.quadrotor_dynamics import get_tilt_angle
from controllers.pid_controller import CascadePIDController


def dual_render(vis_env):
    """Render the CURRENT dynamics state with both target markers under identical
    per-frame DR noise. The global np.random state is saved before the crosshair
    render and restored before the perspective render, so the only pixel difference
    between the two images is the marker itself (clean O swap)."""
    rng_state = np.random.get_state()
    vis_env.target_render = "crosshair"
    img_cross = vis_env._render_fpv()
    np.random.set_state(rng_state)
    vis_env.target_render = "perspective"
    img_persp = vis_env._render_fpv()
    return img_cross, img_persp


def sample_offset(rng, dist_min, dist_max):
    """Horizontal setpoint offset of random magnitude in [dist_min, dist_max] at a
    random azimuth (z=0, so the target altitude matches the near-origin hover start)."""
    dist = rng.uniform(dist_min, dist_max)
    phi = rng.uniform(-np.pi, np.pi)
    return dist * np.array([np.cos(phi), np.sin(phi), 0.0])


def collect(args):
    mode = args.mode
    is_far = (mode == "far")

    base_env = QuadrotorEnvV4(config_path=os.path.join(ROOT, args.quadrotor_config))
    vis_env = QuadrotorVisualEnv(base_env, image_size=args.image_size,
                                 dr_enabled=True, target_render="crosshair",
                                 physical_size=args.physical_size)

    # ---- Init-mode setup ------------------------------------------------------
    # hover: normal reset (initial_pos_range=0.1, target==init) -> the drone holds,
    #        position error stays ~0 throughout (T0: no 1-3 m coverage).
    # far:   anchor init near origin (hover_anchor_prob=1.0, +-0.1 m) so a 1-3 m
    #        offset target stays inside the |pos|<5 m bound; target = init + offset
    #        (T1: the 1-3 m recovery band is the START position error).
    base_env.swift_perturbation_prob = 0.0
    base_env.hover_anchor_prob = 1.0 if is_far else 0.0

    # ---- Teacher: single PID-CTBR, gentle recovery gains (Gate A) -------------
    ctrl = CascadePIDController(base_env.dynamics.params,
                               omega_max=6.0, dt=base_env.dt_outer,
                               vel_max=args.pid_vel_max, Kp_pos=args.pid_kp_pos)

    # ---- Reproducible RNG -----------------------------------------------------
    np.random.seed(args.seed)                       # DR (per-episode + per-frame noise)
    rng_off = np.random.default_rng(args.seed + 777)  # far-offset stream (independent)

    # ---- Output files (one crosshair + one perspective per mode) -------------
    suffix = "_smoke" if args.smoke else ""
    out_cross = os.path.join(ROOT, args.out_dir,
                             f"{args.out_prefix}_{mode}_crosshair{suffix}.h5")
    out_persp = os.path.join(ROOT, args.out_dir,
                             f"{args.out_prefix}_{mode}_persp{suffix}.h5")
    os.makedirs(os.path.dirname(out_cross), exist_ok=True)

    state_dim = base_env.observation_space.shape[0]
    action_dim = base_env.action_space.shape[0]

    print(f"[v7 collect] mode={mode}  teacher=PID-CTBR(vel_max={args.pid_vel_max}, "
          f"Kp_pos={args.pid_kp_pos})  n_episodes={args.n_episodes}")
    print(f"  init mode: {'anchor (|pos|<=0.1m) + setpoint_offset' if is_far else 'normal hover (target==init)'}")
    if is_far:
        print(f"  far offset: |off| ~ U[{args.offset_min}, {args.offset_max}] m, random azimuth, z=0")
    print(f"  -> {out_cross}")
    print(f"  -> {out_persp}")

    total_steps = 0
    crashes = 0
    ep_lengths = []
    ep_offset_dists = []
    ep_init_perr = []   # per-episode initial ||pos_error|| (sanity for coverage)

    with h5py.File(out_cross, "w") as hc, h5py.File(out_persp, "w") as hp:
        for ep in tqdm(range(args.n_episodes), desc=f"{mode} episodes"):
            seed = args.seed + ep
            if is_far:
                offset = sample_offset(rng_off, args.offset_min, args.offset_max)
                obs_dict, _ = vis_env.reset(seed=seed,
                                            options={"setpoint_offset": offset})
            else:
                offset = None
                obs_dict, _ = vis_env.reset(seed=seed)
            # vis_env.reset() returns the Dict obs {'image', 'state'}; we render
            # manually below, so we only keep the 15D state. base_env.step() (used in
            # the loop, the UNWRAPPED env) returns the 15D array directly.
            state_obs = obs_dict["state"]
            ctrl.reset()

            # initial position-error magnitude == ||states[0, 0:3]|| (body == world norm)
            ep_init_perr.append(float(np.linalg.norm(state_obs[0:3])))

            imgs_c, imgs_p = [], []
            actions, states, imu_data = [], [], []
            done = False

            while not done:
                img_c, img_p = dual_render(vis_env)
                imu_vec = base_env.get_imu()
                action = ctrl.compute_ctbr_action(
                    base_env.dynamics.state, base_env.target_position,
                    base_env.F_c_max, base_env.omega_max)

                imgs_c.append(img_c)
                imgs_p.append(img_p)
                actions.append(action)
                states.append(state_obs)
                imu_data.append(imu_vec.astype(np.float32))

                state_obs, _, terminated, truncated, _ = base_env.step(action)
                done = terminated or truncated
                if terminated:
                    crashes += 1

            actions_a = np.array(actions, dtype=np.float32)
            states_a = np.array(states, dtype=np.float32)
            imu_a = np.array(imu_data, dtype=np.float32)

            for hf, imgs in ((hc, imgs_c), (hp, imgs_p)):
                g = hf.create_group(f"episode_{ep}")
                g.create_dataset("images", data=np.array(imgs, dtype=np.uint8),
                                 compression="gzip", compression_opts=4)
                g.create_dataset("actions", data=actions_a)
                g.create_dataset("states", data=states_a)
                g.create_dataset("imu_data", data=imu_a)
                # Episode-type tag drives FlowDatasetV5's task-cond [is_hover, is_recovery].
                # IMPORTANT: hover episodes must NOT carry init_tilt_deg, else the dataset
                # misclassifies them as recovery (train_flow_v5.py:87).
                g.attrs["episode_type"] = "recovery" if is_far else "hover"
                if is_far:
                    g.attrs["offset_dist_m"] = float(np.linalg.norm(offset))
                    g.attrs["offset_vec"] = offset.astype(np.float32)

            ep_lengths.append(len(actions))
            total_steps += len(actions)
            if is_far:
                ep_offset_dists.append(float(np.linalg.norm(offset)))

        for hf, render in ((hc, "crosshair"), (hp, "perspective")):
            hf.attrs["n_episodes"] = args.n_episodes
            hf.attrs["total_steps"] = total_steps
            hf.attrs["image_size"] = args.image_size
            hf.attrs["state_dim"] = state_dim
            hf.attrs["action_dim"] = action_dim
            hf.attrs["action_space"] = "ctbr"
            hf.attrs["version"] = "v7_pidctbr"
            hf.attrs["mode"] = mode
            hf.attrs["target_render"] = render
            hf.attrs["teacher"] = "pid_ctbr"
            hf.attrs["pid_vel_max"] = args.pid_vel_max
            hf.attrs["pid_kp_pos"] = args.pid_kp_pos
            hf.attrs["physical_size"] = args.physical_size
            hf.attrs["seed"] = args.seed
            if is_far:
                hf.attrs["offset_min_m"] = args.offset_min
                hf.attrs["offset_max_m"] = args.offset_max

    survival = 1.0 - crashes / args.n_episodes
    print(f"\n[v7 collect] {mode} done.")
    print(f"  episodes:        {args.n_episodes}")
    print(f"  total steps:     {total_steps:,}")
    print(f"  mean ep length:  {np.mean(ep_lengths):.1f}")
    print(f"  survival:        {survival*100:.1f}%  ({args.n_episodes - crashes}/{args.n_episodes})")
    print(f"  init pos-err:    mean={np.mean(ep_init_perr):.3f}m  "
          f"min={np.min(ep_init_perr):.3f}  max={np.max(ep_init_perr):.3f}")
    if is_far:
        print(f"  offset dist:     mean={np.mean(ep_offset_dists):.3f}m  "
              f"min={np.min(ep_offset_dists):.3f}  max={np.max(ep_offset_dists):.3f}")
    print(f"  saved: {out_cross}")
    print(f"         {out_persp}")
    if survival < 0.9:
        print(f"  WARNING: survival {survival*100:.1f}% < 90% -- teacher may be too "
              f"aggressive (ease --pid-vel-max / --pid-kp-pos).")


def main():
    ap = argparse.ArgumentParser(
        description="Phase 3 v7: PID-CTBR dual-render collection (T x O 2x2)")
    ap.add_argument("--mode", choices=["hover", "far"], required=True)
    ap.add_argument("--n-episodes", type=int, default=500)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--physical-size", type=float, default=0.5,
                    help="world half-size of the perspective target marker (m)")
    ap.add_argument("--quadrotor-config", default="configs/quadrotor_v4.yaml")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--out-prefix", default="expert_demos_v7")
    ap.add_argument("--seed", type=int, default=12345)
    # far-mode setpoint-offset band (the 1-3 m coverage the paper's BC data lacks)
    ap.add_argument("--offset-min", type=float, default=1.0)
    ap.add_argument("--offset-max", type=float, default=3.0)
    # PID-CTBR gentle recovery gains (Gate A: 100% survive / 0.14-0.18 m across 1-4 m)
    ap.add_argument("--pid-vel-max", type=float, default=1.0)
    ap.add_argument("--pid-kp-pos", type=float, default=0.8)
    ap.add_argument("--smoke", action="store_true",
                    help="write *_smoke.h5 outputs (does not clobber the real datasets)")
    args = ap.parse_args()
    collect(args)


if __name__ == "__main__":
    main()

"""
Quadrotor Environment v4.0 — CTBR Action Space + INDI Rate Controller

Action (4D, normalized to [-1, 1]):
    [F_c_norm, wx_norm, wy_norm, wz_norm]
    F_c_norm  -> collective thrust [0, F_c_max]
    w*_norm   -> body rate targets [-omega_max_*, omega_max_*] rad/s

Inner-loop: INDI rate controller at 200 Hz
    Uses actual post-lag motor thrusts + numerical angular acceleration diff
    to compute incremental torque correction — robust to motor lag.

Observation (15D): identical to v3.3 QuadrotorEnv.
IMU (6D):          identical to v3.3 (physics-based, normalized).

Hover equilibrium:
    F_c_hover = 0.5 * 9.81 = 4.905 N
    F_c_norm_hover = (4.905 / 16.0) * 2 - 1 ≈ -0.387
    omega_cmd_hover = [0, 0, 0]
    -> Actor bias init: [-0.387, 0.0, 0.0, 0.0]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional
import yaml

from .quadrotor_dynamics import (
    QuadrotorDynamics, QuadrotorParams,
    quaternion_to_rotation_matrix, rotation_matrix_to_6d, get_tilt_angle,
)

_GYRO_SCALE  = 2.0
_SF_SCALE    = 5.0
_SF_Z_OFFSET = -9.81


class INDIRateController:
    """
    INDI rate controller: maps (F_c, omega_cmd) -> motor thrusts.

    At each call the controller:
      1. Estimates angular acceleration via numerical differentiation.
      2. Computes desired angular acceleration using a P-gain on rate error.
      3. Converts the required incremental torque into motor thrust increments,
         using the actual (post-lag) motor thrusts as the linearisation point.
      4. Clips to physical limits.

    Running at the physics rate (200 Hz) with tau_motor=20 ms means the motor
    lag spans ~4 INDI steps, giving the integrator enough resolution while
    avoiding the overshoot that a 500 Hz loop would cause.
    """

    def __init__(self, params: QuadrotorParams, dt: float,
                 kp: np.ndarray, omega_max: np.ndarray, tau_max: np.ndarray):
        self.params   = params
        self.dt       = dt
        self.kp       = kp
        self.omega_max = omega_max
        self.tau_max  = tau_max

        L = params.arm_length
        c = params.torque_coeff
        # Motor mixing matrix (X-config, identical to QuadrotorDynamics)
        self.mixer = np.array([
            [ 1,     1,     1,     1   ],
            [ L,    -L,    -L,     L   ],
            [ L,     L,    -L,    -L   ],
            [-c,     c,    -c,     c   ],
        ])
        self.mixer_inv    = np.linalg.inv(self.mixer)
        self.torque_mixer = self.mixer[1:4, :]   # (3, 4): thrust -> [tau_x, tau_y, tau_z]

        self._omega_prev = np.zeros(3)
        self._first_step = True

    def reset(self):
        self._omega_prev = np.zeros(3)
        self._first_step = True

    def compute(self, F_c: float, omega_cmd: np.ndarray,
                omega: np.ndarray, motor_thrust_actual: np.ndarray) -> np.ndarray:
        """
        Args:
            F_c:                 collective thrust [N]
            omega_cmd:           desired body rates [rad/s]  (3,)
            omega:               current body rates [rad/s]  (3,)
            motor_thrust_actual: post-lag motor thrusts [N]  (4,)

        Returns:
            f_cmd: motor thrust commands [N], clipped to [0, f_max]  (4,)
        """
        # Angular acceleration: skip first step (no previous omega)
        if self._first_step:
            omega_dot = np.zeros(3)
            self._first_step = False
        else:
            omega_dot = (omega - self._omega_prev) / self.dt

        self._omega_prev = omega.copy()

        # P-controller virtual control (desired angular acceleration)
        nu = self.kp * (omega_cmd - omega)

        # Current torque from actual (post-lag) motors
        tau_actual = self.torque_mixer @ motor_thrust_actual

        # INDI increment: desired torque adds inertia * (desired - actual) accel
        I       = self.params.inertia
        tau_des = tau_actual + I * (nu - omega_dot)

        # Anti-windup: clamp each torque axis
        tau_des = np.clip(tau_des, -self.tau_max, self.tau_max)

        # Compute desired motor thrusts via mixer inverse
        wrench_des = np.array([F_c, tau_des[0], tau_des[1], tau_des[2]])
        f_cmd      = self.mixer_inv @ wrench_des
        f_cmd      = np.clip(f_cmd, 0.0, self.params.motor_max_thrust)

        return f_cmd


class QuadrotorEnvV4(gym.Env):
    """
    6-DOF Quadrotor environment with CTBR action space and INDI inner-loop.

    Outer policy runs at 50 Hz and outputs CTBR commands.
    INDI runs at 200 Hz (each outer step contains n_inner_steps INDI iterations).
    Observation and IMU interfaces are backward-compatible with v3.3.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, config_path: str = "configs/quadrotor_v4.yaml",
                 render_mode: Optional[str] = None):
        super().__init__()

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self._load_config()

        # CTBR action space: [F_c_norm, wx_norm, wy_norm, wz_norm] in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # 15D observation (identical to QuadrotorEnv)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        params = QuadrotorParams(
            mass=self.mass,
            arm_length=self.arm_length,
            inertia=np.array(self.inertia),
            gravity=self.gravity,
            motor_max_thrust=self.motor_max_thrust,
            motor_time_constant=self.motor_time_constant,
            drag_coeff=self.drag_coeff,
            torque_coeff=self.torque_coeff,
        )
        self.dynamics     = QuadrotorDynamics(params, dt=self.dt_inner)
        self.hover_thrust = self.dynamics.get_hover_thrust()

        self.indi = INDIRateController(
            params    = params,
            dt        = self.dt_inner,
            kp        = self.indi_kp,
            omega_max = self.omega_max,
            tau_max   = self.tau_max,
        )

        self.current_step              = 0
        self.target_position           = np.zeros(3)
        self.time_since_target_change  = 0.0
        self.render_mode               = render_mode

        self.disturbance_active           = False
        self.disturbance_time_remaining   = 0.0
        self.disturbance_force            = np.zeros(3)
        self.disturbance_torque           = np.zeros(3)

        self.history = {
            'time': [], 'position': [], 'velocity': [],
            'quaternion': [], 'ang_velocity': [],
            'target': [], 'motor_thrust': [], 'reward': [],
            'ctbr_action': [],
        }

    def _load_config(self):
        q = self.config['quadrotor']
        self.mass                = q['mass']
        self.arm_length          = q['arm_length']
        self.inertia             = q['inertia']
        self.gravity             = q['gravity']
        self.motor_max_thrust    = q['motor_max_thrust']
        self.motor_time_constant = q['motor_time_constant']
        self.drag_coeff          = q['drag_coeff']
        self.torque_coeff        = q['torque_coeff']

        t = self.config['timing']
        self.dt_inner     = t['dt_inner']
        self.dt_outer     = t['dt_outer']
        self.n_inner_steps = t['n_inner_steps']

        c = self.config['ctbr']
        self.F_c_max  = c['F_c_max']
        self.omega_max = np.array([
            c['omega_max_roll'],
            c['omega_max_pitch'],
            c['omega_max_yaw'],
        ])

        i = self.config['indi']
        self.indi_kp  = np.array([i['kp_roll'], i['kp_pitch'], i['kp_yaw']])
        self.tau_max  = np.array([i['tau_max_roll'], i['tau_max_pitch'], i['tau_max_yaw']])

        e = self.config['environment']
        self.max_episode_steps     = e['max_episode_steps']
        self.position_bound        = e['position_bound']
        self.max_tilt_deg          = e['max_tilt_deg']
        self.initial_pos_range     = e['initial_pos_range']
        self.initial_vel_range     = e['initial_vel_range']
        self.target_type           = e['target_type']
        self.waypoint_change_interval = e.get('waypoint_change_interval', 3.0)
        self.waypoint_range        = e.get('waypoint_range', 2.0)

        r = self.config['reward']
        self.sigma_pos    = r['sigma_pos']
        self.sigma_z      = r.get('sigma_z', r['sigma_pos'])
        self.sigma_vel    = r['sigma_vel']
        self.sigma_ang    = r['sigma_ang']
        self.w_pos        = r['w_pos']
        self.w_z          = r.get('w_z', 0.0)
        self.w_vel        = r['w_vel']
        self.w_ang        = r['w_ang']
        self.w_action     = r['w_action']
        self.alive_bonus  = r['alive_bonus']
        self.crash_penalty = r['crash_penalty']
        self.w_brake      = r.get('w_brake', 0.0)
        self.sigma_brake  = r.get('sigma_brake', 0.3)

        d = self.config['disturbance']
        self.disturbance_enabled           = d['enabled']
        self.disturbance_magnitude         = d['magnitude']
        self.disturbance_torque_magnitude  = d['torque_magnitude']
        self.disturbance_probability       = d['probability']
        self.disturbance_duration          = d['duration']

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Anchored curriculum: with prob `hover_anchor_prob`, force a near-hover
        # init (pos<=0.1m, vel<=0.05) to prevent catastrophic forgetting of hover.
        # Set externally via env.hover_anchor_prob = 0.2 from training script.
        anchor_prob = getattr(self, 'hover_anchor_prob', 0.0)
        if anchor_prob > 0.0 and self.np_random.random() < anchor_prob:
            pos_range_now = 0.1
            vel_range_now = 0.05
        else:
            pos_range_now = self.initial_pos_range
            vel_range_now = self.initial_vel_range

        init_pos = self.np_random.uniform(
            -pos_range_now, pos_range_now, size=3
        )
        init_pos[2] = -abs(init_pos[2]) - 1.0

        init_vel = self.np_random.uniform(
            -vel_range_now, vel_range_now, size=3
        )

        self.dynamics.reset(position=init_pos, velocity=init_vel)
        self.indi.reset()

        if self.target_type == "hover":
            self.target_position = init_pos.copy()
        else:
            self._generate_new_waypoint()

        self.time_since_target_change = 0.0
        self.current_step             = 0
        self.disturbance_active       = False
        self.disturbance_time_remaining = 0.0
        self.disturbance_force        = np.zeros(3)
        self.disturbance_torque       = np.zeros(3)

        for key in self.history:
            self.history[key] = []

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        One outer RL step (50 Hz).

        Args:
            action: (4,) CTBR in [-1, 1]:
                    [F_c_norm, wx_norm, wy_norm, wz_norm]
        """
        F_c, omega_cmd = self._decode_action(action)

        total_reward = 0.0

        for _ in range(self.n_inner_steps):
            d_force, d_torque = self._get_disturbance()

            # INDI -> motor thrusts
            f_cmd = self.indi.compute(
                F_c         = F_c,
                omega_cmd   = omega_cmd,
                omega       = self.dynamics.ang_velocity.copy(),
                motor_thrust_actual = self.dynamics.motor_thrust.copy(),
            )

            self.dynamics.step(f_cmd, d_force, d_torque)
            total_reward += self._calculate_reward(action, F_c)

        total_reward /= self.n_inner_steps

        if self.target_type == "waypoint":
            self.time_since_target_change += self.dt_outer
            if self.time_since_target_change >= self.waypoint_change_interval:
                self._generate_new_waypoint()
                self.time_since_target_change = 0.0

        self.current_step += 1
        terminated = self._check_termination()
        truncated  = self.current_step >= self.max_episode_steps

        if terminated:
            total_reward -= self.crash_penalty

        self._log_history(action, total_reward)

        obs = self._get_observation()
        info = {
            'position':     self.dynamics.position.copy(),
            'velocity':     self.dynamics.velocity.copy(),
            'target':       self.target_position.copy(),
            'tilt_deg':     get_tilt_angle(self.dynamics.get_rotation_matrix()),
            'motor_thrust': self.dynamics.motor_thrust.copy(),
            'F_c':          F_c,
            'omega_cmd':    omega_cmd,
        }

        return obs, total_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # IMU interface (v3.3-compatible)
    # ------------------------------------------------------------------

    def get_imu(self) -> np.ndarray:
        """Normalized IMU [gyro(3), specific_force_norm(3)] — same as v3.3."""
        gyro = self.dynamics.ang_velocity.astype(np.float32) / _GYRO_SCALE
        sf   = self.dynamics.get_specific_force_body().astype(np.float32)
        sf_n = np.array([
            sf[0] / _SF_SCALE,
            sf[1] / _SF_SCALE,
            (sf[2] - _SF_Z_OFFSET) / _SF_SCALE,
        ], dtype=np.float32)
        return np.concatenate([gyro, sf_n])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_action(self, action: np.ndarray) -> Tuple[float, np.ndarray]:
        """Map normalized CTBR action to physical units."""
        F_c      = (action[0] + 1.0) * 0.5 * self.F_c_max          # [0, F_c_max]
        omega_cmd = action[1:4] * self.omega_max                     # [-omega_max, omega_max]
        return float(F_c), omega_cmd

    def _get_observation(self) -> np.ndarray:
        R = self.dynamics.get_rotation_matrix()
        pos_error_world = self.target_position - self.dynamics.position
        pos_error_body  = R.T @ pos_error_world
        rot_6d          = rotation_matrix_to_6d(R)
        vel_body        = R.T @ self.dynamics.velocity
        omega           = self.dynamics.ang_velocity

        return np.concatenate([
            pos_error_body, rot_6d, vel_body, omega,
        ]).astype(np.float32)

    def _calculate_reward(self, action: np.ndarray, F_c: float) -> float:
        pos_error = self.target_position - self.dynamics.position
        vel       = self.dynamics.velocity
        omega     = self.dynamics.ang_velocity

        xy_sq    = pos_error[0]**2 + pos_error[1]**2
        pos_rew  = self.w_pos * np.exp(-xy_sq / self.sigma_pos)
        z_rew    = self.w_z   * np.exp(-pos_error[2]**2 / self.sigma_z)
        vel_rew  = self.w_vel * np.exp(-np.sum(vel**2) / self.sigma_vel)
        ang_rew  = self.w_ang * np.exp(-np.sum(omega**2) / self.sigma_ang)

        # Action penalty: deviation of CTBR from hover point
        F_hover_norm = (self.hover_thrust * 4 / self.F_c_max) * 2 - 1   # ~-0.387
        action_dev   = np.array([
            action[0] - F_hover_norm,   # F_c deviation
            action[1],                  # omega_x (target 0)
            action[2],                  # omega_y (target 0)
            action[3],                  # omega_z (target 0)
        ])
        action_pen = self.w_action * np.sum(action_dev**2)

        # Brake penalty (Run 14: w_brake=0.0, disabled).
        # Run 13 used isotropic formula exp(-dist/sigma)*v_xy^2 which caused
        # reverse reward hacking: policy learned to escape to dist>>sigma to
        # nullify the penalty, resulting in RMSE 0.51m (worse than Run 12).
        #
        # Run 15 candidate — radial-only brake (safe formula):
        #   dist_xy     = np.sqrt(pos_error[0]**2 + pos_error[1]**2)
        #   vel_radial  = np.dot(vel[:2], -pos_error[:2] / (dist_xy + 1e-6))
        #   brake_pen   = self.w_brake * max(0.0, vel_radial)**2 \
        #                 * np.exp(-dist_xy / self.sigma_brake)
        # Penalises only inbound radial speed; hover/tangential/outbound = 0 penalty.
        dist_xy   = np.sqrt(pos_error[0]**2 + pos_error[1]**2)
        vel_xy_sq = vel[0]**2 + vel[1]**2
        brake_pen = self.w_brake * np.exp(-dist_xy / self.sigma_brake) * vel_xy_sq

        return pos_rew + z_rew + vel_rew + ang_rew - action_pen - brake_pen + self.alive_bonus

    def _check_termination(self) -> bool:
        if np.any(np.abs(self.dynamics.position) > self.position_bound):
            return True
        if get_tilt_angle(self.dynamics.get_rotation_matrix()) > self.max_tilt_deg:
            return True
        if self.dynamics.position[2] > 0.0:
            return True
        return False

    def _generate_new_waypoint(self):
        self.target_position = self.np_random.uniform(
            -self.waypoint_range, self.waypoint_range, size=3
        )
        self.target_position[2] = -abs(self.target_position[2]) - 0.5

    def _get_disturbance(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.disturbance_enabled:
            return None, None

        if self.disturbance_active:
            self.disturbance_time_remaining -= self.dt_inner
            if self.disturbance_time_remaining <= 0:
                self.disturbance_active = False
                self.disturbance_force  = np.zeros(3)
                self.disturbance_torque = np.zeros(3)
            return self.disturbance_force, self.disturbance_torque

        if self.np_random.random() < self.disturbance_probability:
            self.disturbance_active           = True
            self.disturbance_time_remaining   = self.disturbance_duration
            self.disturbance_force   = self.np_random.uniform(
                -self.disturbance_magnitude, self.disturbance_magnitude, size=3
            )
            self.disturbance_torque  = self.np_random.uniform(
                -self.disturbance_torque_magnitude,
                self.disturbance_torque_magnitude, size=3
            )
            return self.disturbance_force, self.disturbance_torque

        return None, None

    def _log_history(self, action: np.ndarray, reward: float):
        self.history['time'].append(self.current_step * self.dt_outer)
        self.history['position'].append(self.dynamics.position.copy())
        self.history['velocity'].append(self.dynamics.velocity.copy())
        self.history['quaternion'].append(self.dynamics.quaternion.copy())
        self.history['ang_velocity'].append(self.dynamics.ang_velocity.copy())
        self.history['target'].append(self.target_position.copy())
        self.history['motor_thrust'].append(self.dynamics.motor_thrust.copy())
        self.history['reward'].append(reward)
        self.history['ctbr_action'].append(action.copy())

    def get_history(self) -> Dict:
        return self.history

    def render(self):
        pass

    def close(self):
        pass


def make_quadrotor_env_v4(config_path: str = "configs/quadrotor_v4.yaml") -> QuadrotorEnvV4:
    return QuadrotorEnvV4(config_path=config_path)

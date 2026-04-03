"""
Quadrotor Gymnasium Environment

Wraps QuadrotorDynamics into a Gymnasium-compatible environment for RL training.
Provides 15D state observation and 4D direct motor thrust action space.

Observation (15D):
    [0:3]  Position error in body frame  (R^T * (target - pos))
    [3:9]  6D rotation representation    (first 2 columns of R, flattened)
    [9:12] Linear velocity in body frame (R^T * vel)
    [12:15] Angular velocity in body frame (omega)

Action (4D):
    Normalized motor thrusts in [-1, 1], mapped to [0, motor_max_thrust]
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


class QuadrotorEnv(gym.Env):
    """
    6-DOF Quadrotor environment for end-to-end motor control with PPO/DPPO.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, config_path: str = "configs/quadrotor.yaml",
                 render_mode: Optional[str] = None):
        super().__init__()

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self._load_config()

        # Action space: 4 motor thrusts normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observation space: 15D state vector
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        # Physics engine
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
        self.dynamics = QuadrotorDynamics(params, dt=self.dt_inner)
        self.hover_thrust = self.dynamics.get_hover_thrust()

        # Episode state
        self.current_step = 0
        self.target_position = np.zeros(3)
        self.time_since_target_change = 0.0
        self.render_mode = render_mode

        # Disturbance state
        self.disturbance_active = False
        self.disturbance_time_remaining = 0.0
        self.disturbance_force = np.zeros(3)
        self.disturbance_torque = np.zeros(3)

        # History for logging
        self.history = {
            'time': [], 'position': [], 'velocity': [],
            'quaternion': [], 'ang_velocity': [],
            'target': [], 'motor_thrust': [], 'reward': [],
        }

    def _load_config(self):
        """Extract configuration parameters."""
        q = self.config['quadrotor']
        self.mass = q['mass']
        self.arm_length = q['arm_length']
        self.inertia = q['inertia']
        self.gravity = q['gravity']
        self.motor_max_thrust = q['motor_max_thrust']
        self.motor_time_constant = q['motor_time_constant']
        self.drag_coeff = q['drag_coeff']
        self.torque_coeff = q['torque_coeff']

        t = self.config['timing']
        self.dt_inner = t['dt_inner']
        self.dt_outer = t['dt_outer']
        self.n_inner_steps = t['n_inner_steps']

        e = self.config['environment']
        self.max_episode_steps = e['max_episode_steps']
        self.position_bound = e['position_bound']
        self.max_tilt_deg = e['max_tilt_deg']
        self.initial_pos_range = e['initial_pos_range']
        self.initial_z_range = e.get('initial_z_range', 0.0)  # fallback: 0 = target_z == init_z (original behaviour)
        self.initial_vel_range = e['initial_vel_range']
        self.target_type = e['target_type']
        self.waypoint_change_interval = e.get('waypoint_change_interval', 3.0)
        self.waypoint_range = e.get('waypoint_range', 2.0)

        r = self.config['reward']
        self.sigma_pos = r['sigma_pos']
        self.sigma_z = r.get('sigma_z', r['sigma_pos'])  # fallback: same as sigma_pos
        self.sigma_vel = r['sigma_vel']
        self.sigma_ang = r['sigma_ang']
        self.w_pos = r['w_pos']
        self.w_z = r.get('w_z', 0.0)                     # fallback: 0 (backward-compatible)
        self.w_vel = r['w_vel']
        self.w_ang = r['w_ang']
        self.w_action = r['w_action']
        self.alive_bonus = r['alive_bonus']
        self.crash_penalty = r['crash_penalty']

        d = self.config['disturbance']
        self.disturbance_enabled = d['enabled']
        self.disturbance_magnitude = d['magnitude']
        self.disturbance_torque_magnitude = d['torque_magnitude']
        self.disturbance_probability = d['probability']
        self.disturbance_duration = d['duration']

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Random initial position near origin
        init_pos = self.np_random.uniform(
            -self.initial_pos_range, self.initial_pos_range, size=3
        )
        # Start above ground (NED: negative Z is up)
        init_pos[2] = -abs(init_pos[2]) - 1.0  # at least 1m altitude

        init_vel = self.np_random.uniform(
            -self.initial_vel_range, self.initial_vel_range, size=3
        )

        self.dynamics.reset(position=init_pos, velocity=init_vel)

        # Set target
        if self.target_type == "hover":
            self.target_position = init_pos.copy()
            # Add Z offset between drone start and target to force altitude correction.
            # initial_z_range controls how far above/below the target the drone starts.
            # In NED, negative Z = higher altitude, so subtract z_offset to place target higher.
            z_offset = self.np_random.uniform(-self.initial_z_range, self.initial_z_range)
            self.target_position[2] = init_pos[2] - z_offset  # target Z: displaced from start
        else:
            self._generate_new_waypoint()

        self.time_since_target_change = 0.0
        self.current_step = 0

        # Reset disturbance
        self.disturbance_active = False
        self.disturbance_time_remaining = 0.0
        self.disturbance_force = np.zeros(3)
        self.disturbance_torque = np.zeros(3)

        # Clear history
        for key in self.history:
            self.history[key] = []

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one RL step (outer loop) containing n_inner_steps physics steps.

        Args:
            action: (4,) normalized motor thrusts in [-1, 1]
        """
        # Map action from [-1, 1] to [0, motor_max_thrust]
        thrust_cmd = (action + 1.0) * 0.5 * self.motor_max_thrust

        total_reward = 0.0

        # Run inner loop
        for i in range(self.n_inner_steps):
            # Generate disturbance
            d_force, d_torque = self._get_disturbance()

            # Physics step
            self.dynamics.step(thrust_cmd, d_force, d_torque)

            # Accumulate reward
            total_reward += self._calculate_reward(thrust_cmd)

        # Average reward over inner steps
        total_reward /= self.n_inner_steps

        # Update target (waypoint mode)
        if self.target_type == "waypoint":
            self.time_since_target_change += self.dt_outer
            if self.time_since_target_change >= self.waypoint_change_interval:
                self._generate_new_waypoint()
                self.time_since_target_change = 0.0

        # Check termination
        self.current_step += 1
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps

        # Apply crash penalty
        if terminated:
            total_reward -= self.crash_penalty

        # Log history
        self._log_history(thrust_cmd, total_reward)

        obs = self._get_observation()
        info = {
            'position': self.dynamics.position.copy(),
            'velocity': self.dynamics.velocity.copy(),
            'target': self.target_position.copy(),
            'tilt_deg': get_tilt_angle(self.dynamics.get_rotation_matrix()),
            'motor_thrust': thrust_cmd.copy(),
        }

        return obs, total_reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Compute 15D observation vector.

        Returns:
            obs: (15,) float32 array
        """
        R = self.dynamics.get_rotation_matrix()

        # Position error in body frame
        pos_error_world = self.target_position - self.dynamics.position
        pos_error_body = R.T @ pos_error_world

        # 6D rotation representation
        rot_6d = rotation_matrix_to_6d(R)

        # Linear velocity in body frame
        vel_body = R.T @ self.dynamics.velocity

        # Angular velocity (already in body frame)
        omega = self.dynamics.ang_velocity

        obs = np.concatenate([
            pos_error_body,   # 3D
            rot_6d,           # 6D
            vel_body,         # 3D
            omega,            # 3D
        ]).astype(np.float32)

        return obs

    def _calculate_reward(self, thrust_cmd: np.ndarray) -> float:
        """
        Anisotropic Gaussian reward for position tracking and stabilization.

        r = w_pos * exp(-(ex^2 + ey^2) / sigma_pos)   # X/Y horizontal
          + w_z   * exp(-ez^2 / sigma_z)               # Z altitude (tighter sigma)
          + w_vel * exp(-||vel||^2 / sigma_vel)
          + w_ang * exp(-||ang_vel||^2 / sigma_ang)
          - w_action * ||action_normalized||^2
          + alive_bonus

        Separating Z from X/Y allows sigma_z to be tighter than sigma_pos,
        applying stronger gradient pressure on altitude without disturbing the
        already-converged horizontal tracking.
        """
        pos_error = self.target_position - self.dynamics.position
        vel = self.dynamics.velocity
        omega = self.dynamics.ang_velocity

        # Horizontal (X/Y) reward
        xy_sq = pos_error[0]**2 + pos_error[1]**2
        pos_reward = self.w_pos * np.exp(-xy_sq / self.sigma_pos)

        # Vertical (Z) reward — dedicated term with tighter sigma
        z_reward = self.w_z * np.exp(-pos_error[2]**2 / self.sigma_z)

        vel_reward = self.w_vel * np.exp(
            -np.sum(vel**2) / self.sigma_vel
        )
        ang_reward = self.w_ang * np.exp(
            -np.sum(omega**2) / self.sigma_ang
        )

        # Action penalty: penalize deviation from hover thrust, not absolute thrust.
        # Old: w_action * sum((thrust/f_max)^2)  — penalizes the drone for simply staying airborne
        # New: w_action * sum(((thrust - hover)/f_max)^2) — cost = 0 at perfect hover, penalizes corrections
        hover_norm = self.hover_thrust / self.motor_max_thrust
        action_dev = thrust_cmd / self.motor_max_thrust - hover_norm
        action_penalty = self.w_action * np.sum(action_dev**2)

        return pos_reward + z_reward + vel_reward + ang_reward - action_penalty + self.alive_bonus

    def _check_termination(self) -> bool:
        """Check if episode should terminate early."""
        # Position out of bounds
        if np.any(np.abs(self.dynamics.position) > self.position_bound):
            return True

        # Excessive tilt
        R = self.dynamics.get_rotation_matrix()
        tilt = get_tilt_angle(R)
        if tilt > self.max_tilt_deg:
            return True

        # Ground contact (NED: positive Z is down, so position Z > 0 means below ground)
        if self.dynamics.position[2] > 0.0:
            return True

        return False

    def _generate_new_waypoint(self):
        """Generate a random waypoint within bounds."""
        self.target_position = self.np_random.uniform(
            -self.waypoint_range, self.waypoint_range, size=3
        )
        # Keep target above ground (NED)
        self.target_position[2] = -abs(self.target_position[2]) - 0.5

    def _get_disturbance(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random force and torque disturbances."""
        if not self.disturbance_enabled:
            return None, None

        if self.disturbance_active:
            self.disturbance_time_remaining -= self.dt_inner
            if self.disturbance_time_remaining <= 0:
                self.disturbance_active = False
                self.disturbance_force = np.zeros(3)
                self.disturbance_torque = np.zeros(3)
            return self.disturbance_force, self.disturbance_torque

        if self.np_random.random() < self.disturbance_probability:
            self.disturbance_active = True
            self.disturbance_time_remaining = self.disturbance_duration
            self.disturbance_force = self.np_random.uniform(
                -self.disturbance_magnitude, self.disturbance_magnitude, size=3
            )
            self.disturbance_torque = self.np_random.uniform(
                -self.disturbance_torque_magnitude,
                self.disturbance_torque_magnitude, size=3
            )
            return self.disturbance_force, self.disturbance_torque

        return None, None

    def _log_history(self, thrust_cmd: np.ndarray, reward: float):
        """Log current state for visualization."""
        self.history['time'].append(self.current_step * self.dt_outer)
        self.history['position'].append(self.dynamics.position.copy())
        self.history['velocity'].append(self.dynamics.velocity.copy())
        self.history['quaternion'].append(self.dynamics.quaternion.copy())
        self.history['ang_velocity'].append(self.dynamics.ang_velocity.copy())
        self.history['target'].append(self.target_position.copy())
        self.history['motor_thrust'].append(thrust_cmd.copy())
        self.history['reward'].append(reward)

    def get_history(self) -> Dict:
        """Return logged history for visualization."""
        return self.history

    def render(self):
        pass

    def close(self):
        pass


def make_quadrotor_env(config_path: str = "configs/quadrotor.yaml") -> QuadrotorEnv:
    """Factory function to create QuadrotorEnv."""
    return QuadrotorEnv(config_path=config_path)

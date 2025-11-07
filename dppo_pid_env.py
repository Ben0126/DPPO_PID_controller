"""
DPPO PID Environment - Custom Gymnasium Environment for PID Parameter Tuning
Implements a 2nd-order plant system with real-time PID gain adjustment via RL
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional
import yaml


class DPPOPIDEnv(gym.Env):
    """
    Custom Gymnasium Environment for learning PID controller gains using PPO/DPPO.

    The environment simulates a 2nd-order system (plant) and allows an RL agent
    to adjust PID gains (Kp, Ki, Kd) at a slower rate (outer loop, 20 Hz) while
    the PID controller and physics run at a faster rate (inner loop, 200 Hz).
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, config_path: str = "config.yaml", render_mode: Optional[str] = None):
        """
        Initialize the DPPO PID Environment.

        Args:
            config_path: Path to the YAML configuration file
            render_mode: Rendering mode (currently unused)
        """
        super().__init__()

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Extract configuration parameters
        self._load_config()

        # Define action space: [K_p, K_i, K_d]
        self.action_space = spaces.Box(
            low=0.0,
            high=self.k_max,
            shape=(3,),
            dtype=np.float32
        )

        # Define observation space: 9-dimensional state vector
        # [error, error_dot, integral, position, velocity, reference, Kp, Ki, Kd]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32
        )

        # Initialize state variables
        self._initialize_state()

        # Episode tracking
        self.current_step = 0
        self.render_mode = render_mode

        # History for logging/debugging
        self.history = {
            'time': [],
            'position': [],
            'velocity': [],
            'reference': [],
            'error': [],
            'control': [],
            'kp': [],
            'ki': [],
            'kd': []
        }

    def _load_config(self):
        """Load all configuration parameters from the config dictionary."""
        # Plant parameters
        self.J = self.config['plant']['J']
        self.B = self.config['plant']['B']

        # Timing parameters
        self.dt_inner = self.config['timing']['dt_inner']
        self.dt_outer = self.config['timing']['dt_outer']
        self.n_inner_steps = self.config['timing']['n_inner_steps']

        # Reference signal parameters
        self.r_min = self.config['reference']['r_min']
        self.r_max = self.config['reference']['r_max']
        self.change_interval = self.config['reference']['change_interval']

        # Disturbance parameters
        self.disturbance_enabled = self.config['disturbance']['enabled']
        self.disturbance_magnitude = self.config['disturbance']['magnitude']
        self.disturbance_duration = self.config['disturbance']['duration']
        self.disturbance_probability = self.config['disturbance']['probability']

        # PID constraints
        self.k_max = self.config['pid']['k_max']
        self.integral_max = self.config['pid']['integral_max']

        # Observation normalization scales
        obs_config = self.config['observation']
        self.error_scale = obs_config['error_scale']
        self.error_dot_scale = obs_config['error_dot_scale']
        self.integral_scale = obs_config['integral_scale']
        self.position_scale = obs_config['position_scale']
        self.velocity_scale = obs_config['velocity_scale']
        self.reference_scale = obs_config['reference_scale']
        self.gain_scale = obs_config['gain_scale']

        # Reward weights
        reward_config = self.config['reward']
        self.lambda_error = reward_config['lambda_error']
        self.lambda_velocity = reward_config['lambda_velocity']
        self.lambda_control = reward_config['lambda_control']
        self.lambda_overshoot = reward_config['lambda_overshoot']

        # Episode configuration
        self.max_steps = self.config['episode']['max_steps']
        self.termination_threshold = self.config['episode']['termination_threshold']

    def _initialize_state(self):
        """Initialize or reset all state variables."""
        # Plant state
        self.x = 0.0          # Position
        self.x_dot = 0.0      # Velocity

        # PID state
        self.Kp = 1.0
        self.Ki = 0.1
        self.Kd = 0.1
        self.integral = 0.0
        self.last_error = 0.0

        # Reference tracking
        self.reference = 0.0
        self.time_since_ref_change = 0.0

        # Disturbance state
        self.disturbance_active = False
        self.disturbance_time_remaining = 0.0
        self.disturbance_value = 0.0

        # Time tracking
        self.sim_time = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to a random initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)

        # Reset state
        self._initialize_state()

        # Set random initial reference
        self.reference = self.np_random.uniform(self.r_min, self.r_max)

        # Set random initial position (near reference)
        self.x = self.reference + self.np_random.uniform(-0.5, 0.5)

        # Reset episode tracking
        self.current_step = 0

        # Clear history
        for key in self.history:
            self.history[key] = []

        # Return initial observation
        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one RL step (outer loop) which contains N inner loop steps.

        Args:
            action: PID gains [Kp, Ki, Kd] from the agent

        Returns:
            observation: Current state observation
            reward: Accumulated reward over inner loop steps
            terminated: Whether the episode has ended
            truncated: Whether the episode was truncated
            info: Additional information dictionary
        """
        # 1. Update PID Gains (Outer Loop @ 20 Hz)
        self.Kp, self.Ki, self.Kd = np.clip(action, 0.0, self.k_max)

        total_reward = 0.0

        # 2. Run N Inner Loop Steps (Inner Loop @ 200 Hz)
        for inner_step in range(self.n_inner_steps):
            # 2a. Calculate Error
            error = self.reference - self.x
            error_dot = -self.x_dot  # Derivative of error (assuming reference is constant over inner steps)

            # 2b. Calculate PID Control Input
            u = (self.Kp * error +
                 self.Ki * self.integral +
                 self.Kd * (error - self.last_error) / self.dt_inner)

            # Optional: Saturate control input
            u = np.clip(u, -50.0, 50.0)

            # 2c. Apply Disturbance
            disturbance = self._get_disturbance()

            # 2d. Integrate Plant Dynamics
            self._integrate_plant(u + disturbance)

            # 2e. Update PID State
            self.integral += error * self.dt_inner
            # Anti-windup: Limit integral term
            self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)
            self.last_error = error

            # 2f. Accumulate Reward (at inner loop frequency)
            step_reward = self._calculate_reward(error, error_dot, u)
            total_reward += step_reward

            # Update simulation time
            self.sim_time += self.dt_inner

            # Log history (only every 10th step to save memory)
            if inner_step == 0:
                self._log_history(u)

        # 3. Update reference signal (check if it's time to change)
        self.time_since_ref_change += self.dt_outer
        if self.time_since_ref_change >= self.change_interval:
            self.reference = self.np_random.uniform(self.r_min, self.r_max)
            self.time_since_ref_change = 0.0

        # 4. Check Termination
        self.current_step += 1
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps

        # 5. Get Observation
        observation = self._get_observation()

        # 6. Info Dictionary
        info = {
            'position': self.x,
            'velocity': self.x_dot,
            'reference': self.reference,
            'error': error,
            'control': u,
            'Kp': self.Kp,
            'Ki': self.Ki,
            'Kd': self.Kd
        }

        return observation, total_reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Get the current normalized observation vector.

        Returns:
            Normalized 9-dimensional observation vector
        """
        error = self.reference - self.x
        error_dot = -self.x_dot

        # Normalize all components to approximately [-1, 1]
        obs = np.array([
            error / self.error_scale,
            error_dot / self.error_dot_scale,
            self.integral / self.integral_scale,
            self.x / self.position_scale,
            self.x_dot / self.velocity_scale,
            self.reference / self.reference_scale,
            self.Kp / self.gain_scale,
            self.Ki / self.gain_scale,
            self.Kd / self.gain_scale
        ], dtype=np.float32)

        # Clip to ensure within bounds
        obs = np.clip(obs, -1.0, 1.0)

        return obs

    def _integrate_plant(self, u: float):
        """
        Integrate plant dynamics using Euler method.

        Plant equation: J * x_ddot + B * x_dot = u(t) + d(t)
        Rewritten: x_ddot = (u - B * x_dot) / J

        Args:
            u: Total input (control + disturbance)
        """
        # Calculate acceleration
        x_ddot = (u - self.B * self.x_dot) / self.J

        # Euler integration
        self.x_dot += x_ddot * self.dt_inner
        self.x += self.x_dot * self.dt_inner

    def _get_disturbance(self) -> float:
        """
        Generate external disturbance signal.

        Returns:
            Disturbance force value
        """
        if not self.disturbance_enabled:
            return 0.0

        # Check if disturbance is currently active
        if self.disturbance_active:
            self.disturbance_time_remaining -= self.dt_inner
            if self.disturbance_time_remaining <= 0:
                self.disturbance_active = False
                self.disturbance_value = 0.0
            return self.disturbance_value

        # Randomly activate disturbance
        if self.np_random.random() < self.disturbance_probability:
            self.disturbance_active = True
            self.disturbance_time_remaining = self.disturbance_duration
            self.disturbance_value = self.np_random.uniform(
                -self.disturbance_magnitude,
                self.disturbance_magnitude
            )
            return self.disturbance_value

        return 0.0

    def _calculate_reward(self, error: float, error_dot: float, u: float) -> float:
        """
        Calculate the reward for the current step.

        Reward components:
        1. Tracking error: Penalize squared error
        2. Velocity error: Penalize high velocities
        3. Control effort: Penalize excessive control
        4. Overshoot: Heavily penalize when moving away from setpoint

        Args:
            error: Current tracking error
            error_dot: Rate of change of error
            u: Control input

        Returns:
            Scalar reward value
        """
        # 1. Tracking error penalty
        error_penalty = -self.lambda_error * error**2

        # 2. Velocity penalty (to prevent oscillations)
        velocity_penalty = -self.lambda_velocity * self.x_dot**2

        # 3. Control effort penalty
        control_penalty = -self.lambda_control * u**2

        # 4. Overshoot penalty (error and velocity in opposite directions)
        # If error > 0 (below setpoint) and x_dot < 0 (moving down): bad
        # If error < 0 (above setpoint) and x_dot > 0 (moving up): bad
        overshoot_indicator = max(0, -error * self.x_dot)
        overshoot_penalty = -self.lambda_overshoot * overshoot_indicator

        # Total reward
        total_reward = (error_penalty + velocity_penalty +
                       control_penalty + overshoot_penalty)

        return total_reward

    def _check_termination(self) -> bool:
        """
        Check if the episode should terminate early.

        Terminate if:
        - Position exceeds safe bounds (system instability)

        Returns:
            True if episode should terminate, False otherwise
        """
        # Check position bounds
        if abs(self.x) > self.termination_threshold:
            return True

        return False

    def _log_history(self, u: float):
        """Log current state for later visualization."""
        self.history['time'].append(self.sim_time)
        self.history['position'].append(self.x)
        self.history['velocity'].append(self.x_dot)
        self.history['reference'].append(self.reference)
        self.history['error'].append(self.reference - self.x)
        self.history['control'].append(u)
        self.history['kp'].append(self.Kp)
        self.history['ki'].append(self.Ki)
        self.history['kd'].append(self.Kd)

    def get_history(self) -> Dict:
        """Return the logged history for visualization."""
        return self.history

    def render(self):
        """Render the environment (placeholder for future implementation)."""
        if self.render_mode == "human":
            pass  # Could implement real-time plotting here

    def close(self):
        """Clean up resources."""
        pass


# Utility function for creating the environment
def make_env(config_path: str = "config.yaml"):
    """
    Factory function to create a DPPOPIDEnv instance.

    Args:
        config_path: Path to configuration file

    Returns:
        Initialized environment instance
    """
    return DPPOPIDEnv(config_path=config_path)

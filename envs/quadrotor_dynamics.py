"""
Quadrotor 6-DOF Dynamics Module

Pure physics simulation for a quadrotor with quaternion attitude representation.
Decoupled from Gymnasium — used by QuadrotorEnv as the physics backend.

Coordinate conventions:
  - World frame: NED (North-East-Down), gravity along +Z
  - Body frame: X-forward, Y-right, Z-down
  - Motor layout: X-configuration
      Motor 1 (front-right), Motor 2 (front-left)
      Motor 3 (rear-left),   Motor 4 (rear-right)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class QuadrotorParams:
    """Physical parameters of the quadrotor."""
    mass: float = 0.5                        # kg
    arm_length: float = 0.17                 # m
    inertia: np.ndarray = field(default_factory=lambda: np.array([0.0023, 0.0023, 0.004]))
    gravity: float = 9.81                    # m/s^2
    motor_max_thrust: float = 4.0            # N per motor
    motor_time_constant: float = 0.02        # s
    drag_coeff: float = 0.01                 # linear drag
    torque_coeff: float = 0.016              # c_tau ratio


# ============================================================================
# Quaternion Utilities
# ============================================================================

def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product of two quaternions q = [w, x, y, z].

    Args:
        q1, q2: quaternions of shape (4,)

    Returns:
        Product quaternion (4,)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix.

    Returns:
        R: (3, 3) rotation matrix (body-to-world)
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def quaternion_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Compute quaternion time derivative: dq/dt = 0.5 * q * [0, omega].

    Args:
        q: quaternion [w, x, y, z]
        omega: angular velocity in body frame [wx, wy, wz]

    Returns:
        dq/dt: (4,)
    """
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
    return 0.5 * quaternion_multiply(q, omega_quat)


def rotation_matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """
    Extract 6D continuous rotation representation (first two columns of R).
    More suitable for neural networks than Euler angles or raw quaternion.

    Args:
        R: (3, 3) rotation matrix

    Returns:
        (6,) vector [R[:,0], R[:,1]] flattened
    """
    return np.concatenate([R[:, 0], R[:, 1]])


def get_tilt_angle(R: np.ndarray) -> float:
    """
    Compute tilt angle (deviation from upright) in degrees.
    For NED frame, upright means body Z aligns with world Z.
    cos(tilt) = R[2,2] (dot product of body-Z with world-Z).
    """
    cos_tilt = np.clip(R[2, 2], -1.0, 1.0)
    return np.degrees(np.arccos(cos_tilt))


# ============================================================================
# Quadrotor Dynamics
# ============================================================================

class QuadrotorDynamics:
    """
    6-DOF quadrotor dynamics with RK4 integration.

    State vector (13D):
        position     [x, y, z]         (3) - world frame
        quaternion   [qw, qx, qy, qz]  (4) - body-to-world
        velocity     [vx, vy, vz]       (3) - world frame
        ang_velocity [wx, wy, wz]       (3) - body frame
    """

    def __init__(self, params: QuadrotorParams, dt: float = 0.005):
        self.params = params
        self.dt = dt

        # Build motor mixing matrix (X-configuration)
        # Maps [f1, f2, f3, f4] to [F_total, tau_x, tau_y, tau_z]
        L = params.arm_length
        c = params.torque_coeff
        #   Motor:  1(FR)  2(FL)  3(RL)  4(RR)
        self.mixer = np.array([
            [ 1,     1,     1,     1   ],   # F_total = sum(fi)
            [ L,    -L,    -L,     L   ],   # tau_x (roll)
            [ L,     L,    -L,    -L   ],   # tau_y (pitch)
            [-c,     c,    -c,     c   ],   # tau_z (yaw)
        ])

        # Motor state (with first-order lag)
        self.motor_thrust = np.zeros(4)

        self.reset()

    def reset(self, position: np.ndarray = None, velocity: np.ndarray = None):
        """Reset to initial state (hovering at origin or given position)."""
        self.position = position if position is not None else np.zeros(3)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # upright
        self.velocity = velocity if velocity is not None else np.zeros(3)
        self.ang_velocity = np.zeros(3)
        self.motor_thrust = np.zeros(4)

    @property
    def state(self) -> np.ndarray:
        """Return full 13D state vector."""
        return np.concatenate([
            self.position,
            self.quaternion,
            self.velocity,
            self.ang_velocity,
        ])

    @state.setter
    def state(self, s: np.ndarray):
        """Set state from 13D vector."""
        self.position = s[0:3].copy()
        self.quaternion = quaternion_normalize(s[3:7])
        self.velocity = s[7:10].copy()
        self.ang_velocity = s[10:13].copy()

    def get_rotation_matrix(self) -> np.ndarray:
        """Get current body-to-world rotation matrix."""
        return quaternion_to_rotation_matrix(self.quaternion)

    def _motor_dynamics(self, thrust_cmd: np.ndarray):
        """
        First-order motor lag: dF/dt = (F_cmd - F) / tau.
        Uses simple Euler for motor dynamics (fast enough at 200Hz).
        """
        tau = self.params.motor_time_constant
        if tau > 0:
            alpha = self.dt / (tau + self.dt)
            self.motor_thrust += alpha * (thrust_cmd - self.motor_thrust)
        else:
            self.motor_thrust = thrust_cmd.copy()

    def _compute_forces_torques(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute total force (world frame) and torque (body frame)
        from current motor thrusts.

        Returns:
            force_world: (3,) total force in world frame
            torque_body: (3,) total torque in body frame
        """
        p = self.params
        R = self.get_rotation_matrix()

        # Motor mixing: [F_total, tau_x, tau_y, tau_z]
        wrench = self.mixer @ self.motor_thrust
        F_total = wrench[0]
        torque_body = wrench[1:4]

        # Thrust in body frame is along -Z (body Z points down in NED, thrust is up)
        # In NED: body thrust vector = [0, 0, -F_total] in body frame
        thrust_body = np.array([0.0, 0.0, -F_total])
        thrust_world = R @ thrust_body

        # Gravity in world frame (NED: gravity is +Z)
        gravity_world = np.array([0.0, 0.0, p.mass * p.gravity])

        # Aerodynamic drag (simple linear model in world frame)
        drag_world = -p.drag_coeff * self.velocity

        # Total force
        force_world = thrust_world + gravity_world + drag_world

        return force_world, torque_body

    def _state_derivative(self, state: np.ndarray, force_world: np.ndarray,
                          torque_body: np.ndarray) -> np.ndarray:
        """
        Compute state derivative ds/dt for RK4 integration.

        Args:
            state: 13D state vector
            force_world: total force in world frame
            torque_body: total torque in body frame

        Returns:
            ds/dt: 13D state derivative
        """
        p = self.params
        pos = state[0:3]
        quat = quaternion_normalize(state[3:7])
        vel = state[7:10]
        omega = state[10:13]

        # Position derivative: dp/dt = v
        dp = vel

        # Quaternion derivative: dq/dt = 0.5 * q * [0, omega]
        dq = quaternion_derivative(quat, omega)

        # Linear acceleration: dv/dt = F/m
        dv = force_world / p.mass

        # Angular acceleration: dw/dt = I^{-1} (tau - w x (I w))
        I = p.inertia  # diagonal [Ixx, Iyy, Izz]
        Iw = I * omega
        gyroscopic = np.cross(omega, Iw)
        dw = (torque_body - gyroscopic) / I

        return np.concatenate([dp, dq, dv, dw])

    def step(self, thrust_cmd: np.ndarray,
             disturbance_force: np.ndarray = None,
             disturbance_torque: np.ndarray = None):
        """
        Advance dynamics by one timestep using RK4 integration.

        Args:
            thrust_cmd: (4,) commanded motor thrusts, each in [0, motor_max_thrust]
            disturbance_force: (3,) external force disturbance in world frame
            disturbance_torque: (3,) external torque disturbance in body frame
        """
        # Clamp motor commands
        f_max = self.params.motor_max_thrust
        thrust_cmd = np.clip(thrust_cmd, 0.0, f_max)

        # Update motor dynamics (first-order lag)
        self._motor_dynamics(thrust_cmd)

        # Compute forces and torques from motors
        force_world, torque_body = self._compute_forces_torques()

        # Add disturbances
        if disturbance_force is not None:
            force_world = force_world + disturbance_force
        if disturbance_torque is not None:
            torque_body = torque_body + disturbance_torque

        # RK4 integration
        s = self.state
        dt = self.dt

        k1 = self._state_derivative(s, force_world, torque_body)
        k2 = self._state_derivative(s + 0.5 * dt * k1, force_world, torque_body)
        k3 = self._state_derivative(s + 0.5 * dt * k2, force_world, torque_body)
        k4 = self._state_derivative(s + dt * k3, force_world, torque_body)

        s_new = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Update state
        self.state = s_new

    def get_hover_thrust(self) -> float:
        """Compute per-motor thrust needed to hover."""
        return self.params.mass * self.params.gravity / 4.0

"""
Cascade PID Position Controller for Quadrotor (NED frame).

4-level cascade:
  Level 1: Position P   → velocity setpoint
  Level 2: Velocity PI  → desired acceleration → thrust vector → F_total + R_des
  Level 3: Attitude P   → angular rate setpoint  (SO3 error, body frame)
  Level 4: Rate P       → torques               (body frame)
  → Motor inverse mixer → 4D action ∈ [-1, 1]

Coordinate convention:
  World NED: +X North, +Y East, +Z Down.  Ground = Z=0, flying = Z<0.
  Gravity:   [0, 0, g] in world frame.
  Motor thrust in body frame: [0, 0, -F_total] (upward = -Z_body).
  Body-to-world rotation matrix R: R[:,2] = body-Z in world = [0,0,1] when level.

Newton's law:
  m*a = [0,0,m*g] - F_total*R[:,2]
  → desired thrust vector = m*(g_vec - a_des)  where g_vec=[0,0,g]
"""

import numpy as np


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [qw,qx,qy,qz] → 3×3 body-to-world rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qw*qz),   2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)],
    ])


class CascadePIDController:
    """
    Cascade PID for quadrotor waypoint tracking in NED frame.

    Usage::

        ctrl = CascadePIDController(env.dynamics.params)
        ctrl.reset()
        action = ctrl.compute_action(env.dynamics.state, env.target_position)
    """

    def __init__(self, params,
                 Kp_pos:     float = 1.5,
                 Kp_vel:     float = 3.0,
                 Ki_vel:     float = 0.5,
                 vel_int_limit: float = 1.0,
                 Kp_att:     float = 8.0,
                 Kp_att_yaw: float = 2.0,
                 omega_max:  float = 2.0,
                 Kp_rate:    float = 0.15,
                 vel_max:    float = 2.0,
                 yaw_sp:     float = 0.0,
                 dt:         float = 0.02):
        """
        Args:
            params:       QuadrotorParams (mass, arm_length, gravity, etc.)
            Kp_pos:       position → velocity gain [m/s per m]
            Kp_vel:       velocity → acceleration gain [m/s² per m/s]
            Ki_vel:       velocity integral gain [m/s² per (m·s)]
            vel_int_limit: anti-windup clip for velocity integral [m/s]
            Kp_att:       attitude → rate gain, roll/pitch [rad/s per rad]
            Kp_att_yaw:   attitude → rate gain, yaw [rad/s per rad]
            omega_max:    clip for angular rate setpoint [rad/s]
            Kp_rate:      rate → torque gain [Nm per rad/s]
            vel_max:      clip for velocity setpoint [m/s]
            yaw_sp:       fixed yaw setpoint [rad] (0 = North-facing)
            dt:           outer loop timestep [s] (= env.dt_outer = 0.02)
        """
        self.p         = params
        self.Kp_pos    = Kp_pos
        self.Kp_vel    = Kp_vel
        self.Ki_vel    = Ki_vel
        self.vel_int_limit = vel_int_limit
        self.Kp_att    = Kp_att
        self.Kp_att_yaw = Kp_att_yaw
        self.omega_max = omega_max
        self.Kp_rate   = Kp_rate
        self.vel_max   = vel_max
        self.yaw_sp    = yaw_sp
        self.dt        = dt
        self._vel_int  = np.zeros(3)

    def reset(self):
        """Reset integrators — call at the start of every episode."""
        self._vel_int = np.zeros(3)

    def _outer_loop(self, state: np.ndarray, target: np.ndarray):
        """
        Cascade Levels 1–3: position/velocity/attitude → collective thrust + rate
        setpoint. Shared by both the motor-output path (``compute_action``) and the
        CTBR-output path (``compute_ctbr_action``).

        Args:
            state:  13D dynamics state [pos(3), quat(4), vel(3), omega(3)]
            target: 3D target position in world NED frame

        Returns:
            F_total:   collective thrust [N]
            omega_cmd: desired body rates [rad/s] (3,), clipped to ±omega_max
        """
        p = self.p
        pos   = state[0:3]
        quat  = state[3:7]    # [qw, qx, qy, qz]
        vel   = state[7:10]   # world NED

        R = _quat_to_matrix(quat)   # body-to-world

        # ── Level 1: Position P → velocity setpoint (world NED) ──────────────
        pos_err = target - pos
        vel_cmd = np.clip(self.Kp_pos * pos_err, -self.vel_max, self.vel_max)

        # ── Level 2: Velocity PI → acceleration setpoint (world NED) ─────────
        vel_err = vel_cmd - vel
        self._vel_int += vel_err * self.dt
        np.clip(self._vel_int, -self.vel_int_limit, self.vel_int_limit,
                out=self._vel_int)
        accel_cmd = self.Kp_vel * vel_err + self.Ki_vel * self._vel_int

        # ── Level 2b: Thrust vector decomposition ─────────────────────────────
        # Newton: m*a = [0,0,m*g] - F_total*R[:,2]
        # → F_total*R[:,2] = m*(g_vec - a_des) = thrust_vec
        g_vec = np.array([0.0, 0.0, p.gravity])
        thrust_vec = p.mass * (g_vec - accel_cmd)

        F_total = np.linalg.norm(thrust_vec)
        F_total = max(F_total, 0.1 * p.mass * p.gravity)   # floor at 10 % hover thrust

        # ── Level 3: Attitude P → rate setpoint (body frame, SO3 error) ───────
        des_z   = thrust_vec / F_total          # desired body-Z axis in world
        R_des   = self._R_from_des_z(des_z)
        R_err   = R.T @ R_des                   # error rotation in body frame: R_err = R^T R_des
        # Vee-map the skew-symmetric part → 3D attitude error vector
        att_err = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ])
        omega_cmd = np.array([
            self.Kp_att     * att_err[0],
            self.Kp_att     * att_err[1],
            self.Kp_att_yaw * att_err[2],
        ])
        omega_cmd = np.clip(omega_cmd, -self.omega_max, self.omega_max)
        return F_total, omega_cmd

    def compute_action(self, state: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Compute 4D motor action for one outer-loop step.

        Args:
            state:  13D dynamics state [pos(3), quat(4), vel(3), omega(3)]
            target: 3D target position in world NED frame

        Returns:
            action: (4,) float32, each ∈ [-1, 1]
        """
        p = self.p
        omega = state[10:13]  # body frame (current rates, for the Level 4 rate loop)

        F_total, omega_cmd = self._outer_loop(state, target)

        # ── Level 4: Rate P → torques (body frame) ────────────────────────────
        omega_err = omega_cmd - omega
        torques   = self.Kp_rate * omega_err     # [tau_x, tau_y, tau_z]

        # ── Motor inverse mixer (X config) ────────────────────────────────────
        # Mixer: F=ΣTi, tau_x=L*(T1-T2-T3+T4), tau_y=L*(T1+T2-T3-T4), tau_z=c*(-T1+T2-T3+T4)
        L = p.arm_length
        c = p.torque_coeff
        tx, ty, tz = torques
        T = np.array([
            (F_total + tx/L + ty/L - tz/c) / 4.0,   # Motor 1 FR
            (F_total - tx/L + ty/L + tz/c) / 4.0,   # Motor 2 FL
            (F_total - tx/L - ty/L - tz/c) / 4.0,   # Motor 3 RL
            (F_total + tx/L - ty/L + tz/c) / 4.0,   # Motor 4 RR
        ])
        T = np.clip(T, 0.0, p.motor_max_thrust)

        # Map [0, motor_max_thrust] → [-1, 1]
        action = T / p.motor_max_thrust * 2.0 - 1.0
        return action.astype(np.float32)

    def compute_ctbr_action(self, state: np.ndarray, target: np.ndarray,
                            f_c_max: float, omega_max: np.ndarray) -> np.ndarray:
        """
        Compute a normalized CTBR action for the v4 (INDI) env, bypassing the
        motor mixer. Output is the exact inverse of
        ``QuadrotorEnvV4._decode_action``:
            F_c   = (a[0] + 1) * 0.5 * f_c_max      →  a[0]   = F_total / f_c_max * 2 - 1
            omega = a[1:4] * omega_max              →  a[1:4] = omega_cmd / omega_max

        Args:
            state:     13D dynamics state [pos(3), quat(4), vel(3), omega(3)]
            target:    3D target position in world NED frame
            f_c_max:   env.F_c_max  (collective-thrust normalisation, [N])
            omega_max: env.omega_max  (per-axis rate normalisation, [rad/s], (3,))

        Returns:
            action: (4,) float32  [F_c_norm, wx_norm, wy_norm, wz_norm] ∈ [-1, 1]
        """
        F_total, omega_cmd = self._outer_loop(state, target)
        f_c_norm   = np.clip(F_total / f_c_max * 2.0 - 1.0, -1.0, 1.0)
        omega_norm = np.clip(omega_cmd / np.asarray(omega_max, dtype=float),
                             -1.0, 1.0)
        return np.concatenate([[f_c_norm], omega_norm]).astype(np.float32)

    def _R_from_des_z(self, des_z: np.ndarray) -> np.ndarray:
        """
        Build desired rotation matrix given desired body-Z direction and yaw setpoint.

        des_z must be a unit vector.  The resulting R satisfies R[:,2] = des_z
        and the body-X axis points as close as possible to the yaw setpoint heading.
        """
        # Perpendicular to the desired heading in the XY plane
        # (avoids the gimbal-lock degenerate case of cross(heading, des_z))
        y_c = np.array([-np.sin(self.yaw_sp), np.cos(self.yaw_sp), 0.0])
        des_x = np.cross(y_c, des_z)
        norm_x = np.linalg.norm(des_x)
        if norm_x < 1e-6:
            # des_z nearly aligned with y_c (extreme pitch): fall back to world-X
            des_x = np.array([1.0, 0.0, 0.0])
        else:
            des_x /= norm_x
        des_y = np.cross(des_z, des_x)
        return np.column_stack([des_x, des_y, des_z])

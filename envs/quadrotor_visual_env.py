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
from .quadrotor_dynamics import quaternion_to_rotation_matrix


class QuadrotorVisualEnv(gym.Wrapper):
    """
    Visual wrapper that renders synthetic FPV images from quadrotor state.
    """

    def __init__(self, env: QuadrotorEnv, image_size: int = 64, dr_enabled: bool = True):
        super().__init__(env)
        self.image_size = image_size
        self.dr_enabled = dr_enabled

        # Per-episode DR state (initialised to neutral values; overwritten in reset())
        self._dr_sky_offset    = np.zeros(3, dtype=np.int32)
        self._dr_gnd_offset    = np.zeros(3, dtype=np.int32)
        self._dr_brightness    = 1.0
        self._dr_focal_scale   = 0.40
        self._dr_crosshair_d   = 0
        self._dr_horizon_color = np.array([200, 200, 200], dtype=np.uint8)

        # Override observation space to include images
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255,
                shape=(3, image_size, image_size),
                dtype=np.uint8,
            ),
            'state': self.env.observation_space,
        })

    def _randomize_episode(self):
        """Sample per-episode DR parameters."""
        self._dr_sky_offset    = np.random.randint(-40, 41, size=3).astype(np.int32)
        self._dr_gnd_offset    = np.random.randint(-40, 41, size=3).astype(np.int32)
        self._dr_brightness    = float(np.random.uniform(0.7, 1.3))
        self._dr_focal_scale   = float(np.random.uniform(0.30, 0.50))
        self._dr_crosshair_d   = int(np.random.randint(-2, 3))   # -2 to +2 px
        self._dr_horizon_color = np.random.randint(150, 256, size=3).astype(np.uint8)

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        state_obs, info = self.env.reset(**kwargs)
        if self.dr_enabled:
            self._randomize_episode()
        image = self._render_fpv()
        return {'image': image, 'state': state_obs}, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        state_obs, reward, terminated, truncated, info = self.env.step(action)
        image = self._render_fpv()
        obs = {'image': image, 'state': state_obs}
        return obs, reward, terminated, truncated, info

    def _render_fpv(self) -> np.ndarray:
        """
        Render a synthetic FPV image from current drone state.

        Returns:
            image: (3, H, W) uint8 RGB image
        """
        H = W = self.image_size
        image = np.zeros((H, W, 3), dtype=np.uint8)

        dynamics = self.env.dynamics
        R = dynamics.get_rotation_matrix()
        pos = dynamics.position
        target = self.env.target_position

        # Altitude (NED: negative Z is up, so altitude = -pos[2])
        altitude = -pos[2]

        # 1. Sky/ground gradient based on pitch
        # R[2,0] = projection of body X (forward) onto world Z (down): +ve = nose down
        # At level hover R[2,0]=0 → horizon at H/2; nose down → horizon moves up (smaller y)
        pitch_factor = np.clip(R[2, 0], -1, 1)
        horizon_y = int(H * 0.5 - pitch_factor * H * 0.3)

        # Sky (blue gradient) + per-episode color offset
        so = self._dr_sky_offset
        for y in range(horizon_y):
            t = y / max(horizon_y, 1)
            sky_r = int(np.clip(50  + 100 * t + so[0], 0, 255))
            sky_g = int(np.clip(100 + 80  * t + so[1], 0, 255))
            sky_b = int(np.clip(180 + 60  * t + so[2], 0, 255))
            image[y, :] = [sky_r, sky_g, sky_b]

        # Ground (brown/green gradient) + per-episode color offset
        go = self._dr_gnd_offset
        for y in range(horizon_y, H):
            t = (y - horizon_y) / max(H - horizon_y, 1)
            gnd_r = int(np.clip(60  + 40 * t + go[0], 0, 255))
            gnd_g = int(np.clip(100 + 30 * t + go[1], 0, 255))
            gnd_b = int(np.clip(40  + 20 * t + go[2], 0, 255))
            image[y, :] = [gnd_r, gnd_g, gnd_b]

        # 2. Horizon line (per-episode color)
        hl = np.clip(horizon_y, 1, H - 2)
        # Roll shifts horizon angle
        roll_shift = int(R[2, 1] * W * 0.3)  # R[2,1] = sin(roll) for ZYX NED convention
        horizon_rgb = self._dr_horizon_color.tolist()
        for x in range(W):
            y_line = hl + int(roll_shift * (x - W // 2) / (W // 2))
            y_line = np.clip(y_line, 0, H - 1)
            image[y_line, x] = horizon_rgb

        # 3. Target direction marker
        # Compute target direction in body frame
        target_dir_world = target - pos
        target_dist = np.linalg.norm(target_dir_world)
        if target_dist > 0.01:
            target_dir_body = R.T @ (target_dir_world / target_dist)
            # Project to image plane (simple pinhole, per-episode focal scale)
            # body X is forward, Y is right, Z is down
            tx = target_dir_body[1]  # right in body = right on image
            ty = target_dir_body[2]  # down in body = down on image
            forward = target_dir_body[0]  # forward component

            if forward > 0:  # target is in front
                # Map to pixel coordinates
                focal = self._dr_focal_scale
                px = int(W // 2 + tx / (forward + 0.1) * W * focal)
                py = int(H // 2 + ty / (forward + 0.1) * H * focal)
                px = np.clip(px, 4, W - 5)
                py = np.clip(py, 4, H - 5)

                # Draw crosshair (red) with per-episode size perturbation
                size = max(2, min(6, int(6 / (target_dist + 0.5)) + self._dr_crosshair_d))
                for d in range(-size, size + 1):
                    yc = np.clip(py + d, 0, H - 1)
                    xc = np.clip(px + d, 0, W - 1)
                    image[yc, px] = [255, 50, 50]
                    image[py, xc] = [255, 50, 50]

        # 4. Altitude indicator (left edge, green bar)
        alt_normalized = np.clip(altitude / 5.0, 0, 1)
        bar_height = int(alt_normalized * (H - 4))
        for y in range(H - 2 - bar_height, H - 2):
            image[y, 1:4] = [50, 255, 50]

        # 5. Center reticle (white dot)
        cx, cy = W // 2, H // 2
        image[cy - 1:cy + 2, cx - 1:cx + 2] = [255, 255, 255]

        # --- Domain Randomization: per-frame ---
        if self.dr_enabled:
            # Gaussian pixel noise (sigma=5)
            noise = np.random.normal(0, 5, image.shape).astype(np.int32)
            image = np.clip(image.astype(np.int32) + noise, 0, 255).astype(np.uint8)

            # Global brightness scaling (applied after noise)
            image = np.clip(image.astype(np.float32) * self._dr_brightness,
                            0, 255).astype(np.uint8)

        # Convert to CHW format
        return image.transpose(2, 0, 1).copy()


    def _render_depth(self) -> np.ndarray:
        """
        Render a per-pixel depth map from the current drone state.

        Uses ground-plane ray intersection in NED coordinates.
        Each pixel's ray is back-projected through the pinhole camera model,
        rotated to the world frame, and intersected with the flat ground (Z=0).

        Returns:
            depth: (1, H, W) uint8, value 255 corresponds to depth >= 10 m.
                   Pixels whose ray points upward (no ground intersection) are
                   clamped to 255 (max depth).
        """
        H = W = self.image_size
        dynamics = self.env.dynamics
        R   = dynamics.get_rotation_matrix()   # body-to-world (3×3)
        pos = dynamics.position                # world NED [x, y, z]

        focal = self._dr_focal_scale           # per-episode focal scale (0.30–0.50)

        # Build per-pixel ray directions in body frame
        # body X = forward, Y = right, Z = down
        ys = np.arange(H, dtype=np.float32)    # row indices
        xs = np.arange(W, dtype=np.float32)    # col indices

        # Normalised image-plane coordinates
        dy = (ys - H * 0.5) / (H * focal)     # (H,)  positive = body-Z down
        dx = (xs - W * 0.5) / (W * focal)     # (W,)  positive = body-Y right

        # Shape (H, W, 3): [forward=1, right=dx, down=dy]
        rays_body = np.empty((H, W, 3), dtype=np.float32)
        rays_body[..., 0] = 1.0
        rays_body[..., 1] = dx[np.newaxis, :]
        rays_body[..., 2] = dy[:, np.newaxis]

        # Normalise each ray to unit length
        norms = np.linalg.norm(rays_body, axis=-1, keepdims=True)
        rays_body /= norms

        # Rotate rays to world frame: ray_world = R @ ray_body  (body-to-world)
        rays_flat   = rays_body.reshape(-1, 3)          # (H*W, 3)
        rays_world  = (R @ rays_flat.T).T               # (H*W, 3)
        rays_world  = rays_world.reshape(H, W, 3)

        # Ground-plane intersection:  pos[2] + t * dz = 0  →  t = -pos[2] / dz
        # NED: ground is Z = 0; pos[2] < 0 means drone is above ground
        dz          = rays_world[..., 2]                # (H, W)
        going_down  = dz > 1e-4                         # rays heading toward ground
        safe_dz     = np.where(going_down, dz, 1.0)    # avoid div-by-zero
        t           = np.where(going_down, -pos[2] / safe_dz, 50.0)
        t           = np.clip(t, 0.0, 50.0)

        # Normalise to uint8: 10 m → 255; pixels beyond 10 m are saturated at 255
        depth_u8 = np.clip(t / 10.0 * 255.0, 0, 255).astype(np.uint8)
        return depth_u8[np.newaxis, :, :]               # (1, H, W)


def make_visual_env(config_path: str = "configs/quadrotor.yaml",
                    image_size: int = 64,
                    dr_enabled: bool = True) -> QuadrotorVisualEnv:
    """Factory function to create QuadrotorVisualEnv."""
    env = QuadrotorEnv(config_path=config_path)
    return QuadrotorVisualEnv(env, image_size=image_size, dr_enabled=dr_enabled)

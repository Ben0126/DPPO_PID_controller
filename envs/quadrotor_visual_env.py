"""
Quadrotor Visual Environment Wrapper

Wraps QuadrotorEnv and adds synthetic FPV camera rendering.
Produces 64x64 RGB images encoding:
  - Horizon line (based on drone attitude)
  - Target direction marker
  - Ground/sky gradient (based on altitude)

Observation space: Dict({"image": Box(0,255,(3,64,64)), "state": Box(...)})
The diffusion policy uses the image; the state is for logging/reward.
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

    def __init__(self, env: QuadrotorEnv, image_size: int = 64):
        super().__init__(env)
        self.image_size = image_size

        # Override observation space to include images
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255,
                shape=(3, image_size, image_size),
                dtype=np.uint8,
            ),
            'state': self.env.observation_space,
        })

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        state_obs, info = self.env.reset(**kwargs)
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
        # Body Z axis in world frame = R[:, 2]
        # If R[2,2] > 0 (body Z roughly aligned with world Z), drone is upright
        pitch_factor = np.clip(R[2, 2], -1, 1)  # ~cos(tilt)
        horizon_y = int(H * 0.5 * (1 + pitch_factor * 0.3))  # horizon shifts with pitch

        # Sky (blue gradient)
        for y in range(horizon_y):
            t = y / max(horizon_y, 1)
            sky_r = int(50 + 100 * t)
            sky_g = int(100 + 80 * t)
            sky_b = int(180 + 60 * t)
            image[y, :] = [sky_r, sky_g, sky_b]

        # Ground (brown/green gradient)
        for y in range(horizon_y, H):
            t = (y - horizon_y) / max(H - horizon_y, 1)
            gnd_r = int(60 + 40 * t)
            gnd_g = int(100 + 30 * t)
            gnd_b = int(40 + 20 * t)
            image[y, :] = [gnd_r, gnd_g, gnd_b]

        # 2. Horizon line
        hl = np.clip(horizon_y, 1, H - 2)
        # Roll shifts horizon angle
        roll_shift = int(R[0, 2] * W * 0.3)  # R[0,2] relates to roll
        for x in range(W):
            y_line = hl + int(roll_shift * (x - W // 2) / (W // 2))
            y_line = np.clip(y_line, 0, H - 1)
            image[y_line, x] = [200, 200, 200]

        # 3. Target direction marker
        # Compute target direction in body frame
        target_dir_world = target - pos
        target_dist = np.linalg.norm(target_dir_world)
        if target_dist > 0.01:
            target_dir_body = R.T @ (target_dir_world / target_dist)
            # Project to image plane (simple pinhole)
            # body X is forward, Y is right, Z is down
            tx = target_dir_body[1]  # right in body = right on image
            ty = target_dir_body[2]  # down in body = down on image
            forward = target_dir_body[0]  # forward component

            if forward > 0:  # target is in front
                # Map to pixel coordinates
                px = int(W // 2 + tx / (forward + 0.1) * W * 0.4)
                py = int(H // 2 + ty / (forward + 0.1) * H * 0.4)
                px = np.clip(px, 4, W - 5)
                py = np.clip(py, 4, H - 5)

                # Draw crosshair (red)
                size = max(2, min(6, int(6 / (target_dist + 0.5))))
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

        # Convert to CHW format
        return image.transpose(2, 0, 1).copy()


def make_visual_env(config_path: str = "configs/quadrotor.yaml",
                    image_size: int = 64) -> QuadrotorVisualEnv:
    """Factory function to create QuadrotorVisualEnv."""
    env = QuadrotorEnv(config_path=config_path)
    return QuadrotorVisualEnv(env, image_size=image_size)

"""
Flow Matching Policy v5.0 — Cross-Attention + State Prediction Auxiliary Loss.

Diagnosis from v4 distillation (200 updates, crash_rate stuck at 100%):
  Vision feature collapse + information asymmetry.  The H4 vision encoder
  only saw hover-distribution images during BC, so when DAgger drove the
  student into OOD swift-perturbation states, the vision features carried
  no useful physics and the IMU pathway dominated.

v5 architecture (two upgrades, dims chosen so H4 weights still partial-load):

  VisionEncoderV5 exposes BOTH pooled (B,256) AND spatial (B,256,4,4)
  IMUEncoder unchanged (6 -> 1024 -> 512)

  CrossAttentionIMU2Vision:
      Q = Linear(imu_feat, 256)                  # IMU asks "where to look"
      K = V = spatial.flatten(2).T               # (B, 16, 256) vision tokens
      attended = MultiHeadAttn(Q, K, V) -> (B, 256)

  global_cond = cat([attended (256), imu_feat (512)]) = 768D
      (deliberately identical to H4 so flow_net weights transfer 1:1)

  StatePredictor (training-only):
      MLP(pooled vision 256 -> 256 -> 15)
      L_state = MSE(pred, normalised_15d_state)
      auxiliary loss forces vision encoder to learn physics from pixels.

  Total loss: L = L_flow + lambda_state * L_state (+ optional tilt term)

Inference (`_encode`, `predict_action`) routes through the same attended path.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_encoder_v5 import VisionEncoderV5
from .conditional_unet1d import ConditionalUnet1d


class CrossAttentionIMU2Vision(nn.Module):
    """IMU as query attending to spatial vision tokens."""

    def __init__(self, imu_dim: int = 512, vis_dim: int = 256, n_heads: int = 8):
        super().__init__()
        self.q_proj = nn.Linear(imu_dim, vis_dim)
        self.attn   = nn.MultiheadAttention(vis_dim, n_heads, batch_first=True)
        self.norm   = nn.LayerNorm(vis_dim)

    def forward(self, imu_feat: torch.Tensor,
                vis_spatial: torch.Tensor) -> torch.Tensor:
        """
        imu_feat:    (B, imu_dim)
        vis_spatial: (B, vis_dim, H, W)  e.g. (B, 256, 4, 4)
        returns:     (B, vis_dim)
        """
        tokens = vis_spatial.flatten(2).transpose(1, 2)        # (B, H*W, vis_dim)
        q      = self.q_proj(imu_feat).unsqueeze(1)            # (B, 1, vis_dim)
        out, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.norm(out.squeeze(1))                       # (B, vis_dim)


class StatePredictor(nn.Module):
    """Vision-only MLP that predicts normalised 15D state."""

    def __init__(self, vis_dim: int = 256, hidden_dim: int = 256,
                 state_dim: int = 15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vis_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, vis_pooled: torch.Tensor) -> torch.Tensor:
        return self.net(vis_pooled)


class FlowMatchingPolicyV5(nn.Module):
    """
    Flow Matching policy for CTBR control with IMU->Vision cross-attention
    and auxiliary state prediction head.
    """

    def __init__(
        self,
        vision_feature_dim: int = 256,
        imu_feature_dim: int = 512,
        time_embed_dim: int = 128,
        down_dims: tuple = (256, 512),
        T_obs: int = 2,
        T_pred: int = 8,
        action_dim: int = 4,
        n_inference_steps: int = 1,
        t_embed_scale: int = 999,
        cross_attn_heads: int = 8,
        state_predictor_hidden: int = 256,
        state_dim: int = 15,
    ):
        super().__init__()
        self.T_pred = T_pred
        self.action_dim = action_dim
        self.n_inference_steps = n_inference_steps
        self.t_embed_scale = t_embed_scale

        # attended_vision (vision_feature_dim) + imu_feat (imu_feature_dim) = 768
        global_cond_dim = vision_feature_dim + imu_feature_dim

        self.vision_encoder = VisionEncoderV5(
            in_channels=T_obs * 3,
            feature_dim=vision_feature_dim,
        )

        self.imu_encoder = nn.Sequential(
            nn.Linear(6, 1024),
            nn.ReLU(),
            nn.Linear(1024, imu_feature_dim),
            nn.ReLU(),
        )

        self.cross_attn = CrossAttentionIMU2Vision(
            imu_dim=imu_feature_dim,
            vis_dim=vision_feature_dim,
            n_heads=cross_attn_heads,
        )

        self.state_predictor = StatePredictor(
            vis_dim=vision_feature_dim,
            hidden_dim=state_predictor_hidden,
            state_dim=state_dim,
        )

        # Kept for compatibility / optional aux supervision (same as v4)
        self.tilt_head = nn.Linear(imu_feature_dim, 1)

        self.flow_net = ConditionalUnet1d(
            action_dim=action_dim,
            feature_dim=global_cond_dim,
            time_embed_dim=time_embed_dim,
            down_dims=down_dims,
            kernel_size=5,
            n_groups=8,
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, images: torch.Tensor,
                imu: torch.Tensor) -> torch.Tensor:
        """Returns global_cond (B, 768) for flow_net conditioning."""
        _, vis_spatial = self.vision_encoder(images, return_spatial=True)
        imu_feat       = self.imu_encoder(imu)
        attended       = self.cross_attn(imu_feat, vis_spatial)
        return torch.cat([attended, imu_feat], dim=-1)

    def _encode_full(self, images: torch.Tensor,
                     imu: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (global_cond, vis_pooled, imu_feat) — used by compute_loss."""
        vis_pooled, vis_spatial = self.vision_encoder(images, return_spatial=True)
        imu_feat                 = self.imu_encoder(imu)
        attended                 = self.cross_attn(imu_feat, vis_spatial)
        global_cond              = torch.cat([attended, imu_feat], dim=-1)
        return global_cond, vis_pooled, imu_feat

    def _t_to_int(self, t: torch.Tensor) -> torch.Tensor:
        return (t * self.t_embed_scale).long().clamp(0, self.t_embed_scale)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        images: torch.Tensor,
        imu: torch.Tensor,
        actions: torch.Tensor,
        states_gt: Optional[torch.Tensor] = None,
        lambda_state: float = 0.1,
        tilt_gt: Optional[torch.Tensor] = None,
        lambda_tilt: float = 0.1,
        return_components: bool = False,
        state_loss_type: str = 'mse',
    ):
        """
        Args:
            images:    (B, T_obs*3, H, W)
            imu:       (B, 6) normalised IMU
            actions:   (B, action_dim, T_pred) CTBR in [-1, 1]
            states_gt: (B, 15) normalised state (PPO obs_rms) — enables aux loss
            tilt_gt:   (B,)    radians — enables tilt aux loss
            return_components: if True, additionally returns dict of scalar losses

        Returns:
            total loss (and optionally dict of components for logging)
        """
        B = actions.shape[0]
        device = actions.device

        global_cond, vis_pooled, imu_feat = self._encode_full(images, imu)

        t = torch.rand(B, device=device)
        eps = torch.randn_like(actions)
        t_expand = t[:, None, None]
        x_t = (1.0 - t_expand) * actions + t_expand * eps
        v_target = eps - actions

        v_pred = self.flow_net(x_t, self._t_to_int(t), global_cond)
        flow_loss = F.mse_loss(v_pred, v_target)

        total = flow_loss
        components = {'flow_loss': flow_loss.detach()}

        if states_gt is not None:
            state_pred = self.state_predictor(vis_pooled)
            if state_loss_type == 'huber':
                state_loss = F.smooth_l1_loss(state_pred, states_gt, beta=1.0)
            elif state_loss_type == 'mse':
                state_loss = F.mse_loss(state_pred, states_gt)
            else:
                raise ValueError(f"unknown state_loss_type: {state_loss_type}")
            total = total + lambda_state * state_loss
            components['state_loss'] = state_loss.detach()

        if tilt_gt is not None:
            tilt_pred = self.tilt_head(imu_feat).squeeze(-1)
            tilt_loss = F.mse_loss(tilt_pred, tilt_gt)
            total = total + lambda_tilt * tilt_loss
            components['tilt_loss'] = tilt_loss.detach()

        if return_components:
            return total, components
        return total

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(
        self,
        images: torch.Tensor,
        imu: torch.Tensor,
        n_steps: Optional[int] = None,
        _fixed_x1: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        B = images.shape[0]
        device = images.device
        n = n_steps if n_steps is not None else self.n_inference_steps

        global_cond = self._encode(images, imu)

        x = _fixed_x1 if _fixed_x1 is not None else \
            torch.randn(B, self.action_dim, self.T_pred, device=device) * temperature

        dt = 1.0 / n
        for i in range(n):
            t_val = 1.0 - i * dt
            t_batch = torch.full((B,), t_val, device=device)
            v = self.flow_net(x, self._t_to_int(t_batch), global_cond)
            x = x - dt * v

        return x

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location='cpu'))

    def transfer_from_h4(self, h4_path: str, verbose: bool = True
                         ) -> Tuple[int, int]:
        """
        Partial-load weights from a v4 (H4) FlowMatchingPolicyV4 checkpoint.

        Shapes/keys match for: vision_encoder.encoder.*, vision_encoder.fc.*,
        imu_encoder.*, flow_net.*, tilt_head.*.
        cross_attn.* and state_predictor.* remain randomly initialised.
        """
        h4_sd  = torch.load(h4_path, map_location='cpu')
        own_sd = self.state_dict()
        transferred, skipped = 0, 0
        for k, v in h4_sd.items():
            if k in own_sd and own_sd[k].shape == v.shape:
                own_sd[k] = v
                transferred += 1
            else:
                skipped += 1
        self.load_state_dict(own_sd)
        if verbose:
            print(f"[transfer_from_h4] transferred {transferred} tensors, "
                  f"skipped {skipped} (cross_attn / state_predictor stay random)")
        return transferred, skipped

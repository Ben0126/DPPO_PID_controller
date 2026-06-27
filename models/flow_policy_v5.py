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
        task_dim: int = 0,
    ):
        super().__init__()
        self.T_pred = T_pred
        self.action_dim = action_dim
        self.n_inference_steps = n_inference_steps
        self.t_embed_scale = t_embed_scale
        self.task_dim = task_dim

        # attended_vision (vision_feature_dim) + imu_feat (imu_feature_dim) + task_dim
        global_cond_dim = vision_feature_dim + imu_feature_dim + task_dim

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
                imu: torch.Tensor,
                task_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns global_cond (B, global_cond_dim) for flow_net conditioning."""
        _, vis_spatial = self.vision_encoder(images, return_spatial=True)
        imu_feat       = self.imu_encoder(imu)
        attended       = self.cross_attn(imu_feat, vis_spatial)
        cond_list = [attended, imu_feat]
        if self.task_dim > 0:
            if task_cond is None:
                # default to hover
                task_cond = torch.tensor([[1.0, 0.0]], device=images.device).expand(images.shape[0], -1)
            cond_list.append(task_cond)
        return torch.cat(cond_list, dim=-1)

    def _encode_full(self, images: torch.Tensor,
                     imu: torch.Tensor,
                     task_cond: Optional[torch.Tensor] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (global_cond, vis_pooled, imu_feat) — used by compute_loss."""
        vis_pooled, vis_spatial = self.vision_encoder(images, return_spatial=True)
        imu_feat                 = self.imu_encoder(imu)
        attended                 = self.cross_attn(imu_feat, vis_spatial)
        cond_list = [attended, imu_feat]
        if self.task_dim > 0:
            if task_cond is None:
                task_cond = torch.tensor([[1.0, 0.0]], device=images.device).expand(images.shape[0], -1)
            cond_list.append(task_cond)
        global_cond              = torch.cat(cond_list, dim=-1)
        return global_cond, vis_pooled, imu_feat

    def _t_to_int(self, t: torch.Tensor) -> torch.Tensor:
        return (t * self.t_embed_scale).long().clamp(0, self.t_embed_scale)

    @staticmethod
    def _dispersive_loss(features: torch.Tensor) -> torch.Tensor:
        """
        LEGACY (pre-2026-06-22) hand-rolled log-distance repulsion, applied to the
        off-action-path ``vis_pooled``. Retained only for reproducing the original
        P2 ablation and `measure_feature_collapse.py`. NOT faithful to [13]/[14];
        use ``_dispersive_loss_infonce`` on a flow-net intermediate for new work.
        L_disp = -mean( log(||fi - fj|| + ε) )  for i ≠ j
        """
        B = features.shape[0]
        if B < 2:
            return features.sum() * 0.0
        diff = features.unsqueeze(1) - features.unsqueeze(0)    # (B, B, D)
        dist = torch.norm(diff, dim=-1)                          # (B, B)
        mask = 1 - torch.eye(B, device=features.device)
        loss = -torch.log(dist + 1e-6) * mask
        return loss.sum() / (B * (B - 1))

    @staticmethod
    def _dispersive_loss_infonce(features: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
        """
        Faithful Dispersive Loss — InfoNCE-L2 variant of Wang & He (arXiv:2506.09027),
        matching the OFFICIAL reference implementation (github.com/raywang4/DispLoss),
        not just the paper's Algorithm 1:
            D     = ||z_i - z_j||^2 / d        # per-dimension normalised squared L2
            L_disp = log E_{i,j}[ exp( -D / tau ) ]
        The `/ d` (flattened feature dim) is in the official code but OMITTED from the
        paper's printed Algorithm 1; without it the loss saturates (exp(-O(d)/tau)->0,
        zero gradient) on GroupNorm'd flat features. The i=j diagonal (D=0 -> exp(0)=1)
        is intentionally kept, matching the reference's full-BxB-matrix mean.
        Applied to the generative network's intermediate (flow-net mid-block) features.
        Default tau=0.5 per [13]/[14]; temperature is robust over a wide range [13, Tab.4].
        """
        B = features.shape[0]
        if B < 2:
            return features.sum() * 0.0
        d = features.shape[1]
        D = (torch.cdist(features, features, p=2) ** 2) / d     # (B, B) per-dim squared L2
        return torch.log(torch.exp(-D / tau).mean())

    @staticmethod
    def _dispersive_loss_cosine(features: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
        """
        Scale-invariant (unit-sphere) variant of the InfoNCE-L2 Dispersive Loss
        (Phase 6 / RESEARCH_PLAN_v7 Direction 4). Each feature vector is L2-normalised
        onto the unit hypersphere, then repelled by the SAME InfoNCE-L2 form on the
        angular (cosine) distance:
            z      = features / ||features||_2
            D      = ||z_i - z_j||^2   ( = 2 - 2 cos(z_i, z_j),  bounded in [0, 4] )
            L_disp = log E_{i,j}[ exp( -D / tau ) ]
        Unlike the faithful `_dispersive_loss_infonce`, the per-dimension `/d`
        normalisation is DROPPED here: on the unit sphere the squared L2 is already
        O(1) (in [0, 4]), so dividing by d (~1024) would shrink D to ~4/d and saturate
        exp(-D/tau) -> 1 (vanishing gradient). Without /d the repulsion stays in force
        while the unit-sphere projection pins feat_norm to a constant 1 by construction,
        so the norm-inflation channel that lets infonce game its objective (feat_norm
        ~9x on flow_mid, ~287x off-path; §6.1) is STRUCTURALLY IMPOSSIBLE here. Same
        diagonal (i=j) convention as the faithful term.
        """
        B = features.shape[0]
        if B < 2:
            return features.sum() * 0.0
        z = F.normalize(features.float(), dim=1)
        D = torch.cdist(z, z, p=2) ** 2                         # (B, B) angular squared L2
        return torch.log(torch.exp(-D / tau).mean())

    @staticmethod
    def _dispersive_loss_vicreg(features: torch.Tensor, gamma: float = 1.0,
                                eps: float = 1e-4) -> torch.Tensor:
        """
        Scale-invariant VICReg-style regulariser (Bardes et al., ICLR 2022) — the second
        Phase 6 scale-invariant control whose objective cannot be lowered by norm inflation:
            variance   = mean_j max(0, gamma - std_j),   std_j = sqrt(Var(z_:,j) + eps)
            covariance = (1/d) * sum_{i != j} Cov(z)_{i,j}^2
        Both terms are >= 0 and MINIMISED (-> 0) when the batch features reach per-dimension
        std >= gamma (anti-collapse) AND are decorrelated. This is the SAME "more dispersed
        -> lower loss" sign convention as `_dispersive_loss_infonce`, so it is added to the
        total with the SAME +lambda weight. There is no invariance term (no positive pair in
        this setting). Computed in float32 because the off-diagonal covariance-square sum over
        ~d^2 terms overflows fp16 under autocast.
        """
        B = features.shape[0]
        if B < 2:
            return features.sum() * 0.0
        x = features.float()
        d = x.shape[1]
        std = torch.sqrt(x.var(dim=0) + eps)                    # (d,) per-dim std
        var_term = F.relu(gamma - std).mean()
        xc = x - x.mean(dim=0, keepdim=True)
        cov = (xc.T @ xc) / (B - 1)                             # (d, d) covariance
        off_diag = cov - torch.diag(torch.diagonal(cov))
        cov_term = off_diag.pow(2).sum() / d
        return var_term + cov_term

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
        task_cond: Optional[torch.Tensor] = None,
        lambda_dispersive: float = 0.0,
        dispersive_target: str = 'vis_pooled',
        dispersive_tau: float = 0.5,
        dispersive_form: str = 'infonce',
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

        global_cond, vis_pooled, imu_feat = self._encode_full(images, imu, task_cond=task_cond)

        t = torch.rand(B, device=device)
        eps = torch.randn_like(actions)
        t_expand = t[:, None, None]
        x_t = (1.0 - t_expand) * actions + t_expand * eps
        v_target = eps - actions

        need_mid = lambda_dispersive > 0.0 and dispersive_target == 'flow_mid'
        flow_out = self.flow_net(x_t, self._t_to_int(t), global_cond, return_mid=need_mid)
        v_pred, mid_feat = flow_out if need_mid else (flow_out, None)
        flow_loss = F.mse_loss(v_pred, v_target)

        total = flow_loss
        components = {'flow_loss': flow_loss.detach()}

        if lambda_dispersive > 0.0:
            if dispersive_target == 'flow_mid':
                # Faithful to Dispersive Loss [13] / D2PPO [14]: a repulsion on the
                # generative network's intermediate (mid-block) representation.
                #   'infonce' = faithful InfoNCE-L2 (/d) — DEFAULT, byte-identical to P2f
                #   'cosine'  = Phase 6 scale-invariant unit-sphere InfoNCE (no /d)
                #   'vicreg'  = Phase 6 scale-invariant variance+covariance regulariser
                mid_flat = mid_feat.flatten(1)
                if dispersive_form == 'infonce':
                    l_disp_raw = self._dispersive_loss_infonce(mid_flat, dispersive_tau)
                elif dispersive_form == 'cosine':
                    l_disp_raw = self._dispersive_loss_cosine(mid_flat, dispersive_tau)
                elif dispersive_form == 'vicreg':
                    l_disp_raw = self._dispersive_loss_vicreg(mid_flat)
                else:
                    raise ValueError(f"unknown dispersive_form: {dispersive_form}")
                l_disp = l_disp_raw * lambda_dispersive
            else:
                # Legacy off-path placement (hand-rolled log-distance on vis_pooled).
                l_disp = self._dispersive_loss(vis_pooled) * lambda_dispersive
            total = total + l_disp
            components['loss_dispersive'] = l_disp.detach()

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
    # RL fine-tuning (ReinFlow)
    # ------------------------------------------------------------------

    def compute_weighted_loss(
        self,
        images: torch.Tensor,
        imu: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        beta: float,
        fixed_x1: Optional[torch.Tensor] = None,
        positive_mask: bool = False,
        task_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = actions.shape[0]
        device = actions.device

        global_cond = self._encode(images, imu, task_cond=task_cond)

        if fixed_x1 is not None:
            eps = fixed_x1
            t = torch.ones(B, device=device)
        else:
            t = torch.rand(B, device=device)
            eps = torch.randn_like(actions)

        t_expand = t[:, None, None]
        x_t = (1.0 - t_expand) * actions + t_expand * eps
        v_target = eps - actions

        t_int = self._t_to_int(t)
        v_pred = self.flow_net(x_t, t_int, global_cond)

        weights = torch.exp(beta * advantages).clamp(max=20.0).detach()
        mse = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=[1, 2])

        if positive_mask:
            mask = (advantages > 0).float().detach()
            num_positive = mask.sum().clamp(min=1.0)
            return (weights * mse * mask).sum() / num_positive

        return (weights * mse).mean()

    def compute_clipped_loss(
        self,
        images:        torch.Tensor,
        imu:           torch.Tensor,
        actions_taken: torch.Tensor,
        fixed_x1:      torch.Tensor,
        mu_old:        torch.Tensor,
        advantages:    torch.Tensor,
        sde_noise_std: float,
        clip_epsilon:  float,
        task_cond:     Optional[torch.Tensor] = None,
    ):
        B = actions_taken.shape[0]
        sigma2 = sde_noise_std ** 2

        global_cond = self._encode(images, imu, task_cond=task_cond)
        t_batch = torch.ones(B, device=actions_taken.device)
        t_int   = self._t_to_int(t_batch)
        v_new   = self.flow_net(fixed_x1, t_int, global_cond)
        mu_new  = fixed_x1 - v_new

        sq_new = (actions_taken - mu_new).pow(2).sum(dim=[1, 2])
        sq_old = (actions_taken - mu_old).pow(2).sum(dim=[1, 2])
        log_ratio = (-0.5 / sigma2 * (sq_new - sq_old)).clamp(-20.0, 20.0)
        ratio     = torch.exp(log_ratio)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        loss  = -torch.min(surr1, surr2).mean()

        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean().item()
            approx_kl     = 0.5 * log_ratio.pow(2).mean().item()
            mean_ratio    = ratio.mean().item()
            log_ratio_std = log_ratio.std().item()

        return loss, clip_fraction, approx_kl, mean_ratio, log_ratio_std

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
        task_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = images.shape[0]
        device = images.device
        n = n_steps if n_steps is not None else self.n_inference_steps

        global_cond = self._encode(images, imu, task_cond=task_cond)

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

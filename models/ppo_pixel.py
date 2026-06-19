"""
PPO-from-pixels (P1 baseline): CNN actor-critic over the 6x64x64 FPV stack.

This is the end-to-end RL-from-pixels reference for the v6 baseline matrix. It
reuses the SAME ``VisionEncoder`` (6 -> 256) as the flow / BC-vision-only policies,
swapping the state MLP of ``models.ppo_expert.PPOExpert`` for a CNN trunk. Actor and
critic each own a *separate* encoder, mirroring the separate-network design of the
state-based expert (so the value loss never corrupts the actor's features and the
per-network grad-clip in :meth:`update` stays valid).

The actor exposes ``predict_action(images, imu=None, n_steps=None, task_cond=None)
-> (B, action_dim, T_pred)`` — the IDENTICAL contract the flow policies use — so a
trained checkpoint scores through ``scripts.evaluate_baselines_frozen`` unchanged
(imu / n_steps / task_cond are ignored: PPO is a reactive single-step policy, so the
deterministic action is tiled across T_pred).

Memory note: a pixel rollout buffer is stored as **uint8 on CPU**; minibatches are
moved to GPU and divided by 255 inside :meth:`update`. Storing the full
(n_steps*n_envs, 6, 64, 64) buffer as float32 on GPU would need ~13 GB and is the
reason ``update`` is reimplemented here rather than inherited.

CTBR hover bias: F_c_norm_hover = (mg / F_c_max)*2 - 1 ≈ -0.387, omega_cmd = [0,0,0].
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple

from models.vision_encoder import VisionEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PixelActor(nn.Module):
    """CNN encoder + MLP head -> TanhNormal over CTBR actions.

    Also serves as the inference module for the frozen eval (``predict_action``).
    """

    def __init__(self, in_channels: int = 6, feature_dim: int = 256,
                 action_dim: int = 4, hidden_dim: int = 256, T_pred: int = 8):
        super().__init__()
        self.action_dim = action_dim
        self.T_pred = T_pred
        self.vision_encoder = VisionEncoder(in_channels=in_channels,
                                            feature_dim=feature_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        # CTBR hover initialisation: F_c_norm ≈ -0.387, body-rate cmds = 0
        nn.init.zeros_(self.mean_layer.weight)
        with torch.no_grad():
            self.mean_layer.bias.copy_(torch.tensor([-0.387, 0.0, 0.0, 0.0]))

    def forward(self, images_01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """images_01: (B, 6, 64, 64) float in [0, 1]."""
        x = self.net(self.vision_encoder(images_01))
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), -20, 2)
        return mean, log_std

    @torch.no_grad()
    def predict_action(self, images, imu=None, n_steps=None, task_cond=None):
        """Flow-policy-compatible inference: deterministic action tiled to T_pred.

        ``images`` is the [0,1]-normalised (B, 6, 64, 64) tensor the frozen rollout
        passes; ``imu`` / ``n_steps`` / ``task_cond`` are ignored.
        """
        mean, _ = self.forward(images)
        action = torch.tanh(mean)                                  # (B, action_dim)
        return action.unsqueeze(-1).expand(-1, -1, self.T_pred)    # (B, action_dim, T_pred)


class PixelCritic(nn.Module):
    """Separate CNN encoder + MLP -> scalar value."""

    def __init__(self, in_channels: int = 6, feature_dim: int = 256,
                 hidden_dim: int = 256):
        super().__init__()
        self.vision_encoder = VisionEncoder(in_channels=in_channels,
                                            feature_dim=feature_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, images_01: torch.Tensor) -> torch.Tensor:
        return self.net(self.vision_encoder(images_01))


class PPOPixel:
    """PPO agent for end-to-end pixel control (CNN actor-critic, CTBR action)."""

    def __init__(self, in_channels: int = 6, feature_dim: int = 256,
                 action_dim: int = 4, hidden_dim: int = 256, T_pred: int = 8,
                 lr: float = 1e-4, critic_lr_multiplier: float = 1.0,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_range: float = 0.2, ent_coef: float = 0.01,
                 vf_coef: float = 0.5, max_grad_norm: float = 0.5,
                 batch_size: int = 256, n_epochs: int = 4):
        self.actor = PixelActor(in_channels, feature_dim, action_dim,
                                hidden_dim, T_pred).to(device)
        self.critic = PixelCritic(in_channels, feature_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr * critic_lr_multiplier},
        ])
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs

    # -- inference -----------------------------------------------------------
    def _to_img(self, images_u8: np.ndarray) -> torch.Tensor:
        """(N, 6, 64, 64) uint8 -> float [0,1] on device."""
        return torch.from_numpy(images_u8).to(device).float() / 255.0

    @torch.no_grad()
    def get_action_vec(self, images_u8: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample actions for a batch of stacked frames (uint8, (n_envs,6,64,64))."""
        x = self._to_img(images_u8)
        mean, log_std = self.actor(x)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        actions = torch.tanh(z)
        log_prob = dist.log_prob(z).sum(-1)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6).sum(-1)
        values = self.critic(x).squeeze(-1)
        return (actions.cpu().numpy(), log_prob.cpu().numpy(), values.cpu().numpy())

    @torch.no_grad()
    def get_action_deterministic(self, images_u8: np.ndarray) -> np.ndarray:
        mean, _ = self.actor(self._to_img(images_u8))
        return torch.tanh(mean).cpu().numpy()

    def evaluate(self, images_01: torch.Tensor, actions: torch.Tensor):
        """log_prob, value, entropy for a minibatch (images already [0,1])."""
        mean, log_std = self.actor(images_01)
        std = log_std.exp()
        dist = Normal(mean, std)
        a = torch.clamp(actions, -0.999999, 0.999999)
        z = torch.atanh(a)
        log_prob = dist.log_prob(z).sum(-1, keepdim=True)
        log_prob -= torch.log(1 - a.pow(2) + 1e-6).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1)
        value = self.critic(images_01)
        return log_prob, value, entropy

    # -- GAE (identical math to PPOExpert) -----------------------------------
    def compute_gae(self, rewards: List[float], values: List[float],
                    dones: List[bool], last_value: float = 0.0):
        advantages, returns, gae = [], [], 0.0
        values_ext = values + [last_value]
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values_ext[i + 1] * (1 - dones[i]) - values_ext[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values_ext[i])
        return advantages, returns

    # -- update (image-aware: uint8 buffer, per-minibatch GPU transfer) ------
    def update(self, memory: Dict, precomputed_advantages: np.ndarray,
               precomputed_returns: np.ndarray, target_kl: float = None
               ) -> Dict[str, float]:
        states_u8 = memory['states']                 # (N, 6, 64, 64) uint8 on CPU
        actions_t = torch.FloatTensor(np.asarray(memory['actions'])).to(device)
        old_lp_t = torch.FloatTensor(np.asarray(memory['log_probs'])).to(device)
        adv_t = torch.FloatTensor(precomputed_advantages).to(device)
        ret_t = torch.FloatTensor(precomputed_returns).to(device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        N = len(states_u8)
        indices = np.arange(N)
        tot_pl = tot_vl = tot_ent = 0.0
        n_upd = 0
        kl_stop = False

        for _ in range(self.n_epochs):
            if kl_stop:
                break
            np.random.shuffle(indices)
            for start in range(0, N, self.batch_size):
                idx = indices[start:start + self.batch_size]
                b_img = torch.from_numpy(states_u8[idx]).to(device).float() / 255.0
                b_act = actions_t[idx]
                b_old = old_lp_t[idx]
                b_adv = adv_t[idx]
                b_ret = ret_t[idx]

                new_lp, value, entropy = self.evaluate(b_img, b_act)
                ratio = torch.exp(new_lp.squeeze() - b_old)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(value.squeeze(), b_ret)
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                tot_pl += policy_loss.item()
                tot_vl += value_loss.item()
                tot_ent += entropy.mean().item()
                n_upd += 1

                if target_kl is not None:
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    if approx_kl > target_kl:
                        kl_stop = True
                        break

        return {
            'policy_loss': tot_pl / max(n_upd, 1),
            'value_loss': tot_vl / max(n_upd, 1),
            'entropy': tot_ent / max(n_upd, 1),
            'kl_early_stop': kl_stop,
        }

    def save(self, filepath: str):
        torch.save({'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, filepath)

    def load(self, filepath: str):
        ckpt = torch.load(filepath, map_location=device, weights_only=False)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])

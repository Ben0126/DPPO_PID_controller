"""
PPO Expert for Direct Motor Control

State-based PPO agent that maps 15D quadrotor state to 4D motor thrusts.
Used as the "perfect expert" for collecting imitation learning data.

Adapted from ppo_native.py with scaled dimensions and improved architecture.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RunningMeanStd:
    """
    Online normalization tracker for observations.
    Tracks running mean and standard deviation using Welford's algorithm.
    """

    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        if x.ndim == 1:
            batch_mean = x
            batch_var = np.zeros_like(x)
            batch_count = 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

    def state_dict(self) -> Dict:
        return {'mean': self.mean.copy(), 'var': self.var.copy(), 'count': self.count}

    def load_state_dict(self, d: Dict):
        self.mean = d['mean']
        self.var = d['var']
        self.count = d['count']


class Actor(nn.Module):
    """Policy network with TanhNormal distribution for bounded actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std


class Critic(nn.Module):
    """Value network for advantage estimation."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class PPOExpert:
    """
    PPO agent for direct quadrotor motor control.

    Uses TanhNormal distribution: network outputs pre-tanh values,
    tanh squashes to [-1, 1] which the environment maps to [0, f_max].
    """

    def __init__(self, state_dim: int = 15, action_dim: int = 4,
                 hidden_dim: int = 256, lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_range: float = 0.2, ent_coef: float = 0.01,
                 vf_coef: float = 0.5, max_grad_norm: float = 0.5,
                 batch_size: int = 256, n_epochs: int = 10):
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)

        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr},
        ])

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs

    @torch.no_grad()
    def get_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Sample action using TanhNormal policy.

        Args:
            state: normalized observation (state_dim,)

        Returns:
            action_tanh: action in [-1, 1] (4,)
            log_prob: log probability scalar
            value: value estimate scalar
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.actor(state_t)
        std = log_std.exp()

        dist = Normal(mean, std)
        z = dist.rsample()
        action_tanh = torch.tanh(z)

        # Log prob with Tanh correction
        log_prob = dist.log_prob(z).sum(dim=-1)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6).sum(dim=-1)

        value = self.critic(state_t)

        return (
            action_tanh.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    @torch.no_grad()
    def get_action_deterministic(self, state: np.ndarray) -> np.ndarray:
        """Get deterministic action (mean of policy). Used for evaluation/data collection."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, _ = self.actor(state_t)
        return torch.tanh(mean).squeeze(0).cpu().numpy()

    def evaluate(self, states: torch.Tensor,
                 actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log_prob, value, and entropy for a batch."""
        mean, log_std = self.actor(states)
        std = log_std.exp()
        dist = Normal(mean, std)

        # Inverse tanh to recover z
        actions_clamped = torch.clamp(actions, -0.999999, 0.999999)
        z = torch.atanh(actions_clamped)

        log_prob = dist.log_prob(z).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - actions_clamped.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(states)

        return log_prob, value, entropy

    def compute_gae(self, rewards: List[float], values: List[float],
                    dones: List[bool],
                    last_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            last_value: V(s_{T+1}). Use 0.0 if the rollout ended on a terminal
                        state; use the critic's estimate if it was truncated.

        Returns:
            advantages: list of advantage estimates
            returns: list of return targets for value function
        """
        advantages = []
        returns = []
        gae = 0.0

        values_ext = values + [last_value]  # correct bootstrap for truncated rollouts

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values_ext[i + 1] * (1 - dones[i]) - values_ext[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values_ext[i])

        return advantages, returns

    def update(self, memory: Dict,
               last_value: float = 0.0,
               target_kl: float = None) -> Dict[str, float]:
        """
        Perform PPO update on collected trajectories.

        Args:
            memory: dict with keys 'states', 'actions', 'log_probs',
                    'rewards', 'dones', 'values'
            last_value: bootstrap value for the state after the rollout ends.
                        0.0 if the last step was terminal, else V(s_{T+1}).
            target_kl: if set, stop epoch early when mean approx KL exceeds
                       this threshold (prevents policy churn).

        Returns:
            metrics: dict with training loss metrics
        """
        # Compute GAE with correct bootstrap
        advantages, returns = self.compute_gae(
            memory['rewards'], memory['values'], memory['dones'],
            last_value=last_value,
        )

        # Convert to tensors
        states = torch.FloatTensor(np.array(memory['states'])).to(device)
        actions = torch.FloatTensor(np.array(memory['actions'])).to(device)
        old_log_probs = torch.FloatTensor(np.array(memory['log_probs'])).to(device)
        advantages_t = torch.FloatTensor(advantages).to(device)
        returns_t = torch.FloatTensor(returns).to(device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # PPO update epochs
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        kl_early_stop = False

        for epoch in range(self.n_epochs):
            if kl_early_stop:
                break
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                idx = indices[start:end]

                b_states = states[idx]
                b_actions = actions[idx]
                b_old_lp = old_log_probs[idx]
                b_adv = advantages_t[idx]
                b_ret = returns_t[idx]

                new_log_probs, values, entropy = self.evaluate(b_states, b_actions)

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs.squeeze() - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_range,
                                    1 + self.clip_range) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values.squeeze(), b_ret)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

                # KL early stopping: check approx KL after each mini-batch
                if target_kl is not None:
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    if approx_kl > target_kl:
                        kl_early_stop = True
                        break

        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
            'kl_early_stop': kl_early_stop,
        }

    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

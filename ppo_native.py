
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, TransformedDistribution, TanhTransform
import yaml
from datetime import datetime
from dppo_pid_env import make_env

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class RunningMeanStd:
    """
    Tracks the running mean and standard deviation of a data stream.
    Used for normalizing observations.
    """
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        
        # Shared feature extractor or separate? Notes say MLP.
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Output mean and log_std
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2) # Clamp for stability
        return mean, log_std

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        return self.net(state)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 clip_epsilon=0.2, c1=0.5, c2=0.01, batch_size=64, n_epochs=10):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1 # Value loss coeff
        self.c2 = c2 # Entropy coeff
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        self.mse_loss = nn.MSELoss()

    def get_action(self, state):
        state = torch.FloatTensor(state).to(device)
        mean, log_std = self.actor(state)
        std = log_std.exp()
        
        # Create distribution
        dist = Normal(mean, std)
        
        # Sample action (reparameterization trick)
        z = dist.rsample()
        
        # Apply Tanh to squash to [-1, 1] (if needed, but Env expects [0, K_max])
        # The notes say "TanhNormal". 
        # If we use Tanh, the output is [-1, 1]. We need to scale it to [0, K_max] or similar.
        # However, the Env action space is Box(0, K_max).
        # Let's assume the network outputs raw actions and we squash them to [0, 1] then scale, 
        # or we just output raw and let the Env clip.
        # But "TanhNormal" implies squashing.
        # Let's use Tanh and then scale in the Env wrapper or here.
        # For simplicity and stability, let's stick to Gaussian and let the Env clip, 
        # OR use Tanh and scale.
        # The notes explicitly mention "TanhNormal".
        
        action_tanh = torch.tanh(z)
        
        # Calculate log_prob correction for Tanh
        log_prob = dist.log_prob(z).sum(dim=-1)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6).sum(dim=-1)
        
        return action_tanh.cpu().detach().numpy(), log_prob.cpu().detach().numpy()

    def evaluate(self, state, action):
        mean, log_std = self.actor(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        
        # We need to reverse the Tanh to get back to 'z' space to evaluate log_prob correctly
        # But 'action' passed here is already Tanh-ed.
        # Ideally we store 'z' or we just use the 'action' and assume it's Tanh-ed.
        # Numerical stability issue: atanh(1) is inf.
        # Let's use a slightly safer approach or just standard Gaussian if Tanh is too complex for "Teaching Note" level without a library.
        # But the note asks for "Complete Algorithm".
        # Let's stick to standard Gaussian for simplicity unless Tanh is strictly required.
        # The note says "TanhNormal".
        # Okay, let's implement TanhNormal logic properly.
        
        # To evaluate log_prob of a given action 'a' (which is tanh(z)):
        # z = atanh(a)
        # log_prob(a) = log_prob_normal(z) - sum(log(1 - a^2))
        
        action = torch.clamp(action, -0.999999, 0.999999)
        z = torch.atanh(action)
        
        log_prob = dist.log_prob(z).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        entropy = dist.entropy().sum(dim=-1) # Entropy of the base distribution (approx)
        
        value = self.critic(state)
        
        return log_prob, value, entropy

    def update(self, memory):
        # Unpack memory
        states = torch.FloatTensor(np.array(memory['states'])).to(device)
        actions = torch.FloatTensor(np.array(memory['actions'])).to(device)
        old_log_probs = torch.FloatTensor(np.array(memory['log_probs'])).to(device)
        rewards = memory['rewards']
        dones = memory['dones']
        values = memory['values'] # Estimated values from rollout
        
        # Calculate GAE and Returns
        returns = []
        advantages = []
        gae = 0
        
        # Bootstrap value if not done (assuming last state value is 0 or handled outside)
        # We need the value of the next state for the last step.
        # For simplicity, assume 0 or use the last value.
        # Let's assume we pass 'next_value' or handle it.
        # Standard GAE loop:
        
        # We need values for all states. We have them in 'values'.
        # We need value for state[t+1].
        # Let's append 0 to values for the terminal state.
        values_ext = values + [0] 
        
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values_ext[i+1] * (1 - dones[i]) - values_ext[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values_ext[i])
            
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO Update Loop
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                
                # Evaluate
                new_log_probs, state_values, entropy = self.evaluate(batch_states, batch_actions)
                
                # Ratio
                ratio = torch.exp(new_log_probs.squeeze() - batch_old_log_probs)
                
                # Surrogate Loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                value_loss = self.mse_loss(state_values.squeeze(), batch_returns)
                
                # Entropy Loss
                entropy_loss = -entropy.mean()
                
                # Total Loss
                loss = policy_loss + self.c1 * value_loss + self.c2 * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()

def train():
    # Load config
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    env = make_env("config.yaml")
    
    # Hyperparameters
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Action scaling: Env expects [0, K_max]. 
    # Our Agent outputs [-1, 1] (Tanh).
    # We need to map [-1, 1] to [0, K_max].
    # Let's do this mapping in the training loop.
    action_max = env.action_space.high
    action_min = env.action_space.low
    # Assuming min is 0.
    
    agent = PPOAgent(state_dim, action_dim)
    
    # Normalization
    obs_rms = RunningMeanStd(shape=(state_dim,))
    
    # Training parameters
    max_timesteps = config['training'].get('total_timesteps', 100000)
    steps_per_batch = config['training'].get('n_steps', 2048)
    
    timestep = 0
    episode = 0
    
    while timestep < max_timesteps:
        memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'values': []
        }
        
        state, _ = env.reset()
        obs_rms.update(np.array([state]))
        state = obs_rms.normalize(state)
        
        # Collect batch
        for _ in range(steps_per_batch):
            # Get action
            action_tanh, log_prob = agent.get_action(state)
            
            # Map action to env space [0, K_max]
            # action_tanh is in [-1, 1]
            # mapped = (action_tanh + 1) / 2 * (max - min) + min
            action_env = (action_tanh + 1) / 2 * (action_max - action_min) + action_min
            
            # Get value estimate (for GAE)
            state_tensor = torch.FloatTensor(state).to(device)
            value = agent.critic(state_tensor).item()
            
            # Step
            next_state, reward, terminated, truncated, _ = env.step(action_env)
            done = terminated or truncated
            
            # Store
            memory['states'].append(state)
            memory['actions'].append(action_tanh) # Store the raw tanh action
            memory['log_probs'].append(log_prob)
            memory['rewards'].append(reward)
            memory['dones'].append(done)
            memory['values'].append(value)
            
            # Update state
            obs_rms.update(np.array([next_state]))
            state = obs_rms.normalize(next_state)
            timestep += 1
            
            if done:
                state, _ = env.reset()
                obs_rms.update(np.array([state]))
                state = obs_rms.normalize(state)
                episode += 1
                
        # Update Agent
        agent.update(memory)
        
        print(f"Timestep: {timestep}, Episode: {episode}, Last Reward: {reward:.2f}")
        
        # Save occasionally
        if timestep % 10000 < steps_per_batch:
             torch.save(agent.actor.state_dict(), f"ppo_actor_{timestep}.pth")

if __name__ == "__main__":
    train()

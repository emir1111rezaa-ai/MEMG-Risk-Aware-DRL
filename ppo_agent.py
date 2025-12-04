"""Risk-Aware PPO Agent for MEMG Control."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple
from config import PPO_CONFIG, RISK_CONFIG

class ActorNetwork(nn.Module):
    """Policy network (actor)."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super(ActorNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Mean and log_std for Gaussian policy
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        features = self.feature_extractor(state)
        mean = self.mean(features)
        
        # Apply action space constraints
        # battery_power: [-1, 1], chp: [0, 1], grid: [-1, 1], boiler: [0, 1]
        mean_constrained = torch.cat([
            torch.tanh(mean[:, 0:1]),      # Battery: [-1, 1]
            torch.sigmoid(mean[:, 1:2]),   # CHP: [0, 1]
            torch.tanh(mean[:, 2:3]),      # Grid: [-1, 1]
            torch.sigmoid(mean[:, 3:4])    # Boiler: [0, 1]
        ], dim=1)
        
        std = torch.exp(self.log_std).expand_as(mean_constrained)
        
        return mean_constrained, std
    
    def get_action(self, state, deterministic=False):
        mean, std = self.forward(state)
        
        if deterministic:
            return mean, None, None
        
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Constrain actions
        action_constrained = torch.cat([
            torch.clamp(action[:, 0:1], -1, 1),
            torch.clamp(action[:, 1:2], 0, 1),
            torch.clamp(action[:, 2:3], -1, 1),
            torch.clamp(action[:, 3:4], 0, 1)
        ], dim=1)
        
        return action_constrained, log_prob, dist.entropy().sum(dim=-1, keepdim=True)


class CriticNetwork(nn.Module):
    """Value network (critic)."""
    
    def __init__(self, state_dim: int, hidden_dims: List[int]):
        super(CriticNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        return self.network(state)


class RiskAwarePPOAgent:
    """PPO Agent with CVaR risk-aware objective."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict = PPO_CONFIG,
                device: str = 'cpu'):
        self.device = device
        self.config = config
        
        self.actor = ActorNetwork(
            state_dim, action_dim, config['actor_hidden_dims']
        ).to(device)
        
        self.critic = CriticNetwork(
            state_dim, config['critic_hidden_dims']
        ).to(device)
        
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config['learning_rate_actor']
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config['learning_rate_critic']
        )
        
        # Experience buffer
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        # CVaR tracking
        self.episode_returns = []
        self.cvar_threshold = None
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, entropy = self.actor.get_action(state_tensor, deterministic)
            value = self.critic(state_tensor)
        
        return action.cpu().numpy()[0], log_prob, value, entropy
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition in buffer."""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value.cpu().item())
        self.buffer['log_probs'].append(log_prob.cpu().item())
        self.buffer['dones'].append(done)
    
    def compute_gae(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        rewards = np.array(self.buffer['rewards'])
        values = np.array(self.buffer['values'])
        dones = np.array(self.buffer['dones'])
        
        advantages = np.zeros_like(rewards)
        gae = 0
        
        values_next = np.append(values[1:], next_value)
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config['gamma'] * values_next[t] * (1 - dones[t]) - values[t]
            gae = delta + self.config['gamma'] * self.config['gae_lambda'] * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def compute_cvar_loss(self, returns: np.ndarray) -> float:
        """Compute CVaR-based risk penalty."""
        # Update CVaR threshold (alpha-quantile of returns)
        if self.cvar_threshold is None or len(self.episode_returns) < 10:
            self.cvar_threshold = np.percentile(returns, 
                                               (1 - RISK_CONFIG['cvar_alpha']) * 100)
        else:
            # Exponential moving average
            new_threshold = np.percentile(returns, (1 - RISK_CONFIG['cvar_alpha']) * 100)
            self.cvar_threshold = 0.9 * self.cvar_threshold + 0.1 * new_threshold
        
        # CVaR: conditional expectation of worst (1-alpha)% returns
        worst_returns = returns[returns <= self.cvar_threshold]
        
        if len(worst_returns) > 0:
            cvar = np.mean(worst_returns)
        else:
            cvar = self.cvar_threshold
        
        # Risk penalty (encourage higher CVaR, i.e., less negative tail)
        risk_penalty = -cvar * RISK_CONFIG['risk_aversion']
        
        return risk_penalty
    
    def update(self) -> Dict:
        """Update policy and value networks using PPO."""
        if len(self.buffer['states']) < self.config['mini_batch_size']:
            return {}
        
        # Compute advantages and returns
        next_value = 0.0  # Assume terminal state
        advantages, returns = self.compute_gae(next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.buffer['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer['log_probs'])).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # CVaR risk penalty
        cvar_penalty = self.compute_cvar_loss(returns)
        
        # PPO update epochs
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for epoch in range(self.config['ppo_epochs']):
            # Mini-batch sampling
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.config['mini_batch_size']):
                end = start + self.config['mini_batch_size']
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                
                # Evaluate actions
                mean, std = self.actor(batch_states)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
                
                # Ratio for PPO
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 
                                   1 - self.config['clip_epsilon'],
                                   1 + self.config['clip_epsilon']) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()
                
                # Add CVaR penalty to actor loss
                actor_loss_total = actor_loss + \
                                  self.config['entropy_coef'] * entropy_loss + \
                                  cvar_penalty * 0.01  # Scale CVaR penalty
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss_total.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 
                                        self.config['max_grad_norm'])
                self.actor_optimizer.step()
                
                # Critic loss
                values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(values, batch_returns)
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(),
                                        self.config['max_grad_norm'])
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
        
        # Clear buffer
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        n_updates = self.config['ppo_epochs'] * (len(states) // self.config['mini_batch_size'])
        
        return {
            'actor_loss': total_actor_loss / n_updates,
            'critic_loss': total_critic_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'cvar_threshold': self.cvar_threshold,
            'cvar_penalty': cvar_penalty
        }
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

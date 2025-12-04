"""Baseline methods for comparison."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple
from scipy.optimize import linprog, minimize
from config import *

class SimpleDRLAgent:
    """Simple DRL without advanced forecasting and risk-awareness."""
    
    def __init__(self, state_dim: int, action_dim: int, device: str = 'cpu'):
        self.device = device
        
        # Simpler architecture
        hidden_dims = BASELINE_CONFIGS['simple_drl']['hidden_dims']
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        ).to(device)
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        ).to(device)
        
        lr = BASELINE_CONFIGS['simple_drl']['learning_rate']
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor)
        
        if not deterministic:
            # Add exploration noise
            noise = torch.randn_like(action) * 0.1
            action = action + noise
            action = torch.clamp(action, -1, 1)
        
        # Map to action space constraints
        action_np = action.cpu().numpy()[0]
        action_constrained = np.array([
            action_np[0],                    # Battery: [-1, 1]
            (action_np[1] + 1) / 2,         # CHP: [0, 1]
            action_np[2],                    # Grid: [-1, 1]
            (action_np[3] + 1) / 2          # Boiler: [0, 1]
        ])
        
        return action_constrained
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition."""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['next_states'].append(next_state)
        self.buffer['dones'].append(done)
    
    def update(self) -> Dict:
        """Simple policy gradient update."""
        if len(self.buffer['states']) < 32:
            return {}
        
        states = torch.FloatTensor(np.array(self.buffer['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer['actions'])).to(self.device)
        rewards = torch.FloatTensor(np.array(self.buffer['rewards'])).to(self.device)
        next_states = torch.FloatTensor(np.array(self.buffer['next_states'])).to(self.device)
        dones = torch.FloatTensor(np.array(self.buffer['dones'])).to(self.device)
        
        # Compute returns
        returns = torch.zeros_like(rewards)
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + 0.99 * G * (1 - dones[t])
            returns[t] = G
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Critic loss
        values = self.critic(states).squeeze()
        critic_loss = nn.MSELoss()(values, returns)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor loss
        predicted_actions = self.actor(states)
        advantages = returns - self.critic(states).squeeze().detach()
        actor_loss = -(predicted_actions * advantages.unsqueeze(1)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Clear buffer
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }


class ClassicalOptimization:
    """Classical optimization-based EMS (Model Predictive Control style)."""
    
    def __init__(self, horizon: int = 24):
        self.horizon = horizon
    
    def solve(self, forecast_data: Dict) -> np.ndarray:
        """Solve optimization problem for one timestep."""
        # Extract forecasts
        pv_forecast = forecast_data['pv_generation'][:self.horizon]
        wt_forecast = forecast_data['wt_generation'][:self.horizon]
        elec_load_forecast = forecast_data['electrical_load'][:self.horizon]
        thermal_load_forecast = forecast_data['thermal_load'][:self.horizon]
        grid_price_forecast = forecast_data['grid_import_price'][:self.horizon]
        
        # Decision variables for horizon:
        # [battery_power (H), chp_power (H), grid_power (H), boiler_power (H), soc (H)]
        n_vars = 5 * self.horizon
        
        # Objective: minimize total cost
        c = np.zeros(n_vars)
        
        for t in range(self.horizon):
            # Grid cost
            c[2 * self.horizon + t] = grid_price_forecast[t]
            
            # CHP cost
            chp_fuel_cost = CHP_CONFIG['gas_price'] / CHP_CONFIG['electrical_efficiency']
            c[self.horizon + t] = chp_fuel_cost + CHP_CONFIG['maintenance_cost']
            
            # Boiler cost
            boiler_fuel_cost = BOILER_CONFIG['gas_price'] / BOILER_CONFIG['thermal_efficiency']
            c[3 * self.horizon + t] = boiler_fuel_cost + BOILER_CONFIG['maintenance_cost']
            
            # Battery degradation (approximate)
            c[t] = ELECTRICAL_BUS['battery_degradation_cost'] * 0.1
        
        # Constraints
        A_eq = []
        b_eq = []
        A_ub = []
        b_ub = []
        
        for t in range(self.horizon):
            # Power balance constraint (equality)
            # pv + wt + chp + battery + grid = load
            power_balance = np.zeros(n_vars)
            power_balance[t] = 1  # battery
            power_balance[self.horizon + t] = 1  # chp
            power_balance[2 * self.horizon + t] = 1  # grid
            
            renewable = pv_forecast[t] + wt_forecast[t]
            A_eq.append(power_balance)
            b_eq.append(elec_load_forecast[t] - renewable)
            
            # Thermal balance (inequality - can have excess)
            thermal_balance = np.zeros(n_vars)
            thermal_balance[self.horizon + t] = -(CHP_CONFIG['thermal_efficiency'] / 
                                                  CHP_CONFIG['electrical_efficiency'])
            thermal_balance[3 * self.horizon + t] = -1  # boiler
            A_ub.append(thermal_balance)
            b_ub.append(-thermal_load_forecast[t])
            
            # SOC dynamics (simplified)
            if t > 0:
                soc_constraint = np.zeros(n_vars)
                soc_constraint[4 * self.horizon + t] = 1
                soc_constraint[4 * self.horizon + t - 1] = -1
                soc_constraint[t] = -1.0 / ELECTRICAL_BUS['battery_capacity']
                A_eq.append(soc_constraint)
                b_eq.append(0)
        
        # Bounds
        bounds = []
        
        # Battery power bounds
        for t in range(self.horizon):
            bounds.append((-ELECTRICAL_BUS['battery_max_power'], 
                          ELECTRICAL_BUS['battery_max_power']))
        
        # CHP power bounds
        for t in range(self.horizon):
            bounds.append((0, CHP_CONFIG['max_power_elec']))
        
        # Grid power bounds
        for t in range(self.horizon):
            bounds.append((-GRID_CONFIG['max_export'], GRID_CONFIG['max_import']))
        
        # Boiler power bounds
        for t in range(self.horizon):
            bounds.append((0, BOILER_CONFIG['max_thermal_power']))
        
        # SOC bounds
        for t in range(self.horizon):
            bounds.append((ELECTRICAL_BUS['battery_min_soc'], 
                          ELECTRICAL_BUS['battery_max_soc']))
        
        # Solve linear program
        try:
            result = linprog(c, A_ub=np.array(A_ub) if A_ub else None, 
                           b_ub=np.array(b_ub) if b_ub else None,
                           A_eq=np.array(A_eq) if A_eq else None,
                           b_eq=np.array(b_eq) if b_eq else None,
                           bounds=bounds,
                           method='highs')
            
            if result.success:
                # Extract first timestep action
                action = np.array([
                    result.x[0] / ELECTRICAL_BUS['battery_max_power'],  # Normalize
                    result.x[self.horizon] / CHP_CONFIG['max_power_elec'],
                    result.x[2 * self.horizon] / GRID_CONFIG['max_import'] if result.x[2 * self.horizon] > 0 
                        else result.x[2 * self.horizon] / GRID_CONFIG['max_export'],
                    result.x[3 * self.horizon] / BOILER_CONFIG['max_thermal_power']
                ])
                return action
            else:
                # Fallback: simple heuristic
                return self._heuristic_action(forecast_data)
        
        except Exception as e:
            print(f"Optimization failed: {e}")
            return self._heuristic_action(forecast_data)
    
    def _heuristic_action(self, forecast_data: Dict) -> np.ndarray:
        """Simple rule-based fallback."""
        # Use grid to meet all electrical demand
        # Use boiler for all thermal demand
        # No battery or CHP
        return np.array([0.0, 0.0, 0.5, 0.5])

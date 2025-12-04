"""Training script for all methods."""
import numpy as np
import torch
import pickle
from tqdm import tqdm
from typing import Dict
import matplotlib.pyplot as plt

from config import *
from data_generator import MEMGDataGenerator
from forecasting_model import AdvancedForecaster
from memg_environment import MEMGEnvironment
from ppo_agent import RiskAwarePPOAgent
from baseline_methods import SimpleDRLAgent, ClassicalOptimization

def train_forecasters(train_data: Dict, val_data: Dict) -> AdvancedForecaster:
    """Train forecasting models for all variables."""
    print("\n" + "="*60)
    print("TRAINING ADVANCED FORECASTING MODELS")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    forecaster = AdvancedForecaster(config=FORECASTING_CONFIG, device=device)
    
    variables = ['pv_generation', 'wt_generation', 'electrical_load', 'thermal_load']
    
    for var in variables:
        print(f"\n{'='*60}")
        print(f"Training forecaster for: {var}")
        print(f"{'='*60}")
        
        history = forecaster.train_model(
            var,
            train_data[var],
            val_data[var],
            verbose=True
        )
    
    # Save forecaster
    with open('forecaster.pkl', 'wb') as f:
        pickle.dump(forecaster, f)
    
    print("\nForecasting models trained and saved.")
    return forecaster

def train_ppo_agent(env: MEMGEnvironment, n_episodes: int = 300) -> RiskAwarePPOAgent:
    """Train Risk-Aware PPO agent."""
    print("\n" + "="*60)
    print("TRAINING RISK-AWARE PPO AGENT")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = RiskAwarePPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        config=PPO_CONFIG,
        device=device
    )
    
    episode_rewards = []
    episode_costs = []
    episode_violations = []
    
    for episode in tqdm(range(n_episodes), desc="Training PPO"):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, log_prob, value, entropy = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            state = next_state
            episode_reward += reward
        
        # Update agent
        if len(agent.buffer['states']) >= PPO_CONFIG['buffer_size']:
            metrics = agent.update()
            
            if episode % 10 == 0 and metrics:
                print(f"\nEpisode {episode}: Reward={episode_reward:.2f}, "
                      f"Cost={env.total_cost:.2f}, "
                      f"Actor Loss={metrics.get('actor_loss', 0):.4f}, "
                      f"CVaR Threshold={metrics.get('cvar_threshold', 0):.2f}")
        
        episode_rewards.append(episode_reward)
        episode_costs.append(env.total_cost)
        episode_violations.append(sum(env.constraint_violations))
        
        # Store episode return for CVaR
        agent.episode_returns.append(episode_reward)
        if len(agent.episode_returns) > 100:
            agent.episode_returns.pop(0)
    
    # Save agent
    agent.save('ppo_agent.pth')
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(episode_rewards)
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    
    axes[1].plot(episode_costs)
    axes[1].set_title('Episode Costs')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Total Cost ($)')
    
    axes[2].plot(episode_violations)
    axes[2].set_title('Constraint Violations')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Total Violations')
    
    plt.tight_layout()
    plt.savefig('training_curves_ppo.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nRisk-Aware PPO agent trained and saved.")
    return agent

def train_simple_drl(env: MEMGEnvironment, n_episodes: int = 300) -> SimpleDRLAgent:
    """Train simple DRL baseline."""
    print("\n" + "="*60)
    print("TRAINING SIMPLE DRL BASELINE")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = SimpleDRLAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device=device
    )
    
    episode_rewards = []
    episode_costs = []
    
    for episode in tqdm(range(n_episodes), desc="Training Simple DRL"):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # Update periodically
            if len(agent.buffer['states']) >= 64:
                agent.update()
        
        episode_rewards.append(episode_reward)
        episode_costs.append(env.total_cost)
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward={episode_reward:.2f}, Cost={env.total_cost:.2f}")
    
    # Save
    torch.save(agent.actor.state_dict(), 'simple_drl_agent.pth')
    
    print("\nSimple DRL baseline trained and saved.")
    return agent

def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print(" "*10 + "MULTI-ENERGY MICROGRID TRAINING PIPELINE")
    print("="*70)
    
    # Set random seeds
    np.random.seed(SIM_CONFIG['random_seed'])
    torch.manual_seed(SIM_CONFIG['random_seed'])
    
    # Generate data
    print("\nGenerating synthetic MEMG data...")
    generator = MEMGDataGenerator(seed=SIM_CONFIG['random_seed'])
    all_data = generator.generate_all_data()
    train_data, val_data, test_data = generator.split_data(all_data)
    
    print(f"\nData split:")
    print(f"  Training:   {len(train_data['electrical_load'])} timesteps")
    print(f"  Validation: {len(val_data['electrical_load'])} timesteps")
    print(f"  Test:       {len(test_data['electrical_load'])} timesteps")
    
    # Train forecasters
    forecaster = train_forecasters(train_data, val_data)
    
    # Create environments
    print("\nCreating training environments...")
    
    # Environment with advanced forecasting for PPO
    env_ppo = MEMGEnvironment(
        data=train_data,
        forecaster=forecaster,
        use_forecasting=True
    )
    
    # Environment without advanced forecasting for simple DRL
    env_simple = MEMGEnvironment(
        data=train_data,
        forecaster=None,  # Will use perfect foresight
        use_forecasting=False
    )
    
    # Train agents
    ppo_agent = train_ppo_agent(env_ppo, n_episodes=300)
    simple_agent = train_simple_drl(env_simple, n_episodes=300)
    
    print("\n" + "="*70)
    print(" "*20 + "TRAINING COMPLETED")
    print("="*70)
    print("\nTrained models saved:")
    print("  - forecaster.pkl")
    print("  - ppo_agent.pth")
    print("  - simple_drl_agent.pth")
    print("\nProceed to evaluation.py to compare all methods.")

if __name__ == '__main__':
    main()

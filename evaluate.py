"""Comprehensive evaluation and comparison of all methods."""
import numpy as np
import torch
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from tqdm import tqdm

from config import *
from data_generator import MEMGDataGenerator
from forecasting_model import AdvancedForecaster
from memg_environment import MEMGEnvironment
from ppo_agent import RiskAwarePPOAgent
from baseline_methods import SimpleDRLAgent, ClassicalOptimization

sns.set_style('whitegrid')
sns.set_palette('husl')

class MEMGEvaluator:
    """Comprehensive evaluator for all methods."""
    
    def __init__(self, test_data: Dict, forecaster: AdvancedForecaster):
        self.test_data = test_data
        self.forecaster = forecaster
        self.results = {}
    
    def evaluate_forecasting(self) -> Dict:
        """Evaluate forecasting accuracy."""
        print("\n" + "="*60)
        print("EVALUATING FORECASTING MODELS")
        print("="*60)
        
        variables = ['pv_generation', 'wt_generation', 'electrical_load', 'thermal_load']
        metrics_all = {}
        
        for var in variables:
            metrics = self.forecaster.evaluate(var, self.test_data[var])
            metrics_all[var] = metrics
        
        return metrics_all
    
    def evaluate_method(self, method_name: str, agent, env: MEMGEnvironment,
                       n_episodes: int = 10) -> Dict:
        """Evaluate a single method."""
        print(f"\n{'='*60}")
        print(f"EVALUATING: {method_name}")
        print(f"{'='*60}")
        
        episode_results = []
        
        for episode in tqdm(range(n_episodes), desc=f"Evaluating {method_name}"):
            state = env.reset()
            done = False
            
            episode_data = {
                'costs': [],
                'violations': [],
                'actions': [],
                'battery_soc': [],
                'renewable_gen': [],
                'loads': []
            }
            
            while not done:
                # Get action based on method
                if isinstance(agent, RiskAwarePPOAgent):
                    action, _, _, _ = agent.select_action(state, deterministic=True)
                elif isinstance(agent, SimpleDRLAgent):
                    action = agent.select_action(state, deterministic=True)
                elif isinstance(agent, ClassicalOptimization):
                    # Prepare forecast data
                    forecast_data = {
                        'pv_generation': env.data['pv_generation'][env.timestep:],
                        'wt_generation': env.data['wt_generation'][env.timestep:],
                        'electrical_load': env.data['electrical_load'][env.timestep:],
                        'thermal_load': env.data['thermal_load'][env.timestep:],
                        'grid_import_price': env.data['grid_import_price'][env.timestep:]
                    }
                    action = agent.solve(forecast_data)
                else:
                    raise ValueError(f"Unknown agent type: {type(agent)}")
                
                next_state, reward, done, info = env.step(action)
                
                # Record data
                episode_data['costs'].append(info['costs']['total'])
                episode_data['violations'].append(sum(info['violations'].values()))
                episode_data['actions'].append(action)
                episode_data['battery_soc'].append(env.battery_soc)
                episode_data['renewable_gen'].append(info['renewable_generation'])
                episode_data['loads'].append({
                    'electrical': info['elec_load'],
                    'thermal': info['thermal_load']
                })
                
                state = next_state
            
            # Compute episode metrics
            costs = np.array(episode_data['costs'])
            violations = np.array(episode_data['violations'])
            
            episode_metrics = {
                'total_cost': np.sum(costs),
                'mean_cost': np.mean(costs),
                'std_cost': np.std(costs),
                'max_cost': np.max(costs),
                'cvar_cost': self._compute_cvar(costs, alpha=RISK_CONFIG['cvar_alpha']),
                'total_violations': np.sum(violations),
                'violation_rate': np.mean(violations > 0),
                'max_violation': np.max(violations),
                'battery_cycles': env.battery_cycles,
                'renewable_utilization': self._compute_renewable_utilization(episode_data),
                'episode_data': episode_data
            }
            
            episode_results.append(episode_metrics)
        
        # Aggregate results
        aggregated = self._aggregate_results(episode_results)
        self.results[method_name] = aggregated
        
        return aggregated
    
    def _compute_cvar(self, values: np.ndarray, alpha: float = 0.95) -> float:
        """Compute Conditional Value at Risk (CVaR)."""
        threshold = np.percentile(values, (1 - alpha) * 100)
        worst_values = values[values >= threshold]
        return np.mean(worst_values) if len(worst_values) > 0 else threshold
    
    def _compute_renewable_utilization(self, episode_data: Dict) -> float:
        """Compute renewable energy utilization rate."""
        total_renewable = sum(episode_data['renewable_gen'])
        total_load = sum([load['electrical'] for load in episode_data['loads']])
        return min(total_renewable / total_load, 1.0) if total_load > 0 else 0
    
    def _aggregate_results(self, episode_results: List[Dict]) -> Dict:
        """Aggregate results across episodes."""
        metrics = {}
        
        for key in ['total_cost', 'mean_cost', 'cvar_cost', 'total_violations',
                   'battery_cycles', 'renewable_utilization']:
            values = [ep[key] for ep in episode_results]
            metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # Store one episode data for visualization
        metrics['sample_episode'] = episode_results[0]['episode_data']
        
        return metrics
    
    def compare_methods(self) -> pd.DataFrame:
        """Create comparison table."""
        print("\n" + "="*60)
        print("METHOD COMPARISON SUMMARY")
        print("="*60)
        
        comparison_data = []
        
        for method_name, results in self.results.items():
            row = {
                'Method': method_name,
                'Mean Cost ($)': f"{results['mean_cost']['mean']:.2f} ± {results['mean_cost']['std']:.2f}",
                'CVaR Cost ($)': f"{results['cvar_cost']['mean']:.2f}",
                'Violations': f"{results['total_violations']['mean']:.1f}",
                'Battery Cycles': f"{results['battery_cycles']['mean']:.2f}",
                'Renewable Util.': f"{results['renewable_utilization']['mean']:.2%}"
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        print("\n", df.to_string(index=False))
        
        return df
    
    def visualize_results(self):
        """Generate comprehensive visualizations."""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # 1. Cost comparison
        self._plot_cost_comparison()
        
        # 2. CVaR comparison
        self._plot_cvar_comparison()
        
        # 3. Constraint violations
        self._plot_violations_comparison()
        
        # 4. Battery operation
        self._plot_battery_operation()
        
        # 5. Time-series comparison
        self._plot_timeseries_comparison()
        
        print("Visualizations saved.")
    
    def _plot_cost_comparison(self):
        """Plot cost comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(self.results.keys())
        mean_costs = [self.results[m]['mean_cost']['mean'] for m in methods]
        std_costs = [self.results[m]['mean_cost']['std'] for m in methods]
        
        x = np.arange(len(methods))
        bars = ax.bar(x, mean_costs, yerr=std_costs, capsize=5, alpha=0.7)
        
        # Color bars
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Cost per Timestep ($)', fontsize=12, fontweight='bold')
        ax.set_title('Cost Comparison Across Methods', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cost_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cvar_comparison(self):
        """Plot CVaR comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(self.results.keys())
        cvar_costs = [self.results[m]['cvar_cost']['mean'] for m in methods]
        
        x = np.arange(len(methods))
        bars = ax.bar(x, cvar_costs, alpha=0.7)
        
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'CVaR ({RISK_CONFIG["cvar_alpha"]:.0%}) Cost ($)', 
                     fontsize=12, fontweight='bold')
        ax.set_title('Tail Risk (CVaR) Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cvar_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_violations_comparison(self):
        """Plot constraint violations."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(self.results.keys())
        violations = [self.results[m]['total_violations']['mean'] for m in methods]
        
        x = np.arange(len(methods))
        bars = ax.bar(x, violations, alpha=0.7)
        
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Constraint Violations', fontsize=12, fontweight='bold')
        ax.set_title('Constraint Violation Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('violations_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_battery_operation(self):
        """Plot battery SOC comparison."""
        fig, axes = plt.subplots(len(self.results), 1, figsize=(12, 4 * len(self.results)))
        
        if len(self.results) == 1:
            axes = [axes]
        
        for ax, (method_name, results) in zip(axes, self.results.items()):
            soc_data = results['sample_episode']['battery_soc']
            timesteps = range(len(soc_data))
            
            ax.plot(timesteps, soc_data, linewidth=2, label=method_name)
            ax.axhline(y=ELECTRICAL_BUS['battery_min_soc'], color='r', 
                      linestyle='--', alpha=0.5, label='Min SOC')
            ax.axhline(y=ELECTRICAL_BUS['battery_max_soc'], color='r',
                      linestyle='--', alpha=0.5, label='Max SOC')
            
            ax.set_xlabel('Timestep', fontsize=11, fontweight='bold')
            ax.set_ylabel('Battery SOC', fontsize=11, fontweight='bold')
            ax.set_title(f'Battery Operation: {method_name}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('battery_operation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_timeseries_comparison(self):
        """Plot detailed time-series for first 168 hours (1 week)."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Select first method for detailed view
        method_name = list(self.results.keys())[0]
        data = self.results[method_name]['sample_episode']
        
        # Limit to first week
        n_hours = min(168, len(data['costs']))
        hours = range(n_hours)
        
        # Cost trajectory
        axes[0].plot(hours, data['costs'][:n_hours], linewidth=1.5)
        axes[0].set_ylabel('Cost ($)', fontsize=11, fontweight='bold')
        axes[0].set_title(f'Operational Cost Over Time: {method_name}', 
                         fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Battery SOC
        axes[1].plot(hours, data['battery_soc'][:n_hours], linewidth=1.5, color='green')
        axes[1].axhline(y=ELECTRICAL_BUS['battery_min_soc'], color='r',
                       linestyle='--', alpha=0.5)
        axes[1].axhline(y=ELECTRICAL_BUS['battery_max_soc'], color='r',
                       linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Battery SOC', fontsize=11, fontweight='bold')
        axes[1].set_title('Battery State of Charge', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        # Renewable generation
        axes[2].plot(hours, data['renewable_gen'][:n_hours], linewidth=1.5, color='orange')
        axes[2].set_xlabel('Hour', fontsize=11, fontweight='bold')
        axes[2].set_ylabel('Power (kW)', fontsize=11, fontweight='bold')
        axes[2].set_title('Renewable Generation', fontsize=12, fontweight='bold')
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('timeseries_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main evaluation pipeline."""
    print("\n" + "="*70)
    print(" "*15 + "MEMG EVALUATION PIPELINE")
    print("="*70)
    
    # Load data
    print("\nLoading data and models...")
    generator = MEMGDataGenerator(seed=SIM_CONFIG['random_seed'])
    all_data = generator.generate_all_data()
    _, _, test_data = generator.split_data(all_data)
    
    # Load forecaster
    with open('forecaster.pkl', 'rb') as f:
        forecaster = pickle.load(f)
    
    # Evaluate forecasting
    evaluator = MEMGEvaluator(test_data, forecaster)
    forecast_metrics = evaluator.evaluate_forecasting()
    
    # Create test environments
    env_advanced = MEMGEnvironment(
        data=test_data,
        forecaster=forecaster,
        use_forecasting=True
    )
    
    env_simple = MEMGEnvironment(
        data=test_data,
        forecaster=None,
        use_forecasting=False
    )
    
    # Load agents
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Proposed method: Risk-Aware PPO + Advanced Forecasting
    ppo_agent = RiskAwarePPOAgent(
        state_dim=env_advanced.state_dim,
        action_dim=env_advanced.action_dim,
        config=PPO_CONFIG,
        device=device
    )
    ppo_agent.load('ppo_agent.pth')
    
    # Simple DRL baseline
    simple_agent = SimpleDRLAgent(
        state_dim=env_simple.state_dim,
        action_dim=env_simple.action_dim,
        device=device
    )
    simple_agent.actor.load_state_dict(torch.load('simple_drl_agent.pth', map_location=device))
    
    # Classical optimization baseline
    classical_agent = ClassicalOptimization(horizon=24)
    
    # Evaluate all methods
    evaluator.evaluate_method(
        "Proposed (PPO + Forecasting + CVaR)",
        ppo_agent,
        env_advanced,
        n_episodes=10
    )
    
    evaluator.evaluate_method(
        "Simple DRL",
        simple_agent,
        env_simple,
        n_episodes=10
    )
    
    evaluator.evaluate_method(
        "Classical Optimization",
        classical_agent,
        env_simple,
        n_episodes=10
    )
    
    # Generate comparison
    comparison_df = evaluator.compare_methods()
    comparison_df.to_csv('method_comparison.csv', index=False)
    
    # Generate visualizations
    evaluator.visualize_results()
    
    # Calculate improvements
    print("\n" + "="*60)
    print("IMPROVEMENT ANALYSIS")
    print("="*60)
    
    proposed_cost = evaluator.results["Proposed (PPO + Forecasting + CVaR)"]['mean_cost']['mean']
    simple_cost = evaluator.results["Simple DRL"]['mean_cost']['mean']
    classical_cost = evaluator.results["Classical Optimization"]['mean_cost']['mean']
    
    print(f"\nCost Reduction vs Simple DRL: {(simple_cost - proposed_cost) / simple_cost * 100:.2f}%")
    print(f"Cost Reduction vs Classical: {(classical_cost - proposed_cost) / classical_cost * 100:.2f}%")
    
    proposed_cvar = evaluator.results["Proposed (PPO + Forecasting + CVaR)"]['cvar_cost']['mean']
    simple_cvar = evaluator.results["Simple DRL"]['cvar_cost']['mean']
    classical_cvar = evaluator.results["Classical Optimization"]['cvar_cost']['mean']
    
    print(f"\nCVaR Reduction vs Simple DRL: {(simple_cvar - proposed_cvar) / simple_cvar * 100:.2f}%")
    print(f"CVaR Reduction vs Classical: {(classical_cvar - proposed_cvar) / classical_cvar * 100:.2f}%")
    
    print("\n" + "="*70)
    print(" "*20 + "EVALUATION COMPLETED")
    print("="*70)
    print("\nResults saved:")
    print("  - method_comparison.csv")
    print("  - cost_comparison.png")
    print("  - cvar_comparison.png")
    print("  - violations_comparison.png")
    print("  - battery_operation.png")
    print("  - timeseries_detailed.png")

if __name__ == '__main__':
    main()

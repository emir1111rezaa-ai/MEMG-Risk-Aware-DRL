"""Utility functions for analysis and visualization."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

def calculate_improvement_percentage(baseline: float, proposed: float) -> float:
    """Calculate improvement percentage."""
    return ((baseline - proposed) / baseline) * 100

def create_summary_table(results: Dict) -> pd.DataFrame:
    """Create comprehensive summary table."""
    summary_data = []
    
    for method_name, method_results in results.items():
        row = {
            'Method': method_name,
            'Mean Cost': method_results['mean_cost']['mean'],
            'Std Cost': method_results['mean_cost']['std'],
            'CVaR Cost': method_results['cvar_cost']['mean'],
            'Total Violations': method_results['total_violations']['mean'],
            'Battery Cycles': method_results['battery_cycles']['mean'],
            'Renewable Utilization': method_results['renewable_utilization']['mean']
        }
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def plot_action_distribution(episode_data: Dict, method_name: str, save_path: str):
    """Plot distribution of actions taken."""
    actions = np.array(episode_data['actions'])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    action_names = ['Battery Power', 'CHP Power', 'Grid Power', 'Boiler Power']
    
    for i, (ax, name) in enumerate(zip(axes.flatten(), action_names)):
        ax.hist(actions[:, i], bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel(f'{name} (normalized)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{name} Distribution', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
    
    fig.suptitle(f'Action Distribution: {method_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_constraint_violations(episode_data: Dict) -> Dict:
    """Analyze types and severity of constraint violations."""
    violations = episode_data['violations']
    
    violation_types = {
        'battery_soc': [],
        'battery_power': [],
        'grid_limit': [],
        'chp_ramp': [],
        'power_balance': [],
        'thermal_balance': []
    }
    
    for v in violations:
        if isinstance(v, dict):
            for key in violation_types.keys():
                violation_types[key].append(v.get(key, 0))
    
    analysis = {}
    for key, values in violation_types.items():
        values_array = np.array(values)
        analysis[key] = {
            'count': np.sum(values_array > 0),
            'mean_severity': np.mean(values_array[values_array > 0]) if np.any(values_array > 0) else 0,
            'max_severity': np.max(values_array)
        }
    
    return analysis

def compute_economic_metrics(episode_data: Dict, timestep_hours: float = 1.0) -> Dict:
    """Compute detailed economic metrics."""
    costs = episode_data['costs']
    
    total_cost = sum(costs)
    avg_hourly_cost = np.mean(costs)
    peak_cost = np.max(costs)
    
    # Compute cost components if available
    cost_breakdown = {
        'total': total_cost,
        'average_hourly': avg_hourly_cost,
        'peak': peak_cost,
        'std_dev': np.std(costs)
    }
    
    return cost_breakdown

def compute_battery_health_metrics(episode_data: Dict) -> Dict:
    """Compute battery health and usage metrics."""
    soc_data = np.array(episode_data['battery_soc'])
    actions = np.array(episode_data['actions'])
    
    # Extract battery actions (assume first action is battery)
    battery_actions = actions[:, 0] if actions.ndim > 1 else actions
    
    # Metrics
    soc_range = np.max(soc_data) - np.min(soc_data)
    soc_std = np.std(soc_data)
    
    # Count charge/discharge cycles (rough estimate)
    soc_diff = np.diff(soc_data)
    charge_events = np.sum(soc_diff > 0.01)
    discharge_events = np.sum(soc_diff < -0.01)
    
    # Action smoothness
    action_changes = np.abs(np.diff(battery_actions))
    avg_action_change = np.mean(action_changes)
    max_action_change = np.max(action_changes)
    
    return {
        'soc_range': soc_range,
        'soc_std': soc_std,
        'charge_events': charge_events,
        'discharge_events': discharge_events,
        'avg_action_change': avg_action_change,
        'max_action_change': max_action_change,
        'smoothness_score': 1.0 / (1.0 + avg_action_change)  # Higher is smoother
    }

def generate_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Generate LaTeX table code."""
    latex_str = "\\begin{table}[htbp]\n"
    latex_str += "\\centering\n"
    latex_str += f"\\caption{{{caption}}}\n"
    latex_str += f"\\label{{{label}}}\n"
    latex_str += df.to_latex(index=False, escape=False)
    latex_str += "\\end{table}\n"
    
    return latex_str

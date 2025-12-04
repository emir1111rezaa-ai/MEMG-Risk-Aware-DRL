"""Configuration file for Multi-Energy Microgrid system."""
import numpy as np

# Simulation parameters
SIM_CONFIG = {
    'total_days': 365,
    'timesteps_per_day': 24,  # Hourly resolution
    'training_days': 292,     # 80% for training
    'validation_days': 36,    # 10% for validation
    'test_days': 37,          # 10% for testing
    'random_seed': 42
}

# Electrical bus parameters
ELECTRICAL_BUS = {
    'pv_capacity': 200.0,      # kW
    'wt_capacity': 150.0,      # kW
    'battery_capacity': 500.0, # kWh
    'battery_max_power': 100.0,# kW
    'battery_min_soc': 0.1,    # 10%
    'battery_max_soc': 0.9,    # 90%
    'battery_initial_soc': 0.5,
    'battery_charge_eff': 0.95,
    'battery_discharge_eff': 0.95,
    'battery_degradation_cost': 0.05,  # $/kWh cycled
    'battery_cycle_penalty': 0.01,     # Penalty for rapid changes
}

# CHP (Combined Heat and Power) parameters
CHP_CONFIG = {
    'max_power_elec': 100.0,   # kW electrical output
    'min_power_elec': 20.0,    # kW minimum stable operation
    'electrical_efficiency': 0.35,
    'thermal_efficiency': 0.45,
    'total_efficiency': 0.80,
    'ramp_rate': 30.0,         # kW/hour
    'gas_price': 0.03,         # $/kWh (natural gas)
    'maintenance_cost': 0.015, # $/kWh
}

# Gas boiler parameters
BOILER_CONFIG = {
    'max_thermal_power': 150.0,  # kW thermal output
    'min_thermal_power': 0.0,
    'thermal_efficiency': 0.90,
    'gas_price': 0.03,           # $/kWh (natural gas)
    'maintenance_cost': 0.005,   # $/kWh
}

# Grid connection parameters
GRID_CONFIG = {
    'max_import': 250.0,        # kW
    'max_export': 200.0,        # kW
    'import_price_base': 0.12,  # $/kWh (base price)
    'export_price_base': 0.08,  # $/kWh (feed-in tariff)
    'peak_hours': [17, 18, 19, 20, 21],  # Evening peak
    'peak_multiplier': 1.5,     # Price multiplier during peak
    'contract_penalty': 10.0,   # Penalty for exceeding limits
}

# Load profiles (will be generated with forecasting)
LOAD_CONFIG = {
    'electrical_load_mean': 120.0,  # kW
    'electrical_load_std': 30.0,
    'thermal_load_mean': 80.0,      # kW
    'thermal_load_std': 20.0,
    'peak_load_ratio': 1.8,
    'seasonal_variation': 0.3,
}

# Renewable generation profiles
RENEWABLE_CONFIG = {
    'pv_capacity_factor_mean': 0.20,
    'pv_capacity_factor_std': 0.15,
    'wt_capacity_factor_mean': 0.30,
    'wt_capacity_factor_std': 0.20,
    'weather_correlation': 0.7,  # Correlation between forecasting errors
}

# Forecasting module parameters
FORECASTING_CONFIG = {
    'cnn_filters': [64, 128, 64],
    'cnn_kernel_sizes': [3, 3, 3],
    'lstm_hidden_size': 128,
    'lstm_layers': 2,
    'lookback_window': 168,      # 7 days (hourly)
    'forecast_horizon': 24,      # 24 hours ahead
    'wavelet_family': 'db4',     # Daubechies wavelet
    'wavelet_level': 3,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'dropout': 0.2,
}

# PPO agent parameters
PPO_CONFIG = {
    'actor_hidden_dims': [256, 256, 128],
    'critic_hidden_dims': [256, 256, 128],
    'learning_rate_actor': 3e-4,
    'learning_rate_critic': 1e-3,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.01,
    'value_loss_coef': 0.5,
    'max_grad_norm': 0.5,
    'ppo_epochs': 10,
    'mini_batch_size': 64,
    'buffer_size': 2048,
}

# Risk-aware reward parameters
RISK_CONFIG = {
    'cvar_alpha': 0.95,          # CVaR confidence level
    'risk_aversion': 0.3,        # Weight for risk term in reward
    'cost_weight': 1.0,
    'constraint_penalty_weight': 100.0,
    'battery_health_weight': 0.1,
    'smoothness_weight': 0.05,
}

# MDP state space definition
STATE_SPACE = {
    'battery_soc': (0.0, 1.0),
    'pv_forecast': (0.0, ELECTRICAL_BUS['pv_capacity']),
    'wt_forecast': (0.0, ELECTRICAL_BUS['wt_capacity']),
    'elec_load_forecast': (0.0, LOAD_CONFIG['electrical_load_mean'] * 3),
    'thermal_load_forecast': (0.0, LOAD_CONFIG['thermal_load_mean'] * 3),
    'grid_price': (0.0, GRID_CONFIG['import_price_base'] * 2),
    'hour_of_day': (0, 23),
    'day_of_week': (0, 6),
    'chp_power_prev': (0.0, CHP_CONFIG['max_power_elec']),
    'forecast_uncertainty': (0.0, 1.0),  # Estimated forecast error
}

# MDP action space definition
ACTION_SPACE = {
    'battery_power': (-ELECTRICAL_BUS['battery_max_power'], 
                      ELECTRICAL_BUS['battery_max_power']),  # Negative = charge
    'chp_power_elec': (0.0, CHP_CONFIG['max_power_elec']),
    'grid_power': (-GRID_CONFIG['max_export'], 
                   GRID_CONFIG['max_import']),  # Negative = export
    'boiler_power': (0.0, BOILER_CONFIG['max_thermal_power']),
}

# Constraint violation penalties
CONSTRAINT_PENALTIES = {
    'battery_soc_violation': 500.0,
    'battery_power_violation': 300.0,
    'grid_limit_violation': 1000.0,
    'chp_ramp_violation': 200.0,
    'power_balance_violation': 1000.0,
    'thermal_balance_violation': 800.0,
}

# Baseline method configurations
BASELINE_CONFIGS = {
    'classical_optimization': {
        'method': 'MILP',  # Mixed Integer Linear Programming
        'scenarios': 10,    # Number of scenarios for stochastic programming
        'solver': 'GLPK',
        'time_limit': 300,  # seconds
    },
    'simple_drl': {
        'use_advanced_forecasting': False,
        'use_risk_aware_reward': False,
        'hidden_dims': [128, 128],
        'learning_rate': 3e-4,
    },
}

# Evaluation metrics
EVALUATION_METRICS = [
    'mean_total_cost',
    'cvar_cost',
    'constraint_violations_count',
    'battery_degradation_cost',
    'grid_import_cost',
    'chp_fuel_cost',
    'peak_power_utilization',
    'renewable_utilization_rate',
    'battery_cycle_count',
    'forecast_mae',
    'forecast_rmse',
]

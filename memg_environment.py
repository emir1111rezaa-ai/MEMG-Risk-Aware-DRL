"""Multi-Energy Microgrid Environment for DRL."""
import numpy as np
import gym
from gym import spaces
from typing import Dict, Tuple, Optional
from config import *

class MEMGEnvironment(gym.Env):
    """MDP formulation of Multi-Energy Microgrid Energy Management System."""
    
    def __init__(self, data: Dict, forecaster=None, use_forecasting: bool = True):
        super(MEMGEnvironment, self).__init__()
        
        self.data = data
        self.forecaster = forecaster
        self.use_forecasting = use_forecasting and forecaster is not None
        
        self.timestep = 0
        self.max_timesteps = len(data['electrical_load'])
        
        # State space: [soc, pv_forecast, wt_forecast, elec_load_forecast, 
        #                thermal_load_forecast, grid_price, hour, day_of_week,
        #                chp_power_prev, forecast_uncertainty]
        self.state_dim = 10
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Action space: [battery_power_ratio, chp_power_ratio, grid_power_ratio, boiler_power_ratio]
        # All actions normalized to [-1, 1] or [0, 1]
        self.action_dim = 4
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Internal state
        self.battery_soc = ELECTRICAL_BUS['battery_initial_soc']
        self.chp_power_prev = 0.0
        self.battery_power_prev = 0.0
        
        # Tracking variables
        self.total_cost = 0.0
        self.cost_history = []
        self.constraint_violations = []
        self.battery_cycles = 0.0
        
        # Forecasts cache
        self.forecasts = {}
        if self.use_forecasting:
            self._generate_all_forecasts()
    
    def _generate_all_forecasts(self):
        """Pre-generate forecasts for all timesteps."""
        print("Generating forecasts for environment...")
        
        variables = ['pv_generation', 'wt_generation', 'electrical_load', 'thermal_load']
        
        for var in variables:
            self.forecasts[var] = {'predictions': [], 'uncertainties': []}
            
            lookback = FORECASTING_CONFIG['lookback_window']
            horizon = FORECASTING_CONFIG['forecast_horizon']
            
            # Generate rolling forecasts
            for t in range(lookback, self.max_timesteps - horizon + 1):
                historical = self.data[var][t - lookback:t]
                forecast, uncertainty = self.forecaster.predict(var, historical)
                
                self.forecasts[var]['predictions'].append(forecast)
                self.forecasts[var]['uncertainties'].append(uncertainty)
        
        print("Forecasts generated.")
    
    def _get_forecast(self, variable: str, timestep: int, horizon: int = 24) -> Tuple[np.ndarray, float]:
        """Get forecast for a variable at a specific timestep."""
        if self.use_forecasting and variable in self.forecasts:
            lookback = FORECASTING_CONFIG['lookback_window']
            idx = timestep - lookback
            
            if 0 <= idx < len(self.forecasts[variable]['predictions']):
                forecast = self.forecasts[variable]['predictions'][idx]
                uncertainty = np.mean(self.forecasts[variable]['uncertainties'][idx])
                return forecast[:horizon], uncertainty
        
        # Fallback: use actual values (perfect foresight)
        forecast = self.data[variable][timestep:timestep + horizon]
        return forecast, 0.0
    
    def _normalize_state(self, state_dict: Dict) -> np.ndarray:
        """Normalize state to [0, 1] range."""
        normalized = np.array([
            state_dict['battery_soc'],
            state_dict['pv_forecast'] / ELECTRICAL_BUS['pv_capacity'],
            state_dict['wt_forecast'] / ELECTRICAL_BUS['wt_capacity'],
            state_dict['elec_load_forecast'] / (LOAD_CONFIG['electrical_load_mean'] * 3),
            state_dict['thermal_load_forecast'] / (LOAD_CONFIG['thermal_load_mean'] * 3),
            state_dict['grid_price'] / (GRID_CONFIG['import_price_base'] * 2),
            state_dict['hour_of_day'] / 23.0,
            state_dict['day_of_week'] / 6.0,
            state_dict['chp_power_prev'] / CHP_CONFIG['max_power_elec'],
            np.clip(state_dict['forecast_uncertainty'], 0, 1)
        ], dtype=np.float32)
        
        return np.clip(normalized, 0.0, 1.0)
    
    def _get_current_state(self) -> np.ndarray:
        """Get current state observation."""
        # Get forecasts
        pv_forecast, pv_unc = self._get_forecast('pv_generation', self.timestep, 1)
        wt_forecast, wt_unc = self._get_forecast('wt_generation', self.timestep, 1)
        elec_load_forecast, elec_unc = self._get_forecast('electrical_load', self.timestep, 1)
        thermal_load_forecast, th_unc = self._get_forecast('thermal_load', self.timestep, 1)
        
        # Average uncertainty
        avg_uncertainty = np.mean([pv_unc, wt_unc, elec_unc, th_unc])
        
        state_dict = {
            'battery_soc': self.battery_soc,
            'pv_forecast': pv_forecast[0] if len(pv_forecast) > 0 else 0,
            'wt_forecast': wt_forecast[0] if len(wt_forecast) > 0 else 0,
            'elec_load_forecast': elec_load_forecast[0] if len(elec_load_forecast) > 0 else 0,
            'thermal_load_forecast': thermal_load_forecast[0] if len(thermal_load_forecast) > 0 else 0,
            'grid_price': self.data['grid_import_price'][self.timestep],
            'hour_of_day': self.data['hour'][self.timestep],
            'day_of_week': self.data['day_of_week'][self.timestep],
            'chp_power_prev': self.chp_power_prev,
            'forecast_uncertainty': avg_uncertainty
        }
        
        return self._normalize_state(state_dict)
    
    def _denormalize_action(self, action: np.ndarray) -> Dict:
        """Convert normalized action to actual control commands."""
        battery_power = action[0] * ELECTRICAL_BUS['battery_max_power']
        chp_power = action[1] * CHP_CONFIG['max_power_elec']
        grid_power = action[2] * GRID_CONFIG['max_import'] if action[2] > 0 else \
                     action[2] * GRID_CONFIG['max_export']
        boiler_power = action[3] * BOILER_CONFIG['max_thermal_power']
        
        return {
            'battery_power': battery_power,
            'chp_power_elec': chp_power,
            'grid_power': grid_power,
            'boiler_power': boiler_power
        }
    
    def _apply_constraints(self, action_dict: Dict) -> Tuple[Dict, Dict]:
        """Apply physical constraints and compute violations."""
        violations = {
            'battery_soc': 0.0,
            'battery_power': 0.0,
            'grid_limit': 0.0,
            'chp_ramp': 0.0,
            'power_balance': 0.0,
            'thermal_balance': 0.0
        }
        
        # Battery power limits
        if action_dict['battery_power'] > ELECTRICAL_BUS['battery_max_power']:
            violations['battery_power'] = action_dict['battery_power'] - ELECTRICAL_BUS['battery_max_power']
            action_dict['battery_power'] = ELECTRICAL_BUS['battery_max_power']
        elif action_dict['battery_power'] < -ELECTRICAL_BUS['battery_max_power']:
            violations['battery_power'] = abs(action_dict['battery_power']) - ELECTRICAL_BUS['battery_max_power']
            action_dict['battery_power'] = -ELECTRICAL_BUS['battery_max_power']
        
        # CHP ramp rate constraint
        chp_ramp = abs(action_dict['chp_power_elec'] - self.chp_power_prev)
        if chp_ramp > CHP_CONFIG['ramp_rate']:
            violations['chp_ramp'] = chp_ramp - CHP_CONFIG['ramp_rate']
            if action_dict['chp_power_elec'] > self.chp_power_prev:
                action_dict['chp_power_elec'] = self.chp_power_prev + CHP_CONFIG['ramp_rate']
            else:
                action_dict['chp_power_elec'] = self.chp_power_prev - CHP_CONFIG['ramp_rate']
        
        # CHP min/max limits
        if action_dict['chp_power_elec'] > 0:
            action_dict['chp_power_elec'] = np.clip(
                action_dict['chp_power_elec'],
                CHP_CONFIG['min_power_elec'],
                CHP_CONFIG['max_power_elec']
            )
        
        # Grid limits
        if action_dict['grid_power'] > GRID_CONFIG['max_import']:
            violations['grid_limit'] = action_dict['grid_power'] - GRID_CONFIG['max_import']
            action_dict['grid_power'] = GRID_CONFIG['max_import']
        elif action_dict['grid_power'] < -GRID_CONFIG['max_export']:
            violations['grid_limit'] = abs(action_dict['grid_power']) - GRID_CONFIG['max_export']
            action_dict['grid_power'] = -GRID_CONFIG['max_export']
        
        return action_dict, violations
    
    def _simulate_step(self, action_dict: Dict) -> Tuple[float, Dict, Dict]:
        """Simulate one timestep of the MEMG."""
        # Get actual values (ground truth)
        pv_actual = self.data['pv_generation'][self.timestep]
        wt_actual = self.data['wt_generation'][self.timestep]
        elec_load_actual = self.data['electrical_load'][self.timestep]
        thermal_load_actual = self.data['thermal_load'][self.timestep]
        
        # CHP thermal output
        chp_thermal = action_dict['chp_power_elec'] * \
                     (CHP_CONFIG['thermal_efficiency'] / CHP_CONFIG['electrical_efficiency'])
        
        # Battery dynamics
        if action_dict['battery_power'] > 0:  # Discharging
            battery_energy_change = -action_dict['battery_power'] * ELECTRICAL_BUS['battery_discharge_eff']
        else:  # Charging
            battery_energy_change = -action_dict['battery_power'] / ELECTRICAL_BUS['battery_charge_eff']
        
        new_soc = self.battery_soc + battery_energy_change / ELECTRICAL_BUS['battery_capacity']
        
        violations = {}
        
        # SOC constraints
        if new_soc > ELECTRICAL_BUS['battery_max_soc']:
            violations['battery_soc'] = new_soc - ELECTRICAL_BUS['battery_max_soc']
            new_soc = ELECTRICAL_BUS['battery_max_soc']
            # Adjust battery power to respect SOC limit
            action_dict['battery_power'] = (ELECTRICAL_BUS['battery_max_soc'] - self.battery_soc) * \
                                          ELECTRICAL_BUS['battery_capacity'] * \
                                          ELECTRICAL_BUS['battery_charge_eff']
        elif new_soc < ELECTRICAL_BUS['battery_min_soc']:
            violations['battery_soc'] = ELECTRICAL_BUS['battery_min_soc'] - new_soc
            new_soc = ELECTRICAL_BUS['battery_min_soc']
            action_dict['battery_power'] = (self.battery_soc - ELECTRICAL_BUS['battery_min_soc']) * \
                                          ELECTRICAL_BUS['battery_capacity'] / \
                                          ELECTRICAL_BUS['battery_discharge_eff']
        
        # Electrical power balance
        elec_generation = pv_actual + wt_actual + action_dict['chp_power_elec'] + \
                         action_dict['battery_power']
        elec_balance = elec_generation + action_dict['grid_power'] - elec_load_actual
        
        if abs(elec_balance) > 1.0:  # Tolerance of 1 kW
            violations['power_balance'] = abs(elec_balance)
        
        # Thermal power balance
        thermal_generation = chp_thermal + action_dict['boiler_power']
        thermal_balance = thermal_generation - thermal_load_actual
        
        if thermal_balance < -1.0:  # Thermal shortage
            violations['thermal_balance'] = abs(thermal_balance)
        
        # Calculate costs
        costs = self._calculate_costs(action_dict, elec_load_actual)
        
        # Update state
        self.battery_soc = new_soc
        self.chp_power_prev = action_dict['chp_power_elec']
        
        # Track battery cycles
        battery_cycle_increment = abs(action_dict['battery_power']) / \
                                 (2 * ELECTRICAL_BUS['battery_capacity'])
        self.battery_cycles += battery_cycle_increment
        
        info = {
            'costs': costs,
            'violations': violations,
            'battery_cycles': battery_cycle_increment,
            'elec_balance': elec_balance,
            'thermal_balance': thermal_balance,
            'renewable_generation': pv_actual + wt_actual,
            'elec_load': elec_load_actual,
            'thermal_load': thermal_load_actual
        }
        
        return new_soc, action_dict, info
    
    def _calculate_costs(self, action_dict: Dict, elec_load: float) -> Dict:
        """Calculate all cost components."""
        # Grid import/export cost
        grid_price = self.data['grid_import_price'][self.timestep]
        if action_dict['grid_power'] > 0:  # Import
            grid_cost = action_dict['grid_power'] * grid_price
        else:  # Export
            export_price = self.data['grid_export_price'][self.timestep]
            grid_cost = action_dict['grid_power'] * export_price  # Negative cost (revenue)
        
        # CHP fuel and maintenance cost
        chp_gas_consumption = action_dict['chp_power_elec'] / CHP_CONFIG['electrical_efficiency']
        chp_cost = chp_gas_consumption * CHP_CONFIG['gas_price'] + \
                  action_dict['chp_power_elec'] * CHP_CONFIG['maintenance_cost']
        
        # Boiler fuel cost
        boiler_gas_consumption = action_dict['boiler_power'] / BOILER_CONFIG['thermal_efficiency']
        boiler_cost = boiler_gas_consumption * BOILER_CONFIG['gas_price'] + \
                     action_dict['boiler_power'] * BOILER_CONFIG['maintenance_cost']
        
        # Battery degradation cost
        battery_degradation = abs(action_dict['battery_power']) * \
                             ELECTRICAL_BUS['battery_degradation_cost']
        
        # Battery cycling penalty (for smoothness)
        battery_change = abs(action_dict['battery_power'] - self.battery_power_prev)
        battery_cycle_penalty = battery_change * ELECTRICAL_BUS['battery_cycle_penalty']
        
        self.battery_power_prev = action_dict['battery_power']
        
        total_cost = grid_cost + chp_cost + boiler_cost + battery_degradation + battery_cycle_penalty
        
        return {
            'grid': grid_cost,
            'chp': chp_cost,
            'boiler': boiler_cost,
            'battery_degradation': battery_degradation,
            'battery_cycling': battery_cycle_penalty,
            'total': total_cost
        }
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one timestep."""
        # Denormalize action
        action_dict = self._denormalize_action(action)
        
        # Apply constraints
        action_dict, constraint_violations = self._apply_constraints(action_dict)
        
        # Simulate
        new_soc, final_action, info = self._simulate_step(action_dict)
        
        # Merge violations
        for key, value in constraint_violations.items():
            if key in info['violations']:
                info['violations'][key] += value
            else:
                info['violations'][key] = value
        
        # Calculate reward
        reward = self._calculate_reward(info)
        
        # Update tracking
        self.total_cost += info['costs']['total']
        self.cost_history.append(info['costs']['total'])
        self.constraint_violations.append(sum(info['violations'].values()))
        
        # Next state
        self.timestep += 1
        done = self.timestep >= self.max_timesteps - 1
        
        next_state = self._get_current_state() if not done else np.zeros(self.state_dim)
        
        info['timestep'] = self.timestep
        info['total_cost'] = self.total_cost
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, info: Dict) -> float:
        """Calculate risk-aware reward."""
        # Base cost (negative reward)
        cost = info['costs']['total']
        
        # Constraint violation penalties
        violation_penalty = 0.0
        for key, value in info['violations'].items():
            if key in CONSTRAINT_PENALTIES:
                violation_penalty += value * CONSTRAINT_PENALTIES[key]
        
        # Battery health reward (smooth operation)
        smoothness_reward = -info['battery_cycles'] * RISK_CONFIG['battery_health_weight']
        
        # Base reward
        reward = -(cost * RISK_CONFIG['cost_weight'] + 
                  violation_penalty * RISK_CONFIG['constraint_penalty_weight'])
        
        # Add smoothness
        reward += smoothness_reward * RISK_CONFIG['smoothness_weight']
        
        return reward
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.timestep = FORECASTING_CONFIG['lookback_window']  # Start after lookback
        self.battery_soc = ELECTRICAL_BUS['battery_initial_soc']
        self.chp_power_prev = 0.0
        self.battery_power_prev = 0.0
        self.total_cost = 0.0
        self.cost_history = []
        self.constraint_violations = []
        self.battery_cycles = 0.0
        
        return self._get_current_state()
    
    def render(self, mode='human'):
        """Render environment state."""
        if mode == 'human':
            print(f"Timestep: {self.timestep}, SOC: {self.battery_soc:.3f}, "
                  f"Cost: {self.total_cost:.2f}")

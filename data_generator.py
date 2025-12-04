"""Generate synthetic time-series data for MEMG simulation."""
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from config import *

class MEMGDataGenerator:
    """Generate realistic multi-energy microgrid time-series data."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        self.total_timesteps = SIM_CONFIG['total_days'] * SIM_CONFIG['timesteps_per_day']
        
    def generate_all_data(self) -> Dict[str, np.ndarray]:
        """Generate all required time-series data."""
        print("Generating MEMG time-series data...")
        
        timestamps = pd.date_range(
            start='2024-01-01',
            periods=self.total_timesteps,
            freq='H'
        )
        
        # Generate load profiles
        electrical_load = self._generate_electrical_load(timestamps)
        thermal_load = self._generate_thermal_load(timestamps)
        
        # Generate renewable generation
        pv_generation = self._generate_pv_profile(timestamps)
        wt_generation = self._generate_wt_profile(timestamps)
        
        # Generate grid prices
        grid_prices = self._generate_grid_prices(timestamps)
        
        # Generate weather-related features
        temperature = self._generate_temperature(timestamps)
        
        data = {
            'timestamp': timestamps,
            'electrical_load': electrical_load,
            'thermal_load': thermal_load,
            'pv_generation': pv_generation,
            'wt_generation': wt_generation,
            'grid_import_price': grid_prices,
            'grid_export_price': grid_prices * (GRID_CONFIG['export_price_base'] / 
                                                GRID_CONFIG['import_price_base']),
            'temperature': temperature,
            'hour': timestamps.hour.values,
            'day_of_week': timestamps.dayofweek.values,
            'day_of_year': timestamps.dayofyear.values,
        }
        
        print(f"Generated {self.total_timesteps} timesteps of data")
        return data
    
    def _generate_electrical_load(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Generate realistic electrical load profile."""
        load = np.zeros(len(timestamps))
        
        for i, ts in enumerate(timestamps):
            # Base load
            base = LOAD_CONFIG['electrical_load_mean']
            
            # Seasonal variation (higher in summer/winter)
            seasonal = base * LOAD_CONFIG['seasonal_variation'] * \
                      np.cos(2 * np.pi * ts.dayofyear / 365 - np.pi)
            
            # Daily pattern (peak in morning and evening)
            hour_factor = 0.7 + 0.3 * np.cos(2 * np.pi * (ts.hour - 18) / 24)
            if 6 <= ts.hour <= 9:  # Morning peak
                hour_factor += 0.4
            elif 17 <= ts.hour <= 21:  # Evening peak
                hour_factor += 0.6
            
            # Weekend reduction
            if ts.dayofweek >= 5:
                hour_factor *= 0.85
            
            load[i] = base * hour_factor + seasonal
        
        # Add noise with autocorrelation
        noise = self._generate_correlated_noise(len(timestamps), 
                                                LOAD_CONFIG['electrical_load_std'],
                                                correlation=0.8)
        load += noise
        
        return np.maximum(load, 0)
    
    def _generate_thermal_load(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Generate realistic thermal load profile."""
        load = np.zeros(len(timestamps))
        
        for i, ts in enumerate(timestamps):
            # Base load
            base = LOAD_CONFIG['thermal_load_mean']
            
            # Strong seasonal variation (heating in winter)
            seasonal = base * 0.5 * np.cos(2 * np.pi * (ts.dayofyear - 15) / 365)
            
            # Daily pattern (higher during occupied hours)
            hour_factor = 1.0
            if 6 <= ts.hour <= 22:
                hour_factor = 1.3
            else:
                hour_factor = 0.6
            
            # Weekend pattern
            if ts.dayofweek >= 5:
                hour_factor *= 1.1  # More heating on weekends
            
            load[i] = (base + seasonal) * hour_factor
        
        # Add correlated noise
        noise = self._generate_correlated_noise(len(timestamps),
                                                LOAD_CONFIG['thermal_load_std'],
                                                correlation=0.85)
        load += noise
        
        return np.maximum(load, 0)
    
    def _generate_pv_profile(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Generate realistic PV generation profile."""
        generation = np.zeros(len(timestamps))
        
        for i, ts in enumerate(timestamps):
            # Solar availability (0 at night)
            if 6 <= ts.hour <= 18:
                # Bell curve for daily solar irradiance
                hour_angle = np.pi * (ts.hour - 6) / 12
                solar_factor = np.sin(hour_angle)
                
                # Seasonal variation (more in summer)
                seasonal = 0.7 + 0.3 * np.cos(2 * np.pi * (ts.dayofyear - 172) / 365)
                
                # Capacity factor
                cf = RENEWABLE_CONFIG['pv_capacity_factor_mean'] * solar_factor * seasonal
            else:
                cf = 0
            
            generation[i] = ELECTRICAL_BUS['pv_capacity'] * cf
        
        # Add weather-induced variability
        noise = self._generate_correlated_noise(len(timestamps),
                                                ELECTRICAL_BUS['pv_capacity'] * 0.15,
                                                correlation=0.9)
        generation += noise
        
        return np.clip(generation, 0, ELECTRICAL_BUS['pv_capacity'])
    
    def _generate_wt_profile(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Generate realistic wind turbine generation profile."""
        # Wind is more variable than solar
        base_cf = RENEWABLE_CONFIG['wt_capacity_factor_mean']
        
        # Generate using multi-scale noise
        generation = np.zeros(len(timestamps))
        
        # Seasonal wind patterns (more in winter)
        for i, ts in enumerate(timestamps):
            seasonal = 0.8 + 0.2 * np.cos(2 * np.pi * (ts.dayofyear - 15) / 365)
            generation[i] = base_cf * seasonal
        
        # Add highly variable noise (wind intermittency)
        noise = self._generate_correlated_noise(len(timestamps),
                                                base_cf * 0.5,
                                                correlation=0.7)
        generation += noise
        
        # Apply wind turbine power curve (simplified cubic)
        generation = np.clip(generation, 0, 1.0)
        generation = generation ** 1.5  # Non-linear power curve
        
        return generation * ELECTRICAL_BUS['wt_capacity']
    
    def _generate_grid_prices(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Generate dynamic grid electricity prices."""
        prices = np.ones(len(timestamps)) * GRID_CONFIG['import_price_base']
        
        for i, ts in enumerate(timestamps):
            # Peak hour pricing
            if ts.hour in GRID_CONFIG['peak_hours']:
                prices[i] *= GRID_CONFIG['peak_multiplier']
            
            # Seasonal adjustment
            seasonal = 1.0 + 0.15 * np.cos(2 * np.pi * (ts.dayofyear - 15) / 365)
            prices[i] *= seasonal
            
            # Weekend reduction
            if ts.dayofweek >= 5:
                prices[i] *= 0.9
        
        # Add price volatility
        noise = self._generate_correlated_noise(len(timestamps),
                                                GRID_CONFIG['import_price_base'] * 0.1,
                                                correlation=0.85)
        prices += noise
        
        return np.maximum(prices, GRID_CONFIG['import_price_base'] * 0.5)
    
    def _generate_temperature(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Generate ambient temperature profile (Celsius)."""
        temp = np.zeros(len(timestamps))
        
        for i, ts in enumerate(timestamps):
            # Annual variation
            annual_mean = 15.0  # °C
            seasonal = 15.0 * np.cos(2 * np.pi * (ts.dayofyear - 200) / 365)
            
            # Daily variation
            daily = 5.0 * np.cos(2 * np.pi * (ts.hour - 14) / 24)
            
            temp[i] = annual_mean + seasonal + daily
        
        # Add noise
        noise = self._generate_correlated_noise(len(timestamps), 3.0, correlation=0.9)
        temp += noise
        
        return temp
    
    def _generate_correlated_noise(self, length: int, std: float, 
                                   correlation: float = 0.8) -> np.ndarray:
        """Generate autocorrelated noise using AR(1) process."""
        noise = np.zeros(length)
        noise[0] = np.random.normal(0, std)
        
        for i in range(1, length):
            noise[i] = correlation * noise[i-1] + \
                      np.random.normal(0, std * np.sqrt(1 - correlation**2))
        
        return noise
    
    def split_data(self, data: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict]:
        """Split data into train, validation, and test sets."""
        train_end = SIM_CONFIG['training_days'] * SIM_CONFIG['timesteps_per_day']
        val_end = train_end + SIM_CONFIG['validation_days'] * SIM_CONFIG['timesteps_per_day']
        
        train_data = {k: v[:train_end] if isinstance(v, np.ndarray) else v[:train_end] 
                     for k, v in data.items()}
        val_data = {k: v[train_end:val_end] if isinstance(v, np.ndarray) else v[train_end:val_end]
                   for k, v in data.items()}
        test_data = {k: v[val_end:] if isinstance(v, np.ndarray) else v[val_end:]
                    for k, v in data.items()}
        
        return train_data, val_data, test_data

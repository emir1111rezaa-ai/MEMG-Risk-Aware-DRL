"""Advanced forecasting module using wavelet decomposition and CNN-LSTM."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pywt
from typing import Tuple, Dict, List
from config import FORECASTING_CONFIG

class WaveletDecomposition:
    """Wavelet-based signal preprocessing."""
    
    def __init__(self, wavelet: str = 'db4', level: int = 3):
        self.wavelet = wavelet
        self.level = level
    
    def decompose(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Decompose signal into approximation and detail coefficients."""
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        
        components = {
            'approximation': coeffs[0],
            'details': coeffs[1:]
        }
        
        return components
    
    def reconstruct(self, components: Dict[str, np.ndarray]) -> np.ndarray:
        """Reconstruct signal from wavelet components."""
        coeffs = [components['approximation']] + list(components['details'])
        reconstructed = pywt.waverec(coeffs, self.wavelet)
        return reconstructed
    
    def denoise(self, signal: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """Remove noise using wavelet soft thresholding."""
        components = self.decompose(signal)
        
        # Apply soft thresholding to detail coefficients
        denoised_details = []
        for detail in components['details']:
            threshold_val = threshold * np.std(detail)
            denoised = pywt.threshold(detail, threshold_val, mode='soft')
            denoised_details.append(denoised)
        
        denoised_components = {
            'approximation': components['approximation'],
            'details': denoised_details
        }
        
        return self.reconstruct(denoised_components)


class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time-series forecasting."""
    
    def __init__(self, data: np.ndarray, lookback: int, horizon: int):
        self.data = data
        self.lookback = lookback
        self.horizon = horizon
        
    def __len__(self):
        return len(self.data) - self.lookback - self.horizon + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.lookback]
        y = self.data[idx + self.lookback:idx + self.lookback + self.horizon]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class CNNLSTM(nn.Module):
    """CNN-LSTM architecture for time-series forecasting."""
    
    def __init__(self, input_dim: int, output_dim: int, config: Dict):
        super(CNNLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # CNN layers for feature extraction
        cnn_layers = []
        in_channels = input_dim
        
        for i, (filters, kernel_size) in enumerate(zip(
            config['cnn_filters'], config['cnn_kernel_sizes'])):
            cnn_layers.extend([
                nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(config['dropout'])
            ])
            in_channels = filters
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # LSTM layers for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=config['cnn_filters'][-1],
            hidden_size=config['lstm_hidden_size'],
            num_layers=config['lstm_layers'],
            batch_first=True,
            dropout=config['dropout'] if config['lstm_layers'] > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=config['lstm_hidden_size'],
            num_heads=4,
            dropout=config['dropout'],
            batch_first=True
        )
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(config['lstm_hidden_size'], config['lstm_hidden_size'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['lstm_hidden_size'] // 2, output_dim)
        )
    
    def forward(self, x):
        # x shape: (batch, sequence, features)
        batch_size, seq_len, features = x.shape
        
        # CNN expects (batch, channels, sequence)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        
        # Back to (batch, sequence, features) for LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last timestep
        out = attn_out[:, -1, :]
        
        # Project to output
        out = self.fc(out)
        
        return out


class AdvancedForecaster:
    """Advanced forecasting system with wavelet preprocessing and CNN-LSTM."""
    
    def __init__(self, config: Dict = FORECASTING_CONFIG, device: str = 'cpu'):
        self.config = config
        self.device = device
        self.wavelet = WaveletDecomposition(
            wavelet=config['wavelet_family'],
            level=config['wavelet_level']
        )
        self.models = {}  # Separate model for each variable
        self.scalers = {}  # Normalization parameters
        
    def preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply wavelet denoising and normalization."""
        # Denoise using wavelets
        denoised = self.wavelet.denoise(signal)
        return denoised
    
    def normalize(self, data: np.ndarray, variable_name: str, 
                 fit: bool = False) -> np.ndarray:
        """Normalize data to zero mean and unit variance."""
        if fit:
            self.scalers[variable_name] = {
                'mean': np.mean(data),
                'std': np.std(data) + 1e-8
            }
        
        scaler = self.scalers[variable_name]
        return (data - scaler['mean']) / scaler['std']
    
    def denormalize(self, data: np.ndarray, variable_name: str) -> np.ndarray:
        """Denormalize data back to original scale."""
        scaler = self.scalers[variable_name]
        return data * scaler['std'] + scaler['mean']
    
    def create_model(self, variable_name: str, input_features: int = 1):
        """Create CNN-LSTM model for a specific variable."""
        model = CNNLSTM(
            input_dim=input_features,
            output_dim=self.config['forecast_horizon'],
            config=self.config
        ).to(self.device)
        
        self.models[variable_name] = model
        return model
    
    def train_model(self, variable_name: str, train_data: np.ndarray,
                   val_data: np.ndarray, verbose: bool = True) -> Dict:
        """Train forecasting model for a specific variable."""
        print(f"\nTraining forecaster for {variable_name}...")
        
        # Preprocess
        train_processed = self.preprocess_signal(train_data)
        val_processed = self.preprocess_signal(val_data)
        
        # Normalize
        train_norm = self.normalize(train_processed, variable_name, fit=True)
        val_norm = self.normalize(val_processed, variable_name, fit=False)
        
        # Create datasets
        train_dataset = TimeSeriesDataset(
            train_norm, 
            self.config['lookback_window'],
            self.config['forecast_horizon']
        )
        val_dataset = TimeSeriesDataset(
            val_norm,
            self.config['lookback_window'],
            self.config['forecast_horizon']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        
        # Create model
        if variable_name not in self.models:
            self.create_model(variable_name)
        
        model = self.models[variable_name]
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=verbose
        )
        
        # Training loop
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.config['epochs']):
            # Training
            model.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.unsqueeze(-1).to(self.device)  # Add feature dim
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.unsqueeze(-1).to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    predictions = model(x_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save best model
                torch.save(model.state_dict(), f'best_{variable_name}_model.pth')
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config['epochs']}: "
                      f"Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Load best model
        model.load_state_dict(torch.load(f'best_{variable_name}_model.pth'))
        
        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        
        return history
    
    def predict(self, variable_name: str, historical_data: np.ndarray,
               return_uncertainty: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Generate forecast with uncertainty estimation."""
        if variable_name not in self.models:
            raise ValueError(f"No trained model for {variable_name}")
        
        model = self.models[variable_name]
        model.eval()
        
        # Preprocess
        processed = self.preprocess_signal(historical_data)
        normalized = self.normalize(processed, variable_name, fit=False)
        
        # Prepare input
        lookback = self.config['lookback_window']
        x = torch.FloatTensor(normalized[-lookback:]).unsqueeze(0).unsqueeze(-1)
        x = x.to(self.device)
        
        with torch.no_grad():
            # Point forecast
            forecast = model(x).cpu().numpy().flatten()
            
            # Uncertainty estimation via dropout sampling (MC Dropout)
            if return_uncertainty:
                model.train()  # Enable dropout
                samples = []
                for _ in range(30):  # 30 MC samples
                    sample = model(x).cpu().numpy().flatten()
                    samples.append(sample)
                model.eval()
                
                samples = np.array(samples)
                uncertainty = np.std(samples, axis=0)
            else:
                uncertainty = np.zeros_like(forecast)
        
        # Denormalize
        forecast = self.denormalize(forecast, variable_name)
        uncertainty = uncertainty * self.scalers[variable_name]['std']
        
        return forecast, uncertainty
    
    def evaluate(self, variable_name: str, test_data: np.ndarray) -> Dict:
        """Evaluate forecasting performance on test data."""
        print(f"\nEvaluating {variable_name} forecaster...")
        
        lookback = self.config['lookback_window']
        horizon = self.config['forecast_horizon']
        
        n_windows = len(test_data) - lookback - horizon + 1
        
        all_targets = []
        all_forecasts = []
        all_uncertainties = []
        
        for i in range(n_windows):
            historical = test_data[i:i + lookback]
            target = test_data[i + lookback:i + lookback + horizon]
            
            forecast, uncertainty = self.predict(variable_name, historical)
            
            all_targets.append(target)
            all_forecasts.append(forecast)
            all_uncertainties.append(uncertainty)
        
        all_targets = np.array(all_targets)
        all_forecasts = np.array(all_forecasts)
        all_uncertainties = np.array(all_uncertainties)
        
        # Compute metrics
        mae = np.mean(np.abs(all_targets - all_forecasts))
        rmse = np.sqrt(np.mean((all_targets - all_forecasts) ** 2))
        mape = np.mean(np.abs((all_targets - all_forecasts) / (all_targets + 1e-8))) * 100
        
        # Coverage of prediction intervals
        lower_bound = all_forecasts - 1.96 * all_uncertainties
        upper_bound = all_forecasts + 1.96 * all_uncertainties
        coverage = np.mean((all_targets >= lower_bound) & (all_targets <= upper_bound))
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'coverage_95': coverage,
            'mean_uncertainty': np.mean(all_uncertainties)
        }
        
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, Coverage: {coverage:.2%}")
        
        return metrics

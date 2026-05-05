import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from .base_model import BaseForecastingModel

logger = logging.getLogger(__name__)

# --- 1. The Neural Network Architecture ---
class PyTorchLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(PyTorchLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # The LSTM layer (batch_first=True ensures input shape is [batch, seq_len, features])
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer to map the hidden state to a single numerical prediction
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # We only care about the final output of the sequence
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out


# --- 2. The Framework Wrapper ---
class LSTMModel(BaseForecastingModel):
    def __init__(self, cfg):
        model_cfg = cfg.model
        self.window = model_cfg.lookback_window
        self.epochs = model_cfg.epochs
        self.batch_size = model_cfg.batch_size
        self.lr = model_cfg.learning_rate
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Instantiate the architecture and move it to the GPU
        self.model = PyTorchLSTM(
            hidden_size=model_cfg.hidden_size, 
            num_layers=model_cfg.num_layers
        ).to(self.device)
        
        # Neural nets require data to be strictly scaled between -1 and 1 or 0 and 1
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # Store the last sequence of training data to use as the "seed" for prediction
        self.last_sequence = None

    def _create_sequences(self, data: np.ndarray):
        """Slides a window across the 1D data to create X (context) and y (target) pairs."""
        X, y = [], []
        for i in range(len(data) - self.window):
            X.append(data[i : i + self.window])
            y.append(data[i + self.window])
        return np.array(X), np.array(y)

    def train(self, train_data: pd.Series):
        """Normalizes data, builds sequences, and executes the PyTorch training loop."""
        # 1. Normalize and reshape
        data_scaled = self.scaler.fit_transform(train_data.values.reshape(-1, 1))
        
        # Save the very last window so the predict() method has a starting point
        self.last_sequence = data_scaled[-self.window:]
        
        # 2. Build sequences
        X, y = self._create_sequences(data_scaled)
        
        # Convert to PyTorch Tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        # Create DataLoader for batching
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 3. Setup Loss and Optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # 4. Training Loop (tqdm leave=False keeps the console clean during Monte Carlo loops)
        self.model.train()
        for epoch in tqdm(range(self.epochs), desc="Training LSTM", leave=False):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, steps: int) -> list:
        """Autoregressively predicts future steps one by one."""
        self.model.eval()
        predictions = []
        
        # Start with the sequence saved at the end of training (Shape: [1, seq_len, 1])
        current_seq = torch.tensor(self.last_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(steps):
                # Predict the next step
                next_step = self.model(current_seq)
                predictions.append(next_step.item())
                
                # Append the new prediction to the sequence and drop the oldest value
                # (Rolling the window forward by 1)
                next_step_reshaped = next_step.unsqueeze(0)  # Shape: [1, 1, 1]
                current_seq = torch.cat((current_seq[:, 1:, :], next_step_reshaped), dim=1)
                
        # Inverse transform the predictions back to their original numerical scale
        predictions_unscaled = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        return predictions_unscaled.flatten().tolist()
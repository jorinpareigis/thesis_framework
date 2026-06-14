import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import logging
import random
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from .base_model import BaseForecastingModel

logger = logging.getLogger(__name__)

# --- 1. The Neural Network Architecture ---
class PyTorchLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(PyTorchLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=lstm_dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
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
        
        # Capture the isolated model seed from Hydra
        self.seed = cfg.current_model_seed 
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Force deterministic weight initialization
        self._set_seed()
        
        # Instantiate the architecture AFTER setting the seed
        self.model = PyTorchLSTM(
            hidden_size=model_cfg.hidden_size, 
            num_layers=model_cfg.num_layers,
            dropout=model_cfg.get("dropout", 0.2)
        ).to(self.device)
        
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.last_sequence = None

    def _set_seed(self):
        """Forces all underlying C++ and Python random number generators into a fixed state."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            # Enforce deterministic cuDNN algorithms but slows down learning, True means slow but same, False means different but faster
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = False

    def _create_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(len(data) - self.window):
            X.append(data[i : i + self.window])
            y.append(data[i + self.window])
        return np.array(X), np.array(y)

    def train(self, train_data: pd.Series):
        """Normalizes data, builds sequences, and executes the PyTorch training loop."""
        
        # Guard against un-imputed NaNs silently ruining loss metrics
        if train_data.isna().any():
            raise ValueError("Training data contains NaNs. Use an imputation method in your corruption config.")
            
        # 1. Normalize and reshape
        data_scaled = self.scaler.fit_transform(train_data.values.reshape(-1, 1))
        self.last_sequence = data_scaled[-self.window:]
        
        # 2. Build sequences
        X, y = self._create_sequences(data_scaled)
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        # Create DataLoader for batching
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Force the DataLoader's shuffle mechanism to be deterministic
        g = torch.Generator()
        g.manual_seed(self.seed)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=g)
        
        # 3. Setup Loss and Optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # 4. Training Loop
        self.model.train()
        for epoch in tqdm(range(self.epochs), desc="Training LSTM", leave=False):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # Prevent silent broadcasting bugs (Cursor Warning #15)
                loss = criterion(outputs, batch_y.unsqueeze(1)) 
                
                loss.backward()
                optimizer.step()

    def predict(self, steps: int) -> list:
        self.model.eval()
        predictions = []
        current_seq = torch.tensor(self.last_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(steps):
                next_step = self.model(current_seq)
                predictions.append(next_step.item())
                next_step_reshaped = next_step.unsqueeze(0)
                current_seq = torch.cat((current_seq[:, 1:, :], next_step_reshaped), dim=1)
                
        predictions_unscaled = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions_unscaled.flatten().tolist()
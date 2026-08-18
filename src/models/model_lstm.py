import logging
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .base_model import BaseForecastingModel

logger = logging.getLogger(__name__)

class PyTorchLSTM(nn.Module):
    """
    A standard PyTorch LSTM architecture for time-series forecasting.
    
    Consists of an LSTM layer followed by a fully connected linear layer 
    to map the final hidden state to a single continuous output.
    """
    def __init__(
        self, 
        input_size: int = 1, 
        hidden_size: int = 50, 
        num_layers: int = 2, 
        output_size: int = 1, 
        dropout: float = 0.2
    ) -> None:
        """
        Initializes the neural network layers.

        Args:
            input_size (int): Number of expected features in the input sequence.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent LSTM layers.
            output_size (int): Size of the final output projection.
            dropout (float): Dropout probability applied between LSTM layers.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, sequence_length, features).

        Returns:
            torch.Tensor: The predicted output tensor.
        """
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out


class LSTMModel(BaseForecastingModel):
    """
    Framework wrapper for the PyTorch LSTM model.
    
    Manages data normalization, rolling window sequence generation, deterministic 
    seeding, and the autoregressive inference loop.
    """
    def __init__(self, cfg: Any) -> None:
        """
        Initializes the LSTM model wrapper, configures hyperparameters, 
        and prepares the hardware environment.

        Args:
            cfg (Any): The Hydra configuration object.
        """
        model_cfg = cfg.model
        self.window = model_cfg.lookback_window
        self.epochs = model_cfg.epochs
        self.batch_size = model_cfg.batch_size
        self.lr = model_cfg.learning_rate
        
        self.seed = cfg.current_model_seed 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._set_seed()
        
        self.model = PyTorchLSTM(
            hidden_size=model_cfg.hidden_size, 
            num_layers=model_cfg.num_layers,
            dropout=model_cfg.get("dropout", 0.2)
        ).to(self.device)
        
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.last_sequence: np.ndarray | None = None

    def _set_seed(self) -> None:
        """
        Forces the underlying C++ and Python random number generators into a fixed state 
        to ensure reproducibility during weight initialization and data shuffling.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            # Disable cuDNN benchmarking to prevent dynamic selection of algorithms
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = False

    def _create_sequences(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Transforms a 1D time-series array into overlapping 2D sequences for recurrent training.

        Args:
            data (np.ndarray): The scaled 1D time-series data.

        Returns:
            tuple[np.ndarray, np.ndarray]: The feature matrix (X) and target vector (y).
        """
        X, y = [], []
        for i in range(len(data) - self.window):
            X.append(data[i : i + self.window])
            y.append(data[i + self.window])
        return np.array(X), np.array(y)

    def train(self, train_data: pd.Series) -> None:
        """
        Normalizes data, constructs rolling window sequences, and executes the training loop.

        Args:
            train_data (pd.Series): The historical training dataset.

        Raises:
            ValueError: If the training data contains un-imputed NaN values.
        """
        if train_data.isna().any():
            raise ValueError("Training data contains NaNs. Use an imputation method in your corruption config.")
            
        data_scaled = self.scaler.fit_transform(train_data.values.reshape(-1, 1))
        self.last_sequence = data_scaled[-self.window:]
        
        X, y = self._create_sequences(data_scaled)
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # The DataLoader utilizes a separate C++ generator for shuffling. 
        # Injecting a seeded generator enforces deterministic batching orders.
        g = torch.Generator()
        g.manual_seed(self.seed)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=g)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for epoch in tqdm(range(self.epochs), desc="Training LSTM", leave=False):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # Squeeze/unsqueeze guarantees dimensions align to prevent silent broadcasting bugs.
                loss = criterion(outputs, batch_y.unsqueeze(1)) 
                
                loss.backward()
                optimizer.step()

    def predict(self, steps: int) -> list[float]:
        """
        Executes autoregressive multi-step forecasting by iteratively predicting one step ahead 
        and appending the output to the rolling window context.

        Args:
            steps (int): The number of future time steps to predict.

        Returns:
            list[float]: The unscaled point forecasts in the original unit domain.
            
        Raises:
            RuntimeError: If the model has not been trained prior to calling predict.
        """
        if self.last_sequence is None:
            raise RuntimeError("Model must be trained before generating predictions.")

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
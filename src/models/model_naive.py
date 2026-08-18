import numpy as np
import pandas as pd
from typing import Any

from .base_model import BaseForecastingModel

class NaiveModel(BaseForecastingModel):
    """
    Implements baseline statistical forecasting methods.
    
    Acts as a minimum performance threshold to evaluate whether complex 
    machine learning models add tangible predictive value.
    """
    def __init__(self, cfg: Any) -> None:
        """
        Initializes the naive forecasting strategy and seasonal parameters.

        Args:
            cfg (Any): The configuration object containing the chosen strategy 
                       and dataset seasonality parameters.

        Raises:
            ValueError: If the requested strategy is not implemented.
        """
        model_cfg = cfg.model
        self.strategy = model_cfg.strategy
        self.season_length = model_cfg.get("season_length", 1)
        self.history: np.ndarray | None = None

        valid_strategies = ["forward_fill", "mean", "seasonal_average", "seasonal_naive"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"Strategy '{self.strategy}' is not recognized.")

    def train(self, train_data: pd.Series) -> None:
        """
        Stores the historical time-series data required for naive extrapolation.

        Args:
            train_data (pd.Series): The historical training dataset.

        Raises:
            ValueError: If the provided data length is shorter than the configured seasonal cycle.
        """
        self.history = train_data.astype(float).values
        
        if self.strategy in ["seasonal_naive", "seasonal_average"]:
            if len(self.history) < self.season_length:
                raise ValueError(
                    f"Data length ({len(self.history)}) must be >= season_length ({self.season_length})."
                )

    def predict(self, steps: int) -> list[float]:
        """
        Generates predictions based on the configured baseline strategy.

        Args:
            steps (int): The number of future time steps to predict.

        Returns:
            list[float]: The generated point forecasts.

        Raises:
            RuntimeError: If the model has not been trained prior to prediction.
        """
        if self.history is None:
            raise RuntimeError("The model must be trained before prediction.")
            
        predictions = []
        
        if self.strategy == "forward_fill":
            val = float(self.history[-1])
            predictions = [val] * steps
            
        elif self.strategy == "mean":
            val = float(np.mean(self.history))
            predictions = [val] * steps
            
        elif self.strategy == "seasonal_average":
            val = float(np.mean(self.history[-self.season_length:]))
            predictions = [val] * steps
            
        elif self.strategy == "seasonal_naive":
            # Modulo arithmetic cycles backwards through the final known season
            for i in range(steps):
                idx = -(self.season_length) + (i % self.season_length)
                predictions.append(float(self.history[idx]))
            
        return predictions
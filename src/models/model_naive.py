import numpy as np
import pandas as pd
from .base_model import BaseForecastingModel

class NaiveModel(BaseForecastingModel):
    """
    Implements basic baseline forecasting methods. 
    Used to establish a minimum performance threshold for complex models.
    """
    def __init__(self, cfg):
        model_cfg = cfg.model
        self.strategy = model_cfg.strategy
        # Defaults to 1 to prevent division by zero or indexing errors if omitted in YAML
        self.season_length = model_cfg.get("season_length", 1)
        self.history = None

        # Fail-fast validation: catch configuration errors before initiating the data pipeline
        valid_strategies = ["forward_fill", "mean", "seasonal_average", "seasonal_naive"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"Strategy '{self.strategy}' is not recognized.")

    def train(self, train_data: pd.Series):
        """Stores historical data and verifies structural requirements."""
        # Convert to standard float array for faster vectorized numpy operations later
        self.history = train_data.astype(float).values
        
        # Ensure enough historical data exists to calculate a full seasonal cycle
        if self.strategy in ["seasonal_naive", "seasonal_average"]:
            if len(self.history) < self.season_length:
                raise ValueError(
                    f"Data length ({len(self.history)}) must be >= season_length ({self.season_length})."
                )

    def predict(self, steps: int) -> list:
        """Executes the chosen naive forecasting strategy."""
        if self.history is None:
            raise RuntimeError("The model must be trained before prediction.")
            
        predictions = []
        
        if self.strategy == "forward_fill":
            # Propagates the most recent single observation forward
            val = self.history[-1]
            predictions = [val] * steps
            
        elif self.strategy == "mean":
            # Calculates the global mean of all available historical data
            val = np.mean(self.history)
            predictions = [val] * steps
            
        elif self.strategy == "seasonal_average":
            # Averages only the final complete seasonal cycle (e.g., the last 24 hours)
            val = np.mean(self.history[-self.season_length:])
            predictions = [val] * steps
            
        elif self.strategy == "seasonal_naive":
            # Repeats the exact pattern of the last known season
            for i in range(steps):
                # Modulo operator loops the index backward over the season_length boundary.
                # Example for season_length=24: idx cycles from -24 up to -1 repeatedly.
                idx = -(self.season_length) + (i % self.season_length)
                predictions.append(self.history[idx])
            
        return predictions
import numpy as np
import pandas as pd
from .base_model import BaseForecastingModel

class NaiveModel(BaseForecastingModel):
    def __init__(self, cfg):
        model_cfg = cfg.model
        self.strategy = model_cfg.strategy
        self.season_length = model_cfg.get("season_length", 1)
        self.history = None

    def train(self, train_data: pd.Series):
        """Stores the historical data needed for the naive calculations."""
        self.history = train_data.astype(float).values
        
        # Validation constraint
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
            # Flatline of the last known value
            val = self.history[-1]
            predictions = [val] * steps
            
        elif self.strategy == "mean":
            # Flatline of the entire historical average
            val = np.mean(self.history)
            predictions = [val] * steps
            
        elif self.strategy == "seasonal_average":
            # Flatline of the last full season's average
            val = np.mean(self.history[-self.season_length:])
            predictions = [val] * steps
            
        elif self.strategy == "seasonal_naive":
            # Replays the exact sequence of the last known season
            for i in range(steps):
                # Calculate the index to strictly loop through the last season
                idx = -(self.season_length) + (i % self.season_length)
                predictions.append(self.history[idx])
                
        else:
            raise NotImplementedError(f"Strategy '{self.strategy}' is not recognized.")
            
        return predictions
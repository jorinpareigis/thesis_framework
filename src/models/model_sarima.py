import warnings
from typing import Any

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .base_model import BaseForecastingModel

# Globally suppress warnings to hide verbose statsmodels convergence outputs
warnings.filterwarnings("ignore")

class SARIMAModel(BaseForecastingModel):
    """
    Implements the Seasonal Autoregressive Integrated Moving Average (SARIMA) model.
    
    A classic statistical autoregressive model suitable for univariate time-series 
    forecasting, providing explicit support for trend and seasonality.
    """
    def __init__(self, cfg: Any) -> None:
        """
        Initializes the SARIMA model with hyperparameters from the configuration.

        Args:
            cfg (Any): The Hydra configuration object.
        """
        model_cfg = cfg.model
        
        self.order = tuple(model_cfg.order)
        self.seasonal_order = tuple(model_cfg.seasonal_order)
        self.fitted_model: Any | None = None

    def train(self, train_data: pd.Series) -> None:
        """
        Instantiates and fits the SARIMA model to the historical time-series data.

        Args:
            train_data (pd.Series): The historical training dataset.
        """
        train_data = train_data.astype(float)
        
        model = SARIMAX(
            train_data, 
            order=self.order, 
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.fitted_model = model.fit(disp=False)

    def predict(self, steps: int) -> list[float]:
        """
        Generates point forecasts for future time steps using the fitted model.

        Args:
            steps (int): The number of future time steps to predict.

        Returns:
            list[float]: The generated point forecasts.
            
        Raises:
            RuntimeError: If the model has not been trained prior to prediction.
        """
        if self.fitted_model is None:
            raise RuntimeError("The model must be trained before prediction.")
            
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast.tolist()
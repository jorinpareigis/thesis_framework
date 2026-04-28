import warnings
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base_model import BaseForecastingModel

# Suppress standard statsmodels warnings to maintain a clean terminal output
warnings.filterwarnings("ignore")

class SARIMAXModel(BaseForecastingModel):
    def __init__(self, cfg):
        """
        Initializes the SARIMAX model using Hydra configuration parameters.
        """
        model_cfg = cfg.model
        # Convert lists from YAML into tuples required by statsmodels
        self.order = tuple(model_cfg.order)
        self.seasonal_order = tuple(model_cfg.seasonal_order)
        self.fitted_model = None

    def train(self, train_data: pd.Series):
        """
        Trains the SARIMAX model on the provided historical data.
        """
        # Enforce float type to prevent the pandas object dtype error encountered earlier
        train_data = train_data.astype(float)
        
        model = SARIMAX(
            train_data, 
            order=self.order, 
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.fitted_model = model.fit(disp=False)

    def predict(self, steps: int) -> list:
        """
        Generates a forecast for the specified number of future steps.
        """
        if self.fitted_model is None:
            raise RuntimeError("The model must be trained before prediction.")
            
        forecast = self.fitted_model.forecast(steps=steps) # type: ignore
        return forecast.tolist()
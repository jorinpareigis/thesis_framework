import warnings
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base_model import BaseForecastingModel

# Suppress standard statsmodels warnings to maintain a clean terminal output.
# statsmodels frequently throws convergence or non-stationary warnings which 
# are expected when feeding it heavily corrupted data during our stress tests.
warnings.filterwarnings("ignore")

class SARIMAXModel(BaseForecastingModel):
    """
    Implements the SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors) model.
    Acts as the standard statistical baseline for time-series forecasting.
    """
    def __init__(self, cfg):
        """
        Initializes the SARIMAX model using Hydra configuration parameters.
        """
        model_cfg = cfg.model
        
        # Hydra reads YAML arrays as Python lists, but statsmodels strictly 
        # requires tuples for the order parameters. Explicit conversion prevents TypeErrors.
        self.order = tuple(model_cfg.order)
        self.seasonal_order = tuple(model_cfg.seasonal_order)
        self.fitted_model = None

    def train(self, train_data: pd.Series):
        """
        Trains the SARIMAX model on the provided historical data.
        """
        # Enforce float type to prevent the pandas object dtype error encountered earlier.
        # This ensures compatibility if upstream parsing leaves mixed numeric/string artifacts.
        train_data = train_data.astype(float)
        
        model = SARIMAX(
            train_data, 
            order=self.order, 
            seasonal_order=self.seasonal_order,
            # Stationarity and invertibility checks are disabled. 
            # Linear interpolation over large gaps of missing data creates artificial flatlines.
            # These flatlines lack variance and will crash the strict mathematical solvers in statsmodels.
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # disp=False prevents the solver from printing iteration text to the console,
        # keeping the execution logs clean for Weights & Biases.
        self.fitted_model = model.fit(disp=False)

    def predict(self, steps: int) -> list:
        """
        Generates a forecast for the specified number of future steps.
        """
        if self.fitted_model is None:
            raise RuntimeError("The model must be trained before prediction.")
            
        # Generates point forecasts for the defined horizon.
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast.tolist()
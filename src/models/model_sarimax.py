import warnings
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base_model import BaseForecastingModel

warnings.filterwarnings("ignore")

class SARIMAXModel(BaseForecastingModel):
    def __init__(self, cfg):
        model_cfg = cfg.model
        
        self.order = tuple(model_cfg.order)
        self.seasonal_order = tuple(model_cfg.seasonal_order)
        self.fitted_model = None

    def train(self, train_data: pd.Series):
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
        if self.fitted_model is None:
            raise RuntimeError("The model must be trained before prediction.")
            
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast.tolist()
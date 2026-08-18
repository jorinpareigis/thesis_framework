import logging
from typing import Any
import pandas as pd
from prophet import Prophet

from .base_model import BaseForecastingModel

# Suppress heavy internal logging from Prophet and its C++ backend (cmdstanpy)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

class ProphetModel(BaseForecastingModel):
    """
    Implements Facebook's Prophet model for time-series forecasting.
    
    Treats forecasting as a curve-fitting task rather than an autoregressive task, 
    enhancing robustness against Missing Completely At Random (MCAR) data.
    """
    def __init__(self, cfg: Any) -> None:
        """
        Initializes the Prophet model and configures its hyperparameters.

        Args:
            cfg (Any): The Hydra configuration object.
        """
        model_cfg = cfg.model
        
        self.config_freq = cfg.dataset.get("frequency", "h")
        
        self.model = Prophet(
            growth=model_cfg.growth,
            seasonality_mode=model_cfg.seasonality_mode,
            changepoint_prior_scale=model_cfg.changepoint_prior_scale,
            seasonality_prior_scale=model_cfg.seasonality_prior_scale,
            daily_seasonality=model_cfg.daily_seasonality,
            weekly_seasonality=model_cfg.weekly_seasonality,
            yearly_seasonality=model_cfg.yearly_seasonality
        )
        self.freq: str | None = None

    def train(self, train_data: pd.Series) -> None:
        """
        Transforms the time-series data to meet Prophet's strict schema and fits the curve.

        Args:
            train_data (pd.Series): The historical training dataset.
        """
        df = train_data.reset_index()
        df.columns = ['ds', 'y']
        df['y'] = df['y'].astype(float)
        
        self.freq = train_data.index.inferred_freq
        if self.freq is None:
            # Fallback mechanism: Severe missing data can break pandas ability to infer the step frequency mathematically.
            logger.warning(
                f"Could not infer datetime frequency from index. "
                f"Defaulting to config frequency: '{self.config_freq}'."
            )
            self.freq = self.config_freq
            
        self.model.fit(df)

    def predict(self, steps: int) -> list[float]:
        """
        Generates future datestamps and extrapolates the fitted curve.

        Args:
            steps (int): The number of future time steps to predict.

        Returns:
            list[float]: The generated point forecasts.
            
        Raises:
            RuntimeError: If the model has not been trained prior to prediction.
        """
        if self.freq is None:
            raise RuntimeError("Model must be trained before generating predictions.")

        future = self.model.make_future_dataframe(periods=steps, freq=self.freq, include_history=False)
        forecast = self.model.predict(future)
        
        return forecast['yhat'].tolist()
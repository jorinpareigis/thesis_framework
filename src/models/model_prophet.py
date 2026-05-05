import pandas as pd
import logging
# Suppress Prophet's heavy debug logging to keep the terminal clean
import logging as py_logging
py_logging.getLogger('cmdstanpy').setLevel(py_logging.ERROR)
py_logging.getLogger('prophet').setLevel(py_logging.ERROR)

from prophet import Prophet
from .base_model import BaseForecastingModel

logger = logging.getLogger(__name__)

class ProphetModel(BaseForecastingModel):
    """
    Implements Facebook's Prophet model for time-series forecasting.
    Prophet treats forecasting as a curve-fitting task rather than a strict 
    autoregressive task, making it highly robust to missing data (MCAR).
    """
    def __init__(self, cfg):
        model_cfg = cfg.model
        
        # Initialize Prophet explicitly using the parameters defined in prophet.yaml
        self.model = Prophet(
            growth=model_cfg.growth,
            seasonality_mode=model_cfg.seasonality_mode,
            changepoint_prior_scale=model_cfg.changepoint_prior_scale,
            seasonality_prior_scale=model_cfg.seasonality_prior_scale,
            daily_seasonality=model_cfg.daily_seasonality,
            weekly_seasonality=model_cfg.weekly_seasonality,
            yearly_seasonality=model_cfg.yearly_seasonality
        )
        # Placeholder to store the dataset's temporal frequency (e.g., 'D' for days, 'h' for hours)
        self.freq = None

    def train(self, train_data: pd.Series):
        """
        Transforms the standard Series into Prophet's required DataFrame structure 
        and fits the curve.
        """
        # 1. Structural Conversion
        # Prophet strictly requires a DataFrame with 'ds' (datestamp) and 'y' (value) columns.
        df = train_data.reset_index()
        df.columns = ['ds', 'y']
        df['y'] = df['y'].astype(float)
        
        # 2. Frequency Inference
        # Prophet needs to know the time delta between steps to generate future dates.
        self.freq = train_data.index.inferred_freq
        if self.freq is None:
            # Fallback mechanism: If data corruption is so severe that pandas cannot 
            # mathematically infer the step frequency, we default to hourly.
            logger.warning("Could not infer datetime frequency from index. Defaulting to 'h'.")
            self.freq = 'h'
            
        self.model.fit(df)

    def predict(self, steps: int) -> list:
        """
        Generates future datestamps and returns the predicted curve values.
        """
        # make_future_dataframe builds an empty DataFrame with the correct future timestamps.
        # include_history=False prevents Prophet from re-predicting the entire training set.
        future = self.model.make_future_dataframe(periods=steps, freq=self.freq, include_history=False)
        
        # Predict generates a massive DataFrame with confidence intervals (yhat_lower, yhat_upper).
        forecast = self.model.predict(future)
        
        # We only extract the point forecast ('yhat') to match the framework's evaluation logic.
        return forecast['yhat'].tolist()
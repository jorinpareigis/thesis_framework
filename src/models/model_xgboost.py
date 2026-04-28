import numpy as np
import pandas as pd
import xgboost as xgb
from .base_model import BaseForecastingModel

class XGBoostModel(BaseForecastingModel):
    def __init__(self, cfg):
        model_cfg = cfg.model
        self.n_lags = model_cfg.n_lags
        
        # Initialize the underlying XGBoost regressor with YAML parameters
        self.model = xgb.XGBRegressor(
            n_estimators=model_cfg.n_estimators,
            learning_rate=model_cfg.learning_rate,
            max_depth=model_cfg.max_depth,
            objective='reg:squarederror',
            random_state=cfg.seed
        )
        self.last_known_data = None

    def _create_features(self, series: pd.Series):
        """Converts a 1D time series into a 2D feature matrix (X) and target vector (y)."""
        X, y = [], []
        values = series.values
        for i in range(len(values) - self.n_lags):
            X.append(values[i : i + self.n_lags])
            y.append(values[i + self.n_lags])
        return np.array(X), np.array(y)

    def train(self, train_data: pd.Series):
        train_data = train_data.astype(float)
        
        # Save the very last window of data. We need this to initiate the first prediction.
        self.last_known_data = train_data.iloc[-self.n_lags:].values.tolist()
        
        X_train, y_train = self._create_features(train_data)
        self.model.fit(X_train, y_train)

    def predict(self, steps: int) -> list:
        if self.last_known_data is None:
            raise RuntimeError("The model must be trained before prediction.")
            
        predictions = []
        # Create a working copy of the most recent data
        current_window = self.last_known_data.copy()
        
        # Recursive forecasting: predict 1 step, append it, slide the window, repeat.
        for _ in range(steps):
            X_pred = np.array(current_window[-self.n_lags:]).reshape(1, -self.n_lags)
            
            y_hat = self.model.predict(X_pred)[0]
            predictions.append(y_hat)
            
            # Append prediction to use as a feature for the next step
            current_window.append(y_hat)
            
        return predictions
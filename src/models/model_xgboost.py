import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Any

from .base_model import BaseForecastingModel

class XGBoostModel(BaseForecastingModel):
    """
    Implements an XGBoost regressor for time-series forecasting.
    
    Transforms 1D sequential time-series data into a 2D supervised machine learning 
    feature matrix using a sliding window approach (autoregressive lags).
    """
    def __init__(self, cfg: Any) -> None:
        """
        Initializes the XGBoost model and configures hyperparameters.

        Args:
            cfg (Any): The Hydra configuration object containing model parameters.
        """
        model_cfg = cfg.model
        self.n_lags = model_cfg.n_lags
        
        self.model = xgb.XGBRegressor(
            n_estimators=model_cfg.n_estimators,
            learning_rate=model_cfg.learning_rate,
            max_depth=model_cfg.max_depth,
            subsample=model_cfg.get("subsample", 1.0),
            colsample_bytree=model_cfg.get("colsample_bytree", 1.0),
            objective='reg:squarederror',
            random_state=cfg.current_model_seed
        )
        self.last_known_data: list[float] | None = None

    def _create_features(self, time_series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts a 1D time series into a 2D feature matrix (X) and target vector (y)
        via a sliding window of size `n_lags`.

        Args:
            time_series (pd.Series): The historical time-series data.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the feature matrix (X) 
                                           and the target vector (y).
        """
        X, y = [], []
        values = time_series.values
        for i in range(len(values) - self.n_lags):
            X.append(values[i : i + self.n_lags])
            y.append(values[i + self.n_lags])
        return np.array(X), np.array(y)

    def train(self, train_data: pd.Series) -> None:
        """
        Transforms the sequence into tabular features and trains the tree ensemble.

        Args:
            train_data (pd.Series): The historical training dataset.

        Raises:
            ValueError: If the training data contains un-imputed NaN values.
        """
        if train_data.isna().any():
            raise ValueError("Training data contains NaNs. Use an imputation method in your corruption config.")
            
        # Enforce float type to prevent data type conflicts within the XGBoost C++ backend
        train_data = train_data.astype(float)
        
        self.last_known_data = train_data.iloc[-self.n_lags:].values.tolist()
        
        X_train, y_train = self._create_features(train_data)
        self.model.fit(X_train, y_train)

    def predict(self, steps: int) -> list[float]:
        """
        Executes recursive multi-step forecasting by iteratively predicting one step 
        ahead and appending the output to the rolling window context.

        Args:
            steps (int): The number of future time steps to predict.

        Returns:
            list[float]: The generated point forecasts.

        Raises:
            RuntimeError: If the model has not been trained prior to prediction.
        """
        if self.last_known_data is None:
            raise RuntimeError("The model must be trained before prediction.")
            
        predictions = []
        rolling_context = self.last_known_data.copy()
        
        for _ in range(steps):
            X_pred = np.array(rolling_context[-self.n_lags:]).reshape(1, -self.n_lags)
            
            predicted_value = float(self.model.predict(X_pred)[0])
            predictions.append(predicted_value)
            
            # Compounding error vulnerability: Using own predictions as features for future steps
            rolling_context.append(predicted_value)
            
        return predictions
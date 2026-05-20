import numpy as np
import pandas as pd
import xgboost as xgb
from .base_model import BaseForecastingModel

class XGBoostModel(BaseForecastingModel):
    """
    Implements XGBoost for time-series forecasting.
    Unlike SARIMAX, XGBoost does not natively process sequential time-series data.
    This class transforms the 1D time-series into a 2D supervised machine learning 
    format using a sliding window of lag features.
    """
    def __init__(self, cfg):
        model_cfg = cfg.model
        # n_lags defines the size of the sliding window (how many past steps are used to predict the next step).
        self.n_lags = model_cfg.n_lags
        
        # Initialize the underlying XGBoost regressor with YAML parameters.
        self.model = xgb.XGBRegressor(
            n_estimators=model_cfg.n_estimators,
            learning_rate=model_cfg.learning_rate,
            max_depth=model_cfg.max_depth,
            subsample=model_cfg.get("subsample", 1.0),
            colsample_bytree=model_cfg.get("colsample_bytree", 1.0),
            # Set to standard regression loss.
            objective='reg:squarederror',
            # Seed propagation ensures tree building is deterministic across Monte Carlo runs.
            random_state=cfg.seed
        )
        # Stores the final data window from the training set to bootstrap the prediction phase.
        self.last_known_data = None

    def _create_features(self, series: pd.Series):
        """
        Converts a 1D time series into a 2D feature matrix (X) and target vector (y).
        Iterates through the data, extracting chunks of length `n_lags` as features, 
        and the immediately following single value as the target.
        """
        X, y = [], []
        values = series.values
        for i in range(len(values) - self.n_lags):
            X.append(values[i : i + self.n_lags])
            y.append(values[i + self.n_lags])
        return np.array(X), np.array(y)

    def train(self, train_data: pd.Series):
        # CRITICAL FIX 4: Guard against un-imputed NaNs crashing the C++ backend
        if train_data.isna().any():
            raise ValueError("Training data contains NaNs. Use an imputation method in your corruption config.")
            
        # Enforce float type to avoid data type conflicts within the XGBoost C++ backend.
        train_data = train_data.astype(float)
        
        # Save the very last window of data. We need this to initiate the first prediction 
        # because the model requires exactly `n_lags` features to execute an inference step.
        self.last_known_data = train_data.iloc[-self.n_lags:].values.tolist()
        
        # Transform the sequential data and train the tree ensemble.
        X_train, y_train = self._create_features(train_data)
        self.model.fit(X_train, y_train)

    def predict(self, steps: int) -> list:
        if self.last_known_data is None:
            raise RuntimeError("The model must be trained before prediction.")
            
        predictions = []
        # Create a working copy of the most recent data to avoid modifying the trained state.
        current_window = self.last_known_data.copy()
        
        # Recursive forecasting: predict 1 step, append it, slide the window, repeat.
        # This allows multi-step forecasting from a model inherently designed for single-step regression.
        for _ in range(steps):
            # Extract the most recent `n_lags` values and reshape to a 2D matrix (1 row, n_lags columns) 
            # as required by the XGBoost predict API.
            X_pred = np.array(current_window[-self.n_lags:]).reshape(1, -self.n_lags)
            
            y_hat = self.model.predict(X_pred)[0]
            predictions.append(y_hat)
            
            # Append prediction to the window to act as a feature for the next iteration step.
            # This causes errors to compound over long prediction horizons.
            current_window.append(y_hat)
            
        return predictions
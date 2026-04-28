import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_predictions(true_values: pd.Series, predictions: list) -> dict:
    """
    Calculates evaluation metrics comparing ground truth to model predictions.
    
    Args:
        true_values (pd.Series): The actual historical values.
        predictions (list): The predicted values from the model.
        
    Returns:
        dict: A dictionary containing the calculated RMSE and MAE.
    """
    # Ensure inputs are aligned and numeric
    y_true = np.array(true_values, dtype=float)
    y_pred = np.array(predictions, dtype=float)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return {
        "MAE": mae,
        "RMSE": rmse
    }
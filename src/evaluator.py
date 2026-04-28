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
    # Convert inputs to standard NumPy float arrays.
    # This strips away pandas indices from the true_values Series. If indices 
    # are not stripped, scikit-learn may attempt to align the predictions list 
    # with the pandas index, causing shape mismatches or NaN evaluations.
    y_true = np.array(true_values, dtype=float)
    y_pred = np.array(predictions, dtype=float)
    
    # MAE (Mean Absolute Error) provides a linear penalty. 
    # It shows the expected average deviation of the forecast in raw units.
    mae = mean_absolute_error(y_true, y_pred)
    
    # RMSE (Root Mean Squared Error) squares the errors before averaging them.
    # This disproportionately penalizes large errors. Tracking RMSE is critical 
    # for this framework because it acts as an early warning metric when a model 
    # suffers a catastrophic failure due to extreme data corruption (e.g., severe outliers).
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return {
        "MAE": mae,
        "RMSE": rmse
    }
from abc import ABC, abstractmethod
import pandas as pd

class BaseForecastingModel(ABC):
    """
    This is the Abstract Base Class (Template) for all forecasting models in the framework.
    Any new model added to the framework MUST implement these three methods.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the model and its specific parameters.
        (e.g., 'order' for SARIMAX, or 'max_depth' for XGBoost).
        """
        pass

    @abstractmethod
    def train(self, train_data: pd.Series):
        """
        Trains the model on the provided historical data.
        
        Args:
            train_data (pd.Series): A pandas Series containing the time-series data.
        """
        pass

    @abstractmethod
    def predict(self, steps: int) -> list:
        """
        Generates a forecast for the specified number of future steps.
        
        Args:
            steps (int): The number of timestamps to predict into the future.
            
        Returns:
            list: A list or array of the predicted numerical values.
        """
        pass
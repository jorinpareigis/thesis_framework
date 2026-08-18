from abc import ABC, abstractmethod
from typing import Any
import pandas as pd

class BaseForecastingModel(ABC):
    """
    Abstract Base Class defining the uniform interface for all forecasting models within the framework.
    
    Enforces a strict contract for initialization, training, and inference to guarantee 
    interoperability with the automated Monte Carlo evaluation pipeline.
    """

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the forecasting model and parses its specific hyperparameters.

        Args:
            **kwargs (Any): Model-specific configuration parameters.
        """
        pass

    @abstractmethod
    def train(self, train_data: pd.Series) -> None:
        """
        Fits the model to the provided historical time-series data.

        Args:
            train_data (pd.Series): The historical training dataset.
        """
        pass

    @abstractmethod
    def predict(self, steps: int) -> list[float]:
        """
        Generates a sequential point forecast for future time steps.

        Args:
            steps (int): The number of future time intervals to predict.

        Returns:
            list[float]: The forecasted numerical values.
        """
        pass
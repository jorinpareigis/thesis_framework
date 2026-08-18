import logging
from typing import Any
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

from .base_model import BaseForecastingModel

logger = logging.getLogger(__name__)

class ChronosModel(BaseForecastingModel):
    """
    Implements Amazon's Chronos foundation model for zero-shot time-series forecasting.
    
    Utilizes a transformer-based language model architecture to generate 
    probabilistic forecasting trajectories.
    """

    def __init__(self, cfg: Any) -> None:
        """
        Initializes the Chronos pipeline and allocates it to the optimal hardware accelerator.

        Args:
            cfg (Any): The configuration object containing model hyperparameters and random seeds.
        """
        model_cfg = cfg.model
        self.repo_id = model_cfg.repo_id
        self.num_samples = model_cfg.num_samples
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        
        self.seed = cfg.seed 
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            logger.warning("CUDA not detected. Chronos will run on CPU and be severely bottlenecked.")

        # torch.bfloat16 optimizes VRAM usage for modern NVIDIA architectures
        # without suffering the precision degradation typical of standard float16.
        self.pipeline = ChronosPipeline.from_pretrained(
            self.repo_id,
            device_map=self.device,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )
        self.context_data: torch.Tensor | None = None

    def train(self, train_data: pd.Series) -> None:
        """
        Processes historical data to act as the context sequence for zero-shot inference.
        
        Note: Foundation models do not update internal weights during this phase.

        Args:
            train_data (pd.Series): The historical time-series data.
        """
        self.context_data = torch.tensor(train_data.values, dtype=torch.float32)

    def predict(self, steps: int) -> list[float]:
        """
        Executes stochastic inference and extracts the median trajectory as the point forecast.

        Args:
            steps (int): The number of future time steps to predict.

        Returns:
            list[float]: The median point forecast.
            
        Raises:
            RuntimeError: If context data is missing (train() was not executed).
        """
        if self.context_data is None:
            raise RuntimeError("Context data must be provided via train() before prediction.")

        # Seed PyTorch RNG immediately prior to stochastic sampling to guarantee deterministic outputs
        torch.manual_seed(self.seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        forecast_tensor = self.pipeline.predict(
            self.context_data.unsqueeze(0),
            prediction_length=steps,
            num_samples=self.num_samples,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        
        forecast_samples = forecast_tensor[0].cpu().numpy()
        point_forecast = np.median(forecast_samples, axis=0)
        
        return point_forecast.tolist()
import torch
import numpy as np
import pandas as pd
import logging
from chronos import ChronosPipeline
from .base_model import BaseForecastingModel

logger = logging.getLogger(__name__)

class ChronosModel(BaseForecastingModel):
    """
    Implements Amazon's Chronos foundation model for zero-shot time-series forecasting.
    Uses a transformer-based language model architecture.
    """
    def __init__(self, cfg):
        model_cfg = cfg.model
        self.repo_id = model_cfg.repo_id
        self.num_samples = model_cfg.num_samples
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        
        # Capture the seed from Hydra for deterministic inference
        self.seed = cfg.seed 
        
        # Explicitly assign to GPU to ensure acceptable iteration speed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            logger.warning("CUDA not detected. Chronos will run on CPU and be severely bottlenecked.")

        # Initialize the pipeline. This triggers a one-time download on the first execution.
        # torch.bfloat16 optimizes VRAM usage for the RTX 3000 series without losing precision.
        self.pipeline = ChronosPipeline.from_pretrained(
            self.repo_id,
            device_map=self.device,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )
        self.context_data = None

    def train(self, train_data: pd.Series):
        """
        Zero-shot models do not train on local data. 
        This method converts the historical sequence into a PyTorch tensor to be used as context.
        """
        # Ensure data is float and convert directly to a PyTorch tensor
        self.context_data = torch.tensor(train_data.values, dtype=torch.float32)

    def predict(self, steps: int) -> list:
        """
        Executes inference. Chronos returns multiple probabilistic trajectories.
        We extract the median trajectory to serve as our definitive point forecast.
        """
        if self.context_data is None:
            raise RuntimeError("Context data must be provided via train() before prediction.")

        # CRITICAL FIX 6: Seed PyTorch RNG right before stochastic sampling
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
        
        # 1. The tensor from Chronos is stripped of its batch dimension and converted to a NumPy array.
        # This gives you an array with the shape: (20 samples, 24 future time steps)
        forecast_samples = forecast_tensor[0].cpu().numpy()
        
        # 2. THIS is the exact line where the median is calculated.
        point_forecast = np.median(forecast_samples, axis=0)
        
        # 3. The resulting NumPy array is converted back into a standard Python list to match the framework.
        return point_forecast.tolist()
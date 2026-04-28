import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import logging
import numpy as np

from src.data_loader import load_data
from src.corruptions import apply_corruption
from src.evaluator import evaluate_predictions
from src.models.model_xgboost import XGBoostModel
from src.models.model_naive import NaiveModel
from src.models.model_sarimax import SARIMAXModel

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main orchestrator for the machine learning framework.
    """
    # 1. Initialize Weights & Biases
    # OmegaConf.to_container converts the Hydra config into a standard dictionary for W&B
    wandb.init(
        project="thesis_framework",
        name=cfg.experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    logger.info(f"Starting experiment: {cfg.experiment_name}")

    # 2. Load Data
    train_clean, test_truth = load_data(cfg)
    test_size = cfg.dataset.test_size

    num_runs = cfg.get("num_runs", 10)
    corruption_steps = np.arange(
        cfg.corruption_start, 
        cfg.corruption_end + cfg.corruption_step, 
        cfg.corruption_step
    )

    # 3. Execution Setup
    num_runs = cfg.get("num_runs", 10)
    corruption_steps = np.arange(
        cfg.corruption_start, 
        cfg.corruption_end + cfg.corruption_step, 
        cfg.corruption_step
    )
    
    # Dictionary to aggregate results across multiple runs
    aggregated_metrics = {
        round(float(pct), 4): {"RMSE": [], "MAE": []} 
        for pct in corruption_steps
    }

    # 4. Outer Loop: Multiple Seeds
    for run in range(num_runs):
        # Generate a unique static seed for this specific run
        current_seed = cfg.seed + run
        cfg.seed = current_seed  # Update cfg so corruptions.py reads the new seed
        
        logger.info(f"=== Starting Run {run + 1}/{num_runs} with Seed {current_seed} ===")
        
        # 5. Inner Loop: Corruption Scaling
        for missing_pct in corruption_steps:
            missing_pct = round(float(missing_pct), 4)
            
            # A. Corrupt Data
            corrupted_train = apply_corruption(train_clean, cfg, missing_pct)
            
            # B. Instantiate and Train Model
            if cfg.model.name == "sarimax":
                model = SARIMAXModel(cfg)
            elif cfg.model.name == "xgboost":
                model = XGBoostModel(cfg)
            elif cfg.model.name == "naive":
                model = NaiveModel(cfg)
            else:
                raise NotImplementedError(f"Model {cfg.model.name} is not implemented.")
                
            model.train(corrupted_train)
            
            # C. Predict
            predictions = model.predict(steps=test_size)
            
            # D. Evaluate
            metrics = evaluate_predictions(test_truth, predictions)
            
            # E. Store metrics for averaging
            aggregated_metrics[missing_pct]["RMSE"].append(metrics["RMSE"])
            aggregated_metrics[missing_pct]["MAE"].append(metrics["MAE"])

    # 6. Average and Log to W&B
    logger.info("--- Averaging Results and Logging to W&B ---")
    for missing_pct, metrics_dict in aggregated_metrics.items():
        avg_rmse = np.mean(metrics_dict["RMSE"])
        avg_mae = np.mean(metrics_dict["MAE"])
        
        # W&B processes the final averaged curve
        wandb.log({
            "missing_pct": missing_pct,
            "RMSE": avg_rmse,
            "MAE": avg_mae
        })
        
        logger.info(f"Pct: {missing_pct}% | Avg RMSE: {avg_rmse:.2f} | Avg MAE: {avg_mae:.2f}")

    # Close the W&B run cleanly
    wandb.finish()
    logger.info("Experiment completed successfully.")

if __name__ == "__main__":
    main()
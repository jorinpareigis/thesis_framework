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

# The @hydra.main decorator intercepts the execution to parse the YAML configurations 
# in the "configs" directory before passing them to the main function as a DictConfig object.
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main orchestrator for the machine learning framework.
    Manages the data pipeline, Monte Carlo execution loops, and external logging.
    """
    # 1. Initialize Weights & Biases
    # OmegaConf.to_container converts the Hydra config into a standard Python dictionary.
    # This is required because WandB cannot natively serialize Hydra's internal object types.
    # resolve=True evaluates any interpolated variables (like ${dataset.seasonality}) before logging.
    wandb.init(
        project="thesis_framework",
        name=cfg.experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    logger.info(f"Starting experiment: {cfg.experiment_name}")

    # 2. Load Data
    # The train/test split is performed here, prior to any corruption, 
    # guaranteeing that the test set remains an uncorrupted ground truth benchmark.
    train_clean, test_truth = load_data(cfg)
    test_size = cfg.dataset.test_size

    # 3. Execution Setup
    # Define the number of Monte Carlo iterations for statistical robustness.
    num_runs = cfg.get("num_runs", 10)
    
    # Generate the sequence of corruption intensities to test (e.g., 0.0%, 0.5%, 1.0%...).
    corruption_steps = np.arange(
        cfg.corruption_start, 
        cfg.corruption_end + cfg.corruption_step, 
        cfg.corruption_step
    )
    
    # Dictionary to aggregate results across all runs for each specific corruption level.
    aggregated_metrics = {
        round(float(pct), 4): {"RMSE": [], "MAE": []} 
        for pct in corruption_steps
    }

    # 4. Outer Loop: Multiple Seeds (Monte Carlo Simulation)
    # Running the experiment multiple times with different seeds captures both the 
    # expected performance (mean) and the variance (standard deviation) of the models.
    for run in range(num_runs):
        # Generate a unique, deterministic seed for this specific run.
        # This ensures the random variations applied in corruptions.py follow a unique 
        # but reproducible path for every iteration.
        current_seed = cfg.seed + run
        cfg.seed = current_seed  # Update cfg so corruptions.py reads the new seed
        
        logger.info(f"=== Starting Run {run + 1}/{num_runs} with Seed {current_seed} ===")
        
        # 5. Inner Loop: Corruption Scaling
        # Progressively degrades the dataset to test the model's breaking point.
        for missing_pct in corruption_steps:
            # Rounding prevents Python's floating-point precision errors (e.g., 0.500000001)
            # from creating mismatched keys in the aggregated_metrics dictionary.
            missing_pct = round(float(missing_pct), 4)
            
            # A. Corrupt Data
            corrupted_train = apply_corruption(train_clean, cfg, missing_pct)
            
            # B. Instantiate and Train Model
            # Model selection routes the data to the appropriate class wrapper.
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
    # Logging occurs only after all runs complete. Aggregating the data first ensures 
    # the WandB dashboard receives clean statistical summaries rather than noisy, individual run data.
    logger.info("--- Averaging Results and Logging to W&B ---")
    for missing_pct, metrics_dict in aggregated_metrics.items():
        avg_rmse = np.mean(metrics_dict["RMSE"])
        std_rmse = np.std(metrics_dict["RMSE"])
        avg_mae = np.mean(metrics_dict["MAE"])
        std_mae = np.std(metrics_dict["MAE"])
        
        # Push the final computed statistics to the cloud tracking dashboard.
        wandb.log({
            "missing_pct": missing_pct,
            "RMSE_mean": avg_rmse,
            "RMSE_std": std_rmse,
            "MAE_mean": avg_mae,
            "MAE_std": std_mae
        })
        
        logger.info(f"Pct: {missing_pct}% | Avg RMSE: {avg_rmse:.2f} | Avg MAE: {avg_mae:.2f}")

    # Close the W&B run cleanly to free up system memory and finalize the cloud sync.
    wandb.finish()
    logger.info("Experiment completed successfully.")

if __name__ == "__main__":
    main()

# Example terminal commands for execution:
# python main.py experiment_name=run_0.0.1 dataset=energy corruption=mcar model=naive model.strategy=forward_fill
# python main.py experiment_name=run_energy_outliers_xgboost dataset=energy corruption=outliers model=xgboost
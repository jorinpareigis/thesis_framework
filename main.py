import os
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import logging
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

from src.utils.validators import validate_configuration
from src.data_loader import load_data
from src.corruptions import apply_corruption
from src.evaluator import evaluate_predictions
from src.models.model_xgboost import XGBoostModel
from src.models.model_naive import NaiveModel
from src.models.model_sarimax import SARIMAXModel
from src.models.model_prophet import ProphetModel
from src.models.model_chronos import ChronosModel
from src.models.model_lstm import LSTMModel

logger = logging.getLogger(__name__)

# The @hydra.main decorator intercepts the execution to parse the YAML configurations 
# in the "configs" directory before passing them to the main function as a DictConfig object.
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main orchestrator for the machine learning framework.
    Manages the data pipeline, Monte Carlo execution loops, and external logging.
    """
    # 0. Fail-Fast Validation
    # Checks if the dataset + corruption pairing makes scientific sense
    validate_configuration(cfg)

    # 1. Initialize Weights & Biases
    # OmegaConf.to_container converts the Hydra config into a standard Python dictionary.
    # This is required because WandB cannot natively serialize Hydra's internal object types.
    # resolve=True evaluates any interpolated variables (like ${dataset.seasonality}) before logging.
    group_name = cfg.get("group_name", f"{cfg.dataset.name}_{cfg.corruption.type}")
    
    wandb.init(
        project="thesis_framework",
        name=cfg.experiment_name,
        group=group_name,  # Dynamically bundles tasks inside the requested dashboard folder
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    logger.info(f"Starting experiment: {cfg.experiment_name}")

    # 2. Load Data
    # The train/test split is performed here, prior to any corruption, 
    # guaranteeing that the test set remains an uncorrupted ground truth benchmark.
    # removed data load from here inside the monte carlo loop

    # --- 3. Execution Setup ---
    num_runs = cfg.get("num_runs", 50)
    batch_mode = cfg.get("batch_mode", False)
    
    # Load the predefined 0-100% list from config.yaml
    corruption_steps = cfg.get("corruption_steps", [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5, 100.0])
    
    aggregated_metrics = {
        float(level): {"RMSE": [], "MAE": []} 
        for level in corruption_steps
    }

    milestones = {25: False, 50: False, 75: False}

    # 4. Outer Loop: Multiple Seeds (Monte Carlo Simulation)
    # Running the experiment multiple times with different seeds captures both the 
    # expected performance (mean) and the variance (standard deviation) of the models.
    # 4. Outer Loop: Multiple Seeds (Monte Carlo Simulation)
    for run in range(num_runs):
        # Allow dynamic addition/modification of config keys during execution
        OmegaConf.set_struct(cfg, False)
        
        # Calculate isolated seeds. If fixed, use base seed. If varying, increment by run index.
        cfg.current_data_seed = cfg.seed if cfg.fix_dataset_seed else cfg.seed + run
        cfg.current_corr_seed = cfg.seed if cfg.fix_corruption_seed else cfg.seed + run
        cfg.current_model_seed = cfg.seed if cfg.fix_model_seed else cfg.seed + run
        
        # Re-lock the config to prevent accidental mutations elsewhere
        OmegaConf.set_struct(cfg, True)

        # --- Milestone Reporting ---
        progress_pct = (run / num_runs) * 100
        if progress_pct >= 25 and not milestones[25]:
            logger.info("BATCH_PROGRESS_25")
            milestones[25] = True
        elif progress_pct >= 50 and not milestones[50]:
            logger.info("BATCH_PROGRESS_50")
            milestones[50] = True
        elif progress_pct >= 75 and not milestones[75]:
            logger.info("BATCH_PROGRESS_75")
            milestones[75] = True
        
        if not batch_mode:
            logger.info(f"=== Starting Run {run + 1}/{num_runs} ===")
            logger.info(f"Active Seeds -> Data: {cfg.current_data_seed} | Corruption: {cfg.current_corr_seed} | Model: {cfg.current_model_seed}")

        # MOVED HERE: Load a new randomized data slice for this specific run
        train_clean, test_truth = load_data(cfg)
        test_size = cfg.dataset.test_size
        
        # --- 5. Inner Loop: Corruption Scaling ---
        for corruption_level in tqdm(corruption_steps, desc=f"Run {run + 1} Progress", leave=True, disable=batch_mode):
            corruption_level = float(corruption_level)
            
            # A. Corrupt Data
            corrupted_train = apply_corruption(train_clean, cfg, corruption_level)
            
            # B. Instantiate and Train Model
            # Model selection routes the data to the appropriate class wrapper.
            if cfg.model.name == "sarimax":
                model = SARIMAXModel(cfg)
            elif cfg.model.name == "xgboost":
                model = XGBoostModel(cfg)
            elif cfg.model.name == "naive":
                model = NaiveModel(cfg)
            elif cfg.model.name == "prophet":
                model = ProphetModel(cfg)
            elif cfg.model.name == "chronos":
                model = ChronosModel(cfg)
            elif cfg.model.name == "lstm":
                model = LSTMModel(cfg)
            else:
                raise NotImplementedError(f"Model {cfg.model.name} is not implemented.")
                
            model.train(corrupted_train)
            
            # C. Predict
            predictions = model.predict(steps=test_size)
            
            # D. Evaluate
            metrics = evaluate_predictions(test_truth, predictions)
            
            # E. Store metrics for averaging
            aggregated_metrics[corruption_level]["RMSE"].append(metrics["RMSE"])
            aggregated_metrics[corruption_level]["MAE"].append(metrics["MAE"])

# --- 6. Average and Log to W&B ---
    logger.info("--- Averaging Results and Logging to W&B ---")
    for corruption_level, metrics_dict in aggregated_metrics.items():
        avg_rmse = np.mean(metrics_dict["RMSE"])
        std_rmse = np.std(metrics_dict["RMSE"])
        med_rmse = np.median(metrics_dict["RMSE"])
        
        avg_mae = np.mean(metrics_dict["MAE"])
        std_mae = np.std(metrics_dict["MAE"])
        med_mae = np.median(metrics_dict["MAE"])
        
        wandb.log({
            "corruption_level": corruption_level,
            "RMSE_mean": avg_rmse,
            "RMSE_std": std_rmse,
            "RMSE_median": med_rmse,
            "MAE_mean": avg_mae,
            "MAE_std": std_mae,
            "MAE_median": med_mae
        })
        
        logger.info(f"Level: {corruption_level}% | Avg RMSE: {avg_rmse:.2f} | Med RMSE: {med_rmse:.2f} | Avg MAE: {avg_mae:.2f}")

    # --- Generate Interactive Plotly Degradation Curve ---
    logger.info("Generating Plotly degradation chart with Spaghetti Plot overlay...")
    
    # Ensure levels are strictly sorted for accurate trapezoidal integration
    levels = sorted(list(aggregated_metrics.keys()))
    rmse_means = [np.mean(aggregated_metrics[l]["RMSE"]) for l in levels]
    rmse_medians = [np.median(aggregated_metrics[l]["RMSE"]) for l in levels]
    rmse_maxes = [np.max(aggregated_metrics[l]["RMSE"]) for l in levels]
    rmse_mins = [np.min(aggregated_metrics[l]["RMSE"]) for l in levels]

    # --- 7. Calculate and Log Summary Table Metrics ---
    # Normalize x-axis to 0.0 - 1.0 scale
    normalized_levels = [l / 100.0 for l in levels]
    
    # Calculate Area Under the Degradation Curve
    audc = np.trapz(y=rmse_medians, x=normalized_levels)
    
    # Extract Baseline (0% corruption)
    baseline_rmse = rmse_medians[0]
    
    # Calculate Degradation Factor
    degradation_factor = audc / baseline_rmse if baseline_rmse != 0 else float('inf')

    # Log to W&B Summary (Creates dedicated columns in the project table)
    wandb.run.summary["Table_Baseline_RMSE"] = baseline_rmse
    wandb.run.summary["Table_Degradation_AUDC"] = audc
    wandb.run.summary["Table_Degradation_Factor"] = degradation_factor

    fig = go.Figure()
    
    # ... (Keep the rest of your fig.add_trace plotting code exactly as it is) ...

    # 1. Add Spaghetti Plot: Plot every single Monte Carlo run individual line
    num_simulations = len(list(aggregated_metrics.values())[0]["RMSE"])
    
    for run_idx in range(num_simulations):
        run_y_values = [m["RMSE"][run_idx] for m in aggregated_metrics.values()]
        show_legend_flag = True if run_idx == 0 else False
        
        fig.add_trace(go.Scatter(
            x=levels,
            y=run_y_values,
            mode='lines',
            line=dict(color='rgba(0, 150, 130, 0.22)', width=1), 
            name='Individual MC Runs',
            legendgroup='spaghetti',  
            showlegend=show_legend_flag,
            hoverinfo="skip" 
        ))

    # 2. Add Envelope Boundary Band (Shaded Area between Min and Max observed performance)
    fig.add_trace(go.Scatter(
        x=levels + levels[::-1],
        y=rmse_maxes + rmse_mins[::-1],  
        fill='toself',
        fillcolor='rgba(0,100,80,0.05)', 
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='Observed Range (Min/Max)',
        showlegend=True 
    ))

    # 3. Add Main Mean Line with Fixed Markers (Circles)
    fig.add_trace(go.Scatter(
        x=levels,
        y=rmse_means,
        mode='lines+markers',  
        marker=dict(size=7, symbol='circle', color='rgb(0,100,80)'),
        line=dict(width=2.5, color='rgb(0,100,80)'),
        name='RMSE Mean',
    ))

    # 4. Add Median Line with Fixed Markers (Squares)
    fig.add_trace(go.Scatter(
        x=levels,
        y=rmse_medians,
        mode='lines+markers',  
        marker=dict(size=7, symbol='square', color='rgb(200,80,0)'), # Distinct contrasting color
        line=dict(width=2.5, color='rgb(200,80,0)'),
        name='RMSE Median',
    ))

    fig.update_layout(
        title=f"RMSE: {cfg.model.name.upper()} on {cfg.dataset.name.upper()} with {cfg.corruption.type.upper()}", 
        xaxis_title="Corruption Level (%)", 
        yaxis_title="RMSE",
        yaxis=dict(rangemode="nonnegative"), 
        template="plotly_white",
        hovermode="x unified"
    )
    
    # Log the interactive chart as an HTML object to a panel in W&B
    wandb.log({"RMSE_Detailed_Degradation_Chart": wandb.Html(fig.to_html(full_html=False, include_plotlyjs='cdn'))})

    # --- Export High-Resolution PDF for LaTeX ---
    logger.info("Exporting high-resolution PDF via Kaleido...")
    
    # Create dynamic names based on the Hydra config
    folder_name = f"{cfg.dataset.name}_{cfg.corruption.type}"
    file_name = f"{cfg.dataset.name}_{cfg.corruption.type}_{cfg.model.name}{cfg.run_suffix}.pdf"
    
    # Safely build the directory path: images/[dataset]_[corruption]/
    save_dir = os.path.join("images", folder_name)
    os.makedirs(save_dir, exist_ok=True) 
    
    # Build final file path and write
    save_path = os.path.join(save_dir, file_name)
    fig.write_image(save_path, engine="kaleido", width=1400, height=600)
    
    logger.info(f"PDF successfully saved to: {save_path}")

    # Close the W&B run cleanly
    wandb.finish()
    logger.info("Experiment completed successfully.")

if __name__ == "__main__":
    main()

# Example terminal commands for execution:
# python main.py dataset=air_quality corruption=outliers model=naive model.strategy=forward_fill run_suffix="_median_test"
# python main.py dataset=iot_temp corruption=gaussian_noise model=xgboost run_suffix="_0.0.1"
# python main.py dataset=sp500 corruption=outliers model=xgboost run_suffix="_test"
# python -m scripts.visualize_corruptions
# python -m scripts.visualize_grid
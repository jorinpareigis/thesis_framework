"""
Main orchestrator module for the time-series forecasting robustness framework.
Handles configuration parsing, Monte Carlo simulations, metric aggregation, and visualization.
"""

import os
import logging
import numpy as np
import plotly.graph_objects as go
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.utils.validators import validate_configuration
from src.data_loader import load_data
from src.corruptions import apply_corruption
from src.evaluator import evaluate_predictions
from src.models.model_xgboost import XGBoostModel
from src.models.model_naive import NaiveModel
from src.models.model_sarima import SARIMAModel
from src.models.model_prophet import ProphetModel
from src.models.model_chronos import ChronosModel
from src.models.model_lstm import LSTMModel

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main orchestrator for the machine learning framework.
    
    Args:
        cfg (DictConfig): The parsed configuration object provided by Hydra.
        
    Raises:
        NotImplementedError: If the requested model is not supported.
    """
    validate_configuration(cfg)

    group_name = cfg.get("group_name", f"{cfg.dataset.name}_{cfg.corruption.type}")
    
    # OmegaConf.to_container is required to convert Hydra's internal object types 
    # into a standard dictionary, as WandB cannot serialize DictConfig natively.
    wandb.init(
        project="thesis_framework",
        name=cfg.experiment_name,
        group=group_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    logger.info(f"Starting experiment: {cfg.experiment_name}")

    num_runs = cfg.get("num_runs", 50)
    batch_mode = cfg.get("batch_mode", False)
    
    corruption_steps = cfg.get(
        "corruption_steps", 
        [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5, 100.0]
    )
    
    aggregated_metrics = {
        float(level): {"RMSE": [], "MAE": []} 
        for level in corruption_steps
    }

    logged_milestones = {25: False, 50: False, 75: False}

    # Running multiple seeds isolates expected performance (mean) from variance (std deviation).
    for run in range(num_runs):
        OmegaConf.set_struct(cfg, False)
        
        cfg.current_data_seed = cfg.seed if cfg.fix_dataset_seed else cfg.seed + run
        cfg.current_corr_seed = cfg.seed if cfg.fix_corruption_seed else cfg.seed + run
        cfg.current_model_seed = cfg.seed if cfg.fix_model_seed else cfg.seed + run
        
        OmegaConf.set_struct(cfg, True)

        progress_percentage = (run / num_runs) * 100
        for milestone in [25, 50, 75]:
            if progress_percentage >= milestone and not logged_milestones[milestone]:
                logger.info(f"BATCH_PROGRESS_{milestone}")
                logged_milestones[milestone] = True
        
        if not batch_mode:
            logger.info(f"=== Starting Run {run + 1}/{num_runs} ===")
            logger.info(
                f"Active Seeds -> Data: {cfg.current_data_seed} | "
                f"Corruption: {cfg.current_corr_seed} | "
                f"Model: {cfg.current_model_seed}"
            )

        train_clean, test_truth = load_data(cfg)
        test_size = cfg.dataset.test_size
        
        for corruption_level in tqdm(
            corruption_steps, 
            desc=f"Run {run + 1} Progress", 
            leave=True, 
            disable=batch_mode
        ):
            corruption_level = float(corruption_level)
            corrupted_train = apply_corruption(train_clean, cfg, corruption_level)
            
            if cfg.model.name == "sarima":
                model = SARIMAModel(cfg)
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
            predictions = model.predict(steps=test_size)
            metrics = evaluate_predictions(test_truth, predictions)
            
            aggregated_metrics[corruption_level]["RMSE"].append(metrics["RMSE"])
            aggregated_metrics[corruption_level]["MAE"].append(metrics["MAE"])

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
        
        logger.info(
            f"Level: {corruption_level}% | Avg RMSE: {avg_rmse:.2f} | "
            f"Med RMSE: {med_rmse:.2f} | Avg MAE: {avg_mae:.2f}"
        )

    logger.info("Generating Plotly degradation chart with Spaghetti Plot overlay...")
    
    levels = sorted(list(aggregated_metrics.keys()))
    rmse_means = [np.mean(aggregated_metrics[l]["RMSE"]) for l in levels]
    rmse_medians = [np.median(aggregated_metrics[l]["RMSE"]) for l in levels]
    rmse_maxes = [np.max(aggregated_metrics[l]["RMSE"]) for l in levels]
    rmse_mins = [np.min(aggregated_metrics[l]["RMSE"]) for l in levels]

    normalized_levels = [l / 100.0 for l in levels]
    audc = np.trapz(y=rmse_medians, x=normalized_levels)
    baseline_rmse = rmse_medians[0]
    degradation_factor = audc / baseline_rmse if baseline_rmse != 0 else float('inf')

    wandb.run.summary["Table_Baseline_RMSE"] = baseline_rmse
    wandb.run.summary["Table_Degradation_AUDC"] = audc
    wandb.run.summary["Table_Degradation_Factor"] = degradation_factor

    fig = go.Figure()
    
    num_simulations = len(list(aggregated_metrics.values())[0]["RMSE"])
    for run_idx in range(num_simulations):
        run_rmse_scores = [m["RMSE"][run_idx] for m in aggregated_metrics.values()]
        show_legend_flag = True if run_idx == 0 else False
        
        fig.add_trace(go.Scatter(
            x=levels,
            y=run_rmse_scores,
            mode='lines',
            line=dict(color='rgba(0, 150, 130, 0.22)', width=1), 
            name='Individual Runs',
            legendgroup='spaghetti',  
            showlegend=show_legend_flag,
            hoverinfo="skip" 
        ))

    fig.add_trace(go.Scatter(
        x=levels + levels[::-1],
        y=rmse_maxes + rmse_mins[::-1],  
        fill='toself',
        fillcolor='rgba(0,100,80,0.05)', 
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='Full Range',
        showlegend=True 
    ))

    fig.add_trace(go.Scatter(
        x=levels,
        y=rmse_means,
        mode='lines+markers',  
        marker=dict(size=7, symbol='circle', color='rgb(0,100,80)'),
        line=dict(width=2.5, color='rgb(0,100,80)'),
        name='RMSE Mean',
    ))

    fig.add_trace(go.Scatter(
        x=levels,
        y=rmse_medians,
        mode='lines+markers',  
        marker=dict(size=7, symbol='square', color='rgb(200,80,0)'),
        line=dict(width=2.5, color='rgb(200,80,0)'),
        name='RMSE Median',
    ))

    fig.update_layout(
        title=dict(
            text=f"RMSE: {cfg.model.name.upper()} on {cfg.dataset.name.upper()} with {cfg.corruption.type.upper()}",
            font=dict(size=25)
        ),
        xaxis=dict(
            title="Corruption Level (%)",
            title_font=dict(size=23),
            tickfont=dict(size=19)
        ),
        yaxis=dict(
            title="RMSE",
            title_font=dict(size=23),
            tickfont=dict(size=19),
            rangemode="nonnegative"
        ),
        template="plotly_white",
        hovermode="x unified",
        margin=dict(t=80, b=120),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=20)
        )
    )
    
    wandb.log({"RMSE_Detailed_Degradation_Chart": wandb.Html(fig.to_html(full_html=False, include_plotlyjs='cdn'))})

    logger.info("Exporting high-resolution PDF via Kaleido...")
    folder_name = f"{cfg.dataset.name}_{cfg.corruption.type}"
    file_name = f"{cfg.dataset.name}_{cfg.corruption.type}_{cfg.model.name}{cfg.run_suffix}.pdf"
    
    save_dir = os.path.join("images", folder_name)
    os.makedirs(save_dir, exist_ok=True) 
    
    save_path = os.path.join(save_dir, file_name)
    fig.write_image(save_path, engine="kaleido", width=1400, height=600)
    
    logger.info(f"PDF successfully saved to: {save_path}")

    wandb.finish()
    logger.info("Experiment completed successfully.")

if __name__ == "__main__":
    main()

# Example terminal commands for execution:
# python main.py dataset=air_quality corruption=outliers model=naive model.strategy=forward_fill run_suffix="_median_test"
# python main.py dataset=iot_temp corruption=gaussian_noise model=xgboost run_suffix="_0.0.1"
# python main.py dataset=sp500 corruption=outliers model=xgboost run_suffix="_test"
# python main.py dataset=sp500 corruption=mcar model=sarima run_suffix="_need_viz"
# python main.py dataset=energy corruption=outliers model=sarima run_suffix="_need_viz"
# python main.py dataset=energy corruption=gaussian_noise model=xgboost run_suffix="_framework_test"
# python -m scripts.visualize_corruptions
# python -m scripts.visualize_grid
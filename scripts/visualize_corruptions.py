import os
import logging
import wandb
import plotly.graph_objects as go
from omegaconf import DictConfig
from hydra import initialize, compose

from src.utils.validators import validate_configuration
from src.data_loader import load_data
from src.corruptions import apply_corruption

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def generate_visualizations(cfg: DictConfig):
    """
    Core logic to generate and export visualizations for a single configuration.
    """
    validate_configuration(cfg)
    
    wandb.init(
        project="thesis_framework",
        name=f"viz_{cfg.dataset.name}_{cfg.corruption.type}",
        job_type="visualization",
        reinit=True # Ensures W&B allows consecutive runs in the same script
    )
    
    train_clean, _ = load_data(cfg)
    
    corruption_type = cfg.corruption.type
    original_method = cfg.corruption.method
    steps_to_plot = [0.0, 25.0, 50.0, 75.0, 100.0]
    
    for corruption_level in steps_to_plot:
        fig = go.Figure()
        
        # 1. Plot the Clean Baseline
        fig.add_trace(go.Scatter(
            x=train_clean.index, 
            y=train_clean.values, 
            mode='lines', 
            name='Clean Data',
            line=dict(color='rgba(100, 100, 100, 0.5)', width=1)
        ))
        
        # 2. Generate the Corrupted Data
        if corruption_type in ["mcar", "sensor_outage"]:
            cfg.corruption.method = "none"
            corrupted = apply_corruption(train_clean, cfg, corruption_level)
            cfg.corruption.method = original_method 
            
            missing_mask = corrupted.isna()
            affected_points = train_clean[missing_mask]
            
            if not affected_points.empty:
                fig.add_trace(go.Scatter(
                    x=affected_points.index, 
                    y=affected_points.values, 
                    mode='markers', 
                    name=f'Dropped Data ({corruption_level}%)',
                    marker=dict(color='red', size=6, symbol='x')
                ))
                
        elif corruption_type == "outliers":
            corrupted = apply_corruption(train_clean, cfg, corruption_level)
            affected_mask = (corrupted - train_clean).abs() > 1e-6
            affected_points = corrupted[affected_mask]
            
            if not affected_points.empty:
                fig.add_trace(go.Scatter(
                    x=affected_points.index, 
                    y=affected_points.values, 
                    mode='markers', 
                    name=f'Outliers ({corruption_level}%)',
                    marker=dict(color='red', size=5)
                ))
                
        else:
            corrupted = apply_corruption(train_clean, cfg, corruption_level)
            affected_mask = (corrupted - train_clean).abs() > 1e-6
            
            if affected_mask.any():
                fig.add_trace(go.Scatter(
                        x=corrupted.index, 
                        y=corrupted.values, 
                        mode='lines', 
                        name=f'Corrupted Data ({corruption_level}%)',
                        line=dict(color='rgba(255, 0, 0, 0.7)', width=1)
                ))
        
        fig.update_layout(
            title=f"Dataset: {cfg.dataset.name} | Corruption: {cfg.corruption.type} | Intensity: {corruption_level}%",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_white",
            hovermode="x unified"
        )
        
        wandb.log({f"Step_{corruption_level}": wandb.Html(fig.to_html(full_html=False, include_plotlyjs='cdn'))})
        
        # --- Export High-Resolution PDF for LaTeX ---
        save_dir = os.path.join("images", "viz", cfg.corruption.type)
        os.makedirs(save_dir, exist_ok=True) 
        
        file_name = f"{cfg.dataset.name}_{cfg.corruption.type}_{int(corruption_level)}.pdf"
        save_path = os.path.join(save_dir, file_name)
        
        fig.write_image(save_path, engine="kaleido", width=1400, height=600)
        logger.info(f"Visualizer PDF saved to: {save_path}")
        
    wandb.finish()

def main():
    """
    Iterates through all defined logical dataset and corruption combinations,
    initializing Hydra dynamically for each run.
    """
    valid_combinations = {
        "sp500": ["mcar", "outliers"],
        "energy": ["mcar", "outliers", "sensor_outage", "gaussian_noise"],
        "iot_temp": ["mcar", "outliers", "sensor_outage", "sensor_drift", "gaussian_noise"],
        "air_quality": ["mcar", "outliers", "sensor_outage", "sensor_drift", "gaussian_noise"]
    }

    # Initialize Hydra once pointing to the configs directory
    with initialize(version_base=None, config_path="../configs"):
        for dataset, corruptions in valid_combinations.items():
            for corruption in corruptions:
                logger.info(f"=== Starting Visualization Batch: {dataset.upper()} + {corruption.upper()} ===")
                
                # Compose the nested YAML configuration dynamically
                cfg = compose(config_name="config", overrides=[
                    f"dataset={dataset}", 
                    f"corruption={corruption}"
                ])
                
                generate_visualizations(cfg)

if __name__ == "__main__":
    main()
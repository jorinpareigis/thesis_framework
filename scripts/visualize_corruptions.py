import os
import hydra
from omegaconf import DictConfig
import wandb
import plotly.graph_objects as go
import logging

from src.utils.validators import validate_configuration
from src.data_loader import load_data
from src.corruptions import apply_corruption

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Generates visualizations of data corruptions at specific intensity steps 
    and logs them as interactive Plotly charts to Weights & Biases.
    """
    validate_configuration(cfg)
    
    wandb.init(
        project="thesis_framework",
        name=f"viz_{cfg.dataset.name}_{cfg.corruption.type}",
        job_type="visualization"
    )
    
    train_clean, _ = load_data(cfg)
    
    corruption_type = cfg.corruption.type
    original_method = cfg.corruption.method
    
    # Updated to reflect the new 0-100% scale
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
            # Disable imputation temporarily to expose exact NaN locations
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
            
            # Mask to find EXACTLY which points were shifted
            # Using > 1e-6 handles microscopic floating-point rounding errors
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
            # For gaussian_noise and sensor_drift
            corrupted = apply_corruption(train_clean, cfg, corruption_level)
            
            # Only plot the red line if there is an actual difference (bypasses the 0% issue)
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
        
        # Log via HTML wrapper to force manual dashboard rendering if needed
        wandb.log({f"Step_{corruption_level}": wandb.Html(fig.to_html(full_html=False, include_plotlyjs='cdn'))})
        logger.info(f"Logged chart for step {corruption_level} to W&B.")
        
        # --- Export High-Resolution PDF for LaTeX ---
        # Safely build the directory path: images/viz/[corruption_type]/
        save_dir = os.path.join("images", "viz", cfg.corruption.type)
        os.makedirs(save_dir, exist_ok=True) 
        
        # Build final file path (Appending the intensity step so files don't overwrite)
        file_name = f"{cfg.dataset.name}_{cfg.corruption.type}_{int(corruption_level)}.pdf"
        save_path = os.path.join(save_dir, file_name)
        
        # Export with the same wide 2:1 aspect ratio used in main.py
        fig.write_image(save_path, engine="kaleido", width=1400, height=600)
        logger.info(f"Visualizer PDF saved to: {save_path}")
        
    wandb.finish()

if __name__ == "__main__":
    main()
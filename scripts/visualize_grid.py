import os
import logging
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from omegaconf import DictConfig
from hydra import initialize, compose

from src.utils.validators import validate_configuration
from src.data_loader import load_data
from src.corruptions import apply_corruption

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Define our 5 specific target experiments
    targets = [
        ("energy", "gaussian_noise", "Gaussian Noise"),
        ("energy", "outliers", "Outliers"),
        ("sp500", "mcar", "MCAR"),
        ("iot_temp", "sensor_outage", "Sensor Outage"),
        ("air_quality", "sensor_drift", "Sensor Drift")
    ]
    
    intensity = 50.0
    
    # Create a 3x2 grid. The bottom right (row 3, col 2) will be left empty for the legend.
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[f"{ds.upper()} | {name} (50%)" for ds, _, name in targets] + [""],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )

    # Keep track of which legend items we've already added so we don't get duplicates
    added_legends = set()

    with initialize(version_base=None, config_path="../configs"):
        for idx, (dataset, corruption, title_name) in enumerate(targets):
            logger.info(f"Processing: {dataset.upper()} + {corruption.upper()}")
            
            # Calculate grid position (1-indexed for Plotly)
            row = (idx // 2) + 1
            col = (idx % 2) + 1
            
            cfg = compose(config_name="config", overrides=[
                f"dataset={dataset}", 
                f"corruption={corruption}"
            ])
            validate_configuration(cfg)
            
            train_clean, _ = load_data(cfg)
            
            # --- 1. Plot Clean Data ---
            show_clean_legend = "Clean Data" not in added_legends
            fig.add_trace(go.Scatter(
                x=train_clean.index, 
                y=train_clean.values, 
                mode='lines', 
                name='Clean Data',
                line=dict(color='rgba(100, 100, 100, 0.4)', width=1.5),
                showlegend=show_clean_legend
            ), row=row, col=col)
            if show_clean_legend: added_legends.add("Clean Data")

            # --- 2. Generate and Plot Corrupted Data ---
            original_method = cfg.corruption.method
            
            if corruption in ["mcar", "sensor_outage"]:
                # Force 'none' to get the raw NaNs for visualization
                cfg.corruption.method = "none"
                corrupted = apply_corruption(train_clean, cfg, intensity)
                cfg.corruption.method = original_method 
                
                missing_mask = corrupted.isna()
                
                if corruption == "mcar":
                    affected_points = train_clean[missing_mask]
                    fig.add_trace(go.Scatter(
                        x=affected_points.index, 
                        y=affected_points.values, 
                        mode='markers', 
                        name=title_name,
                        marker=dict(color='red', size=3, symbol='x'),
                        showlegend=True
                    ), row=row, col=col)
                    
                elif corruption == "sensor_outage":
                    # To draw solid lines ONLY where data dropped, we replace non-missing data with NaNs.
                    # Plotly naturally breaks lines at NaNs, creating distinct red blocks.
                    outage_lines = train_clean.copy()
                    outage_lines[~missing_mask] = np.nan
                    fig.add_trace(go.Scatter(
                        x=outage_lines.index, 
                        y=outage_lines.values, 
                        mode='lines', 
                        name=title_name,
                        line=dict(color='red', width=2),
                        showlegend=True
                    ), row=row, col=col)

            elif corruption == "outliers":
                corrupted = apply_corruption(train_clean, cfg, intensity)
                affected_mask = (corrupted - train_clean).abs() > 1e-6
                affected_points = corrupted[affected_mask]
                
                fig.add_trace(go.Scatter(
                    x=affected_points.index, 
                    y=affected_points.values, 
                    mode='markers', 
                    name=title_name,
                    marker=dict(color='red', size=4),
                    showlegend=True
                ), row=row, col=col)
                
            elif corruption == "gaussian_noise":
                corrupted = apply_corruption(train_clean, cfg, intensity)
                fig.add_trace(go.Scatter(
                    x=corrupted.index, 
                    y=corrupted.values, 
                    mode='lines', 
                    name=title_name,
                    # Dotted line to clearly show the noise overlaying the clean baseline
                    line=dict(color='rgba(255, 0, 0, 0.7)', width=0.75, dash='dot'),
                    showlegend=True
                ), row=row, col=col)
                
            elif corruption == "sensor_drift":
                corrupted = apply_corruption(train_clean, cfg, intensity)
                affected_mask = (corrupted - train_clean).abs() > 1e-6
                
                # Find the exact timestamp where the drift begins
                if affected_mask.any():
                    first_idx = affected_mask.idxmax()
                    drift_data = corrupted.loc[first_idx:]
                    
                    fig.add_trace(go.Scatter(
                        x=drift_data.index, 
                        y=drift_data.values, 
                        mode='lines', 
                        name=title_name,
                        # Plot only from the onset index forward
                        line=dict(color='rgba(255, 0, 0, 0.8)', width=2),
                        showlegend=True
                    ), row=row, col=col)

    # Apply global layout constraints
    fig.update_layout(
        title="Visual Representation of Framework Corruptions at 50% Intensity",
        template="plotly_white",
        height=900,  # Taller aspect ratio to fit the 3x2 grid comfortably
        width=1400,
        # Force the legend into the empty bottom-right slot
        legend=dict(
            x=0.75, 
            y=0.15, 
            xanchor="center", 
            yanchor="middle",
            bordercolor="Black",
            borderwidth=1
        )
    )
    
    # Hide axis titles for interior plots to keep it clean, but keep ticks
    fig.update_xaxes(showline=True, linewidth=1, linecolor='lightgray')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgray')

    # Export High-Resolution PDF
    save_dir = os.path.join("images", "final_corr_viz")
    os.makedirs(save_dir, exist_ok=True) 
    
    save_path = os.path.join(save_dir, "corruption_grid_50pct.pdf")
    fig.write_image(save_path, engine="kaleido")
    
    logger.info(f"Success! Visualizer PDF grid saved to: {save_path}")

if __name__ == "__main__":
    main()
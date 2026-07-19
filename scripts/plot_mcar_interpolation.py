import os
import pandas as pd
import plotly.graph_objects as go
from hydra import initialize, compose

from src.data_loader import load_data
from src.corruptions import apply_corruption

def generate_mcar_smoothing_plot():
    """
    Generates visualizations demonstrating the artificial smoothing effect 
    of linear interpolation under different MCAR corruption intensities.
    """
    ZOOM_STEPS = 100 
    
    with initialize(version_base=None, config_path="../configs"):
        for intensity in [50.0, 100.0]:
            cfg = compose(config_name="config", overrides=[
                "dataset=sp500", "corruption=mcar", "model=naive" 
            ])
            
            cfg.current_data_seed = cfg.seed
            cfg.current_corr_seed = cfg.seed
            
            train_clean, test_truth = load_data(cfg)
            full_clean_data = pd.concat([train_clean, test_truth])
            
            corrupted_data = apply_corruption(full_clean_data, cfg, intensity)
            interpolated_data = corrupted_data.interpolate(method='linear')
            
            viz_clean = full_clean_data.iloc[-ZOOM_STEPS:]
            viz_corrupted = corrupted_data.iloc[-ZOOM_STEPS:]
            viz_interpolated = interpolated_data.iloc[-ZOOM_STEPS:]
            
            fig = go.Figure()
            
            # A. Clean Ground Truth (Background)
            fig.add_trace(go.Scatter(
                x=viz_clean.index.to_pydatetime(),
                y=viz_clean.values,
                mode='lines',
                name='Raw S&P 500 Data',
                line=dict(color='rgba(150, 150, 150, 0.7)', width=4) 
            ))
            
            # B. Interpolated Signal (Clean dotted line)
            fig.add_trace(go.Scatter(
                x=viz_interpolated.index.to_pydatetime(),
                y=viz_interpolated.values,
                mode='lines',
                name='Interpolated Signal',
                line=dict(color='#D62728', width=3, dash='dot') # Changed to 'dot'
            ))
            
            fig.update_layout(
                title=dict(
                    text=f"Artificial Smoothing on S&P 500 (MCAR {int(intensity)}%)",
                    font=dict(size=24)
                ),
                xaxis=dict(
                    title="Time",
                    title_font=dict(size=22),
                    tickfont=dict(size=18)
                ),
                yaxis=dict(
                    title="S&P 500 Value",
                    title_font=dict(size=22),
                    tickfont=dict(size=18)
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
                    font=dict(size=18)
                )
            )
            
            save_dir = os.path.join("images", "viz", "thesis_plots")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"sp500_mcar_{int(intensity)}_interpolation.pdf")
            
            fig.write_image(save_path, engine="kaleido", width=1400, height=600)
            print(f"Visualization saved to: {save_path}")

if __name__ == "__main__":
    generate_mcar_smoothing_plot()
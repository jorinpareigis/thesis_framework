import os
import glob
import pandas as pd
import plotly.graph_objects as go
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Fixed allocation for colors and line styles (Symbols removed, forced to circle below)
MODEL_STYLES = {
    "naive":   {"name": "Naive Baseline", "color": "#7F7F7F", "dash": "dash"},
    "sarimax": {"name": "SARIMAX",        "color": "#1F77B4", "dash": "solid"},
    "prophet": {"name": "PROPHET",        "color": "#2CA02C", "dash": "solid"},
    "xgboost": {"name": "XGBOOST",        "color": "#D62728", "dash": "solid"},
    "lstm":    {"name": "LSTM",           "color": "#9467BD", "dash": "solid"},
    "chronos": {"name": "CHRONOS",        "color": "#FF7F0E", "dash": "solid"}
}

def main():
    # --- EDIT TITLES HERE ---
    PLOT_TITLE = "Energy with Outliers: RMSE Median Curve"
    X_AXIS_TITLE = "Corruption Level (%)"
    Y_AXIS_TITLE = "RMSE"
    # ------------------------

    input_dir = os.path.join("images", "CSV")
    output_dir = os.path.join("images", "CSV", "converted")
    
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {input_dir}. Please place your W&B exports there.")
        return

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        
        logger.info(f"Processing: {file_name}")
        df = pd.read_csv(file_path)
        
        if "corruption_level" not in df.columns:
            logger.warning(f"'corruption_level' missing in {file_name}. Skipping this file.")
            continue
            
        x_vals = df["corruption_level"]
        target_columns = [col for col in df.columns if col.endswith("RMSE_median")]
        
        if not target_columns:
            logger.warning(f"No 'RMSE_median' columns found in {file_name}. Skipping.")
            continue

        fig = go.Figure()

        for col in target_columns:
            # Check if column matches any known model
            matched_key = None
            col_lower = col.lower()
            for key in MODEL_STYLES.keys():
                if key in col_lower:
                    matched_key = key
                    break
            
            # Skip models not in the dictionary
            if not matched_key:
                logger.debug(f"Skipping unknown or unmapped column: {col}")
                continue

            style = MODEL_STYLES[matched_key]
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=df[col],
                mode='lines+markers',
                marker=dict(
                    size=9, 
                    symbol="circle", # Hardcoded to circle for all traces
                    color=style["color"]
                ),
                line=dict(
                    width=3, 
                    color=style["color"], 
                    dash=style["dash"]
                ),
                name=style["name"]
            ))
            
        fig.update_layout(
            title=PLOT_TITLE, 
            xaxis_title=X_AXIS_TITLE, 
            yaxis_title=Y_AXIS_TITLE,
            yaxis=dict(rangemode="nonnegative"), 
            template="plotly_white",
            hovermode="x unified"
        )
        
        save_path = os.path.join(output_dir, f"{base_name}.pdf")
        
        fig.write_image(save_path, engine="kaleido", width=1400, height=600)
        logger.info(f"Successfully saved PDF to: {save_path}")

if __name__ == "__main__":
    main()
import os
import glob
import pandas as pd
import plotly.graph_objects as go
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def extract_model_name(raw_column_name: str) -> str:
    """Extracts the clean model name from a messy W&B column export string."""
    known_models = ["lstm", "chronos", "xgboost", "sarimax", "prophet", "naive"]
    raw_lower = raw_column_name.lower()
    
    for model in known_models:
        if model in raw_lower:
            return model.upper() if model != "naive" else "Naive Baseline"
    
    # Fallback if the model name isn't in our known list
    return raw_column_name.split(" - ")[0].strip()

def main():
    # Define directories
    input_dir = os.path.join("images", "CSV")
    output_dir = os.path.join("images", "CSV", "converted")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files in the input folder
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {input_dir}. Please place your W&B exports there.")
        return

    # Different marker symbols for up to 6 lines to ensure accessibility/printability
    symbols = ['circle', 'square', 'diamond', 'triangle-up', 'x', 'cross']

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        
        logger.info(f"Processing: {file_name}")
        df = pd.read_csv(file_path)
        
        # 1. Identify the X-Axis
        if "corruption_level" not in df.columns:
            logger.warning(f"'corruption_level' missing in {file_name}. Skipping this file.")
            continue
            
        x_vals = df["corruption_level"]
        
        # 2. Identify the Y-Axes (Only exact RMSE_median columns, ignoring MIN/MAX)
        target_columns = [col for col in df.columns if col.endswith("RMSE_median")]
        
        if not target_columns:
            logger.warning(f"No 'RMSE_median' columns found in {file_name}. Skipping.")
            continue

        # 3. Build the Plotly Figure
        fig = go.Figure()

        for idx, col in enumerate(target_columns):
            model_name = extract_model_name(col)
            symbol = symbols[idx % len(symbols)]
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=df[col],
                mode='lines+markers',
                marker=dict(size=9, symbol=symbol), # Distinct shapes for every line
                line=dict(width=3),                 # Thick, academic-style lines
                name=model_name
            ))
            
        # 4. Apply Academic Layout (Matched exactly to main.py structure)
        fig.update_layout(
            title=f"RMSE Median Curve", 
            xaxis_title="Corruption Level (%)", 
            yaxis_title="RMSE",
            yaxis=dict(rangemode="nonnegative"), 
            template="plotly_white",
            hovermode="x unified"
        )
        
        # 5. Export High-Resolution PDF
        save_path = os.path.join(output_dir, f"{base_name}.pdf")
        
        # Forces the exact aspect ratio used in main.py
        fig.write_image(save_path, engine="kaleido", width=1400, height=600)
        logger.info(f"Successfully saved PDF to: {save_path}")

if __name__ == "__main__":
    main()
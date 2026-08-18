import os
import pandas as pd
import plotly.graph_objects as go
from hydra import initialize, compose

from src.data_loader import load_data
from src.corruptions import apply_corruption
from src.models.model_prophet import ProphetModel
from src.models.model_sarima import SARIMAModel

def generate_comparative_outlier_plot():
    """
    Generates a visualization demonstrating how Prophet (curve-fitting) 
    and SARIMA (autoregressive) react differently to extreme outliers.
    """
    ZOOM_STEPS = 50 
    
    with initialize(version_base=None, config_path="../configs"):
        cfg_prophet = compose(config_name="config", overrides=[
            "dataset=sp500", "corruption=outliers", "model=prophet"
        ])
        cfg_sarima = compose(config_name="config", overrides=[
            "dataset=sp500", "corruption=outliers", "model=sarima"
        ])
        
        cfg_prophet.current_data_seed = cfg_prophet.seed
        cfg_prophet.current_corr_seed = cfg_prophet.seed
        cfg_prophet.current_model_seed = cfg_prophet.seed
        
        cfg_sarima.current_model_seed = cfg_sarima.seed
        
        train_clean, test_truth = load_data(cfg_prophet)
        full_clean_data = pd.concat([train_clean, test_truth])
        
        train_corrupted = apply_corruption(train_clean, cfg_prophet, 50.0)
        
        model_p = ProphetModel(cfg_prophet)
        model_p.train(train_corrupted)
        
        test_pred_p = pd.Series(model_p.predict(steps=len(test_truth)), index=test_truth.index)
        
        future_train = pd.DataFrame({'ds': train_corrupted.index})
        train_fitted_df_p = model_p.model.predict(future_train)
        train_fitted_p = pd.Series(train_fitted_df_p['yhat'].values, index=train_corrupted.index)
        
        full_forecast_prophet = pd.concat([train_fitted_p, test_pred_p])
        
        model_s = SARIMAModel(cfg_sarima)
        model_s.train(train_corrupted)
        
        test_pred_s = pd.Series(model_s.predict(steps=len(test_truth)), index=test_truth.index)
        
        train_fitted_s = model_s.fitted_model.fittedvalues
        
        full_forecast_sarima = pd.concat([train_fitted_s, test_pred_s])

        viz_clean = full_clean_data.iloc[-ZOOM_STEPS:]
        viz_train_corrupted = train_corrupted[train_corrupted.index.isin(viz_clean.index)]
        viz_test_truth = test_truth[test_truth.index.isin(viz_clean.index)]
        
        viz_forecast_prophet = full_forecast_prophet.iloc[-ZOOM_STEPS:]
        viz_forecast_sarima = full_forecast_sarima.iloc[-ZOOM_STEPS:]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=viz_clean.index.to_pydatetime(),
            y=viz_clean.values,
            mode='lines',
            name='Clean Data',
            line=dict(color='rgba(0, 0, 0, 0.8)', width=2.5, dash='dot') 
        ))
        
        fig.add_trace(go.Scatter(
            x=viz_train_corrupted.index.to_pydatetime(),
            y=viz_train_corrupted.values,
            mode='lines',
            name='Corrupted Data',
            line=dict(color='rgba(214, 39, 40, 0.6)', width=2.5, dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=viz_forecast_prophet.index.to_pydatetime(),
            y=viz_forecast_prophet.values,
            mode='lines',
            name='Prophet',
            line=dict(color='#2CA02C', width=2.5) 
        ))
        
        fig.add_trace(go.Scatter(
            x=viz_forecast_sarima.index.to_pydatetime(),
            y=viz_forecast_sarima.values,
            mode='lines',
            name='SARIMA',
            line=dict(color='#1F77B4', width=2.5) 
        ))
        
        split_date = train_corrupted.index[-1].to_pydatetime()
        
        if viz_clean.index[0].to_pydatetime() <= split_date <= viz_clean.index[-1].to_pydatetime():
            fig.add_vline(x=split_date, line_width=2, line_dash="dash", line_color="black")
        
        visible_train_len = len(viz_train_corrupted)
        visible_test_len = len(viz_test_truth)
        
        if visible_train_len > 0:
            fig.add_annotation(
                x=viz_train_corrupted.index[visible_train_len // 2].to_pydatetime(),
                y=1.05,
                yref="paper",
                showarrow=False,
                text="<b>Training Data</b>",
                font=dict(size=18)
            )
            
        if visible_test_len > 0:
            fig.add_annotation(
                x=viz_test_truth.index[visible_test_len // 2].to_pydatetime(),
                y=1.05,
                yref="paper",
                showarrow=False,
                text="<b>Test Data</b>",
                font=dict(size=18)
            )
        
        fig.update_layout(
            title=dict(
                text="Prophet and SARIMA on S&P500 with Outliers at 50% Intensity",
                font=dict(size=25)
            ),
            xaxis=dict(
                title="Time",
                title_font=dict(size=23),
                tickfont=dict(size=19)
            ),
            yaxis=dict(
                title="S&P 500 Value",
                title_font=dict(size=23),
                tickfont=dict(size=19)
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
        
        save_dir = os.path.join("images", "viz", "thesis_plots")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "comparative_sp500_outliers_50_zoomed.pdf")
        
        fig.write_image(save_path, engine="kaleido", width=1400, height=600)
        print(f"Visualization successfully saved to: {save_path}")

if __name__ == "__main__":
    generate_comparative_outlier_plot()
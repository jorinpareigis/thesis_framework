# Thesis Framework: Time-Series Forecasting

A modular machine learning framework using Hydra and Weights & Biases to evaluate the degradation of forecasting models (SARIMAX, XGBoost, Naive) under varying data corruption conditions (MCAR, Outliers).

## Structure
* `configs/`: Hydra configuration files.
* `src/`: Core logic (data loading, corruption, models, evaluation).
* `main.py`: Framework orchestrator.
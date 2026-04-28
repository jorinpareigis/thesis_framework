# Thesis Framework: Time-Series Forecasting Robustness

A modular machine learning framework designed to evaluate the degradation curves of forecasting models under progressively increasing data corruption. It uses Hydra for dynamic configuration and Weights & Biases (W&B) for logging Monte Carlo simulation results.

## Features
* **Modular Architecture:** Isolate data loading, corruption logic, and model execution.
* **Monte Carlo Evaluation:** Averages multiple runs per configuration to capture statistically significant performance metrics (Mean and Standard Deviation of RMSE/MAE).
* **Deterministic Corruption:** Ensures strict data subsetting across intensity steps (e.g., data dropped at 10% corruption is explicitly retained within the 11% corruption step) by managing RNG seeds.
* **Supported Models:** SARIMAX, XGBoost, Naive Baselines.
* **Supported Corruptions:** MCAR (Missing Completely At Random), Outliers.

## Project Structure
```text
thesis_framework/
├── configs/                 # Hydra configuration YAMLs
│   ├── corruption/          # MCAR, Outliers settings
│   ├── dataset/             # Energy, S&P 500 settings
│   ├── model/               # SARIMAX, XGBoost, Naive settings
│   └── config.yaml          # Global orchestrator parameters
├── data/                    # Local CSV datasets (git-ignored)
├── src/
│   ├── models/              # Model wrappers inheriting BaseForecastingModel
│   ├── corruptions.py       # Anomaly injection logic
│   ├── data_loader.py       # Data parsing and train/test splitting
│   └── evaluator.py         # RMSE and MAE calculations
├── main.py                  # Execution orchestrator
├── download_sp500.py        # Script to fetch S&P 500 data via yfinance
├── requirements.txt         # Project dependencies
└── .gitignore
```

## Installation

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR_USERNAME/thesis_framework.git](https://github.com/YOUR_USERNAME/thesis_framework.git)
cd thesis_framework
```

**2. Set up the Python environment**
Create and activate your preferred virtual environment (venv, conda, etc.), then install the dependencies:
```bash
python -m venv thesis_env
source thesis_env/bin/activate  # On Windows use: thesis_env\Scripts\activate
pip install -r requirements.txt
```

**3. Configure Weights & Biases**
The framework requires a W&B account to log metrics. Authenticate your local environment:
```bash
wandb login
```

**4. Prepare Datasets**
Ensure your local data files are present in the `data/` directory. For the S&P 500 dataset, execute the included fetch script:
```bash
python download_sp500.py
```
*(Ensure `PJME_hourly.csv` is manually placed in `data/` if testing the energy dataset).*

## Running an Experiment

The framework is driven by Hydra. You execute `main.py` and override the default YAML configurations directly from the command line.

### Basic Execution
Running the script without arguments executes the defaults defined in `configs/config.yaml` (Energy dataset, MCAR corruption, SARIMAX model).
```bash
python main.py
```

### Command-Line Overrides
Specify your target parameters using the `key=value` syntax. Always set a descriptive `experiment_name` to track the run cleanly in W&B.

**Example 1: XGBoost on Energy Data with Outliers**
```bash
python main.py experiment_name=energy_xgboost_outliers dataset=energy model=xgboost corruption=outliers
```

**Example 2: Naive Model (Seasonal Average) on S&P 500 with MCAR**
```bash
python main.py experiment_name=sp500_naive_mcar dataset=sp500 model=naive model.strategy=seasonal_average corruption=mcar
```

**Example 3: Overriding Global Parameters**
You can adjust the corruption boundaries or Monte Carlo run count on the fly:
```bash
python main.py experiment_name=quick_test corruption_start=0 corruption_end=10 corruption_step=2 num_runs=3
```

## View Results
Upon execution, `main.py` aggregates the metrics and pushes them to your W&B cloud dashboard. Navigate to your [Weights & Biases Project](https://wandb.ai) to view the plotted degradation curves (Mean and Standard Deviation for RMSE/MAE).
```
# Thesis Framework: Time-Series Forecasting Robustness

A modular machine learning framework designed to evaluate the degradation curves of forecasting models under progressively increasing data corruption. It uses Hydra for dynamic configuration and Weights & Biases (W&B) for logging Monte Carlo simulation results. Designed and optimized for execution on Windows.

## Feature Overview

* **Monte Carlo Evaluation:** Averages multiple runs per configuration to capture statistically significant performance metrics (Mean and Standard Deviation of RMSE/MAE).
* **Deterministic Corruption:** Ensures strict data subsetting across intensity steps by managing RNG seeds.
* **Implemented Datasets:**
  * **Finance:** S&P 500 Daily (`sp500`)
  * **Energy:** PJME Hourly Energy Consumption (`energy`)
  * **IoT:** Machine Temperature 5-min (`iot_temp`)
  * **Environment:** Beijing PM2.5 Hourly Air Quality (`air_quality`)
* **Implemented Models:**
  * **Statistical:** SARIMA, Prophet
  * **Machine Learning:** XGBoost
  * **Deep Learning/Foundation:** LSTM (PyTorch), Chronos (Amazon)
  * **Naive Baselines:** Forward Fill, Global Mean, Seasonal Naive, Seasonal Average
* **Implemented Corruptions:**
  * MCAR (Missing Completely At Random)
  * Outliers
  * Gaussian Noise
  * Sensor Outage
  * Sensor Drift

## Project Structure

```text
thesis_framework/
├── configs/                 # Hydra configuration YAMLs
│   ├── corruption/          # MCAR, Outliers, Noise, Outage, Drift settings
│   ├── dataset/             # Energy, S&P 500, IoT, Air Quality settings
│   ├── model/               # Model-specific hyperparameters
│   └── config.yaml          # Global orchestrator parameters
├── data/                    # Local CSV datasets (git-ignored)
├── scripts/                 # Execution and utility scripts
│   ├── download_*.py        # Data fetching scripts
│   ├── calculate_sarima_baselines.py
├── src/
│   ├── models/              # Model wrappers (XGBoost, LSTM, Chronos, etc.)
│   ├── utils/               # Validators and helpers
│   ├── corruptions.py       # Anomaly injection logic
│   ├── data_loader.py       # Data parsing and train/test splitting
│   └── evaluator.py         # RMSE and MAE calculations
├── main.py                  # Single-experiment execution orchestrator
├── run_experiments.py       # Multi-experiment concurrent batch runner
├── pyproject.toml           # uv project dependencies
└── README.md
```

## Installation (Windows Setup)

This project uses `uv`, a Python package and project manager.

**1. Clone the repository**

```bash
git clone https://github.com/jorinpareigis/thesis_framework.git
cd thesis_framework
```

**2. Install dependencies via uv**
Ensure you have `uv` installed (`pip install uv`). Then, sync the project to automatically create a virtual environment and install all core dependencies.

```bash
uv sync
```

**3. Activate the environment**

```bash
.venv\Scripts\activate
```

**4. Install PyTorch (Hardware Specific)**
*Note: PyTorch is explicitly excluded from the core dependencies. Deep learning models (Chronos, LSTM) require significant matrix multiplication capabilities. To ensure optimal execution speed, install the PyTorch version that matches your hardware accelerator.*

For NVIDIA GPUs (CUDA 12.1):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only:

```bash
uv pip install torch
```

**5. Configure Weights & Biases**
The framework requires a W&B account to log metrics. Authenticate your local environment:

```bash
wandb login
```

**6. Prepare Datasets**
Execute the included fetch scripts to download and format the datasets:

```bash
python scripts/download_air_quality.py
python scripts/download_iot_temp.py
python scripts/download_sp500.py
```

*Note: Download the `PJME_hourly.csv` dataset manually (e.g., from Kaggle) and place it in the `data/` folder.*

## Running a Single Experiment (`main.py`)

The framework is driven by Hydra. You execute `main.py` and override the default YAML configurations directly from the command line.

### Basic Execution

```bash
python main.py
```

### Command-Line Overrides

Specify target parameters using the `key=value` syntax.

**Model Selection (`model=`)**

* `xgboost`, `sarima`, `prophet`, `lstm`, `chronos`
* `model=naive model.strategy=forward_fill` (Options: `forward_fill`, `mean`, `seasonal_naive`, `seasonal_average`)

**Dataset Selection (`dataset=`)**

* `energy`, `sp500`, `iot_temp`, `air_quality`

**Corruption Selection (`corruption=`)**

* `mcar`, `outliers`, `gaussian_noise`, `sensor_outage`, `sensor_drift`

### Dataset & Corruption Limitations

Validation logic prevents scientifically illogical combinations. Ensure you adhere to the following mapping:

* **`sp500`**: `mcar`, `outliers`
* **`energy`**: `mcar`, `outliers`, `sensor_outage`, `gaussian_noise`
* **`iot_temp`**: `mcar`, `outliers`, `sensor_outage`, `sensor_drift`, `gaussian_noise`
* **`air_quality`**: `mcar`, `outliers`, `sensor_outage`, `sensor_drift`, `gaussian_noise`

### Execution Examples

```bash
# Run Chronos on Energy data with MCAR corruption
python main.py run_suffix="_1" dataset=energy model=chronos corruption=mcar

# Quick test overriding experiment name and total iterations count
python main.py experiment_name="quick_test_run" num_runs=5
```

## Running Multiple Experiments (`run_experiments.py`)

To execute multiple configurations sequentially or concurrently without re-entering commands, use `run_experiments.py`.

1. **Tailor to your setup**: Open `run_experiments.py` and adjust `MAX_CPU_WORKERS` and `MAX_GPU_WORKERS` to match your system specifications to prevent Out of Memory (OOM) errors.
2. **Define the grid**: Edit the `DATASETS`, `CORRUPTIONS`, and `MODELS` arrays in the script to define the combinations you wish to run.
3. **Execute**:
```bash
python run_experiments.py
```

The script automatically routes CPU-bound models to a concurrent pool and GPU-bound models to a sequential queue, monitoring progress without cluttering the terminal.

## View Results

The framework aggregates metrics (Mean and Standard Deviation of RMSE/MAE) only after all Monte Carlo iterations complete for a corruption step. Navigate to your [Weights & Biases Dashboard](https://wandb.ai) to view the plotted degradation curves.

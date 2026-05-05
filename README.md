# Thesis Framework: Time-Series Forecasting Robustness

A modular machine learning framework designed to evaluate the degradation curves of forecasting models under progressively increasing data corruption. It uses Hydra for dynamic configuration and Weights & Biases (W&B) for logging Monte Carlo simulation results. 

Designed and optimized for execution on Windows.

## Features
* **Modular Architecture:** Isolate data loading, corruption logic, and model execution.
* **Monte Carlo Evaluation:** Averages multiple runs per configuration to capture statistically significant performance metrics (Mean and Standard Deviation of RMSE/MAE).
* **Deterministic Corruption:** Ensures strict data subsetting across intensity steps (e.g., data dropped at 10% corruption is explicitly retained within the 11% corruption step) by managing RNG seeds.
* **Datasets:** 
  * **Implemented:** Hourly Energy Consumption, Finance (S&P 500).
  * **Planned:** 2-3 additional datasets.
* **Supported Models:** 
  * **Implemented:** SARIMAX, XGBoost, Prophet, Chronos (Amazon Foundation Model), LSTM (PyTorch), and 4 Naive Baselines (Forward Fill, Global Mean, Seasonal Last, Seasonal Average).
  * **Planned:** Gaussian Process Regression (GPR).
* **Supported Corruptions:** 
  * **Implemented:** MCAR (Missing Completely At Random), Outliers.
  * **Planned:** Gaussian Noise, Sensor Outage, Sensor Drift, Adversarial Data Injection.

## Project Structure
```text
thesis_framework/
├── configs/                 # Hydra configuration YAMLs
│   ├── corruption/          # MCAR, Outliers settings
│   ├── dataset/             # Energy, S&P 500 settings
│   ├── model/               # Model-specific hyperparameters
│   └── config.yaml          # Global orchestrator parameters
├── data/                    # Local CSV datasets (git-ignored)
├── src/
│   ├── models/              # Model wrappers (XGBoost, LSTM, Chronos, etc.)
│   ├── corruptions.py       # Anomaly injection logic
│   ├── data_loader.py       # Data parsing and train/test splitting
│   └── evaluator.py         # RMSE and MAE calculations
├── main.py                  # Execution orchestrator
├── download_sp500.py        # Script to fetch S&P 500 data via yfinance
├── requirements.txt         # Project dependencies (includes PyTorch CUDA links)
└── .gitignore
```

## Installation (Windows Setup)

**1. Clone the repository**
```bash
git clone [[https://github.com/jorinpareigis/thesis_framework.git](https://github.com/jorinpareigis/thesis_framework.git)]
cd thesis_framework
```

**2. Set up the Python environment (VS Code Terminal)**
Create and activate a virtual environment:
```cmd
python -m venv thesis_env
.\thesis_env\Scripts\activate
```

**3. Install Dependencies & CUDA Support**
Deep learning models (Chronos, LSTM) require GPU acceleration to run feasibly within a Monte Carlo loop. The `requirements.txt` is pre-configured to fetch the PyTorch CUDA 12.1 backend for NVIDIA GPUs.
```cmd
pip install -r requirements.txt
```

**4. Configure Weights & Biases**
The framework requires a W&B account to log metrics. Authenticate your local environment:
```cmd
wandb login
```

**5. Prepare Datasets**
Ensure your local data files are present in the `data/` directory. 
* **Finance:** Execute the included fetch script:
  ```cmd
  python download_sp500.py
  
```
* **Energy:** Download the `PJME_hourly.csv` dataset and manually place it in the `data/` folder.

## Running an Experiment

The framework is driven by Hydra. You execute `main.py` and override the default YAML configurations directly from the command line.

### Basic Execution
Running the script without arguments executes the defaults defined in `configs/config.yaml`.
```cmd
python main.py
```

### Command-Line Overrides
Specify your target parameters using the `key=value` syntax. Always set a descriptive `experiment_name` to track the run cleanly in W&B.

**1. Model Selection**
* `model=xgboost`
* `model=sarimax`
* `model=prophet`
* `model=lstm`
* `model=chronos`
* `model=naive model.strategy=forward_fill` (Options: `forward_fill`, `mean`, `seasonal_naive`, `seasonal_average`)
* *(Planned)*: `model=gpr`

**2. Dataset Selection**
* `dataset=energy`
* `dataset=sp500`

**3. Corruption Selection**
* `corruption=mcar`
* `corruption=outliers`
* *(Planned)*: `corruption=gaussian_noise`, `corruption=sensor_outage`, `corruption=sensor_drift`, `corruption=adversarial`

### Execution Examples

**Example 1: GPU-Accelerated Foundation Model on Energy Data**
```cmd
python main.py experiment_name=energy_chronos_mcar dataset=energy model=chronos corruption=mcar
```

**Example 2: PyTorch LSTM on S&P 500 with Outliers**
```cmd
python main.py experiment_name=sp500_lstm_outliers dataset=sp500 model=lstm corruption=outliers
```

**Example 3: Overriding Global Loop Parameters**
You can adjust the corruption boundaries or Monte Carlo run count on the fly for rapid testing:
```cmd
python main.py experiment_name=quick_test corruption_start=0 corruption_end=10 corruption_step=2 num_runs=3
```

## View Results
Upon execution, `main.py` utilizes `tqdm` to display nested progress bars in the terminal. Once a full simulation completes, it aggregates the metrics and pushes them to the cloud. Navigate to your [Weights & Biases Project](https://wandb.ai) to view the plotted degradation curves (Mean and Standard Deviation for RMSE/MAE).
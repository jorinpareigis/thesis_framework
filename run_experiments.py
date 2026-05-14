import os
import subprocess
import itertools
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. DEFINE YOUR EXPERIMENT GRID
# ==========================================
DATASETS = ["sp500"]  
CORRUPTIONS = ["mcar"]         
MODELS = ["naive", "sarimax", "xgboost", "prophet"] 

RUN_SUFFIX = "_batch_1"

# ==========================================
# 2. HARDWARE CONCURRENCY LIMITS
# ==========================================
MAX_CPU_WORKERS = 4 
MAX_GPU_WORKERS = 1 

GPU_MODELS = {"lstm", "chronos"}

def run_single_experiment(dataset, corruption, model):
    cmd = [
        "python", "main.py",
        f"dataset={dataset}",
        f"corruption={corruption}",
        f"model={model}",
        f"run_suffix={RUN_SUFFIX}",
        "+batch_mode=True"  # Dynamically adds the batch_mode flag to Hydra
    ]
    
    #if model == "naive":
    #    cmd.append("model.strategy=forward_fill")
        
    logger.info(f"STARTING: [{dataset} | {corruption} | {model}]")
    
    # Force Python to bypass internal buffers for real-time logging
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Merge stderr into stdout (subprocess.STDOUT) to read everything linearly
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        env=env
    )
    
    full_log = []
    
    # Read the clean, tqdm-free log stream
    for line in process.stdout:
        full_log.append(line)
        if "BATCH_PROGRESS_25" in line:
            logger.info(f"PROGRESS [{dataset} | {model}]: 25% completed")
        elif "BATCH_PROGRESS_50" in line:
            logger.info(f"PROGRESS [{dataset} | {model}]: 50% completed")
        elif "BATCH_PROGRESS_75" in line:
            logger.info(f"PROGRESS [{dataset} | {model}]: 75% completed")

    process.wait()
    
    if process.returncode != 0:
        error_msg = "".join(full_log[-15:])
        logger.error(f"FAILED: [{dataset} | {model}]\nError Snippet:\n{error_msg}")
    else:
        logger.info(f"COMPLETED: [{dataset} | {model}]")

def main():
    experiments = list(itertools.product(DATASETS, CORRUPTIONS, MODELS))
    cpu_tasks = []
    gpu_tasks = []
    
    for dataset, corruption, model in experiments:
        if model in GPU_MODELS:
            gpu_tasks.append((dataset, corruption, model))
        else:
            cpu_tasks.append((dataset, corruption, model))

    logger.info(f"Total experiments queued: {len(experiments)} ({len(cpu_tasks)} CPU, {len(gpu_tasks)} GPU)")

    if cpu_tasks:
        logger.info(f"--- Starting CPU Queue ({MAX_CPU_WORKERS} workers) ---")
        with ThreadPoolExecutor(max_workers=MAX_CPU_WORKERS) as executor:
            for task in cpu_tasks:
                executor.submit(run_single_experiment, *task)

    if gpu_tasks:
        logger.info(f"--- Starting GPU Queue ({MAX_GPU_WORKERS} workers) ---")
        with ThreadPoolExecutor(max_workers=MAX_GPU_WORKERS) as executor:
            for task in gpu_tasks:
                executor.submit(run_single_experiment, *task)
                
    logger.info("All batches completed successfully.")

if __name__ == "__main__":
    main()
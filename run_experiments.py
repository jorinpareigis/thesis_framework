import os
import sys
import subprocess
import itertools
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

DATASETS: list[str] = ["energy"]
CORRUPTIONS: list[str] = ["gaussian_noise"]
MODELS: list[str] = ["xgboost"]
RUN_SUFFIX: str = "_final_framework_test"

MAX_CPU_WORKERS: int = 4 
MAX_GPU_WORKERS: int = 1 
GPU_MODELS: set[str] = {"lstm", "chronos"}

def run_single_experiment(dataset: str, corruption: str, model: str) -> None:
    """
    Executes a single Hydra configuration run as a subprocess.
    
    Bypasses internal Python buffers to allow real-time stdout monitoring
    for predefined milestone markers.
    
    Args:
        dataset (str): The target dataset configuration key.
        corruption (str): The target corruption configuration key.
        model (str): The target model configuration key.
    """
    target_group = f"{dataset}_{corruption}"

    cmd = [
        sys.executable, "main.py",
        f"dataset={dataset}",
        f"corruption={corruption}",
        f"model={model}",
        f"run_suffix={RUN_SUFFIX}",
        f"+group_name={target_group}",
        "+batch_mode=True"
    ]
        
    logger.info(f"STARTING: [{dataset} | {corruption} | {model}]")
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        env=env
    )
    
    log_buffer = deque(maxlen=20)
    
    if process.stdout:
        for line in process.stdout:
            log_buffer.append(line)
            if "BATCH_PROGRESS_25" in line:
                logger.info(f"PROGRESS [{dataset} | {corruption} | {model}]: 25% completed")
            elif "BATCH_PROGRESS_50" in line:
                logger.info(f"PROGRESS [{dataset} | {corruption} | {model}]: 50% completed")
            elif "BATCH_PROGRESS_75" in line:
                logger.info(f"PROGRESS [{dataset} | {corruption} | {model}]: 75% completed")

    process.wait()
    
    if process.returncode != 0:
        error_msg = "".join(log_buffer)
        logger.error(f"FAILED: [{dataset} | {corruption} | {model}]\nError Snippet:\n{error_msg}")
    else:
        logger.info(f"COMPLETED: [{dataset} | {corruption} | {model}]")

def main() -> None:
    """
    Orchestrates the batch execution of ML experiments.
    
    Splits tasks into CPU-bound and GPU-bound queues to prevent 
    hardware oversubscription (e.g., CUDA Out-Of-Memory errors).
    """
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
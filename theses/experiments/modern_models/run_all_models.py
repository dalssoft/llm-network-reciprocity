"""
Convenience script to run Experiment A for all modern models.

Usage:
    python -m theses.experiments.modern_models.run_all_models [--parallel N]

This script runs all model configurations (including reasoning variants)
and continues even if some models fail (e.g., due to API credit limits).
Results from successful runs are preserved.
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import List, Tuple

from .model_config import MODERN_CONFIGS, get_config_description, get_config_id
from src.infra.llm.models import Models

# Setup batch runner logger
os.makedirs("logs", exist_ok=True)
_batch_log_filename = f"logs/batch_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

batch_logger = logging.getLogger("batch_runner")
batch_logger.setLevel(logging.DEBUG)

_file_handler = logging.FileHandler(_batch_log_filename)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))
batch_logger.addHandler(_file_handler)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter('%(message)s'))
batch_logger.addHandler(_console_handler)


def run_model_experiment(model: Models, with_reasoning: bool) -> Tuple[int, str]:
    """Run experiment for a single model configuration.

    Returns:
        (return_code, config_id): Return code from subprocess and config identifier
    """
    config_desc = get_config_description(model, with_reasoning)
    config_id = get_config_id(model, with_reasoning)

    print(f"\n{'='*80}")
    print(f"Starting: {config_desc}")
    print(f"{'='*80}\n")
    
    batch_logger.info(f"Starting model: {config_desc}")
    start_time = datetime.now()

    cmd = [
        sys.executable,
        "-m",
        "theses.experiments.modern_models.run_experiment_a",
        "--model",
        model.model_name,
    ]

    # Add reasoning flag if needed
    if with_reasoning:
        cmd.append("--reasoning")

    # Capture output for logging
    result = subprocess.run(cmd, capture_output=False)
    
    duration = (datetime.now() - start_time).total_seconds()

    if result.returncode == 0:
        print(f"\n✅ Completed: {config_desc}")
        batch_logger.info(f"✅ Completed: {config_desc} (duration: {duration:.1f}s)")
    else:
        print(f"\n❌ Failed: {config_desc}")
        print(f"   (Results from completed replications were saved)")
        batch_logger.error(f"❌ Failed: {config_desc} (return code: {result.returncode}, duration: {duration:.1f}s)")

    return result.returncode, config_id


def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment A for all modern models (including reasoning variants)"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of models to run in parallel (default: 1, sequential)",
    )
    parser.add_argument(
        "--skip-on-error",
        action="store_true",
        help="Continue running other models even if one fails (default: True, always enabled)",
    )

    args = parser.parse_args()

    # Get all configurations
    configs = MODERN_CONFIGS

    print(f"\n{'='*80}")
    print("Modern Models Experiment - Batch Runner")
    print(f"{'='*80}")
    print(f"\nConfigurations to test ({len(configs)}):")
    for i, (model, with_reasoning) in enumerate(configs, 1):
        desc = get_config_description(model, with_reasoning)
        print(f"  {i}. {desc}")

    print(f"\nExecution mode: {'Parallel' if args.parallel > 1 else 'Sequential'}")
    print(f"Error handling: Continue on failure (results preserved)")
    print(f"📋 Batch log file: {_batch_log_filename}")
    print(f"{'='*80}\n")
    
    batch_logger.info(
        f"Batch run started with {len(configs)} configurations"
    )

    if args.parallel > 1:
        print(f"⚠️  Parallel execution not yet implemented. Running sequentially.\n")

    # Run experiments
    failed_configs = []
    successful_configs = []

    for model, with_reasoning in configs:
        try:
            returncode, config_id = run_model_experiment(model, with_reasoning)
            if returncode != 0:
                failed_configs.append((model, with_reasoning, config_id))
            else:
                successful_configs.append((model, with_reasoning, config_id))
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Showing results so far...")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            failed_configs.append(
                (model, with_reasoning, get_config_id(model, with_reasoning))
            )

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"Total configurations: {len(configs)}")
    print(f"Successful: {len(successful_configs)}")
    print(f"Failed: {len(failed_configs)}")

    batch_logger.info(
        f"Batch run complete: {len(successful_configs)}/{len(configs)} succeeded, "
        f"{len(failed_configs)} failed"
    )

    if successful_configs:
        print("\n✅ Successful configurations:")
        for model, with_reasoning, config_id in successful_configs:
            desc = get_config_description(model, with_reasoning)
            print(f"  - {desc}")
            batch_logger.info(f"  ✅ {desc}")

    if failed_configs:
        print("\n❌ Failed configurations:")
        for model, with_reasoning, config_id in failed_configs:
            desc = get_config_description(model, with_reasoning)
            print(f"  - {desc}")
            batch_logger.error(f"  ❌ {desc}")
        print("\nNote: Results from completed replications were still saved.")
        print("You can re-run failed configurations individually using:")
        print(
            "  python -m theses.experiments.modern_models.run_experiment_a --model <model_name> [--reasoning]"
        )
        print(f"\n📋 Full logs available at: {_batch_log_filename}")
        sys.exit(1)
    else:
        print("\n✅ All configurations completed successfully!")
        print(f"📋 Full logs available at: {_batch_log_filename}")
        sys.exit(0)


if __name__ == "__main__":
    main()

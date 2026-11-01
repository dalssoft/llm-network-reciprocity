"""
Batch runner for Experiment B with selected models.

Target: Gemini 3 Flash, Gemini 3 Flash (reasoning), Gemini 2.5 Flash Lite
Replications: 15 per condition
Conditions: 9 (k ∈ {2,4,6} × b/c ∈ {2,4,6})

Usage:
    python -m theses.experiments.modern_models.run_experiment_b_batch

This runs all 3 model configurations sequentially. Each configuration runs
9 conditions × 15 replications = 135 simulations.

Total: 3 models × 135 = 405 simulations
Total decisions: 3 × 9 × 15 × 25 agents × 15 rounds = 151,875 API calls
"""

import argparse
import subprocess
import sys
from datetime import datetime

# Model configurations to run
# Format: (model_name, with_reasoning)
MODEL_CONFIGS = [
    ("gemini-3-flash-preview", False),   # Gemini 3 Flash (standard)
    ("gemini-3-flash-preview", True),    # Gemini 3 Flash (reasoning)
    ("gemini-2.5-flash-lite", False),    # Gemini 2.5 Flash Lite (standard)
]

REPLICATIONS = 15
MAX_WORKERS = 5  # Parallel replications per condition


def run_model_config(model_name: str, with_reasoning: bool) -> int:
    """Run Experiment B for a single model configuration."""
    reasoning_str = " (reasoning)" if with_reasoning else ""
    print(f"\n{'='*70}")
    print(f"Starting: {model_name}{reasoning_str}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Replications: {REPLICATIONS}")
    print(f"{'='*70}\n")

    cmd = [
        sys.executable, "-m",
        "theses.experiments.modern_models.run_experiment_b",
        "--model", model_name,
        "--replications", str(REPLICATIONS),
        "--max-workers", str(MAX_WORKERS),
    ]

    if with_reasoning:
        cmd.append("--reasoning")

    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n⚠️  {model_name}{reasoning_str} failed with return code {result.returncode}")
    else:
        print(f"\n✅ {model_name}{reasoning_str} completed successfully")

    return result.returncode


def main():
    print(f"\n{'#'*70}")
    print("# Experiment B Batch Runner")
    print("# Parameter Sweep for Modern Models")
    print(f"{'#'*70}")
    print(f"\nModels to run:")
    for model_name, with_reasoning in MODEL_CONFIGS:
        reasoning_str = " (reasoning)" if with_reasoning else ""
        print(f"  - {model_name}{reasoning_str}")
    print(f"\nReplications per condition: {REPLICATIONS}")
    print(f"Conditions: 9 (k ∈ {{2,4,6}} × b/c ∈ {{2,4,6}})")
    print(f"Total simulations per model: {9 * REPLICATIONS}")
    print(f"Total simulations overall: {len(MODEL_CONFIGS) * 9 * REPLICATIONS}")

    # Estimate
    decisions_per_model = 9 * REPLICATIONS * 25 * 15  # conditions × reps × agents × rounds
    print(f"\nEstimated API calls per model: {decisions_per_model:,}")
    print(f"Estimated total API calls: {len(MODEL_CONFIGS) * decisions_per_model:,}")

    print(f"\nStarting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    results = []
    for model_name, with_reasoning in MODEL_CONFIGS:
        returncode = run_model_config(model_name, with_reasoning)
        results.append((model_name, with_reasoning, returncode))

    # Summary
    print(f"\n{'#'*70}")
    print("# Batch Complete")
    print(f"{'#'*70}")
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults:")
    for model_name, with_reasoning, returncode in results:
        reasoning_str = " (reasoning)" if with_reasoning else ""
        status = "✅ Success" if returncode == 0 else f"❌ Failed (code {returncode})"
        print(f"  {model_name}{reasoning_str}: {status}")

    # Exit with failure if any model failed
    failed = [r for r in results if r[2] != 0]
    if failed:
        print(f"\n⚠️  {len(failed)} model(s) failed. Check logs for details.")
        sys.exit(1)
    else:
        print(f"\n✅ All {len(MODEL_CONFIGS)} models completed successfully!")


if __name__ == "__main__":
    # No options; argparse only makes --help print instead of starting the batch.
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()

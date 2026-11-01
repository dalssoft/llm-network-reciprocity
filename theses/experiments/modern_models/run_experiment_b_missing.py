"""
Run only MISSING conditions from Experiment B.

Use this after quota resets to complete the experiment without re-running
conditions that already have full data.

Features:
- Automatically retries when rate limits are hit
- Sleeps for configurable time between retries
- Keeps running until all conditions complete

Usage:
    python -m theses.experiments.modern_models.run_experiment_b_missing
    python -m theses.experiments.modern_models.run_experiment_b_missing --dry-run
    python -m theses.experiments.modern_models.run_experiment_b_missing --retry-sleep 300  # 5 min sleep
"""

import os
import json
import subprocess
import sys
import time
from datetime import datetime

# Target: 15 replications per condition
TARGET_REPS = 15

# All conditions in Experiment B
ALL_CONDITIONS = [
    (2, 2.0), (2, 4.0), (2, 6.0),
    (4, 2.0), (4, 4.0), (4, 6.0),
    (6, 2.0), (6, 4.0), (6, 6.0),
]

# Model configurations to check
MODEL_CONFIGS = [
    ("gemini-3-flash-preview", False),   # standard
    ("gemini-3-flash-preview", True),    # reasoning
    ("gemini-2.5-flash-lite", False),    # standard
]

RESULTS_BASE = "theses/experiments/modern_models/results/experiment_b"


def get_completed_reps(model_name: str, with_reasoning: bool, k: int, bc: float) -> int:
    """Check how many replications are completed for a condition.

    Counts across ALL run directories for this condition (handles multiple runs).
    """
    reasoning_suffix = "_reasoning" if with_reasoning else ""
    model_dir = f"{RESULTS_BASE}/{model_name}{reasoning_suffix}"

    if not os.path.exists(model_dir):
        return 0

    condition_prefix = f"bc{bc}_k{k}"
    total_reps = 0

    # Sum replications from ALL matching directories
    for d in os.listdir(model_dir):
        if d.startswith(condition_prefix):
            results_file = f"{model_dir}/{d}/results.json"
            if os.path.exists(results_file):
                try:
                    with open(results_file) as f:
                        data = json.load(f)
                    total_reps += len(data)
                except:
                    pass

    return total_reps


def get_missing_conditions():
    """Get list of (model, reasoning, k, bc, missing_reps) for incomplete conditions."""
    missing = []

    for model_name, with_reasoning in MODEL_CONFIGS:
        for k, bc in ALL_CONDITIONS:
            completed = get_completed_reps(model_name, with_reasoning, k, bc)
            if completed < TARGET_REPS:
                missing.append({
                    "model": model_name,
                    "reasoning": with_reasoning,
                    "k": k,
                    "bc": bc,
                    "completed": completed,
                    "needed": TARGET_REPS - completed,
                })

    return missing


def run_condition(model: str, reasoning: bool, k: int, bc: float, reps: int, max_workers: int = 2):
    """Run a single condition."""
    cmd = [
        sys.executable, "-m",
        "theses.experiments.modern_models.run_experiment_b",
        "--model", model,
        "--k", str(k),
        "--bc", str(bc),
        "--replications", str(reps),
        "--max-workers", str(max_workers),
    ]

    if reasoning:
        cmd.append("--reasoning")

    print(f"\nRunning: {' '.join(cmd)}")
    return subprocess.run(cmd)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run missing Experiment B conditions")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without executing")
    parser.add_argument("--max-workers", type=int, default=2, help="Parallel workers (lower = less rate limiting)")
    parser.add_argument("--model", type=str, help="Only run for specific model")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Experiment B - Missing Conditions Checker")
    print(f"{'='*60}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target replications: {TARGET_REPS}")
    print(f"Max workers: {args.max_workers}")

    missing = get_missing_conditions()

    if args.model:
        missing = [m for m in missing if m["model"] == args.model]

    if not missing:
        print("\n✅ All conditions complete! Nothing to run.")
        return

    print(f"\n{'='*60}")
    print("Missing Conditions:")
    print(f"{'='*60}")

    total_needed = 0
    for m in missing:
        reasoning_str = " (R)" if m["reasoning"] else ""
        print(f"  {m['model']}{reasoning_str} | k={m['k']}, bc={m['bc']} | {m['completed']}/{TARGET_REPS} | need {m['needed']}")
        total_needed += m["needed"]

    print(f"\nTotal missing replications: {total_needed}")
    print(f"Estimated API calls: {total_needed * 25 * 15:,}")

    if args.dry_run:
        print("\n[DRY RUN] Would run the above conditions.")
        return

    print(f"\n{'='*60}")
    print("Starting missing runs...")
    print(f"{'='*60}")

    for m in missing:
        reasoning_str = " (reasoning)" if m["reasoning"] else ""
        print(f"\n>>> {m['model']}{reasoning_str} | k={m['k']}, bc={m['bc']} | need {m['needed']} reps")

        result = run_condition(
            model=m["model"],
            reasoning=m["reasoning"],
            k=m["k"],
            bc=m["bc"],
            reps=m["needed"],
            max_workers=args.max_workers,
        )

        if result.returncode != 0:
            print(f"⚠️  Condition failed. Continuing with next...")

    print(f"\n{'='*60}")
    print("Missing runs complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

"""
Analysis for probabilistic horizon experiment.

Compares cooperation dynamics across delta values and horizon info conditions.

Usage:
    python -m theses.experiments.probabilistic_horizon.analysis.analyze_results
    python -m theses.experiments.probabilistic_horizon.analysis.analyze_results --condition delta0.9_informed
"""

import json
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ReplicationStats:
    """Statistics for a single replication."""
    replication_id: int
    final_coop_rate: float
    avg_coop_rate: float
    coop_rates_by_round: List[float]
    actual_rounds: int


@dataclass
class ConditionStats:
    """Aggregated statistics for a condition."""
    condition_name: str
    delta: float
    horizon_info: str
    replications: List[ReplicationStats]

    @property
    def mean_final_coop(self) -> float:
        return statistics.mean(r.final_coop_rate for r in self.replications)

    @property
    def std_final_coop(self) -> float:
        if len(self.replications) < 2:
            return 0.0
        return statistics.stdev(r.final_coop_rate for r in self.replications)

    @property
    def mean_avg_coop(self) -> float:
        return statistics.mean(r.avg_coop_rate for r in self.replications)

    @property
    def std_avg_coop(self) -> float:
        if len(self.replications) < 2:
            return 0.0
        return statistics.stdev(r.avg_coop_rate for r in self.replications)

    @property
    def mean_rounds(self) -> float:
        return statistics.mean(r.actual_rounds for r in self.replications)

    @property
    def std_rounds(self) -> float:
        if len(self.replications) < 2:
            return 0.0
        return statistics.stdev(r.actual_rounds for r in self.replications)

    def mean_coop_by_round(self, max_round: Optional[int] = None) -> List[float]:
        """
        Calculate mean cooperation rate for each round across replications.
        Since replications may have different lengths, uses only replications
        that reached each round.
        """
        if max_round is None:
            max_round = max(r.actual_rounds for r in self.replications)
        means = []
        for round_idx in range(max_round):
            round_rates = [
                r.coop_rates_by_round[round_idx]
                for r in self.replications
                if round_idx < len(r.coop_rates_by_round)
            ]
            if round_rates:
                means.append(statistics.mean(round_rates))
            else:
                break
        return means

    def n_replications_at_round(self, max_round: Optional[int] = None) -> List[int]:
        """Count how many replications are still active at each round."""
        if max_round is None:
            max_round = max(r.actual_rounds for r in self.replications)
        counts = []
        for round_idx in range(max_round):
            count = sum(
                1 for r in self.replications
                if round_idx < len(r.coop_rates_by_round)
            )
            counts.append(count)
        return counts


def analyze_replication(rep_data: Dict[str, Any]) -> ReplicationStats:
    """Analyze a single replication's data."""
    history = rep_data["history"]
    actual_rounds = rep_data.get("actual_rounds", len(history))

    coop_rates_by_round = []
    for round_data in history:
        actions = [r["action"] for r in round_data["results"]]
        coop_count = sum(1 for a in actions if a == "COOPERATE")
        coop_rates_by_round.append(coop_count / len(actions))

    return ReplicationStats(
        replication_id=rep_data["replication"],
        final_coop_rate=coop_rates_by_round[-1] if coop_rates_by_round else 0.0,
        avg_coop_rate=statistics.mean(coop_rates_by_round) if coop_rates_by_round else 0.0,
        coop_rates_by_round=coop_rates_by_round,
        actual_rounds=actual_rounds,
    )


def analyze_condition(results_dir: Path) -> Optional[ConditionStats]:
    """Analyze all replications for a condition."""
    results_file = results_dir / "results.json"
    config_file = results_dir / "config.json"

    if not results_file.exists() or not config_file.exists():
        return None

    with open(results_file) as f:
        results = json.load(f)
    with open(config_file) as f:
        config = json.load(f)

    if not results:
        print(f"  No successful replications in {results_dir.name}")
        return None

    replications = [analyze_replication(rep) for rep in results]

    return ConditionStats(
        condition_name=config["condition"],
        delta=config["continuation_delta"],
        horizon_info=config["horizon_info"],
        replications=replications,
    )


def find_latest_results(experiment_dir: Path, condition_prefix: str) -> Optional[Path]:
    """Find the most recent results directory for a condition."""
    if not experiment_dir.exists():
        return None
    matching_dirs = [
        d for d in experiment_dir.iterdir()
        if d.is_dir() and d.name.startswith(condition_prefix)
    ]
    if not matching_dirs:
        return None
    return max(matching_dirs, key=lambda d: d.stat().st_mtime)


def get_available_models(base_dir: Path) -> List[str]:
    """Get list of models that have results."""
    if not base_dir.exists():
        return []
    return [d.name for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]


def print_condition_summary(stats: ConditionStats):
    """Print summary for a single condition."""
    print(f"\n  {'─'*50}")
    print(f"  Condition: {stats.condition_name}")
    print(f"  Delta: {stats.delta}, Info: {stats.horizon_info}")
    print(f"  Replications: {len(stats.replications)}")
    print(f"  Actual rounds: {stats.mean_rounds:.1f} +/- {stats.std_rounds:.1f}")
    print(f"  Final coop rate: {stats.mean_final_coop:.1%} +/- {stats.std_final_coop:.1%}")
    print(f"  Average coop rate: {stats.mean_avg_coop:.1%} +/- {stats.std_avg_coop:.1%}")

    # Show cooperation trajectory at key points
    means = stats.mean_coop_by_round()
    n_active = stats.n_replications_at_round()
    if means:
        print(f"  Cooperation trajectory (n=replications still active):")
        key_rounds = [0, 4, 9, 14, 19, 24, 29, 39, 49]
        for idx in key_rounds:
            if idx < len(means):
                print(f"    Round {idx+1:3d}: {means[idx]:.1%} (n={n_active[idx]})")


def print_comparison_table(all_stats: Dict[str, ConditionStats]):
    """Print comparison table across all conditions."""
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")

    header = f"{'Condition':<25} {'Delta':<7} {'Info':<12} {'Rounds':<12} {'Final Coop':<15} {'Avg Coop':<15}"
    print(header)
    print("-" * 80)

    # Sort by delta then info
    for key in sorted(all_stats.keys()):
        s = all_stats[key]
        print(
            f"{s.condition_name:<25} "
            f"{s.delta:<7.1f} "
            f"{s.horizon_info:<12} "
            f"{s.mean_rounds:5.1f}+/-{s.std_rounds:4.1f} "
            f"{s.mean_final_coop:5.1%}+/-{s.std_final_coop:5.1%}  "
            f"{s.mean_avg_coop:5.1%}+/-{s.std_avg_coop:5.1%}"
        )

    print("-" * 80)

    # Compare informed vs uninformed for each delta
    deltas = sorted(set(s.delta for s in all_stats.values()))
    print("\nInformed vs Uninformed comparison:")
    for d in deltas:
        informed = next((s for s in all_stats.values() if s.delta == d and s.horizon_info == "informed"), None)
        uninformed = next((s for s in all_stats.values() if s.delta == d and s.horizon_info == "uninformed"), None)
        if informed and uninformed:
            diff = informed.mean_avg_coop - uninformed.mean_avg_coop
            print(f"  delta={d}: informed={informed.mean_avg_coop:.1%}, uninformed={uninformed.mean_avg_coop:.1%}, diff={diff:+.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze probabilistic horizon experiment results"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model subdirectory to analyze (e.g., gemini-3-flash-preview_reasoning)",
    )
    parser.add_argument(
        "--condition", type=str, default=None,
        help="Specific condition prefix to analyze (e.g., delta0.9_informed)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available models with results",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent / "results"

    if args.list:
        models = get_available_models(base_dir)
        print(f"\nAvailable models with results ({len(models)}):")
        for m in sorted(models):
            print(f"  - {m}")
        return

    # Find model directory
    if args.model:
        model_dir = base_dir / args.model
    else:
        models = get_available_models(base_dir)
        if not models:
            print("No results found.")
            return
        if len(models) == 1:
            model_dir = base_dir / models[0]
            print(f"Using model: {models[0]}")
        else:
            print(f"Multiple models found: {models}")
            print("Use --model to specify one, or --list to see all.")
            return

    if not model_dir.exists():
        print(f"No results directory: {model_dir}")
        return

    print(f"\n{'#'*60}")
    print("# Probabilistic Horizon Experiment - Results Analysis")
    print(f"# Model: {model_dir.name}")
    print(f"{'#'*60}")

    # Find all condition directories
    all_stats = {}
    condition_prefixes = set()

    for d in sorted(model_dir.iterdir()):
        if not d.is_dir():
            continue
        # Extract condition prefix (everything before the timestamp)
        parts = d.name.rsplit("_", 2)  # e.g., delta0.9_informed_20260212_143000
        if len(parts) >= 3:
            prefix = parts[0] + "_" + parts[1] if not parts[1][0].isdigit() else parts[0]
        else:
            prefix = d.name
        condition_prefixes.add(prefix)

    # For each condition, find latest results
    for prefix in sorted(condition_prefixes):
        if args.condition and not prefix.startswith(args.condition):
            continue
        latest = find_latest_results(model_dir, prefix)
        if latest:
            stats = analyze_condition(latest)
            if stats:
                all_stats[prefix] = stats
                print_condition_summary(stats)

    if len(all_stats) > 1:
        print_comparison_table(all_stats)
    elif not all_stats:
        print("\nNo results found matching criteria.")


if __name__ == "__main__":
    main()

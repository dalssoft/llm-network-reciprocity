"""
Analysis module for Han et al. (2024) reproduction experiment results.

Usage:
    python -m theses.experiments.han_reproduction.analysis.analyze_results [results_dir]
    
    If no results_dir is provided, uses the latest results from experiment_a.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import statistics


@dataclass
class ReplicationStats:
    """Statistics for a single replication."""
    replication_id: int
    final_coop_rate: float
    avg_coop_rate: float
    coop_rates_by_round: List[float]
    agent_coop_rates: Dict[int, float]  # Per-agent cooperation rate


@dataclass
class ConditionStats:
    """Aggregated statistics for a condition."""
    condition_name: str
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
    
    def mean_coop_by_round(self) -> List[float]:
        """Calculate mean cooperation rate for each round across replications."""
        num_rounds = len(self.replications[0].coop_rates_by_round)
        means = []
        for round_idx in range(num_rounds):
            round_rates = [r.coop_rates_by_round[round_idx] for r in self.replications]
            means.append(statistics.mean(round_rates))
        return means
    
    def std_coop_by_round(self) -> List[float]:
        """Calculate std of cooperation rate for each round across replications."""
        num_rounds = len(self.replications[0].coop_rates_by_round)
        stds = []
        for round_idx in range(num_rounds):
            round_rates = [r.coop_rates_by_round[round_idx] for r in self.replications]
            stds.append(statistics.stdev(round_rates) if len(round_rates) > 1 else 0.0)
        return stds


def load_results(results_path: Path) -> List[Dict[str, Any]]:
    """Load results from JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load config from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def analyze_replication(rep_data: Dict[str, Any]) -> ReplicationStats:
    """Analyze a single replication's data."""
    rep_id = rep_data["replication"]
    history = rep_data["history"]
    
    # Calculate cooperation rate per round
    coop_rates_by_round = []
    agent_actions: Dict[int, List[str]] = {}
    
    for round_data in history:
        actions = []
        for result in round_data["results"]:
            agent_id = result["agent_id"]
            action = result["action"]
            actions.append(action)
            
            if agent_id not in agent_actions:
                agent_actions[agent_id] = []
            agent_actions[agent_id].append(action)
        
        coop_count = sum(1 for a in actions if a == "COOPERATE")
        coop_rate = coop_count / len(actions)
        coop_rates_by_round.append(coop_rate)
    
    # Calculate per-agent cooperation rates
    agent_coop_rates = {}
    for agent_id, actions in agent_actions.items():
        coop_count = sum(1 for a in actions if a == "COOPERATE")
        agent_coop_rates[agent_id] = coop_count / len(actions)
    
    return ReplicationStats(
        replication_id=rep_id,
        final_coop_rate=coop_rates_by_round[-1],
        avg_coop_rate=statistics.mean(coop_rates_by_round),
        coop_rates_by_round=coop_rates_by_round,
        agent_coop_rates=agent_coop_rates,
    )


def analyze_condition(results_dir: Path) -> ConditionStats:
    """Analyze all replications for a condition."""
    results_file = results_dir / "results.json"
    config_file = results_dir / "config.json"
    
    results = load_results(results_file)
    config = load_config(config_file)
    
    replications = [analyze_replication(rep) for rep in results]
    
    return ConditionStats(
        condition_name=config["condition"],
        replications=replications,
    )


def find_latest_results(experiment_dir: Path, condition_prefix: str) -> Optional[Path]:
    """Find the most recent results directory for a condition."""
    matching_dirs = [
        d for d in experiment_dir.iterdir()
        if d.is_dir() and d.name.startswith(condition_prefix)
    ]
    if not matching_dirs:
        return None
    return max(matching_dirs, key=lambda d: d.stat().st_mtime)


def print_condition_summary(stats: ConditionStats):
    """Print summary statistics for a condition."""
    print(f"\n{'='*60}")
    print(f"Condition: {stats.condition_name}")
    print(f"{'='*60}")
    print(f"Replications: {len(stats.replications)}")
    print(f"\nFinal Cooperation Rate:")
    print(f"  Mean: {stats.mean_final_coop:.1%} ± {stats.std_final_coop:.1%}")
    for rep in stats.replications:
        print(f"    Rep {rep.replication_id + 1}: {rep.final_coop_rate:.1%}")
    
    print(f"\nAverage Cooperation Rate:")
    print(f"  Mean: {stats.mean_avg_coop:.1%} ± {stats.std_avg_coop:.1%}")
    for rep in stats.replications:
        print(f"    Rep {rep.replication_id + 1}: {rep.avg_coop_rate:.1%}")
    
    print(f"\nCooperation by Round (mean across replications):")
    means = stats.mean_coop_by_round()
    stds = stats.std_coop_by_round()
    # Print every 5th round + first and last
    key_rounds = [0, 4, 9, 14, 19, 24] if len(means) >= 25 else list(range(len(means)))
    for idx in key_rounds:
        if idx < len(means):
            print(f"    Round {idx + 1:2d}: {means[idx]:.1%} ± {stds[idx]:.1%}")


def print_agent_patterns(stats: ConditionStats):
    """Print per-agent cooperation patterns."""
    print(f"\nPer-Agent Cooperation Rates:")
    
    # Aggregate across replications
    all_agent_rates: Dict[int, List[float]] = {}
    for rep in stats.replications:
        for agent_id, rate in rep.agent_coop_rates.items():
            if agent_id not in all_agent_rates:
                all_agent_rates[agent_id] = []
            all_agent_rates[agent_id].append(rate)
    
    for agent_id in sorted(all_agent_rates.keys()):
        rates = all_agent_rates[agent_id]
        mean_rate = statistics.mean(rates)
        std_rate = statistics.stdev(rates) if len(rates) > 1 else 0.0
        print(f"    Agent {agent_id}: {mean_rate:.1%} ± {std_rate:.1%}")


def compare_conditions(wm_stats: ConditionStats, st_stats: ConditionStats):
    """Compare well-mixed vs structured conditions."""
    print(f"\n{'#'*60}")
    print("# COMPARISON: Well-Mixed vs Structured")
    print(f"{'#'*60}")
    
    print(f"\n{'Metric':<25} {'Well-Mixed':<15} {'Structured':<15} {'Difference':<15}")
    print("-" * 70)
    
    # Final cooperation
    wm_final = stats.mean_final_coop if (stats := wm_stats) else 0
    st_final = stats.mean_final_coop if (stats := st_stats) else 0
    diff_final = wm_final - st_final
    print(f"{'Final Coop Rate':<25} {wm_stats.mean_final_coop:.1%}±{wm_stats.std_final_coop:.1%}      {st_stats.mean_final_coop:.1%}±{st_stats.std_final_coop:.1%}      {diff_final:+.1%}")
    
    # Average cooperation
    diff_avg = wm_stats.mean_avg_coop - st_stats.mean_avg_coop
    print(f"{'Average Coop Rate':<25} {wm_stats.mean_avg_coop:.1%}±{wm_stats.std_avg_coop:.1%}      {st_stats.mean_avg_coop:.1%}±{st_stats.std_avg_coop:.1%}      {diff_avg:+.1%}")
    
    print("\n" + "-" * 70)
    print("Han et al. (2024) Reference (GPT-3.5, Setting 1):")
    print("  Well-Mixed:  Final ~62.5%, Avg ~53.5%")
    print("  Structured:  Final ~25.0%, Avg ~52.0%")
    print("  Difference:  +37.5% (Well-Mixed > Structured)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Han et al. reproduction experiment results"
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=None,
        help="Path to results directory (uses latest if not specified)",
    )
    parser.add_argument(
        "--well-mixed",
        default=None,
        help="Path to well-mixed results directory",
    )
    parser.add_argument(
        "--structured",
        default=None,
        help="Path to structured results directory",
    )
    parser.add_argument(
        "--agents",
        action="store_true",
        help="Show per-agent cooperation patterns",
    )
    args = parser.parse_args()
    
    # Default experiment directory
    base_dir = Path(__file__).parent.parent / "results" / "experiment_a"
    
    # Find results directories
    if args.well_mixed:
        wm_dir = Path(args.well_mixed)
    else:
        wm_dir = find_latest_results(base_dir, "well_mixed")
    
    if args.structured:
        st_dir = Path(args.structured)
    else:
        st_dir = find_latest_results(base_dir, "structured_ring")
    
    print(f"\n{'#'*60}")
    print("# Han et al. (2024) Reproduction - Results Analysis")
    print(f"{'#'*60}")
    
    # Analyze conditions
    wm_stats = None
    st_stats = None
    
    if wm_dir and wm_dir.exists():
        print(f"\nLoading well-mixed results from: {wm_dir.name}")
        wm_stats = analyze_condition(wm_dir)
        print_condition_summary(wm_stats)
        if args.agents:
            print_agent_patterns(wm_stats)
    else:
        print("\nNo well-mixed results found.")
    
    if st_dir and st_dir.exists():
        print(f"\nLoading structured results from: {st_dir.name}")
        st_stats = analyze_condition(st_dir)
        print_condition_summary(st_stats)
        if args.agents:
            print_agent_patterns(st_stats)
    else:
        print("\nNo structured results found.")
    
    # Compare if both available
    if wm_stats and st_stats:
        compare_conditions(wm_stats, st_stats)
    
    print()


if __name__ == "__main__":
    main()


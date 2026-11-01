"""
Analysis module for Han et al. (2024) reproduction experiment results.

Usage:
    python -m theses.experiments.modern_models.analysis.analyze_results --model DeepSeek-V3.2
    python -m theses.experiments.modern_models.analysis.analyze_results --all
    
Supports multi-model results structure with per-model subdirectories.
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


def analyze_condition(results_dir: Path) -> Optional[ConditionStats]:
    """Analyze all replications for a condition. Returns None if no results."""
    results_file = results_dir / "results.json"
    config_file = results_dir / "config.json"
    
    results = load_results(results_file)
    config = load_config(config_file)
    
    # Handle empty results (all replications failed)
    if not results:
        failed_count = config.get("failed_replications", 0)
        print(f"  ⚠️  No successful replications ({failed_count} failed)")
        return None
    
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


def get_available_models(base_dir: Path) -> List[str]:
    """Get list of models that have results."""
    if not base_dir.exists():
        return []
    return [d.name for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]


def find_model_results(base_dir: Path, model_name: str, condition_prefix: str) -> Optional[Path]:
    """Find latest results for a specific model and condition."""
    model_dir = base_dir / model_name
    if not model_dir.exists():
        return None
    return find_latest_results(model_dir, condition_prefix)


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


def analyze_model(base_dir: Path, model_name: str, show_agents: bool = False):
    """Analyze results for a single model."""
    print(f"\n{'#'*60}")
    print(f"# Model: {model_name}")
    print(f"{'#'*60}")
    
    wm_dir = find_model_results(base_dir, model_name, "well_mixed")
    st_dir = find_model_results(base_dir, model_name, "structured_ring")
    
    wm_stats = None
    st_stats = None
    
    if wm_dir and wm_dir.exists():
        print(f"\nLoading well-mixed results from: {wm_dir.name}")
        wm_stats = analyze_condition(wm_dir)
        if wm_stats:
            print_condition_summary(wm_stats)
            if show_agents:
                print_agent_patterns(wm_stats)
    else:
        print("\nNo well-mixed results found.")
    
    if st_dir and st_dir.exists():
        print(f"\nLoading structured results from: {st_dir.name}")
        st_stats = analyze_condition(st_dir)
        if st_stats:
            print_condition_summary(st_stats)
            if show_agents:
                print_agent_patterns(st_stats)
    else:
        print("\nNo structured results found.")
    
    if wm_stats and st_stats:
        compare_conditions(wm_stats, st_stats)
    elif wm_stats or st_stats:
        print("\n⚠️  Cannot compare - missing one condition")
    
    return wm_stats, st_stats


def print_summary_table(all_results: Dict[str, tuple]):
    """Print a summary table comparing all models."""
    print(f"\n{'#'*70}")
    print("# SUMMARY: All Models Comparison")
    print(f"{'#'*70}")
    
    print(f"\n{'Model':<35} {'WM Final':<12} {'ST Final':<12} {'Diff':<10} {'Network Effect':<15}")
    print("-" * 85)
    
    for model_name, (wm_stats, st_stats) in sorted(all_results.items()):
        wm_final = f"{wm_stats.mean_final_coop:.1%}" if wm_stats else "N/A"
        st_final = f"{st_stats.mean_final_coop:.1%}" if st_stats else "N/A"
        
        if wm_stats and st_stats:
            diff = st_stats.mean_final_coop - wm_stats.mean_final_coop
            diff_str = f"{diff:+.1%}"
            # Human-like: structured > well-mixed
            effect = "✅ Human-like" if diff > 0.05 else ("⚠️ Opposite" if diff < -0.05 else "➖ No effect")
        else:
            diff_str = "N/A"
            effect = "Incomplete"
        
        print(f"{model_name:<35} {wm_final:<12} {st_final:<12} {diff_str:<10} {effect:<15}")
    
    print("-" * 85)
    print("\nReference (Han et al. 2024):")
    print("  GPT-3.5: WM 62.5%, ST 25.0% → Diff -37.5% (Opposite of humans)")
    print("  Humans:  WM ~45%, ST ~75%  → Diff +30%   (Structured promotes cooperation)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Han et al. reproduction experiment results"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to analyze (e.g., DeepSeek-V3.2)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all available models",
    )
    parser.add_argument(
        "--well-mixed",
        default=None,
        help="Path to well-mixed results directory (overrides model lookup)",
    )
    parser.add_argument(
        "--structured",
        default=None,
        help="Path to structured results directory (overrides model lookup)",
    )
    parser.add_argument(
        "--agents",
        action="store_true",
        help="Show per-agent cooperation patterns",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models with results",
    )
    args = parser.parse_args()
    
    # Default experiment directory
    base_dir = Path(__file__).parent.parent / "results" / "experiment_a"
    
    # List available models
    if args.list:
        models = get_available_models(base_dir)
        print(f"\nAvailable models with results ({len(models)}):")
        for m in sorted(models):
            print(f"  - {m}")
        return
    
    print(f"\n{'#'*60}")
    print("# Han et al. (2024) Reproduction - Results Analysis")
    print(f"{'#'*60}")
    
    # Manual path override
    if args.well_mixed or args.structured:
        wm_dir = Path(args.well_mixed) if args.well_mixed else None
        st_dir = Path(args.structured) if args.structured else None
        
        wm_stats = None
        st_stats = None
        
        if wm_dir and wm_dir.exists():
            print(f"\nLoading well-mixed from: {wm_dir}")
            wm_stats = analyze_condition(wm_dir)
            if wm_stats:
                print_condition_summary(wm_stats)
        
        if st_dir and st_dir.exists():
            print(f"\nLoading structured from: {st_dir}")
            st_stats = analyze_condition(st_dir)
            if st_stats:
                print_condition_summary(st_stats)
        
        if wm_stats and st_stats:
            compare_conditions(wm_stats, st_stats)
        return
    
    # Analyze all models
    if args.all:
        models = get_available_models(base_dir)
        if not models:
            print("\nNo results found.")
            return
        
        all_results = {}
        for model_name in sorted(models):
            wm_stats, st_stats = analyze_model(base_dir, model_name, args.agents)
            all_results[model_name] = (wm_stats, st_stats)
        
        print_summary_table(all_results)
        return
    
    # Analyze specific model
    if args.model:
        analyze_model(base_dir, args.model, args.agents)
        return
    
    # Default: show available models and prompt
    models = get_available_models(base_dir)
    if models:
        print(f"\nAvailable models ({len(models)}):")
        for m in sorted(models):
            print(f"  - {m}")
        print("\nUsage:")
        print("  --model <name>  Analyze specific model")
        print("  --all           Analyze all models")
        print("  --list          List available models")
    else:
        print("\nNo results found in experiment_a directory.")
    
    print()


if __name__ == "__main__":
    main()


"""
Comprehensive Analysis: Experiment B (Modern Models vs Han et al. 2024)

Compares Gemini 3 Flash, Gemini 3 Flash (Reasoning), and Gemini 2.5 Flash Lite
against Han's GPT-3.5 baseline across all 9 parameter conditions.

Key research questions:
1. Do modern models respond to b/c > k threshold (unlike GPT-3.5)?
2. Does reasoning mode affect cooperation patterns?
3. Are oscillation patterns present in modern models?
4. Do models show end-game collapse?
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats as scipy_stats

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import patheffects
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plots will be skipped.")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_replications(model_dir: Path) -> Dict[str, List[Dict]]:
    """
    Load ALL replications for a model (not just the latest run per condition).

    Returns: Dict[condition] -> List[replication_data]
    """
    all_replications = defaultdict(list)

    for run_dir in model_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("bc"):
            continue

        # Parse condition name (e.g., "bc4.0_k6_20260123_154642" -> "bc4.0_k6")
        parts = run_dir.name.split("_")
        condition = f"{parts[0]}_{parts[1]}"

        results_file = run_dir / "results.json"
        if not results_file.exists():
            continue

        with open(results_file) as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                continue

        # Each result file contains multiple replications
        for rep in results:
            if "coop_rates" in rep and len(rep["coop_rates"]) > 0:
                all_replications[condition].append(rep)

    return dict(all_replications)


def load_model_data(base_dir: Path) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Load data for all models.

    Returns: Dict[model_name] -> Dict[condition] -> List[replications]
    """
    models = {}

    # Map directory names to display names
    dir_to_name = {
        "gemini-3-flash-preview": "Gemini 3 Flash",
        "gemini-3-flash-preview_reasoning": "Gemini 3 Flash (R)",
        "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    }

    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue

        dir_name = model_dir.name
        if dir_name in dir_to_name:
            display_name = dir_to_name[dir_name]
            models[display_name] = load_all_replications(model_dir)

    return models


# ============================================================================
# STATISTICS COMPUTATION
# ============================================================================

def compute_condition_stats(replications: List[Dict]) -> Dict:
    """Compute statistics for a single condition."""
    if not replications:
        return None

    # Extract cooperation rates
    avg_coops = [np.mean(rep["coop_rates"]) for rep in replications]
    final_coops = [rep["coop_rates"][-1] for rep in replications]

    # Cooperation by round (matrix: n_reps x n_rounds)
    coop_matrix = np.array([rep["coop_rates"] for rep in replications])
    mean_by_round = np.mean(coop_matrix, axis=0)
    std_by_round = np.std(coop_matrix, axis=0)

    # End-game collapse detection (drop from round 14 to 15)
    if len(mean_by_round) >= 15:
        endgame_drop = mean_by_round[-2] - mean_by_round[-1]
    else:
        endgame_drop = 0.0

    # Oscillation detection
    osc_info = detect_oscillation(mean_by_round)

    return {
        "n": len(replications),
        "avg_coop_mean": np.mean(avg_coops),
        "avg_coop_std": np.std(avg_coops),
        "final_coop_mean": np.mean(final_coops),
        "final_coop_std": np.std(final_coops),
        "mean_by_round": mean_by_round,
        "std_by_round": std_by_round,
        "endgame_drop": endgame_drop,
        "oscillation": osc_info,
    }


def detect_oscillation(coop_rates: np.ndarray, threshold: float = 0.05) -> Dict:
    """Detect period-2 oscillation pattern."""
    if len(coop_rates) < 4:
        return {"has_oscillation": False, "amplitude": 0.0, "autocorr_lag1": 0.0}

    # Skip first round (always 100%)
    rates = coop_rates[1:]

    # Compute differences between consecutive rounds
    diffs = np.diff(rates)

    # Check for sign alternation (period-2 pattern)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    max_sign_changes = len(diffs) - 1
    alternation_ratio = sign_changes / max_sign_changes if max_sign_changes > 0 else 0

    # Compute autocorrelation at lag 1
    mean_centered = rates - np.mean(rates)
    if np.std(rates) > 0.01:
        autocorr_lag1 = np.corrcoef(mean_centered[:-1], mean_centered[1:])[0, 1]
    else:
        autocorr_lag1 = 0.0

    # Amplitude of oscillation
    amplitude = np.mean(np.abs(diffs))

    return {
        "has_oscillation": alternation_ratio > 0.6 and amplitude > threshold,
        "amplitude": amplitude,
        "autocorr_lag1": autocorr_lag1 if not np.isnan(autocorr_lag1) else 0.0,
        "alternation_ratio": alternation_ratio,
    }


def categorize_condition(bc_ratio: float, k: int) -> str:
    """Categorize condition by theoretical prediction."""
    if bc_ratio > k:
        return "b/c > k"
    elif bc_ratio == k:
        return "b/c = k"
    else:
        return "b/c < k"


def parse_condition(condition: str) -> Tuple[float, int]:
    """Parse condition string into (bc_ratio, k)."""
    parts = condition.split("_")
    bc = float(parts[0][2:])  # "bc4.0" -> 4.0
    k = int(parts[1][1:])     # "k6" -> 6
    return bc, k


# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

def compute_threshold_sensitivity(model_stats: Dict[str, Dict]) -> Dict:
    """
    Compute sensitivity to b/c > k threshold.

    This is the KEY test: do models cooperate MORE when b/c > k?
    """
    bc_gt_k = []  # Conditions where b/c > k
    bc_le_k = []  # Conditions where b/c <= k

    for condition, stats in model_stats.items():
        if stats is None:
            continue
        bc, k = parse_condition(condition)
        if bc > k:
            bc_gt_k.append(stats["avg_coop_mean"])
        else:
            bc_le_k.append(stats["avg_coop_mean"])

    if not bc_gt_k or not bc_le_k:
        return None

    mean_gt = np.mean(bc_gt_k)
    mean_le = np.mean(bc_le_k)
    diff = mean_gt - mean_le

    # Welch's t-test
    t_stat, p_value = scipy_stats.ttest_ind(bc_gt_k, bc_le_k, equal_var=False)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(bc_gt_k)**2 + np.std(bc_le_k)**2) / 2)
    cohens_d = diff / pooled_std if pooled_std > 0 else 0

    return {
        "mean_bc_gt_k": mean_gt,
        "mean_bc_le_k": mean_le,
        "difference": diff,
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "n_gt_k": len(bc_gt_k),
        "n_le_k": len(bc_le_k),
    }


def compute_model_summary(model_data: Dict[str, List[Dict]]) -> Dict:
    """Compute summary statistics for a model across all conditions."""
    condition_stats = {}
    for condition, reps in model_data.items():
        condition_stats[condition] = compute_condition_stats(reps)

    # Overall statistics
    all_avg_coops = []
    all_final_coops = []
    total_reps = 0

    for condition, stats in condition_stats.items():
        if stats is None:
            continue
        all_avg_coops.append(stats["avg_coop_mean"])
        all_final_coops.append(stats["final_coop_mean"])
        total_reps += stats["n"]

    threshold_sensitivity = compute_threshold_sensitivity(condition_stats)

    return {
        "conditions": condition_stats,
        "total_replications": total_reps,
        "overall_avg_coop": np.mean(all_avg_coops) if all_avg_coops else 0,
        "overall_avg_std": np.std(all_avg_coops) if all_avg_coops else 0,
        "overall_final_coop": np.mean(all_final_coops) if all_final_coops else 0,
        "threshold_sensitivity": threshold_sensitivity,
    }


# ============================================================================
# REPORTING
# ============================================================================

def print_data_summary(all_models: Dict[str, Dict]):
    """Print summary of loaded data."""
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)

    total_reps = 0
    for model_name, model_data in all_models.items():
        reps = sum(len(reps) for reps in model_data.values())
        conditions = len(model_data)
        total_reps += reps
        print(f"\n{model_name}:")
        print(f"  Conditions: {conditions}/9")
        print(f"  Replications: {reps}")

        # Per-condition counts
        for condition in sorted(model_data.keys()):
            n = len(model_data[condition])
            print(f"    {condition}: n={n}")

    print(f"\n{'='*40}")
    print(f"TOTAL REPLICATIONS: {total_reps}")
    print("=" * 80)


def print_model_comparison_table(summaries: Dict[str, Dict]):
    """Print comparison table across models."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON: EXPERIMENT B RESULTS")
    print("=" * 80)

    # Header
    print(f"\n{'Model':<25} {'n':<6} {'Avg Coop':<15} {'Final Coop':<15} {'b/c>k Diff':<12} {'p-value':<10}")
    print("-" * 80)

    for model_name, summary in summaries.items():
        ts = summary["threshold_sensitivity"]
        diff_str = f"+{ts['difference']*100:.1f}%" if ts else "N/A"
        p_str = f"{ts['p_value']:.4f}" if ts else "N/A"

        print(f"{model_name:<25} {summary['total_replications']:<6} "
              f"{summary['overall_avg_coop']*100:>5.1f}% ± {summary['overall_avg_std']*100:>4.1f}%  "
              f"{summary['overall_final_coop']*100:>5.1f}%          "
              f"{diff_str:<12} {p_str:<10}")

    # Add Han baseline for comparison
    print("-" * 80)
    print(f"{'GPT-3.5 (Han baseline)':<25} {'45':<6} "
          f"{'55.3% ± 0.8%':>15}  {'40.0%':>14}          "
          f"{'+1.2%':<12} {'n/s':<10}")

    print("=" * 80)


def print_threshold_analysis(summaries: Dict[str, Dict]):
    """Print detailed threshold sensitivity analysis."""
    print("\n" + "=" * 80)
    print("THRESHOLD SENSITIVITY ANALYSIS: b/c > k")
    print("=" * 80)
    print("\nTheory: Cooperation should be HIGHER when b/c > k (benefit exceeds neighbor count)")
    print("Human behavior: ~40 percentage point difference")
    print("GPT-3.5 (Han): ~1 percentage point difference (insensitive)")
    print("-" * 80)

    for model_name, summary in summaries.items():
        ts = summary["threshold_sensitivity"]
        if ts is None:
            continue

        print(f"\n{model_name}:")
        print(f"  b/c > k conditions (n={ts['n_gt_k']}): {ts['mean_bc_gt_k']*100:.1f}%")
        print(f"  b/c <= k conditions (n={ts['n_le_k']}): {ts['mean_bc_le_k']*100:.1f}%")
        print(f"  Difference: {ts['difference']*100:+.1f}%")
        print(f"  t-statistic: {ts['t_statistic']:.3f}")
        print(f"  p-value: {ts['p_value']:.4f}")
        print(f"  Cohen's d: {ts['cohens_d']:.3f}")

        # Interpretation
        if ts["p_value"] < 0.05 and ts["difference"] > 0.05:
            print(f"  Interpretation: SIGNIFICANT sensitivity to threshold")
        elif ts["p_value"] < 0.10 and ts["difference"] > 0.03:
            print(f"  Interpretation: MARGINAL sensitivity to threshold")
        else:
            print(f"  Interpretation: NO sensitivity (like GPT-3.5)")

    print("\n" + "=" * 80)


def print_condition_details(summaries: Dict[str, Dict]):
    """Print detailed results by condition."""
    print("\n" + "=" * 80)
    print("DETAILED RESULTS BY CONDITION")
    print("=" * 80)

    conditions = ["bc2.0_k2", "bc4.0_k2", "bc6.0_k2",
                  "bc2.0_k4", "bc4.0_k4", "bc6.0_k4",
                  "bc2.0_k6", "bc4.0_k6", "bc6.0_k6"]

    for condition in conditions:
        bc, k = parse_condition(condition)
        category = categorize_condition(bc, k)

        print(f"\n{condition} ({category}):")
        print("-" * 60)

        for model_name, summary in summaries.items():
            stats = summary["conditions"].get(condition)
            if stats is None:
                print(f"  {model_name}: No data")
                continue

            print(f"  {model_name} (n={stats['n']}):")
            print(f"    Avg: {stats['avg_coop_mean']*100:.1f}% ± {stats['avg_coop_std']*100:.1f}%")
            print(f"    Final: {stats['final_coop_mean']*100:.1f}% ± {stats['final_coop_std']*100:.1f}%")
            print(f"    End-game drop: {stats['endgame_drop']*100:+.1f}%")
            osc = stats["oscillation"]
            print(f"    Oscillation: {'Yes' if osc['has_oscillation'] else 'No'} "
                  f"(autocorr={osc['autocorr_lag1']:.2f})")


def print_endgame_analysis(summaries: Dict[str, Dict]):
    """Analyze end-game collapse patterns."""
    print("\n" + "=" * 80)
    print("END-GAME COLLAPSE ANALYSIS")
    print("=" * 80)
    print("\nEnd-game drop = cooperation at round 14 minus cooperation at round 15")
    print("Positive values indicate collapse in final round.")
    print("-" * 80)

    for model_name, summary in summaries.items():
        drops = []
        for condition, stats in summary["conditions"].items():
            if stats is not None:
                drops.append((condition, stats["endgame_drop"]))

        drops.sort(key=lambda x: x[1], reverse=True)
        avg_drop = np.mean([d[1] for d in drops])

        print(f"\n{model_name}:")
        print(f"  Average end-game drop: {avg_drop*100:+.1f}%")
        print(f"  Worst conditions:")
        for cond, drop in drops[:3]:
            print(f"    {cond}: {drop*100:+.1f}%")

    print("\n" + "=" * 80)


def print_oscillation_summary(summaries: Dict[str, Dict]):
    """Print oscillation detection summary."""
    print("\n" + "=" * 80)
    print("OSCILLATION ANALYSIS (Period-2 Cycling)")
    print("=" * 80)
    print("\nGPT-3.5 (Han): 6/9 conditions showed period-2 oscillations")
    print("-" * 80)

    for model_name, summary in summaries.items():
        oscillating = 0
        total = 0

        print(f"\n{model_name}:")
        print(f"  {'Condition':<12} {'Autocorr':<10} {'Amplitude':<10} {'Oscillating'}")

        for condition in sorted(summary["conditions"].keys()):
            stats = summary["conditions"][condition]
            if stats is None:
                continue
            total += 1
            osc = stats["oscillation"]
            if osc["has_oscillation"]:
                oscillating += 1
            status = "Yes" if osc["has_oscillation"] else "No"
            print(f"  {condition:<12} {osc['autocorr_lag1']:>8.3f}  {osc['amplitude']:>8.3f}  {status}")

        print(f"  Summary: {oscillating}/{total} conditions oscillating")

    print("\n" + "=" * 80)


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_model_comparison_plot(summaries: Dict[str, Dict], output_dir: Path):
    """Create bar chart comparing all models."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    models = list(summaries.keys()) + ["GPT-3.5 (Han)"]
    avg_coops = [s["overall_avg_coop"] * 100 for s in summaries.values()] + [55.3]

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#95a5a6"]

    bars = ax.bar(models, avg_coops, color=colors[:len(models)], edgecolor="black")

    # Add value labels
    for bar, val in zip(bars, avg_coops):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Average Cooperation (%)", fontsize=12)
    ax.set_title("Experiment B: Average Cooperation by Model\n(9 conditions, k={2,4,6} x b/c={2,4,6})",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% baseline")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: model_comparison.png")


def create_threshold_comparison_plot(summaries: Dict[str, Dict], output_dir: Path):
    """Create grouped bar chart showing b/c>k vs b/c<=k for each model."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(summaries.keys()) + ["GPT-3.5 (Han)"]
    x = np.arange(len(models))
    width = 0.35

    gt_k_vals = []
    le_k_vals = []

    for summary in summaries.values():
        ts = summary["threshold_sensitivity"]
        gt_k_vals.append(ts["mean_bc_gt_k"] * 100 if ts else 0)
        le_k_vals.append(ts["mean_bc_le_k"] * 100 if ts else 0)

    # Han baseline
    gt_k_vals.append(56.2)
    le_k_vals.append(54.9)

    bars1 = ax.bar(x - width/2, gt_k_vals, width, label="b/c > k (should cooperate)",
                   color="#2ecc71", edgecolor="black")
    bars2 = ax.bar(x + width/2, le_k_vals, width, label="b/c <= k (defection ok)",
                   color="#e74c3c", edgecolor="black")

    ax.set_ylabel("Average Cooperation (%)", fontsize=12)
    ax.set_title("Threshold Sensitivity: b/c > k Effect by Model",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)

    # Add difference labels
    for i, (gt, le) in enumerate(zip(gt_k_vals, le_k_vals)):
        diff = gt - le
        ax.annotate(f"{diff:+.1f}%", xy=(i, max(gt, le) + 3),
                   ha="center", fontsize=10, fontweight="bold",
                   color="green" if diff > 5 else "gray")

    plt.tight_layout()
    plt.savefig(output_dir / "threshold_sensitivity.png", dpi=150)
    plt.close()
    print(f"  Saved: threshold_sensitivity.png")


def create_time_series_plot(summaries: Dict[str, Dict], output_dir: Path):
    """Create time series plot showing cooperation over rounds."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = {
        "Gemini 3 Flash": "#3498db",
        "Gemini 3 Flash (R)": "#e74c3c",
        "Gemini 2.5 Flash Lite": "#2ecc71",
    }

    # Plot for each k value
    for idx, k in enumerate([2, 4, 6]):
        ax = axes[idx]

        for model_name, summary in summaries.items():
            # Average across b/c values for this k
            all_rounds = []
            for bc in [2.0, 4.0, 6.0]:
                condition = f"bc{bc}_k{k}"
                if condition in summary["conditions"]:
                    stats = summary["conditions"][condition]
                    if stats is not None:
                        all_rounds.append(stats["mean_by_round"])

            if all_rounds:
                mean_rounds = np.mean(all_rounds, axis=0)
                rounds = np.arange(1, len(mean_rounds) + 1)
                ax.plot(rounds, mean_rounds * 100, color=colors.get(model_name, "gray"),
                       linewidth=2, label=model_name, marker="o", markersize=3)

        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Cooperation (%)", fontsize=12)
        ax.set_title(f"k = {k}", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Experiment B: Cooperation Over Rounds by Network Degree",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "time_series_by_k.png", dpi=150)
    plt.close()
    print(f"  Saved: time_series_by_k.png")


# Threshold symbols for the heatmaps: blue/orange stay distinct under
# color-vision deficiency; a white halo keeps them readable on any green cell.
THRESHOLD_SYMBOLS = {
    "above": ("+", "#1f5fa8"),
    "boundary": ("=", "#555555"),
    "below": ("\u2212", "#c05a00"),
}


def create_heatmaps(summaries: Dict[str, Dict], output_dir: Path):
    """Create heatmaps for each model."""
    if not HAS_MATPLOTLIB:
        return

    n_models = len(summaries)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    if n_models == 1:
        axes = [axes]

    ks = [2, 4, 6]
    bcs = [2.0, 4.0, 6.0]

    # Oldest model first; the two Gemini 3 Flash configurations side by side.
    order = ["Gemini 2.5 Flash Lite", "Gemini 3 Flash", "Gemini 3 Flash (R)"]
    names = sorted(summaries, key=lambda m: order.index(m) if m in order else len(order))

    matrices = {}
    for model_name in names:
        summary = summaries[model_name]
        matrix = np.zeros((3, 3))
        for i, k in enumerate(ks):
            for j, bc in enumerate(bcs):
                condition = f"bc{bc}_k{k}"
                if condition in summary["conditions"]:
                    stats = summary["conditions"][condition]
                    if stats is not None:
                        matrix[i, j] = stats["avg_coop_mean"] * 100
        matrices[model_name] = matrix

    # All values sit near the ceiling; a 0-100 scale would paint every cell
    # the same green. Start the scale just below the lowest observed value.
    vmin = np.floor(min(m.min() for m in matrices.values()))
    vmax = 100
    # Middle of the Greens ramp: light green at the minimum, without the
    # near-white and near-black ends, so the narrow scale does not overstate
    # the differences.
    cmap = LinearSegmentedColormap.from_list(
        "greens_mid", plt.cm.Greens(np.linspace(0.3, 0.8, 256)))

    for idx, (model_name, matrix) in enumerate(matrices.items()):
        ax = axes[idx]
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        # Add text annotations (white on dark cells, black on light ones);
        # a thin white halo keeps the threshold symbol readable on any green
        for i in range(3):
            for j in range(3):
                bc = bcs[j]
                k = ks[i]
                symbol, symbol_color = THRESHOLD_SYMBOLS[
                    "above" if bc > k else ("boundary" if bc == k else "below")]
                r, g, b, _ = im.cmap(im.norm(matrix[i, j]))
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                ax.text(j, i - 0.14, f"{matrix[i,j]:.0f}%",
                       ha="center", va="center", fontsize=16, fontweight="bold",
                       color="white" if luminance < 0.42 else "black")
                ax.text(j, i + 0.24, symbol, ha="center", va="center",
                       fontsize=24, fontweight="bold", color=symbol_color,
                       path_effects=[patheffects.withStroke(linewidth=3,
                                                            foreground="white")])

        ax.set_xticks(range(3))
        ax.set_xticklabels([f"b/c={int(bc)}" for bc in bcs])
        ax.set_yticks(range(3))
        ax.set_yticklabels([f"k={k}" for k in ks])
        ax.tick_params(labelsize=14)
        ax.set_title(model_name, fontsize=16, fontweight="bold")

    key = [plt.Line2D([], [], linestyle="none", marker=f"${sym}$",
                      markersize=14, color=color, label=label)
           for (sym, color), label in [
               (THRESHOLD_SYMBOLS["above"], "b/c > k: theory predicts more cooperation"),
               (THRESHOLD_SYMBOLS["boundary"], "b/c = k: boundary"),
               (THRESHOLD_SYMBOLS["below"], "b/c < k: theory predicts less cooperation")]]
    fig.legend(handles=key, loc="upper center", ncol=3, frameon=False, fontsize=14)
    plt.tight_layout(rect=(0, 0, 0.93, 0.9))
    cax = fig.add_axes([0.945, 0.13, 0.012, 0.66])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Mean cooperation (%)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    plt.savefig(output_dir / "heatmaps.png", dpi=150)
    plt.close()
    print(f"  Saved: heatmaps.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run comprehensive Experiment B analysis."""
    print("=" * 80)
    print("EXPERIMENT B: COMPREHENSIVE ANALYSIS")
    print("Modern Models vs Han et al. (2024) GPT-3.5 Baseline")
    print("=" * 80)

    # Find results directory
    script_dir = Path(__file__).parent.parent
    results_dir = script_dir / "results" / "experiment_b"

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return

    # Load all model data
    print(f"\nLoading data from: {results_dir}")
    all_models = load_model_data(results_dir)

    if not all_models:
        print("ERROR: No model data found!")
        return

    # Print data summary
    print_data_summary(all_models)

    # Compute summaries
    print("\nComputing statistics...")
    summaries = {}
    for model_name, model_data in all_models.items():
        summaries[model_name] = compute_model_summary(model_data)

    # Print analyses
    print_model_comparison_table(summaries)
    print_threshold_analysis(summaries)
    print_condition_details(summaries)
    print_endgame_analysis(summaries)
    print_oscillation_summary(summaries)

    # Create visualizations
    print("\nGenerating visualizations...")
    output_dir = results_dir
    create_model_comparison_plot(summaries, output_dir)
    create_threshold_comparison_plot(summaries, output_dir)
    create_time_series_plot(summaries, output_dir)
    create_heatmaps(summaries, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return summaries


if __name__ == "__main__":
    main()

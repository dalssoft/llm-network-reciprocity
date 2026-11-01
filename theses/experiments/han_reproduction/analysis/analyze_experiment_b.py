"""
Deep Dive Analysis: Experiment B (Setting 2) - Parameter Sweep

Compares our GPT-3.5 results with Han et al. (2024) Figures 4 & 5.

Key Han findings for GPT-3.5:
- Cooperation ~50% regardless of parameters (insensitive to b/c and k)
- Period-two oscillations in cooperation levels
- No response to theoretical threshold b/c > k

This script:
1. Loads all 9 conditions from Experiment B
2. Computes cooperation time series and final levels
3. Generates comparison plots matching Han's Figures 4 & 5
4. Analyzes oscillation patterns
5. Tests sensitivity to b/c > k threshold
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

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


def load_experiment_b_results(results_dir: str) -> Dict:
    """Load all Experiment B results, using most recent run for each condition."""
    results_path = Path(results_dir)

    # Group directories by condition name
    condition_dirs = defaultdict(list)
    for d in results_path.iterdir():
        if d.is_dir() and d.name.startswith("bc"):
            # Parse condition name (e.g., "bc4.0_k6_20251222_154407" -> "bc4.0_k6")
            parts = d.name.split("_")
            condition = f"{parts[0]}_{parts[1]}"
            condition_dirs[condition].append(d)

    # Use most recent (by timestamp) for each condition
    all_results = {}
    for condition, dirs in condition_dirs.items():
        latest = max(dirs, key=lambda x: x.name)

        results_file = latest / "results.json"
        config_file = latest / "config.json"

        if results_file.exists() and config_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            with open(config_file) as f:
                config = json.load(f)

            # Skip empty results (failed runs)
            if results:
                all_results[condition] = {
                    "results": results,
                    "config": config,
                    "dir": str(latest),
                }

    return all_results


def compute_cooperation_time_series(
    results: List[Dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std of cooperation rate over rounds across replications."""
    # Each replication has a list of coop_rates (one per round)
    all_coop_rates = []
    for rep in results:
        if "coop_rates" in rep:
            all_coop_rates.append(rep["coop_rates"])

    if not all_coop_rates:
        return np.array([]), np.array([])

    # Convert to numpy array: shape (num_reps, num_rounds)
    coop_matrix = np.array(all_coop_rates)

    # Compute mean and std across replications
    mean = np.mean(coop_matrix, axis=0)
    std = np.std(coop_matrix, axis=0)

    return mean, std


def compute_final_cooperation(results: List[Dict]) -> Tuple[float, float]:
    """Compute mean and std of final round cooperation rate."""
    final_rates = []
    for rep in results:
        if "coop_rates" in rep and rep["coop_rates"]:
            final_rates.append(rep["coop_rates"][-1])

    if not final_rates:
        return 0.0, 0.0

    return np.mean(final_rates), np.std(final_rates)


def compute_average_cooperation(results: List[Dict]) -> Tuple[float, float]:
    """Compute mean and std of average cooperation rate across all rounds."""
    avg_rates = []
    for rep in results:
        if "coop_rates" in rep and rep["coop_rates"]:
            avg_rates.append(np.mean(rep["coop_rates"]))

    if not avg_rates:
        return 0.0, 0.0

    return np.mean(avg_rates), np.std(avg_rates)


def detect_oscillation(coop_rates: np.ndarray, threshold: float = 0.1) -> Dict:
    """
    Detect period-2 oscillation pattern (characteristic of GPT-3.5 in Han).

    Returns dict with:
    - has_oscillation: bool
    - amplitude: mean difference between consecutive rounds
    - autocorr_lag1: autocorrelation at lag 1 (negative suggests oscillation)
    """
    if len(coop_rates) < 4:
        return {"has_oscillation": False, "amplitude": 0.0, "autocorr_lag1": 0.0}

    # Compute differences between consecutive rounds
    diffs = np.diff(coop_rates)

    # Check for sign alternation (period-2 pattern)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    max_sign_changes = len(diffs) - 1
    alternation_ratio = sign_changes / max_sign_changes if max_sign_changes > 0 else 0

    # Compute autocorrelation at lag 1
    mean_centered = coop_rates - np.mean(coop_rates)
    if np.std(coop_rates) > 0.01:
        autocorr_lag1 = np.corrcoef(mean_centered[:-1], mean_centered[1:])[0, 1]
    else:
        autocorr_lag1 = 0.0

    # Amplitude of oscillation
    amplitude = np.mean(np.abs(diffs))

    return {
        "has_oscillation": alternation_ratio > 0.6 and amplitude > threshold,
        "amplitude": amplitude,
        "autocorr_lag1": autocorr_lag1,
        "alternation_ratio": alternation_ratio,
    }


def categorize_condition(bc_ratio: float, k: int) -> str:
    """Categorize condition by theoretical prediction."""
    if bc_ratio > k:
        return "b/c > k (cooperation favored)"
    elif bc_ratio == k:
        return "b/c = k (boundary)"
    else:
        return "b/c < k (defection favored)"


def print_summary_table(all_results: Dict):
    """Print a summary table of all conditions."""
    print("\n" + "=" * 80)
    print("EXPERIMENT B SUMMARY: GPT-3.5 Parameter Sweep (Setting 2)")
    print("=" * 80)
    print(
        f"{'Condition':<12} {'k':<4} {'b/c':<6} {'Theory':<25} {'Avg Coop':<12} {'Final Coop':<12}"
    )
    print("-" * 80)

    # Sort conditions by k, then by b/c
    sorted_conditions = sorted(
        all_results.keys(),
        key=lambda x: (int(x.split("_")[1][1:]), float(x.split("_")[0][2:])),
    )

    for condition in sorted_conditions:
        data = all_results[condition]
        config = data["config"]
        results = data["results"]

        k = config["k"]
        bc_ratio = config["bc_ratio"]
        theory = categorize_condition(bc_ratio, k)

        avg_coop, avg_std = compute_average_cooperation(results)
        final_coop, final_std = compute_final_cooperation(results)

        print(
            f"{condition:<12} {k:<4} {bc_ratio:<6.1f} {theory:<25} "
            f"{avg_coop*100:>5.1f}% ± {avg_std*100:>4.1f}%  "
            f"{final_coop*100:>5.1f}% ± {final_std*100:>4.1f}%"
        )

    print("=" * 80)


def print_han_comparison(all_results: Dict):
    """Print detailed comparison with Han et al. findings."""
    print("\n" + "=" * 80)
    print("COMPARISON WITH HAN ET AL. (2024) - GPT-3.5 FINDINGS")
    print("=" * 80)

    # Compute overall statistics
    all_avg_coops = []
    all_final_coops = []
    oscillation_detected = 0
    total_conditions = 0

    for condition, data in all_results.items():
        results = data["results"]
        avg_coop, _ = compute_average_cooperation(results)
        final_coop, _ = compute_final_cooperation(results)
        all_avg_coops.append(avg_coop)
        all_final_coops.append(final_coop)

        # Check oscillation
        mean_coop, _ = compute_cooperation_time_series(results)
        if len(mean_coop) > 0:
            osc = detect_oscillation(mean_coop)
            if osc["has_oscillation"]:
                oscillation_detected += 1
        total_conditions += 1

    # Han's key findings for GPT-3.5
    print("\n📊 HAN'S KEY FINDINGS FOR GPT-3.5:")
    print("-" * 60)
    print("1. Cooperation ~50% regardless of parameters")
    print("2. Insensitive to b/c > k threshold (unlike humans)")
    print("3. Period-two oscillations in cooperation levels")
    print("4. No adaptation to network structure benefits")

    print("\n📈 OUR REPRODUCTION RESULTS:")
    print("-" * 60)

    overall_avg = np.mean(all_avg_coops) * 100
    overall_std = np.std(all_avg_coops) * 100
    print(f"1. Overall cooperation: {overall_avg:.1f}% ± {overall_std:.1f}%")
    print(f"   Han expected: ~50%")
    print(f"   Match: {'✅ YES' if 45 <= overall_avg <= 60 else '❌ NO'}")

    # Check sensitivity to b/c > k
    print("\n2. Sensitivity to b/c > k threshold:")

    bc_gt_k_coops = []
    bc_le_k_coops = []

    for condition, data in all_results.items():
        config = data["config"]
        results = data["results"]
        k = config["k"]
        bc = config["bc_ratio"]
        avg_coop, _ = compute_average_cooperation(results)

        if bc > k:
            bc_gt_k_coops.append(avg_coop)
        else:
            bc_le_k_coops.append(avg_coop)

    if bc_gt_k_coops and bc_le_k_coops:
        bc_gt_k_mean = np.mean(bc_gt_k_coops) * 100
        bc_le_k_mean = np.mean(bc_le_k_coops) * 100
        diff = bc_gt_k_mean - bc_le_k_mean

        print(f"   b/c > k conditions: {bc_gt_k_mean:.1f}%")
        print(f"   b/c ≤ k conditions: {bc_le_k_mean:.1f}%")
        print(f"   Difference: {diff:+.1f}%")
        print(f"   Han expected: No significant difference (GPT-3.5 insensitive)")
        print(f"   Human expected: Large difference (~40-60% vs ~20-30%)")
        print(
            f"   Match with Han: {'✅ YES (insensitive)' if abs(diff) < 10 else '⚠️ PARTIAL (some sensitivity)'}"
        )

    print(f"\n3. Oscillation detection:")
    print(
        f"   Conditions with period-2 oscillation: {oscillation_detected}/{total_conditions}"
    )
    print(f"   Han expected: Period-two cyclicity common in GPT-3.5")

    # Analyze by k value
    print("\n4. Cooperation by network degree (k):")
    k_coops = defaultdict(list)
    for condition, data in all_results.items():
        k = data["config"]["k"]
        avg_coop, _ = compute_average_cooperation(data["results"])
        k_coops[k].append(avg_coop)

    for k in sorted(k_coops.keys()):
        mean_coop = np.mean(k_coops[k]) * 100
        print(f"   k={k}: {mean_coop:.1f}%")

    print(f"   Han expected: Little variation across k for GPT-3.5")

    print("\n" + "=" * 80)


def print_detailed_oscillation_analysis(all_results: Dict):
    """Analyze oscillation patterns in detail."""
    print("\n" + "=" * 80)
    print("OSCILLATION ANALYSIS (Period-2 Cycling)")
    print("=" * 80)
    print(
        f"{'Condition':<12} {'Autocorr(1)':<12} {'Amplitude':<12} {'Alt Ratio':<12} {'Oscillating'}"
    )
    print("-" * 80)

    sorted_conditions = sorted(
        all_results.keys(),
        key=lambda x: (int(x.split("_")[1][1:]), float(x.split("_")[0][2:])),
    )

    for condition in sorted_conditions:
        data = all_results[condition]
        results = data["results"]
        mean_coop, _ = compute_cooperation_time_series(results)

        if len(mean_coop) > 0:
            osc = detect_oscillation(mean_coop)
            status = "✅ Yes" if osc["has_oscillation"] else "❌ No"
            print(
                f"{condition:<12} {osc['autocorr_lag1']:>10.3f}  {osc['amplitude']:>10.3f}  "
                f"{osc['alternation_ratio']:>10.3f}  {status}"
            )

    print("-" * 80)
    print("Note: Negative autocorrelation at lag 1 suggests period-2 oscillation")
    print(
        "      High alternation ratio (>0.6) with amplitude >0.1 indicates oscillation"
    )
    print("=" * 80)


def create_figure_4_style_plot(all_results: Dict, output_dir: str):
    """
    Create plots matching Han's Figure 4 style.

    Han's Figure 4: Average cooperation levels for GPT-3.5 (panels A, B, C)
    - Panel A: k=2 with b/c ∈ {2, 4, 6}
    - Panel B: k=4 with b/c ∈ {2, 4, 6}
    - Panel C: k=6 with b/c ∈ {2, 4, 6}
    """
    if not HAS_MATPLOTLIB:
        print("Skipping Figure 4 plot (matplotlib not available)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = {2.0: "#2ecc71", 4.0: "#3498db", 6.0: "#e74c3c"}  # b/c ratio colors

    for idx, k in enumerate([2, 4, 6]):
        ax = axes[idx]

        for bc in [2.0, 4.0, 6.0]:
            condition = f"bc{bc}_k{k}"
            if condition in all_results:
                data = all_results[condition]
                mean_coop, std_coop = compute_cooperation_time_series(data["results"])

                if len(mean_coop) > 0:
                    rounds = np.arange(1, len(mean_coop) + 1)

                    # Plot mean with confidence band
                    ax.plot(
                        rounds,
                        mean_coop * 100,
                        color=colors[bc],
                        linewidth=2,
                        label=f"b/c={int(bc)}",
                    )
                    ax.fill_between(
                        rounds,
                        (mean_coop - std_coop) * 100,
                        (mean_coop + std_coop) * 100,
                        color=colors[bc],
                        alpha=0.2,
                    )

        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Cooperation Level (%)", fontsize=12)
        ax.set_title(f"k = {k}", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50%")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Add theory annotation
        if k == 2:
            ax.text(
                0.02,
                0.98,
                "A",
                transform=ax.transAxes,
                fontsize=16,
                fontweight="bold",
                va="top",
            )
        elif k == 4:
            ax.text(
                0.02,
                0.98,
                "B",
                transform=ax.transAxes,
                fontsize=16,
                fontweight="bold",
                va="top",
            )
        else:
            ax.text(
                0.02,
                0.98,
                "C",
                transform=ax.transAxes,
                fontsize=16,
                fontweight="bold",
                va="top",
            )

    plt.suptitle(
        "Experiment B: Cooperation over Rounds (GPT-3.5)\nReproduction of Han et al. (2024) Figure 4",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = Path(output_dir) / "figure_4_reproduction.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Figure 4 style plot saved to: {output_path}")
    plt.close()


def create_figure_5_style_plot(all_results: Dict, output_dir: str):
    """
    Create bar chart matching Han's Figure 5 style.

    Han's Figure 5: Final cooperation level after 15 rounds
    - Grouped by theoretical category (b/c < k, b/c = k, b/c > k)
    - Bars labeled with "b/c<k" format
    """
    if not HAS_MATPLOTLIB:
        print("Skipping Figure 5 plot (matplotlib not available)")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Define categories and colors
    categories = {
        "b/c < k": {"color": "#e74c3c", "conditions": []},  # Red - defection favored
        "b/c = k": {"color": "#f39c12", "conditions": []},  # Orange - boundary
        "b/c > k": {
            "color": "#2ecc71",
            "conditions": [],
        },  # Green - cooperation favored
    }

    # Categorize conditions
    for condition, data in all_results.items():
        config = data["config"]
        bc = config["bc_ratio"]
        k = config["k"]

        final_coop, final_std = compute_final_cooperation(data["results"])
        label = (
            f"{int(bc)}<{k}"
            if bc < k
            else (f"{int(bc)}={k}" if bc == k else f"{int(bc)}>{k}")
        )

        entry = {
            "condition": condition,
            "label": label,
            "bc": bc,
            "k": k,
            "final_coop": final_coop,
            "final_std": final_std,
        }

        if bc < k:
            categories["b/c < k"]["conditions"].append(entry)
        elif bc == k:
            categories["b/c = k"]["conditions"].append(entry)
        else:
            categories["b/c > k"]["conditions"].append(entry)

    # Sort within each category
    for cat in categories.values():
        cat["conditions"].sort(key=lambda x: (x["k"], x["bc"]))

    # Plot bars
    x_pos = 0
    x_ticks = []
    x_labels = []

    for cat_name, cat_data in categories.items():
        for entry in cat_data["conditions"]:
            ax.bar(
                x_pos,
                entry["final_coop"] * 100,
                color=cat_data["color"],
                edgecolor="black",
                linewidth=1,
                yerr=entry["final_std"] * 100,
                capsize=3,
            )
            x_ticks.append(x_pos)
            x_labels.append(entry["label"])
            x_pos += 1
        x_pos += 0.5  # Gap between categories

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("Final Cooperation Level (%)", fontsize=12)
    ax.set_xlabel("Condition (b/c < k format)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)

    # Add legend
    patches = [
        mpatches.Patch(color="#e74c3c", label="b/c < k (defection)"),
        mpatches.Patch(color="#f39c12", label="b/c = k (boundary)"),
        mpatches.Patch(color="#2ecc71", label="b/c > k (cooperation)"),
    ]
    ax.legend(handles=patches, loc="upper right")

    ax.set_title(
        "Experiment B: Final Cooperation by Condition (GPT-3.5)\nReproduction of Han et al. (2024) Figure 5",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_dir) / "figure_5_reproduction.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Figure 5 style plot saved to: {output_path}")
    plt.close()


def create_heatmap(all_results: Dict, output_dir: str):
    """Create a heatmap of cooperation levels by k and b/c."""
    if not HAS_MATPLOTLIB:
        print("Skipping heatmap (matplotlib not available)")
        return

    # Create 3x3 matrix
    ks = [2, 4, 6]
    bcs = [2.0, 4.0, 6.0]

    matrix = np.zeros((3, 3))

    for i, k in enumerate(ks):
        for j, bc in enumerate(bcs):
            condition = f"bc{bc}_k{k}"
            if condition in all_results:
                avg_coop, _ = compute_average_cooperation(
                    all_results[condition]["results"]
                )
                matrix[i, j] = avg_coop * 100

    fig, ax = plt.subplots(figsize=(8, 6))

    # Same style as the modern-model heatmaps (Figure 10): middle of the
    # Greens ramp on a narrowed scale around the observed ~55%, so the small
    # differences between conditions show without being overstated.
    cmap = LinearSegmentedColormap.from_list(
        "greens_mid", plt.cm.Greens(np.linspace(0.3, 0.8, 256)))
    im = ax.imshow(matrix, cmap=cmap, vmin=50, vmax=60, aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Mean cooperation (%)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Threshold symbols: blue/orange stay distinct under color-vision
    # deficiency; a white halo keeps them readable on any green cell.
    symbols = {
        "above": ("+", "#1f5fa8"),
        "boundary": ("=", "#555555"),
        "below": ("\u2212", "#c05a00"),
    }

    # Add text annotations
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i - 0.1,
                f"{matrix[i, j]:.1f}%",
                ha="center",
                va="center",
                color="black",
                fontsize=16,
                fontweight="bold",
            )

            bc = bcs[j]
            k = ks[i]
            symbol, symbol_color = symbols[
                "above" if bc > k else ("boundary" if bc == k else "below")
            ]
            ax.text(
                j,
                i + 0.2,
                symbol,
                ha="center",
                va="center",
                color=symbol_color,
                fontsize=22,
                fontweight="bold",
                path_effects=[patheffects.withStroke(linewidth=3, foreground="white")],
            )

    ax.set_xticks(range(3))
    ax.set_xticklabels([f"b/c={int(bc)}" for bc in bcs])
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"k={k}" for k in ks])
    ax.tick_params(labelsize=13)
    ax.set_xlabel("Benefit-to-Cost Ratio (b/c)", fontsize=14)
    ax.set_ylabel("Network Degree (k)", fontsize=14)

    key = [
        plt.Line2D([], [], linestyle="none", marker=f"${sym}$", markersize=13,
                   color=color, label=label)
        for (sym, color), label in [
            (symbols["above"], "b/c > k: theory predicts more cooperation"),
            (symbols["boundary"], "b/c = k: boundary"),
            (symbols["below"], "b/c < k: theory predicts less cooperation"),
        ]
    ]
    ax.legend(handles=key, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=1, frameon=False, fontsize=13)

    plt.tight_layout()

    output_path = Path(output_dir) / "cooperation_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Heatmap saved to: {output_path}")
    plt.close()


def main():
    """Run the full Experiment B analysis."""
    # Find results directory
    script_dir = Path(__file__).parent.parent
    results_dir = script_dir / "results" / "experiment_b"

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return

    print("=" * 80)
    print("EXPERIMENT B DEEP DIVE ANALYSIS")
    print("Han et al. (2024) Reproduction - Setting 2 Parameter Sweep")
    print("=" * 80)

    # Load results
    print(f"\n📂 Loading results from: {results_dir}")
    all_results = load_experiment_b_results(str(results_dir))
    print(f"✅ Loaded {len(all_results)} conditions")

    if len(all_results) < 9:
        print(f"⚠️  Warning: Expected 9 conditions, found {len(all_results)}")

    # Print analyses
    print_summary_table(all_results)
    print_han_comparison(all_results)
    print_detailed_oscillation_analysis(all_results)

    # Create plots
    output_dir = results_dir
    create_figure_4_style_plot(all_results, str(output_dir))
    create_figure_5_style_plot(all_results, str(output_dir))
    create_heatmap(all_results, str(output_dir))

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

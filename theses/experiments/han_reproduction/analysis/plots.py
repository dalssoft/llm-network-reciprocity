"""
Analysis and plotting module for Han et al. (2024) reproduction.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results(results_path: str):
    with open(results_path, "r") as f:
        return json.load(f)


def calculate_cooperation_rate(results):
    """Calculate cooperation rate per round across replications."""
    all_round_data = []
    
    for rep_data in results:
        rep_id = rep_data["replication"]
        for round_data in rep_data["history"]:
            round_num = round_data["round"]
            actions = [r["action"] for r in round_data["results"]]
            coop_rate = sum(1 for a in actions if a == "COOPERATE") / len(actions)
            all_round_data.append({
                "replication": rep_id,
                "round": round_num,
                "coop_rate": coop_rate
            })
            
    return pd.DataFrame(all_round_data)


def plot_experiment_a(well_mixed_path: str, structured_path: str, output_path: str):
    """Replicate Figure 3: Cooperation time series."""
    wm_data = load_results(well_mixed_path)
    st_data = load_results(structured_path)
    
    wm_df = calculate_cooperation_rate(wm_data)
    wm_df["condition"] = "Well-Mixed"
    
    st_df = calculate_cooperation_rate(st_data)
    st_df["condition"] = "Structured (Ring k=2)"
    
    df = pd.concat([wm_df, st_df])
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="round", y="coop_rate", hue="condition", marker="o")
    plt.title("Cooperation Levels: Well-Mixed vs. Structured (Experiment A)")
    plt.xlabel("Round")
    plt.ylabel("Cooperation Rate")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    # Example usage:
    # plot_experiment_a("path/to/wm/results.json", "path/to/st/results.json", "experiment_a_comparison.png")
    pass


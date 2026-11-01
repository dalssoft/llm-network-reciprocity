"""
Contagion analysis for probabilistic horizon experiment.

Analyzes the spatial pattern of cooperation breakdown in uninformed conditions.
Tests whether defection spreads through network neighbors (contagion) vs random.

Claims tested:
  1. Defection originates from a single seed agent
  2. New defectors are neighbors of existing defectors (network contagion)
  3. Defection set remains a contiguous arc on the ring lattice
  4. Spread rate is consistent (~1-2 agents per front per round)
  5. The pattern differs from random (non-spatial) defection

Verification approaches:
  A. Direct neighbor check: fraction of new defectors adjacent to prior defectors
  B. Contiguity test: are defectors a single connected component on the ring?
  C. Null model comparison: simulate random defection matching observed rates,
     measure how often it would produce the same contiguity
  D. Wavefront speed: measure agents gained per round per front

Usage:
    python -m theses.experiments.probabilistic_horizon.analysis.analyze_contagion
    python -m theses.experiments.probabilistic_horizon.analysis.analyze_contagion --model gemini-3-flash-preview_reasoning
"""

import json
import argparse
import random
import statistics
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plots will be skipped.")


NUM_AGENTS = 25


def ring_neighbors(agent_id: int, k: int = 2) -> Set[int]:
    """Get neighbors on a k-regular ring lattice with NUM_AGENTS nodes."""
    half_k = k // 2
    return {(agent_id + d) % NUM_AGENTS for d in range(-half_k, half_k + 1) if d != 0}


def ring_distance(a: int, b: int) -> int:
    """Shortest distance between two agents on a ring of NUM_AGENTS."""
    diff = abs(a - b)
    return min(diff, NUM_AGENTS - diff)


def count_contiguous_arcs(defectors: Set[int]) -> int:
    """
    Count the number of contiguous arcs formed by defectors on the ring.
    A single connected cluster = 1 arc. Two separated groups = 2 arcs. Etc.
    Returns 0 if no defectors.
    """
    if not defectors:
        return 0
    if len(defectors) == NUM_AGENTS:
        return 1

    # Walk around the ring and count transitions from non-defector to defector
    arcs = 0
    for i in range(NUM_AGENTS):
        if i in defectors and (i - 1) % NUM_AGENTS not in defectors:
            arcs += 1
    return arcs


def arc_span(defectors: Set[int]) -> Optional[int]:
    """
    If defectors form a single contiguous arc, return its length (span).
    Otherwise return None.
    """
    if not defectors:
        return 0
    if count_contiguous_arcs(defectors) != 1:
        return None
    return len(defectors)


@dataclass
class RoundTransition:
    """Analysis of the defection transition between two consecutive rounds."""
    round_num: int
    prev_defectors: Set[int]
    curr_defectors: Set[int]
    new_defectors: Set[int]  # agents who switched C -> D
    recovered: Set[int]  # agents who switched D -> C
    new_defectors_adjacent_to_prior: int  # how many new D are neighbors of prior D
    total_new_defectors: int
    fraction_adjacent: float  # new_defectors_adjacent / total_new (1.0 = pure contagion)
    contiguous_arcs: int  # number of separate defection clusters


@dataclass
class ReplicationContagion:
    """Contagion analysis for a single replication."""
    replication_id: int
    actual_rounds: int
    has_degradation: bool  # whether cooperation degrades at all
    seed_defectors: Set[int]  # defectors in round 1
    transitions: List[RoundTransition]
    final_defectors: Set[int]
    # Summary metrics
    mean_fraction_adjacent: float  # avg fraction of new defectors adjacent to prior
    all_single_arc: bool  # are defectors always a single contiguous arc?
    max_arcs: int
    wavefront_speeds: List[float]  # new defectors per round


@dataclass
class NullModelResult:
    """Comparison against random (non-spatial) defection null model."""
    n_simulations: int
    # For each observed replication, how often does random defection produce single-arc
    observed_always_single_arc: bool
    null_always_single_arc_rate: float  # fraction of null sims that are always single-arc
    # Average contiguity of null model
    null_mean_arcs_per_round: float
    observed_mean_arcs_per_round: float


def analyze_replication_contagion(rep_data: dict, k: int = 2) -> ReplicationContagion:
    """Analyze contagion pattern for one replication."""
    history = rep_data["history"]
    actual_rounds = rep_data.get("actual_rounds", len(history))

    if not history:
        return ReplicationContagion(
            replication_id=rep_data["replication"],
            actual_rounds=0, has_degradation=False,
            seed_defectors=set(), transitions=[], final_defectors=set(),
            mean_fraction_adjacent=0.0, all_single_arc=True, max_arcs=0,
            wavefront_speeds=[],
        )

    # Extract defector sets per round
    defectors_by_round = []
    for round_data in history:
        defectors = {
            r["agent_id"] for r in round_data["results"] if r["action"] == "DEFECT"
        }
        defectors_by_round.append(defectors)

    seed_defectors = defectors_by_round[0]
    has_degradation = any(len(d) > 0 for d in defectors_by_round)

    # Analyze transitions
    transitions = []
    for i in range(1, len(defectors_by_round)):
        prev = defectors_by_round[i - 1]
        curr = defectors_by_round[i]
        new_d = curr - prev
        recovered = prev - curr

        # Check how many new defectors are adjacent to prior defectors
        adjacent_count = 0
        for agent in new_d:
            neighbors = ring_neighbors(agent, k)
            if neighbors & prev:  # at least one neighbor was a defector
                adjacent_count += 1

        total_new = len(new_d)
        frac = adjacent_count / total_new if total_new > 0 else 1.0

        transitions.append(RoundTransition(
            round_num=i + 1,
            prev_defectors=prev,
            curr_defectors=curr,
            new_defectors=new_d,
            recovered=recovered,
            new_defectors_adjacent_to_prior=adjacent_count,
            total_new_defectors=total_new,
            fraction_adjacent=frac,
            contiguous_arcs=count_contiguous_arcs(curr),
        ))

    # Summary metrics
    fracs = [t.fraction_adjacent for t in transitions if t.total_new_defectors > 0]
    mean_frac = statistics.mean(fracs) if fracs else 1.0

    arc_counts = [count_contiguous_arcs(d) for d in defectors_by_round if d]
    all_single = all(a <= 1 for a in arc_counts)
    max_arcs = max(arc_counts) if arc_counts else 0

    speeds = [t.total_new_defectors for t in transitions]

    return ReplicationContagion(
        replication_id=rep_data["replication"],
        actual_rounds=actual_rounds,
        has_degradation=has_degradation,
        seed_defectors=seed_defectors,
        transitions=transitions,
        final_defectors=defectors_by_round[-1],
        mean_fraction_adjacent=mean_frac,
        all_single_arc=all_single,
        max_arcs=max_arcs,
        wavefront_speeds=speeds,
    )


def run_null_model(
    observed_rates: List[float],
    n_simulations: int = 5000,
    seed: int = 42,
) -> NullModelResult:
    """
    Null model: for each round, defectors are chosen uniformly at random
    (matching the observed count), ignoring network structure.

    Compare contiguity of random defection vs observed.
    """
    rng = random.Random(seed)
    agents = list(range(NUM_AGENTS))

    null_always_single = 0
    null_arc_counts = []

    for _ in range(n_simulations):
        sim_always_single = True
        sim_arc_total = 0
        sim_rounds = 0

        for rate in observed_rates:
            n_defectors = round(rate * NUM_AGENTS)
            if n_defectors == 0:
                continue
            random_defectors = set(rng.sample(agents, min(n_defectors, NUM_AGENTS)))
            arcs = count_contiguous_arcs(random_defectors)
            if arcs > 1:
                sim_always_single = False
            sim_arc_total += arcs
            sim_rounds += 1

        if sim_always_single:
            null_always_single += 1
        if sim_rounds > 0:
            null_arc_counts.append(sim_arc_total / sim_rounds)

    # Observed stats
    obs_arcs = [count_contiguous_arcs(
        set()  # placeholder — computed externally
    )]  # not used directly; we pass observed_always_single separately

    return NullModelResult(
        n_simulations=n_simulations,
        observed_always_single_arc=True,  # set externally
        null_always_single_arc_rate=null_always_single / n_simulations,
        null_mean_arcs_per_round=statistics.mean(null_arc_counts) if null_arc_counts else 0,
        observed_mean_arcs_per_round=0,  # set externally
    )


def _extract_action_matrix(rep_data: dict) -> "np.ndarray":
    """Build a (num_agents x num_rounds) matrix: 1=cooperate, 0=defect."""
    history = rep_data["history"]
    n_rounds = len(history)
    n_agents = len(history[0]["results"])
    matrix = np.ones((n_agents, n_rounds))
    for rnd_idx, round_data in enumerate(history):
        for r in round_data["results"]:
            if r["action"] == "DEFECT":
                matrix[r["agent_id"], rnd_idx] = 0
    return matrix


def _reorder_for_seed(matrix: "np.ndarray", seed_agent: int) -> "np.ndarray":
    """
    Reorder agent rows so the seed defector is in the middle.
    This centers the contagion wedge visually.
    """
    n = matrix.shape[0]
    order = [(seed_agent + i) % n for i in range(-(n // 2), n // 2 + 1)]
    # Remove duplicates keeping order (for even n)
    seen = set()
    unique_order = []
    for a in order:
        if a not in seen:
            seen.add(a)
            unique_order.append(a)
    return matrix[unique_order, :], unique_order


def create_spacetime_heatmaps(all_results: Dict[str, list], output_dir: str):
    """
    Create space-time heatmaps showing defection contagion on the ring.

    One panel per degrading replication. Agents on Y-axis (reordered so seed
    is centered), rounds on X-axis, colored by C (green) / D (red).
    """
    if not HAS_MATPLOTLIB:
        print("Skipping space-time heatmaps (matplotlib not available)")
        return

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#e74c3c", "#2ecc71"])  # 0=defect(red), 1=cooperate(green)

    # Collect degrading replications across all uninformed conditions
    degrading_reps = []
    for condition_name, results in sorted(all_results.items()):
        if "uninformed" not in condition_name:
            continue
        for rep_data in results:
            avg_coop = statistics.mean(rep_data["coop_rates"]) if rep_data["coop_rates"] else 1.0
            if avg_coop < 0.99 and rep_data["actual_rounds"] > 2:
                # Find seed defector
                first_round = rep_data["history"][0]["results"]
                seeds = [r["agent_id"] for r in first_round if r["action"] == "DEFECT"]
                seed = seeds[0] if seeds else 0
                degrading_reps.append((condition_name, rep_data, seed))

    if not degrading_reps:
        print("No degrading replications with >2 rounds to plot.")
        return

    n_panels = len(degrading_reps)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    for idx, (cond_name, rep_data, seed_agent) in enumerate(degrading_reps):
        ax = axes[idx]
        matrix = _extract_action_matrix(rep_data)
        reordered, agent_order = _reorder_for_seed(matrix, seed_agent)

        ax.imshow(reordered, cmap=cmap, aspect="auto", vmin=0, vmax=1,
                  interpolation="nearest")

        ax.set_xlabel("Round", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Agent (centered on seed)", fontsize=12)

        ax.set_xticks(range(0, matrix.shape[1], max(1, matrix.shape[1] // 10)))
        ax.set_xticklabels(range(1, matrix.shape[1] + 1, max(1, matrix.shape[1] // 10)))

        # Show a few agent labels
        n_agents = len(agent_order)
        tick_step = max(1, n_agents // 5)
        ytick_pos = list(range(0, n_agents, tick_step))
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels([agent_order[i] for i in ytick_pos])

        delta = cond_name.split("_")[0].replace("delta", "d=")
        rep_id = rep_data["replication"]
        n_rounds = rep_data["actual_rounds"]
        final_coop = rep_data["coop_rates"][-1]
        ax.set_title(f"{delta}, Rep {rep_id}\n{n_rounds}R, final={final_coop:.0%}",
                     fontsize=11, fontweight="bold")

        # Panel letter
        ax.text(0.02, 0.98, chr(65 + idx), transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top", color="white")

    # Legend
    patches = [
        mpatches.Patch(color="#2ecc71", label="Cooperate"),
        mpatches.Patch(color="#e74c3c", label="Defect"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Defection Contagion: Space-Time Heatmaps (Uninformed Conditions)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = Path(output_dir) / "contagion_spacetime.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSpace-time heatmap saved to: {output_path}")
    plt.close()


def create_cooperation_trajectories(all_results: Dict[str, list], output_dir: str):
    """
    Create cooperation trajectory line plots.

    One panel per delta value. Each replication is a thin line, colored by
    condition (informed=green, uninformed=red/orange).
    """
    if not HAS_MATPLOTLIB:
        print("Skipping cooperation trajectories (matplotlib not available)")
        return

    # Group by delta
    by_delta: Dict[str, Dict[str, list]] = {}
    for cond_name, results in all_results.items():
        delta_str = cond_name.split("_")[0]  # e.g. "delta0.9"
        info = cond_name.split("_")[1]  # "informed" or "uninformed"
        by_delta.setdefault(delta_str, {}).setdefault(info, []).extend(results)

    deltas = sorted(by_delta.keys())
    n_panels = len(deltas)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    colors = {"informed": "#2ecc71", "uninformed": "#e74c3c"}

    for idx, delta_str in enumerate(deltas):
        ax = axes[idx]
        conditions = by_delta[delta_str]

        for info in ["informed", "uninformed"]:
            if info not in conditions:
                continue
            reps = conditions[info]
            for i, rep in enumerate(reps):
                rates = rep["coop_rates"]
                rounds = np.arange(1, len(rates) + 1)
                ax.plot(rounds, np.array(rates) * 100, color=colors[info],
                        alpha=0.35, linewidth=1.0,
                        label=f"{info} (individual)" if i == 0 else None)

            # Plot mean as thicker line
            max_len = max(len(r["coop_rates"]) for r in reps)

            mean_rates = []
            for rnd_idx in range(max_len):
                vals = [r["coop_rates"][rnd_idx] for r in reps
                        if rnd_idx < len(r["coop_rates"])]
                if vals:
                    mean_rates.append(statistics.mean(vals))
            if mean_rates:
                ax.plot(np.arange(1, len(mean_rates) + 1), np.array(mean_rates) * 100,
                        color=colors[info], linewidth=2.5,
                        label=f"{info} (mean, n={len(reps)})")

        ax.set_xlabel("Round", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Cooperation Level (%)", fontsize=12)
        ax.set_ylim(-2, 102)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3)

        delta_val = delta_str.replace("delta", "")
        ax.set_title(f"delta = {delta_val}", fontsize=14, fontweight="bold")

        ax.text(0.02, 0.98, chr(65 + idx), transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top")

    plt.suptitle("Cooperation Trajectories: Informed vs Uninformed",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = Path(output_dir) / "cooperation_trajectories.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nCooperation trajectories saved to: {output_path}")
    plt.close()


def create_null_model_comparison(all_results: Dict[str, list], output_dir: str, k: int = 2):
    """
    Create bar chart comparing observed vs null model arc counts.

    For each degrading replication, shows observed mean arcs/round vs the
    null model distribution (random defector placement).
    """
    if not HAS_MATPLOTLIB:
        print("Skipping null model comparison (matplotlib not available)")
        return

    # Collect degrading reps with their null model results
    entries = []
    for cond_name, results in sorted(all_results.items()):
        if "uninformed" not in cond_name:
            continue
        for rep_data in results:
            avg_coop = statistics.mean(rep_data["coop_rates"]) if rep_data["coop_rates"] else 1.0
            if avg_coop < 0.99 and rep_data["actual_rounds"] > 2:
                # Compute observed arc counts
                defectors_by_round = []
                for round_data in rep_data["history"]:
                    defectors = {
                        r["agent_id"] for r in round_data["results"]
                        if r["action"] == "DEFECT"
                    }
                    defectors_by_round.append(defectors)

                obs_arcs = [count_contiguous_arcs(d) for d in defectors_by_round if d]
                obs_mean = statistics.mean(obs_arcs) if obs_arcs else 0

                # Run null model
                defection_rates = [1.0 - c for c in rep_data["coop_rates"]]
                null = run_null_model(defection_rates, n_simulations=5000)

                delta_str = cond_name.split("_")[0].replace("delta", "d=")
                label = f"{delta_str}\nRep {rep_data['replication']}"
                entries.append({
                    "label": label,
                    "obs_mean_arcs": obs_mean,
                    "null_mean_arcs": null.null_mean_arcs_per_round,
                    "n_rounds": rep_data["actual_rounds"],
                })

    if not entries:
        print("No degrading replications with >2 rounds for null model plot.")
        return

    n = len(entries)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, 3 * n), 5))

    bars_obs = ax.bar(x - width / 2, [e["obs_mean_arcs"] for e in entries],
                      width, color="#3498db", edgecolor="black", linewidth=1,
                      label="Observed")
    bars_null = ax.bar(x + width / 2, [e["null_mean_arcs"] for e in entries],
                       width, color="#e74c3c", edgecolor="black", linewidth=1,
                       label="Null model (random)")

    ax.set_xticks(x)
    ax.set_xticklabels([e["label"] for e in entries], fontsize=10)
    ax.set_ylabel("Mean Defection Arcs per Round", fontsize=12)
    ax.set_title("Observed Contiguity vs Random Defection Null Model",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate bars with values
    for bar in bars_obs:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{h:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars_null:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{h:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Reference line at 1 arc (perfect contiguity)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Single arc")

    plt.tight_layout()

    output_path = Path(output_dir) / "null_model_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nNull model comparison saved to: {output_path}")
    plt.close()


def print_replication_detail(rep: ReplicationContagion):
    """Print detailed contagion analysis for one replication."""
    print(f"\n  Rep {rep.replication_id} ({rep.actual_rounds} rounds):")
    print(f"    Seed defectors (round 1): {sorted(rep.seed_defectors)}")
    print(f"    Always single arc: {rep.all_single_arc} (max arcs: {rep.max_arcs})")
    print(f"    Mean fraction of new defectors adjacent to prior: {rep.mean_fraction_adjacent:.1%}")

    if rep.transitions:
        print(f"    Wavefront speeds (new D/round): {rep.wavefront_speeds}")
        if rep.wavefront_speeds:
            nonzero = [s for s in rep.wavefront_speeds if s > 0]
            if nonzero:
                print(f"    Mean speed (nonzero rounds): {statistics.mean(nonzero):.1f} agents/round")

    # Round-by-round detail
    if rep.has_degradation and rep.actual_rounds > 1:
        print(f"    Round-by-round:")
        for t in rep.transitions:
            if t.total_new_defectors > 0 or t.recovered:
                adj_label = f"{t.new_defectors_adjacent_to_prior}/{t.total_new_defectors} adjacent" if t.total_new_defectors > 0 else ""
                rec_label = f", {len(t.recovered)} recovered" if t.recovered else ""
                print(f"      Round {t.round_num:2d}: +{t.total_new_defectors}D ({adj_label}{rec_label}), "
                      f"arcs={t.contiguous_arcs}")


def analyze_condition(
    results_dir: Path, k: int = 2, verbose: bool = True,
) -> Optional[Tuple[str, list]]:
    """Analyze contagion patterns for all replications in a condition.

    Returns (condition_name, raw_results) for use in plotting, or None.
    """
    results_file = results_dir / "results.json"
    config_file = results_dir / "config.json"

    if not results_file.exists():
        print(f"  No results.json in {results_dir}")
        return None

    with open(results_file) as f:
        results = json.load(f)
    with open(config_file) as f:
        config = json.load(f)

    condition = config["condition"]
    print(f"\n{'='*60}")
    print(f"Condition: {condition}")
    print(f"Delta: {config['continuation_delta']}, Info: {config['horizon_info']}")
    print(f"Network: {config['network_type']}, k={config['k']}, n={config['num_agents']}")
    print(f"{'='*60}")

    reps = [analyze_replication_contagion(r, k) for r in results]
    degrading = [r for r in reps if len(r.final_defectors) > 0 and r.actual_rounds > 1]

    print(f"\nTotal replications: {len(reps)}")
    print(f"Degrading replications (any defection, >1 round): {len(degrading)}")

    if not degrading:
        print("  No degrading replications to analyze.")
        return (condition, results)

    # ── Claim 1: Defection starts from a single seed ──
    print(f"\n{'─'*50}")
    print("Claim 1: Defection originates from a single seed agent")
    seed_sizes = [len(r.seed_defectors) for r in degrading]
    print(f"  Seed defector counts (round 1): {seed_sizes}")
    print(f"  All single-seed: {all(s == 1 for s in seed_sizes)}")

    # ── Claim 2: New defectors are neighbors of existing defectors ──
    print(f"\n{'─'*50}")
    print("Claim 2: New defectors are neighbors of prior defectors (contagion)")
    all_fracs = []
    total_new = 0
    total_adjacent = 0
    for r in degrading:
        for t in r.transitions:
            if t.total_new_defectors > 0:
                all_fracs.append(t.fraction_adjacent)
                total_new += t.total_new_defectors
                total_adjacent += t.new_defectors_adjacent_to_prior
    if all_fracs:
        print(f"  Total new defectors across all transitions: {total_new}")
        print(f"  Adjacent to prior defector: {total_adjacent} ({total_adjacent/total_new:.1%})")
        print(f"  Per-transition fraction (mean): {statistics.mean(all_fracs):.1%}")
        non_adjacent = total_new - total_adjacent
        if non_adjacent > 0:
            print(f"  Non-adjacent new defectors: {non_adjacent}")
            # Show which ones
            for r in degrading:
                for t in r.transitions:
                    for agent in t.new_defectors:
                        neighbors = ring_neighbors(agent, k)
                        if not (neighbors & t.prev_defectors):
                            print(f"    Rep {r.replication_id}, Round {t.round_num}: "
                                  f"agent {agent} (neighbors: {sorted(neighbors)}, "
                                  f"prev defectors: {sorted(t.prev_defectors)})")

    # ── Claim 3: Defectors form a contiguous arc on the ring ──
    print(f"\n{'─'*50}")
    print("Claim 3: Defectors always form a single contiguous arc on the ring")
    arc_violations = []
    for r in degrading:
        for i, t in enumerate(r.transitions):
            if t.contiguous_arcs > 1:
                arc_violations.append((r.replication_id, t.round_num, t.contiguous_arcs))
    all_single = all(r.all_single_arc for r in degrading)
    print(f"  All degrading reps always single-arc: {all_single}")
    if arc_violations:
        print(f"  Violations: {len(arc_violations)} round-transitions with >1 arc")
        for rep_id, rnd, arcs in arc_violations:
            print(f"    Rep {rep_id}, Round {rnd}: {arcs} arcs")
    else:
        print(f"  No violations: defection is always a single connected cluster")

    # ── Claim 4: Wavefront speed ──
    print(f"\n{'─'*50}")
    print("Claim 4: Consistent spread rate (~1-2 agents per front per round)")
    all_speeds = []
    for r in degrading:
        nonzero = [s for s in r.wavefront_speeds if s > 0]
        all_speeds.extend(nonzero)
    if all_speeds:
        print(f"  All nonzero speeds: {all_speeds}")
        print(f"  Mean: {statistics.mean(all_speeds):.1f}, Median: {statistics.median(all_speeds):.1f}")
        if len(all_speeds) >= 2:
            print(f"  Stdev: {statistics.stdev(all_speeds):.1f}")
        # On a ring with single seed, two fronts expand: expected ~2 agents/round
        print(f"  (Expected for 2 fronts on ring: ~2 agents/round)")

    # ── Claim 5: Null model comparison ──
    print(f"\n{'─'*50}")
    print("Claim 5: Pattern differs from random (non-spatial) defection")
    print("  Running null model (random defector placement matching observed counts)...")

    # Collect observed defection rates across all degrading reps
    for r in degrading:
        history_path = results_dir / "results.json"
        rep_data = [rd for rd in results if rd["replication"] == r.replication_id][0]
        observed_rates = rep_data["coop_rates"]
        defection_rates = [1.0 - c for c in observed_rates]

        null_result = run_null_model(defection_rates, n_simulations=5000)

        # Compute observed arc stats
        defectors_by_round = []
        for round_data in rep_data["history"]:
            defectors = {
                rd["agent_id"] for rd in round_data["results"] if rd["action"] == "DEFECT"
            }
            defectors_by_round.append(defectors)

        obs_arcs = [count_contiguous_arcs(d) for d in defectors_by_round if d]
        obs_mean_arcs = statistics.mean(obs_arcs) if obs_arcs else 0
        obs_always_single = all(a <= 1 for a in obs_arcs)

        print(f"\n  Rep {r.replication_id} ({r.actual_rounds} rounds, "
              f"final coop={observed_rates[-1]:.0%}):")
        print(f"    Observed: always single arc = {obs_always_single}, "
              f"mean arcs/round = {obs_mean_arcs:.2f}")
        print(f"    Null model (n={null_result.n_simulations}): "
              f"always single arc = {null_result.null_always_single_arc_rate:.1%} of sims, "
              f"mean arcs/round = {null_result.null_mean_arcs_per_round:.2f}")
        if obs_always_single:
            print(f"    → Observed contiguity is {'expected' if null_result.null_always_single_arc_rate > 0.5 else 'unlikely'} "
                  f"under random defection (p={null_result.null_always_single_arc_rate:.3f})")

    # ── Detailed per-replication output ──
    if verbose:
        print(f"\n{'─'*50}")
        print("Detailed per-replication analysis:")
        for r in degrading:
            print_replication_detail(r)

    return (condition, results)


def find_latest_results(experiment_dir: Path, condition_prefix: str) -> Optional[Path]:
    """Find the most recent results directory for a condition."""
    if not experiment_dir.exists():
        return None
    matching = [d for d in experiment_dir.iterdir()
                if d.is_dir() and d.name.startswith(condition_prefix)]
    if not matching:
        return None
    return max(matching, key=lambda d: d.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze defection contagion patterns in probabilistic horizon experiment"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model subdirectory (e.g., gemini-3-flash-preview_reasoning)",
    )
    parser.add_argument(
        "--condition", type=str, default=None,
        help="Specific condition prefix (e.g., delta0.9_uninformed)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Skip per-replication detail",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent / "results"

    # Find model directory
    if args.model:
        model_dir = base_dir / args.model
    else:
        models = [d.name for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if not models:
            print("No results found.")
            return
        if len(models) == 1:
            model_dir = base_dir / models[0]
        else:
            print(f"Multiple models found: {models}. Use --model to specify.")
            return

    if not model_dir.exists():
        print(f"No results: {model_dir}")
        return

    print(f"{'#'*60}")
    print("# Defection Contagion Analysis")
    print(f"# Model: {model_dir.name}")
    print(f"{'#'*60}")

    # Find condition directories
    condition_prefixes = set()
    run_dirs = [d for d in sorted(model_dir.iterdir()) if (d / "config.json").exists()]
    for d in run_dirs:
        parts = d.name.rsplit("_", 2)
        if len(parts) >= 3:
            prefix = parts[0] + "_" + parts[1] if not parts[1][0].isdigit() else parts[0]
        else:
            prefix = d.name
        condition_prefixes.add(prefix)

    config = json.load(open(run_dirs[0] / "config.json"))
    k = config.get("k", 2)

    all_results: Dict[str, list] = {}

    for prefix in sorted(condition_prefixes):
        if args.condition and not prefix.startswith(args.condition):
            continue
        latest = find_latest_results(model_dir, prefix)
        if latest:
            result = analyze_condition(latest, k=k, verbose=not args.quiet)
            if result:
                cond_name, raw_results = result
                all_results[cond_name] = raw_results

    # Generate plots
    if all_results:
        output_dir = str(model_dir)
        create_spacetime_heatmaps(all_results, output_dir)
        create_cooperation_trajectories(all_results, output_dir)
        create_null_model_comparison(all_results, output_dir, k=k)


if __name__ == "__main__":
    main()

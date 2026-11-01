"""
Activity A of the says-vs-does study: build the decision-level table.

Walks every results.json under the three experiment families and emits one row
per (run, replication, round, agent) decision, carrying exactly the information
the agent observed at decision time (own action at t-1, actions of its t-1
neighbors — the same content as the round-prompt feedback) plus condition
metadata from config.json.

Validation is built in: for every replication, per-round cooperation rates are
recomputed from the reconstructed decisions and asserted against the
coop_rates / final_coop_rate / avg_coop_rate already stored in results.json.
Any mismatch is a hard failure.

Usage:
    python -m theses.experiments.says_vs_does.build_decision_table
    python -m theses.experiments.says_vs_does.build_decision_table --out /path/table.csv.gz

Output:
    data/decision_table.csv.gz   one row per decision
    data/decision_table_meta.json  build stats + validation summary
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]

# Families scanned, in a stable order. v1 runs (superseded early batch of
# modern_models experiment A) are included but flagged results_version="v1"
# so downstream analyses can filter them out.
RESULT_ROOTS = [
    "han_reproduction/results",
    "modern_models/results",
    "probabilistic_horizon/results",
]

COOP, DEFECT = "COOPERATE", "DEFECT"
RATE_TOL = 1e-6


def classify_run(run_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Derive provenance fields from the run directory path and config."""
    rel = run_dir.relative_to(EXPERIMENTS_ROOT)
    parts = rel.parts
    family = parts[0]
    results_version = "v1" if "v1" in parts else "main"

    exp_name = config.get("experiment", "")
    if "Experiment A" in exp_name:
        experiment = "A"
    elif "Experiment B" in exp_name:
        experiment = "B"
    elif "Probabilistic Horizon" in exp_name:
        experiment = "horizon"
    else:
        raise ValueError(f"Unknown experiment label {exp_name!r} in {run_dir}")

    with_reasoning = bool(config.get("with_reasoning", False))
    model = config["model"]
    model_variant = f"{model}_reasoning" if with_reasoning else model

    # Timestamp suffix of the run dir name (condition_YYYYMMDD_HHMMSS)
    run_ts = "_".join(run_dir.name.rsplit("_", 2)[-2:])

    return {
        "family": family,
        "results_version": results_version,
        "experiment": experiment,
        "run_path": str(rel.parent / run_dir.name),
        "run_ts": run_ts,
        "condition": config["condition"],
        "model": model,
        "model_provider": config.get("model_provider", ""),
        "with_reasoning": with_reasoning,
        "model_variant": model_variant,
        "network_type": config["network_type"],
        "static_network": config["network_type"] != "well_mixed",
        "num_agents": config["num_agents"],
        "k": config["k"],
        "bc_ratio": config["bc_ratio"],
        "num_rounds_planned": config.get("num_rounds"),  # None for horizon
        "continuation_delta": config.get("continuation_delta"),
        "horizon_info": config.get("horizon_info", ""),
    }


def validate_replication(
    rep: Dict[str, Any],
    run_meta: Dict[str, Any],
    problems: List[str],
) -> None:
    """Hard consistency checks between history and the stored aggregates."""
    where = f"{run_meta['run_path']}#r{rep['replication']}"
    history = rep["history"]
    coop_rates = rep["coop_rates"]

    if len(coop_rates) != len(history):
        problems.append(f"{where}: len(coop_rates)={len(coop_rates)} != len(history)={len(history)}")
        return

    if "actual_rounds" in rep and rep["actual_rounds"] != len(history):
        problems.append(f"{where}: actual_rounds={rep['actual_rounds']} != len(history)={len(history)}")

    n_agents = run_meta["num_agents"]
    recomputed = []
    for t_idx, h in enumerate(history):
        if h["round"] != t_idx + 1:
            problems.append(f"{where}: round field {h['round']} at index {t_idx}")
        ids = sorted(r["agent_id"] for r in h["results"])
        if ids != list(range(n_agents)):
            problems.append(f"{where}: round {t_idx + 1} agent ids {ids[:5]}... != 0..{n_agents - 1}")
        rate = sum(r["action"] == COOP for r in h["results"]) / len(h["results"])
        recomputed.append(rate)
        if abs(rate - coop_rates[t_idx]) > RATE_TOL:
            problems.append(
                f"{where}: round {t_idx + 1} recomputed coop {rate:.4f} != stored {coop_rates[t_idx]:.4f}"
            )

    if abs(rep["final_coop_rate"] - recomputed[-1]) > RATE_TOL:
        problems.append(f"{where}: final_coop_rate {rep['final_coop_rate']} != recomputed {recomputed[-1]:.4f}")
    if abs(rep["avg_coop_rate"] - sum(recomputed) / len(recomputed)) > RATE_TOL:
        problems.append(f"{where}: avg_coop_rate {rep['avg_coop_rate']} != recomputed mean")


def rows_for_replication(rep: Dict[str, Any], run_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per (round, agent), with lagged observables from round t-1."""
    history = rep["history"]
    n_rounds_rep = len(history)
    num_rounds_planned = run_meta["num_rounds_planned"]
    rep_key = f"{run_meta['run_path']}#r{rep['replication']}"

    rows: List[Dict[str, Any]] = []
    prev_by_agent: Dict[int, Dict[str, Any]] = {}
    prev_actions: Dict[int, str] = {}

    for t_idx, h in enumerate(history):
        round_num = t_idx + 1
        cur_by_agent = {r["agent_id"]: r for r in h["results"]}

        for agent_id, res in sorted(cur_by_agent.items()):
            row: Dict[str, Any] = dict(run_meta)
            row.update(
                replication=rep["replication"],
                rep_key=rep_key,
                round=round_num,
                agent_id=agent_id,
                action="C" if res["action"] == COOP else "D",
                coop=int(res["action"] == COOP),
                payoff=res["payoff"],
                n_neighbors=len(res["neighbors"]),
                neighbor_ids=";".join(map(str, res["neighbors"])),
                n_rounds_rep=n_rounds_rep,
            )

            # Known-horizon experiments: agent was told total rounds upfront.
            if num_rounds_planned is not None:
                row["rounds_remaining"] = num_rounds_planned - round_num
                row["is_last_round"] = round_num == num_rounds_planned
            else:
                row["rounds_remaining"] = None
                row["is_last_round"] = None

            # Lagged observables = the feedback shown at decision time:
            # own action/payoff at t-1 and the actions of the agent's t-1
            # neighbors (in well-mixed these differ from the current ones).
            if t_idx > 0:
                prev_res = prev_by_agent[agent_id]
                prev_nbs = prev_res["neighbors"]
                n_coop = sum(prev_actions[j] == COOP for j in prev_nbs)
                row.update(
                    own_coop_prev=int(prev_res["action"] == COOP),
                    own_payoff_prev=prev_res["payoff"],
                    n_nb_prev=len(prev_nbs),
                    n_nb_coop_prev=n_coop,
                    frac_nb_coop_prev=n_coop / len(prev_nbs),
                )
            else:
                row.update(
                    own_coop_prev=None,
                    own_payoff_prev=None,
                    n_nb_prev=None,
                    n_nb_coop_prev=None,
                    frac_nb_coop_prev=None,
                )
            rows.append(row)

        prev_by_agent = cur_by_agent
        prev_actions = {aid: r["action"] for aid, r in cur_by_agent.items()}

    return rows


def check_neighbor_symmetry(rep: Dict[str, Any]) -> int:
    """Count directed neighbor entries without a reciprocal (expected 0)."""
    asym = 0
    for h in rep["history"]:
        nbs = {r["agent_id"]: set(r["neighbors"]) for r in h["results"]}
        asym += sum(1 for i, js in nbs.items() for j in js if i not in nbs.get(j, set()))
    return asym


def build_table(verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    run_dirs = sorted(
        p.parent
        for root in RESULT_ROOTS
        for p in (EXPERIMENTS_ROOT / root).rglob("results.json")
    )
    if not run_dirs:
        raise SystemExit(f"No results.json found under {EXPERIMENTS_ROOT}")

    all_rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    warnings: List[str] = []
    per_family: Dict[str, Dict[str, int]] = {}
    asym_total = 0

    for run_dir in run_dirs:
        config = json.loads((run_dir / "config.json").read_text())
        reps = json.loads((run_dir / "results.json").read_text())
        run_meta = classify_run(run_dir, config)

        if "completed_replications" in config and config["completed_replications"] != len(reps):
            warnings.append(
                f"{run_meta['run_path']}: completed_replications={config['completed_replications']}"
                f" != len(results)={len(reps)}"
            )

        fam_key = f"{run_meta['family']}/{run_meta['experiment']}" + (
            " (v1)" if run_meta["results_version"] == "v1" else ""
        )
        stats = per_family.setdefault(fam_key, {"runs": 0, "reps": 0, "decisions": 0})
        stats["runs"] += 1

        for rep in reps:
            validate_replication(rep, run_meta, problems)
            asym_total += check_neighbor_symmetry(rep)
            rows = rows_for_replication(rep, run_meta)
            all_rows.extend(rows)
            stats["reps"] += 1
            stats["decisions"] += len(rows)

    df = pd.DataFrame(all_rows)

    # rep_key must identify a replication uniquely even when replication ids
    # collide across accumulated runs of the same condition.
    dup = df.duplicated(subset=["rep_key", "round", "agent_id"]).sum()
    if dup:
        problems.append(f"{dup} duplicate (rep_key, round, agent_id) rows")

    if asym_total:
        warnings.append(f"{asym_total} asymmetric neighbor entries across all runs")

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiments_root": "theses/experiments",
        "n_run_dirs": len(run_dirs),
        "n_replications": int(df["rep_key"].nunique()),
        "n_decisions": int(len(df)),
        "per_family": per_family,
        "validation": {
            "hard_problems": problems,
            "warnings": warnings,
            "rate_tolerance": RATE_TOL,
        },
        "columns": list(df.columns),
    }

    if verbose:
        print(f"Scanned {len(run_dirs)} run dirs -> {len(df):,} decisions, "
              f"{meta['n_replications']} replications")
        for fam, s in sorted(per_family.items()):
            print(f"  {fam:35s} runs={s['runs']:3d} reps={s['reps']:4d} decisions={s['decisions']:7,d}")
        for w in warnings:
            print(f"  WARNING: {w}")
        if problems:
            print(f"  {len(problems)} HARD VALIDATION PROBLEMS:")
            for p in problems[:20]:
                print(f"    {p}")
        else:
            print("  Validation: all replications match stored coop_rates/final/avg. OK")

    return df, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = Path(__file__).resolve().parent / "data" / "decision_table.csv.gz"
    parser.add_argument("--out", type=Path, default=default_out)
    args = parser.parse_args()

    df, meta = build_table()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    meta_path = args.out.parent / "decision_table_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB) and {meta_path.name}")

    if meta["validation"]["hard_problems"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

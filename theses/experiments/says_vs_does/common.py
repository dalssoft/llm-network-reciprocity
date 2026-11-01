"""Shared helpers for the says-vs-does study scripts."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
DECISION_TABLE = DATA_DIR / "decision_table.csv.gz"

# Analysis slices: structure variation only exists within a slice, so every
# fit/ablation runs slice-wise. v1 (superseded early batch) is excluded.
SLICES = {
    "EXP-A": dict(experiment="A"),          # 8 agents, 25 rounds, WM vs ring
    "EXP-B": dict(experiment="B"),          # 25 agents, 15 rounds, lattice, k x b/c
    "HORIZON": dict(experiment="horizon"),  # 25 agents, lattice, unknown horizon
}


def load_decisions(slice_name: str, min_round: int = 2) -> pd.DataFrame:
    """Load the decision table restricted to one slice, main results only.

    min_round=2 keeps the rows where the lagged observables (the agent's
    round t-1 feedback) exist; round-1 decisions have no information set.
    """
    df = pd.read_csv(DECISION_TABLE, low_memory=False)
    df = df[df.results_version == "main"]
    df = df[df.experiment == SLICES[slice_name]["experiment"]].copy()
    if min_round > 1:
        df = df[df["round"] >= min_round]
        df = df.dropna(subset=["own_coop_prev", "frac_nb_coop_prev"])

    # Regressors the fits use, as plain numerics.
    df["own_coop_prev"] = df["own_coop_prev"].astype(float)
    df["frac_nb_coop_prev"] = df["frac_nb_coop_prev"].astype(float)
    df["static_network"] = df["static_network"].astype(int)
    if df["num_rounds_planned"].notna().all():
        # Known-horizon slices: the tutorial announces the total rounds, so
        # relative position and the final round are in the agent's info set.
        df["round_frac"] = df["round"] / df["num_rounds_planned"]
        df["is_last"] = (df["round"] == df["num_rounds_planned"]).astype(int)
    df["above_threshold"] = (df["bc_ratio"] > df["k"]).astype(int)
    return df


def coop_series(df: pd.DataFrame) -> pd.DataFrame:
    """Population cooperation rate per (rep_key, round), long format."""
    return df.groupby(["model_variant", "condition", "rep_key", "round"], as_index=False)[
        "coop"
    ].mean()

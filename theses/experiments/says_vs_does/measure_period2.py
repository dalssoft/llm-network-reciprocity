"""
Activity B (part 2): measure the period-2 oscillation ("dente de serra").

§5.2.2 currently asserts the sawtooth is "present/visible"; this script turns
it into a statistic. For each replication's population cooperation-rate
series (rounds >= 2, dropping the round-1 transient):

  1. remove a linear trend (OLS) -> residuals e_t
  2. lag-1 and lag-2 autocorrelation of e_t; the period-2 signature is
     rho1 < 0 together with rho2 > 0
  3. one-sided permutation tests (5000 shuffles, deterministic seed per
     replication, mirroring the Monte Carlo convention of
     analyze_contagion.py)
  4. sign-alternation ratio of consecutive first differences (the legacy
     detect_oscillation heuristic, kept for comparability)

NOTE: autocorrelation is computed on detrended LEVELS, not on first
differences - differencing induces a mechanical rho1 = -0.5 even for iid
noise, which would fake the oscillation signature.

Aggregation: per (slice, variant, condition), mean rho2 with a bootstrap CI
over replications, and the share of replications individually significant.
Prints the per-condition verdict for GPT-3.5 Experiment B (§5.2.2).

Usage:
    python -m theses.experiments.says_vs_does.measure_period2
Outputs:
    data/period2_stats.csv     one row per replication
    data/period2_summary.csv   one row per (slice, variant, condition)
"""

import zlib
from typing import Optional

import numpy as np
import pandas as pd

from .common import DATA_DIR, load_decisions, coop_series

N_PERM = 5000
N_BOOT = 10000
MIN_LEN = 8  # residual series shorter than this: no meaningful lag-2 ACF
ALPHA = 0.05


def autocorr(e: np.ndarray, lag: int) -> float:
    e = e - e.mean()
    denom = float(np.dot(e, e))
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(e[:-lag], e[lag:]) / denom)


def detrend(x: np.ndarray) -> np.ndarray:
    t = np.arange(len(x), dtype=float)
    slope, intercept = np.polyfit(t, x, 1)
    return x - (intercept + slope * t)


def rep_period2(x: np.ndarray, rep_key: str) -> Optional[dict]:
    """Period-2 statistics for one replication's coop-rate series."""
    if len(x) < MIN_LEN:
        return None
    e = detrend(x)
    if float(np.std(e)) <= 1e-12:  # constant series (ceiling/floor)
        return {"rho1": np.nan, "rho2": np.nan, "p_rho2": np.nan,
                "p_rho1_neg": np.nan, "alternation": np.nan,
                "series_len": len(x), "degenerate": True}

    rho1 = autocorr(e, 1)
    rho2 = autocorr(e, 2)

    rng = np.random.default_rng(zlib.crc32(rep_key.encode()))
    perm_rho1 = np.empty(N_PERM)
    perm_rho2 = np.empty(N_PERM)
    for i in range(N_PERM):
        p = rng.permutation(e)
        perm_rho1[i] = autocorr(p, 1)
        perm_rho2[i] = autocorr(p, 2)
    p_rho2 = (1 + np.sum(perm_rho2 >= rho2)) / (1 + N_PERM)
    p_rho1_neg = (1 + np.sum(perm_rho1 <= rho1)) / (1 + N_PERM)

    d = np.diff(x)
    nz = d[d != 0]
    alternation = (
        float(np.mean(np.sign(nz[1:]) != np.sign(nz[:-1]))) if len(nz) > 2 else np.nan
    )
    return {"rho1": rho1, "rho2": rho2, "p_rho2": p_rho2,
            "p_rho1_neg": p_rho1_neg, "alternation": alternation,
            "series_len": len(x), "degenerate": False}


def main() -> None:
    stats_rows = []
    for slice_name in ["EXP-A", "EXP-B", "HORIZON"]:
        df = load_decisions(slice_name, min_round=2)
        series = coop_series(df)
        for (variant, condition, rep_key), grp in series.groupby(
            ["model_variant", "condition", "rep_key"]
        ):
            x = grp.sort_values("round")["coop"].to_numpy()
            res = rep_period2(x, rep_key)
            if res is None:
                continue
            stats_rows.append({
                "slice": slice_name, "variant": variant,
                "condition": condition, "rep_key": rep_key, **res,
            })

    stats = pd.DataFrame(stats_rows)
    stats.to_csv(DATA_DIR / "period2_stats.csv", index=False)

    # ---- aggregate per condition ----------------------------------------
    rng = np.random.default_rng(20260704)
    summary_rows = []
    for (slice_name, variant, condition), grp in stats.groupby(["slice", "variant", "condition"]):
        rho2 = grp["rho2"].dropna().to_numpy()
        row = {
            "slice": slice_name, "variant": variant, "condition": condition,
            "n_reps": len(grp), "n_measurable": len(rho2),
            "mean_rho1": grp["rho1"].mean(), "mean_rho2": np.nan,
            "rho2_ci_lo": np.nan, "rho2_ci_hi": np.nan,
            "share_sig_rho2": np.nan, "mean_alternation": grp["alternation"].mean(),
        }
        if len(rho2) > 0:
            row["mean_rho2"] = float(rho2.mean())
            row["share_sig_rho2"] = float((grp["p_rho2"].dropna() < ALPHA).mean())
            if len(rho2) > 1:
                boots = rng.choice(rho2, size=(N_BOOT, len(rho2)), replace=True).mean(axis=1)
                row["rho2_ci_lo"], row["rho2_ci_hi"] = np.percentile(boots, [2.5, 97.5])
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(DATA_DIR / "period2_summary.csv", index=False)

    with pd.option_context("display.width", 220):
        print("PERIOD-2 SUMMARY (per condition; signature = rho1 < 0 and rho2 > 0)")
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- per-condition verdict (GPT-3.5, Experiment B) ------------------
    g35 = summary[(summary["slice"] == "EXP-B") & (summary.variant == "gpt-3.5-turbo")].copy()
    g35["verdict"] = (g35["rho2_ci_lo"] > 0) & (g35["mean_rho1"] < 0)
    print()
    print("GPT-3.5 EXP-B, per-condition verdict (CI of mean rho2 above 0 and mean rho1 < 0):")
    print(g35[["condition", "mean_rho1", "mean_rho2", "rho2_ci_lo", "rho2_ci_hi",
               "share_sig_rho2", "verdict"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nConditions with period-2 signature: {int(g35['verdict'].sum())}/9")


if __name__ == "__main__":
    main()

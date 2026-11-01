"""
Activity B (part 1): the "does" layer - memory-1 reactive model + ablation.

The model is a logistic regression of the decision to cooperate: a weighted
sum z of input variables squashed into a probability P(coop) = 1/(1+e^-z).
Positive weight = pushes toward cooperating; negative = toward defecting.
It is fitted four times per cell (slice x model variant), each time keeping
the previous variables and adding a block ("ablation"); SEs clustered by
replication (decisions within one game are not independent):

    M0: intercept only. The fixed guess (base rate); the floor every other
        layer is measured against.
    MR: + own_coop_prev      did I cooperate in t-1? Weight reads as
                             persistence (+) vs alternation (-).
        + frac_nb_coop_prev  fraction of my t-1 neighbors that cooperated.
                             Weight reads as reciprocity (+) vs exploitation (-).
        Together: the memory-1 stimulus-response hypothesis, using exactly
        the actionable content of the round-t-1 feedback prompt.
    MT: + round_frac         share of the game elapsed: smooth drift.
        + is_last            announced final round: the endgame jump.
        The agent knows both (the tutorial announces total rounds). In
        HORIZON there is no announced end, so MT uses the round number only.
    MS: + structure block    what theory says SHOULD matter, entered last so
        it can only claim predictive power beyond everything the agent
        demonstrably uses (burden of proof). EXP-A: static_network (fixed
        ring vs reshuffled partners). EXP-B: k (neighbors), bc_ratio
        (benefit/cost) and above_threshold (Ohtsuki's rule b/c > k).

The network-blindness measurement is the MT -> MS step: if the structure
block adds ~nothing (delta AIC near 0 or positive, delta pseudo-R2 in the
0.001 range, clustered p large), structure does not enter the individual
decision even though outcomes may differ across structures at the population
level (composition via contagion, not individual response).

Degenerate cells (minority class below --min-events decisions) are reported
descriptively instead of fitted; that split IS the ceiling/floor answer
(DeepSeek at ceiling, gpt-5-mini / gpt-oss / lite-R at floor).
Every cell also gets a model-free conditional table P(coop | what was seen):
after all-C, after any-D, after own-D; it describes deterministic rules
(grim-trigger-like) better than any coefficient.

Usage:
    python -m theses.experiments.says_vs_does.fit_behavioral_model
Outputs:
    data/behavioral_fits.csv       one row per fitted coefficient
    data/behavioral_ablation.csv   one row per (slice, variant, step)
    data/behavioral_cells.csv      descriptive row per (slice, variant)
"""

import argparse
import json
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .common import DATA_DIR, load_decisions

BASE = ["own_coop_prev", "frac_nb_coop_prev"]

SLICE_SPECS: Dict[str, Dict[str, List[str]]] = {
    "EXP-A": {
        "time": ["round_frac", "is_last"],
        "structure": ["static_network"],
    },
    "EXP-B": {
        "time": ["round_frac", "is_last"],
        "structure": ["k", "bc_ratio", "above_threshold"],
    },
    "HORIZON": {
        "time": ["round"],
        "structure": [],
    },
}


SEPARATION_COEF = 15.0  # |coef| beyond this = MLE at the boundary (separation)


def fit_logit(df: pd.DataFrame, rhs: List[str]):
    """Newton first; on singular Hessian fall back to BFGS.

    Quasi-complete separation (e.g. 'saw any defection -> defect' being
    deterministic for the reasoning models) makes the Newton Hessian
    singular. BFGS still converges in log-likelihood, so AIC / pseudo-R2
    (and hence the ablation deltas) remain valid; the separated
    coefficients themselves are flagged unidentified downstream.
    """
    formula = "coop ~ " + (" + ".join(rhs) if rhs else "1")
    kw = dict(cov_type="cluster", cov_kwds={"groups": df["rep_key"]}, disp=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.logit(formula, data=df)
        try:
            return model.fit(maxiter=200, **kw)
        except Exception:
            pass
        try:
            return model.fit(method="bfgs", maxiter=1000, **kw)
        except Exception:
            # Hessian exactly singular at the optimum (hard separation):
            # keep llf / AIC / pseudo-R2, give up on any covariance.
            return model.fit(method="bfgs", maxiter=1000, disp=0, skip_hessian=True)


def reactive_rule_table(df: pd.DataFrame) -> dict:
    """Model-free memory-1 signature: P(coop) by what the agent saw at t-1.

    Comparable across every cell, including degenerate and separated ones.
    """
    after_all_c = df[(df.own_coop_prev == 1) & (df.frac_nb_coop_prev == 1.0)]
    after_any_d = df[(df.own_coop_prev == 0) | (df.frac_nb_coop_prev < 1.0)]
    own_d = df[df.own_coop_prev == 0]
    return {
        "p_coop_after_all_C": after_all_c["coop"].mean() if len(after_all_c) else np.nan,
        "n_after_all_C": len(after_all_c),
        "p_coop_after_any_D": after_any_d["coop"].mean() if len(after_any_d) else np.nan,
        "n_after_any_D": len(after_any_d),
        "p_coop_after_own_D": own_d["coop"].mean() if len(own_d) else np.nan,
        "n_after_own_D": len(own_d),
    }


def fit_cell(
    df: pd.DataFrame,
    slice_name: str,
    variant: str,
    min_events: int,
    fits_rows: List[dict],
    ablation_rows: List[dict],
    cells_rows: List[dict],
    pooled_fe: bool = False,
) -> None:
    """Fit the M0->MR->MT->MS sequence for one (slice, variant) cell."""
    spec = SLICE_SPECS[slice_name]
    n = len(df)
    n_coop = int(df["coop"].sum())
    minority = min(n_coop, n - n_coop)
    cell = {
        "slice": slice_name,
        "variant": variant,
        "n_decisions": n,
        "n_replications": df["rep_key"].nunique(),
        "coop_rate": n_coop / n,
        "alternation_rate": float((df["coop"] != df["own_coop_prev"]).mean()),
        "minority_class_n": minority,
        **reactive_rule_table(df),
    }

    if minority < min_events:
        cell["status"] = "degenerate_ceiling" if cell["coop_rate"] > 0.5 else "degenerate_floor"
        cells_rows.append(cell)
        return
    cell["status"] = "fitted"

    fe = ["C(model_variant)"] if pooled_fe else []
    steps = [
        ("M0", fe if fe else []),
        ("MR", fe + BASE),
        ("MT", fe + BASE + spec["time"]),
    ]
    if spec["structure"]:
        steps.append(("MS", fe + BASE + spec["time"] + spec["structure"]))

    prev_aic: Optional[float] = None
    prev_pr2: Optional[float] = None
    prev_rhs: List[str] = []
    last_ok = None
    any_separation = False
    for step_name, rhs in steps:
        try:
            res = fit_logit(df, rhs)
        except Exception as exc:  # not even BFGS converges
            cell["status"] = f"fitted_through_{last_ok} ({step_name}: {type(exc).__name__})"
            cells_rows.append(cell)
            return
        separated = bool(res.params.abs().max() > SEPARATION_COEF)
        any_separation = any_separation or separated
        row = {
            "slice": slice_name,
            "variant": variant,
            "step": step_name,
            "n": int(res.nobs),
            "n_clusters": df["rep_key"].nunique(),
            "aic": res.aic,
            "pseudo_r2": res.prsquared,
            "delta_aic_vs_prev": None if prev_aic is None else res.aic - prev_aic,
            "delta_pseudo_r2_vs_prev": None if prev_pr2 is None else res.prsquared - prev_pr2,
            "converged": bool(res.mle_retvals.get("converged", True)),
            "quasi_separation": separated,
        }
        # Joint (clustered) Wald test of the block added at this step
        # (skipped under separation, where the sandwich covariance is invalid).
        added = [t for t in rhs if t not in prev_rhs and not t.startswith("C(")]
        if added and prev_rhs is not rhs and prev_aic is not None and not separated:
            try:
                wt = res.wald_test(" = 0, ".join(added) + " = 0", scalar=True)
                row["block_wald_chi2"] = float(wt.statistic)
                row["block_wald_p"] = float(wt.pvalue)
                row["block_df"] = len(added)
            except Exception:
                row["block_wald_chi2"] = row["block_wald_p"] = None
        ablation_rows.append(row)
        prev_aic, prev_pr2, prev_rhs, last_ok = res.aic, res.prsquared, rhs, step_name

        for term in res.params.index:
            if term.startswith("C(model_variant)"):
                continue
            se = z = p = np.nan
            if not separated:
                try:
                    se, z, p = res.bse[term], res.tvalues[term], res.pvalues[term]
                except Exception:
                    pass
            fits_rows.append({
                "slice": slice_name,
                "variant": variant,
                "step": step_name,
                "term": term,
                "coef": res.params[term],
                "se_cluster": se,
                "z": z,
                "p": p,
                "odds_ratio": float(np.exp(np.clip(res.params[term], -500, 500))),
                "identified": not separated,
            })

    if any_separation:
        cell["status"] = "fitted_with_separation"
    cells_rows.append(cell)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-events", type=int, default=30,
                        help="minimum minority-class decisions to fit a cell")
    args = parser.parse_args()

    fits_rows: List[dict] = []
    ablation_rows: List[dict] = []
    cells_rows: List[dict] = []

    for slice_name in ["EXP-A", "EXP-B", "HORIZON"]:
        df = load_decisions(slice_name)
        if slice_name == "HORIZON":
            # Single model variant across all six runs; the meaningful split
            # is informed vs uninformed (informed is 100% cooperation by
            # design, hence degenerate, and is reported as such).
            df = df.assign(cell_key=df.model_variant + "/" + df.horizon_info)
        else:
            df = df.assign(cell_key=df.model_variant)
        for variant, sub in df.groupby("cell_key"):
            fit_cell(sub, slice_name, variant, args.min_events,
                     fits_rows, ablation_rows, cells_rows)

        # Pooled fit with model fixed effects over the non-degenerate variants.
        fitted = [c["variant"] for c in cells_rows
                  if c["slice"] == slice_name and c["status"].startswith("fitted")
                  and not c["status"].startswith("fitted_through")]
        if len(fitted) > 1:
            pooled = df[df.model_variant.isin(fitted)]
            fit_cell(pooled, slice_name, "POOLED(+model FE)", args.min_events,
                     fits_rows, ablation_rows, cells_rows, pooled_fe=True)

    fits = pd.DataFrame(fits_rows)
    ablation = pd.DataFrame(ablation_rows)
    cells = pd.DataFrame(cells_rows)

    DATA_DIR.mkdir(exist_ok=True)
    fits.to_csv(DATA_DIR / "behavioral_fits.csv", index=False)
    ablation.to_csv(DATA_DIR / "behavioral_ablation.csv", index=False)
    cells.to_csv(DATA_DIR / "behavioral_cells.csv", index=False)

    # ---- printed report -------------------------------------------------
    print("=" * 100)
    print("CELLS (ceiling/floor split)")
    print(cells.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print()
    print("=" * 100)
    print("ABLATION (per step: AIC, pseudo-R2, deltas, clustered block Wald)")
    with pd.option_context("display.width", 200):
        cols = ["slice", "variant", "step", "n", "n_clusters", "aic", "pseudo_r2",
                "delta_aic_vs_prev", "delta_pseudo_r2_vs_prev", "block_wald_chi2", "block_wald_p"]
        print(ablation[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print()
    print("=" * 100)
    print("COEFFICIENTS at the full model (MS where available, else MT)")
    last_step = fits.groupby(["slice", "variant"])["step"].transform(
        lambda s: "MS" if (s == "MS").any() else "MT")
    full = fits[fits.step == last_step]
    with pd.option_context("display.width", 200):
        print(full[["slice", "variant", "term", "coef", "se_cluster", "z", "p", "odds_ratio"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    meta = {
        "spec": {
            "base": BASE, "slices": SLICE_SPECS, "min_events": args.min_events,
            "estimator": "statsmodels Logit, cluster-robust SE by rep_key",
        },
        "n_cells": len(cells),
        "n_fitted": int((cells.status == "fitted").sum()),
    }
    (DATA_DIR / "behavioral_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

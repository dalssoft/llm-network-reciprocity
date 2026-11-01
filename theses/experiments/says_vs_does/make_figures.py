"""
Figure of the memory-1 reactive rules (dissertation, Section 5.7).

fig_reactive_rules.png
    Slopegraph of the model-free reactive rule: P(cooperate) after
    "everyone cooperated" vs after "saw a defection", per model (EXP-A).
    Shows the whole typology at once: ceiling, floors, grim-trigger drops,
    the forgiving reciprocator, and gpt-3.5's INVERTED (rising) line.
    Only strata with n >= 30 decisions are drawn as lines; DeepSeek (never
    saw a defection) and the three floors are single annotated points.

Labels are in Portuguese, as in the dissertation (decimal commas).

Usage:
    python -m theses.experiments.says_vs_does.make_figures
Input:
    data/behavioral_cells.csv (from fit_behavioral_model)
Output:
    figures/fig_reactive_rules.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

from .common import DATA_DIR

FIG_DIR = Path(__file__).resolve().parent / "figures"

BLUE, AQUA, YELLOW, VIOLET, MAGENTA, RED = (
    "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e87ba4", "#e34948")
YELLOW_TEXT = "#8a5f00"  # label-relief ink for the yellow series
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2de"

plt.rcParams.update({
    "font.family": "Palatino",
    "font.size": 11,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 300,
})

comma = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))
comma1 = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))


def _despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def _fmt(v: float) -> str:
    return f"{v:.2f}".replace(".", ",")


def fig_reactive_rules() -> None:
    cells = pd.read_csv(DATA_DIR / "behavioral_cells.csv")
    a = cells[(cells["slice"] == "EXP-A") & (~cells.variant.str.startswith("POOLED"))]
    a = a.set_index("variant")

    # variant, display label, color, right-label slot (y in data coords)
    lines = [
        ("gpt-3.5-turbo", "GPT-3.5 (alternador)", RED, 0.576),
        ("gemini-2.5-flash-lite", "Gemini 2.5 Lite (perdoador)", AQUA, 0.458),
        ("gpt-4-0613", "GPT-4 (reciprocador fraco)", VIOLET, 0.322),
        ("gemini-3-flash-preview", "Gemini 3 Flash (rancoroso)", BLUE, 0.175),
        ("gemini-3-flash-preview_reasoning", "Gemini 3 Flash R (rancoroso)", BLUE, 0.095),
    ]
    x0, x1 = 0.0, 1.0
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    for variant, label, color, slot in lines:
        r = a.loc[variant]
        y0, y1 = r["p_coop_after_all_C"], r["p_coop_after_any_D"]
        ax.plot([x0, x1], [y0, y1], color=color, lw=2, zorder=3)
        ax.plot([x0, x1], [y0, y1], "o", color=color, ms=8,
                mec="white", mew=2, zorder=4)
        if abs(slot - y1) > 0.02:  # leader line to a displaced label slot
            ax.plot([x1 + 0.015, x1 + 0.055], [y1, slot], color=INK2,
                    lw=0.7, zorder=2)
        ax.annotate(f"{label}   {_fmt(y0)} $\\rightarrow$ {_fmt(y1)}", (x1, slot),
                    xytext=(16, 0), textcoords="offset points",
                    va="center", fontsize=10, color=INK)

    # single-sided entities
    ax.plot([x0], [1.0], "o", color=INK2, ms=8, mec="white", mew=2, zorder=5)
    ax.plot([0.02, 0.10], [1.006, 1.052], color=INK2, lw=0.7, zorder=2)
    ax.annotate("DeepSeek-V3.2 (teto): 1,00 sempre; nunca viu deserção",
                (0.105, 1.052), va="center", fontsize=9, color=INK2)
    floors = a.loc[["gpt-5-mini", "gpt-5-mini_reasoning", "gpt-oss-120b",
                    "gemini-2.5-flash-lite_reasoning"]]
    y_floor = floors["p_coop_after_any_D"].mean()
    ax.plot([x1], [y_floor], "o", color=INK2, ms=8, mec="white", mew=2, zorder=5)
    ax.plot([x1 + 0.015, x1 + 0.055], [y_floor, 0.005], color=INK2, lw=0.7, zorder=2)
    ax.annotate("pisos (~0,02): GPT-5-mini (±R), GPT-OSS, Lite R",
                (x1 + 0.07, 0.005), va="center", fontsize=9, color=INK2)

    ax.set_xlim(-0.15, 2.2)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["após todos os vizinhos\ncooperarem (t−1)",
                        "após ver ao menos\numa deserção (t−1)"])
    ax.set_ylabel("P(cooperar no round t)")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(comma)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_reactive_rules.png", bbox_inches="tight")
    plt.close(fig)
    print("F1 fig_reactive_rules.png")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_reactive_rules()

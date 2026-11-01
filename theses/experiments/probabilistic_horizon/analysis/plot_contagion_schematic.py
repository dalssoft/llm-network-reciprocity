"""
Schematic of the spatial contagion analysis on the 25-agent ring (k = 2).

Produces the illustration used in the dissertation, Section 4.6.2
(Analise Espacial do Contagio). It is a hand-built example, not data:

  Panel (a): observed contagion at a round t. The defectors form a single
             contiguous arc that grew from a seed; the two new defectors of
             round t are neighbours of round t-1 defectors (contiguous
             adhesion = 1.0); the two ends of the arc are the fronts.
  Panel (b): null model. The same number of defectors is placed at
             uniformly random ring positions, ignoring the network; the
             draw shown yields 4 arcs.

Arc counting follows analyze_contagion.count_contiguous_arcs: walk the ring
and count cooperator -> defector transitions.

Usage:
    python -m theses.experiments.probabilistic_horizon.analysis.plot_contagion_schematic
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Arc, Circle  # noqa: E402

N = 25
GREEN = "#2ecc71"
GREEN_INK = "#14532d"
RED = "#e74c3c"
RED_DARK = "#7b241c"
RING = "#9a9a9a"
INK = "#333333"
NODE_R = 0.085

OUT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "fig_contagio_esquema.png"
)


def angle_deg(i: int) -> float:
    """Node i sits clockwise from the top; node 0 (the seed) is at 90 degrees."""
    return 90.0 - 360.0 * i / N


def xy(i: int, r: float = 1.0) -> np.ndarray:
    a = np.deg2rad(angle_deg(i))
    return np.array([r * np.cos(a), r * np.sin(a)])


def count_arcs(defectors: set) -> int:
    return sum(1 for i in range(N) if i in defectors and (i - 1) % N not in defectors)


def contiguous_groups(defectors: set) -> list:
    """Return the arcs as lists of consecutive node ids (clockwise order)."""
    groups = []
    for i in range(N):
        if i in defectors and (i - 1) % N not in defectors:
            g = [i]
            j = (i + 1) % N
            while j in defectors and j != i:
                g.append(j)
                j = (j + 1) % N
            groups.append(g)
    return groups


def draw_ring(ax, prev_defectors: set, new_defectors: set, seed=None):
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ec=RING, lw=1.6, zorder=1))
    for i in range(N):
        is_new = i in new_defectors
        is_def = i in prev_defectors or is_new
        fc = RED if is_def else GREEN
        ec = RED_DARK if is_new else "white"
        lw = 2.4 if is_new else 1.0
        ax.add_patch(Circle(xy(i), NODE_R, fc=fc, ec=ec, lw=lw, zorder=3))
        ax.text(
            *xy(i),
            str(i),
            ha="center",
            va="center",
            fontsize=5.2,
            color="white" if is_def else GREEN_INK,
            zorder=4,
        )
    if seed is not None:
        ax.annotate(
            "semente\n(round 1)",
            xy=xy(seed, 1.0 - NODE_R),
            xytext=(0.0, 0.52),
            ha="center",
            va="center",
            fontsize=7.5,
            color=INK,
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, shrinkA=0, shrinkB=2),
            zorder=5,
        )


def draw_arcs(ax, defectors: set, r: float = 1.24, pad_deg: float = 5.0):
    for g in contiguous_groups(defectors):
        theta_hi = angle_deg(g[0]) + pad_deg  # first node clockwise = larger angle
        theta_lo = angle_deg(g[-1]) - pad_deg
        ax.add_patch(
            Arc((0, 0), 2 * r, 2 * r, theta1=theta_lo, theta2=theta_hi, color=RED_DARK, lw=2.6, zorder=2)
        )
        for th in (theta_lo, theta_hi):  # small end ticks
            a = np.deg2rad(th)
            ax.plot(
                [(r - 0.035) * np.cos(a), (r + 0.035) * np.cos(a)],
                [(r - 0.035) * np.sin(a), (r + 0.035) * np.sin(a)],
                color=RED_DARK,
                lw=2.0,
                zorder=2,
            )


def draw_front(ax, last_defector: int, direction: int, r: float = 1.12):
    """Arrow along the ring from the last defector towards the next cooperator."""
    start = angle_deg(last_defector) - direction * 3.0
    end = angle_deg(last_defector + direction) + direction * 1.0
    p0 = np.array([r * np.cos(np.deg2rad(start)), r * np.sin(np.deg2rad(start))])
    p1 = np.array([r * np.cos(np.deg2rad(end)), r * np.sin(np.deg2rad(end))])
    rad = 0.18 * direction
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6, connectionstyle=f"arc3,rad={rad}"),
        zorder=5,
    )
    mid = 0.5 * (start + end)
    lab = np.array([1.47 * np.cos(np.deg2rad(mid)), 1.47 * np.sin(np.deg2rad(mid))])
    ax.text(*lab, "frente", ha="center", va="center", fontsize=7.5, color=INK, style="italic")


def finish(ax, title: str, note: str):
    ax.set_aspect("equal")
    ax.set_xlim(-1.62, 1.62)
    ax.set_ylim(-2.02, 1.62)
    ax.axis("off")
    ax.set_title(title, fontsize=8.5, color=INK, pad=4)
    ax.text(0, -1.5, note, ha="center", va="top", fontsize=7.3, color=INK, linespacing=1.4)


def main() -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 4.35))

    # (a) contagion at round t: seed 0 and its neighbours defected before t,
    #     nodes 23 and 2 join at t (both adjacent to a round t-1 defector).
    prev = {24, 0, 1}
    new = {23, 2}
    assert count_arcs(prev | new) == 1
    draw_ring(ax_a, prev, new, seed=0)
    draw_arcs(ax_a, prev | new)
    draw_front(ax_a, 2, +1)
    draw_front(ax_a, 23, -1)
    ax_a.text(0, 1.44, "1 arco", ha="center", va="bottom", fontsize=8.5, color=RED_DARK, weight="bold")
    finish(
        ax_a,
        "(a) Contágio observado em um round $t$",
        "Novos desertores de $t$: 23 e 2, ambos vizinhos\n"
        "de um desertor de $t-1$ (adesão contígua = 1,0).\n"
        "Um único arco, duas frentes.",
    )

    # (b) null model: the same 5 defectors at random positions.
    null = {1, 2, 8, 14, 20}
    assert len(null) == len(prev | new)
    draw_ring(ax_b, null, set())
    draw_arcs(ax_b, null)
    ax_b.text(
        0, 1.44, f"{count_arcs(null)} arcos", ha="center", va="bottom", fontsize=8.5, color=RED_DARK, weight="bold"
    )
    finish(
        ax_b,
        "(b) Modelo nulo: os mesmos 5 desertores sorteados",
        "Posições sorteadas ignorando a rede: 4 arcos.\n"
        "Valor-$p$ = fração dos 5.000 sorteios com tão\n"
        "poucos arcos quanto o observado.",
    )

    handles = [
        Line2D([], [], marker="o", ls="", ms=9, mfc=GREEN, mec="white", label="cooperador"),
        Line2D([], [], marker="o", ls="", ms=9, mfc=RED, mec="white", label="desertor desde $t-1$"),
        Line2D([], [], marker="o", ls="", ms=9, mfc=RED, mec=RED_DARK, mew=2.2, label="novo desertor em $t$"),
        Line2D([], [], color=RED_DARK, lw=2.6, label="arco (grupo contíguo)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5, handletextpad=0.5)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.11, wspace=0.05)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

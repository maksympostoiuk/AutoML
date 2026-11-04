"""Static charts for explanations (matplotlib, saved as PNG).

Style: light surface, recessive grid, one blue for magnitudes; for contributions a
colour-blind-safe pair, red = pushes the prediction up, blue = pushes it down. The sign is
also written next to every bar, so colour is never the only cue.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure  # the object API: no GUI backend, nothing to close

from automl.explain import Explanation

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
BLUE, RED = "#2a78d6", "#e34948"
STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "figure.dpi": 110,
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "x",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK,
    "ytick.left": False,
    "text.color": INK,
}


def plot_importance(explanation: Explanation, top: int = 12) -> Figure:
    """Global importance: mean absolute contribution per column."""
    importance = explanation.importance().head(top)[::-1]  # largest at the top
    with mpl.rc_context(STYLE):
        fig = Figure(figsize=(7.5, 0.42 * len(importance) + 1.4))
        ax = fig.subplots()
        bars = ax.barh(importance.index, importance.to_numpy(), height=0.7, color=BLUE)
        ax.bar_label(bars, fmt="%.3g", padding=4, fontsize=9, color=INK_2)
        ax.set_xlabel(f"mean |contribution| ({explanation.units})")
        ax.set_title("Which columns drive the predictions")
        ax.margins(x=0.12)
        fig.tight_layout()
    return fig


def plot_row(explanation: Explanation, i: int, top: int = 10) -> Figure:
    """Local explanation: why one row got its prediction."""
    table = explanation.row(i)
    shown, rest = table.head(top), table.iloc[top:]
    labels = [f"{c} = {_short(v)}" for c, v in zip(shown["column"], shown["value"], strict=True)]
    values = list(shown["contribution"])
    if len(rest):
        labels.append(f"{len(rest)} other column{'s' if len(rest) > 1 else ''}")
        values.append(float(rest["contribution"].sum()))
    labels, values = labels[::-1], np.array(values[::-1])

    with mpl.rc_context(STYLE):
        fig = Figure(figsize=(7.5, 0.42 * len(values) + 1.8))
        ax = fig.subplots()
        colors = [RED if v > 0 else BLUE for v in values]
        bars = ax.barh(labels, values, height=0.7, color=colors)
        ax.bar_label(bars, labels=[f"{v:+.3g}" for v in values], padding=4, fontsize=9, color=INK_2)
        ax.axvline(0, color=INK_2, linewidth=1)
        ax.set_xlabel(f"contribution ({explanation.units}); red pushes up, blue pushes down")
        ax.set_title(f"Why this prediction: {explanation.describe_row(i)}", fontsize=10)
        ax.margins(x=0.18)
        fig.tight_layout()
    return fig


def _short(value: object, width: int = 24) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    text = f"{value:.4g}" if isinstance(value, float) else str(value)
    return text if len(text) <= width else text[: width - 1] + "…"

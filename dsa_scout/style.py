"""Shared visual style for DSA-Scout plots."""

from __future__ import annotations

import matplotlib as mpl

SCORER_COLORS = {
    "random": "#8A8F98",
    "recency": "#2F6B9A",
    "window_sink": "#1B9AAA",
    "linear": "#D28B26",
    "preview_attn": "#3D8B5B",
    "lightning_untrained": "#B85C5C",
    "lightning_trained": "#7653A6",
    "lightning_trained_plus_recency": "#C75D9B",
}

SCORER_LABELS = {
    "random": "Random",
    "recency": "Recency",
    "window_sink": "Window+Sink",
    "linear": "Linear",
    "preview_attn": "Preview Attn",
    "lightning_untrained": "Lightning (untrained)",
    "lightning_trained": "Lightning (trained)",
    "lightning_trained_plus_recency": "Lightning+Recency",
}


def apply_style() -> None:
    """Apply consistent matplotlib styling."""
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "savefig.pad_inches": 0.08,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.facecolor": "white",
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#30343B",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#AEB7C2",
            "grid.alpha": 0.24,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.markersize": 4.8,
        }
    )

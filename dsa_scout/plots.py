"""Polished plot helpers for DSA-Scout."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dsa_scout.style import SCORER_COLORS, SCORER_LABELS, apply_style

apply_style()

TITLE_COLOR = "#1F2933"
SUBTLE_TEXT = "#52606D"
MID_LAYER_SHADE = "#EFF4F8"
COMPACT_SCORER_LABELS = {
    "random": "Random",
    "recency": "Recency",
    "window_sink": "Window+Sink",
    "linear": "Linear",
    "preview_attn": "Preview",
    "lightning_untrained": "Untrained",
    "lightning_trained": "Trained",
    "lightning_trained_plus_recency": "Hybrid",
}


def _save(fig: Figure, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)


def _middle_layer_band(ax: Axes) -> None:
    ax.axvspan(3.5, 8.5, color=MID_LAYER_SHADE, alpha=0.75, zorder=0)


def save_oracle_heatmap(attn: torch.Tensor, token_count: int, path: Path) -> None:
    """Save an oracle-attention sanity heatmap with a log color scale.

    Args:
        attn: Attention tensor with shape ``[num_layers, seq, seq]``.
        token_count: Number of decoded tokens.
        path: Output PNG path.
    """
    layer_attn = attn[5].detach().cpu().numpy()
    display = np.where(layer_attn > 0, layer_attn, np.nan)
    positive = display[np.isfinite(display)]
    vmin = max(float(np.nanmin(positive)) if positive.size else 1e-6, 1e-6)
    vmax = float(np.nanmax(positive)) if positive.size else 1.0
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        display,
        cmap=cmap,
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_title(f"Layer 5 oracle attention, log scale ({token_count} tokens)")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Attention mass (log scale)")
    _save(fig, path)


def save_scorer_distributions(scores_by_name: Mapping[str, torch.Tensor], path: Path) -> None:
    """Save log-scale histograms with p1-p99 score clipping."""
    names = list(scores_by_name)
    fig, axes = plt.subplots(2, 4, figsize=(12.2, 5.4), sharey=False)
    for idx, (ax, name) in enumerate(zip(axes.flat, names, strict=False)):
        scores = scores_by_name[name].detach().cpu().flatten().numpy()
        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            ax.set_visible(False)
            continue
        low, high = np.percentile(scores, [1, 99])
        if high - low < 1e-9:
            low, high = float(scores.min()), float(scores.max())
        bins = np.linspace(low, high, 60)
        ax.hist(
            np.clip(scores, low, high),
            bins=bins,
            color=SCORER_COLORS[name],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.15,
        )
        ax.set_yscale("log")
        ax.set_title(SCORER_LABELS[name], fontsize=9.2, loc="left", color=TITLE_COLOR)
        ax.text(
            0.98,
            0.92,
            f"n={scores.size:,}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color=SUBTLE_TEXT,
        )
        if idx // 4 == 1:
            ax.set_xlabel("Score, p1-p99 clipped", fontsize=8.5)
        if idx % 4 == 0:
            ax.set_ylabel("Count, log scale", fontsize=8.5)
        if scores.min() < low:
            ax.axvline(low, color="#8A8F98", linestyle=":", linewidth=0.9, alpha=0.75)
        if scores.max() > high:
            ax.axvline(high, color="#8A8F98", linestyle=":", linewidth=0.9, alpha=0.75)
    for ax in list(axes.flat)[len(names) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Scorer Score Distributions",
        fontsize=11.5,
        fontweight="semibold",
        color=TITLE_COLOR,
        y=0.98,
    )
    fig.text(
        0.5,
        0.025,
        "Dotted bounds mark clipped tails.",
        ha="center",
        fontsize=8,
        color=SUBTLE_TEXT,
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.86, bottom=0.13, wspace=0.25, hspace=0.48)
    _save(fig, path)


def save_recall_by_layer(
    results: Mapping[str, np.ndarray],
    scorer_names: Sequence[str],
    path: Path,
) -> None:
    """Save two-panel recall-by-layer plot.

    Args:
        results: Mapping from scorer to arrays with shape ``[seeds, 12, texts]``.
        scorer_names: Ordered scorer names.
        path: Output PNG path.
    """
    strong = ["recency", "window_sink", "linear", "preview_attn", "lightning_trained_plus_recency"]
    weak = ["random", "lightning_untrained", "lightning_trained"]
    panels = [
        ([name for name in strong if name in scorer_names], "Locality-driven scorers"),
        ([name for name in weak if name in scorer_names], "Learned and random scorers"),
    ]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.4, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3.1, 2.1], "hspace": 0.16},
    )
    for ax, (names, title) in zip(axes, panels, strict=True):
        _middle_layer_band(ax)
        for name in names:
            arr = results[name]
            mean = arr.mean(axis=(0, 2))
            layers = np.arange(mean.shape[0])
            ax.plot(
                layers,
                mean,
                marker="o",
                linewidth=1.85 if name in {"window_sink", "recency", "lightning_trained"} else 1.55,
                markersize=4.0,
                color=SCORER_COLORS[name],
                label=SCORER_LABELS[name],
            )
        ax.set_ylabel("Top-8 recall", fontsize=9.2)
        ax.set_title(title, fontsize=10.2, loc="left", color=TITLE_COLOR)
        ax.set_ylim(0.0, 0.9 if title.startswith("Locality") else 0.84)
        ax.legend(loc="upper right", fontsize=7.8, ncol=3 if len(names) > 3 else len(names))
    axes[0].text(
        6.0,
        0.025,
        "middle layers",
        fontsize=7.8,
        color=SUBTLE_TEXT,
        ha="center",
        va="bottom",
    )
    axes[1].set_xlabel("GPT-2 layer", fontsize=9.2)
    axes[1].set_xticks(np.arange(next(iter(results.values())).shape[1]))
    fig.suptitle(
        "Sparse Scorer Recall by Layer", fontsize=11.8, fontweight="semibold", color=TITLE_COLOR
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.08, hspace=0.18)
    _save(fig, path)


def save_recall_vs_k(
    results_k: Mapping[str, Mapping[int, Sequence[float]]],
    scorer_names: Sequence[str],
    k_values: Sequence[int],
    path: Path,
) -> None:
    """Save recall as a function of selection budget."""
    panels = [
        ["recency", "window_sink", "linear", "preview_attn", "lightning_trained_plus_recency"],
        ["random", "lightning_untrained", "lightning_trained"],
    ]
    x = np.arange(len(k_values))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.8, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.8], "hspace": 0.14},
    )
    for ax, names in zip(axes, panels, strict=True):
        for name in [candidate for candidate in names if candidate in scorer_names]:
            means = np.array([np.mean(results_k[name][k_val]) for k_val in k_values], dtype=float)
            cis = np.array(
                [
                    1.96
                    * np.std(results_k[name][k_val], ddof=1)
                    / np.sqrt(len(results_k[name][k_val]))
                    for k_val in k_values
                ],
                dtype=float,
            )
            ax.errorbar(
                x,
                means,
                yerr=cis,
                marker="o",
                linewidth=1.8,
                elinewidth=1.1,
                capsize=3,
                color=SCORER_COLORS[name],
                label=SCORER_LABELS[name],
            )
        ax.set_ylabel("Recall", fontsize=9.2)
        ax.legend(loc="lower right", fontsize=7.8, ncol=2)
    axes[0].set_title("Locality and hybrid scorers", loc="left", fontsize=10.2, color=TITLE_COLOR)
    axes[1].set_title("Learned and random scorers", loc="left", fontsize=10.2, color=TITLE_COLOR)
    axes[0].set_ylim(0.48, 0.82)
    axes[1].set_ylim(0.02, 0.60)
    axes[1].set_xlabel("Selected compressed blocks (k)", fontsize=9.2)
    axes[1].set_xticks(x, [str(k_val) for k_val in k_values])
    fig.suptitle(
        "Recall vs Sparse Selection Budget",
        fontsize=11.8,
        fontweight="semibold",
        color=TITLE_COLOR,
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.9, bottom=0.095, hspace=0.22)
    _save(fig, path)


def save_hybrid_weight_sweep(
    sweep: Mapping[str, Mapping[str, float]],
    recency_mean: float,
    trained_mean: float,
    path: Path,
) -> None:
    """Save the post-hoc hybrid-weight ablation plot."""
    weights = np.array(sorted(float(weight) for weight in sweep), dtype=float)
    means = np.array([sweep[f"{weight:.1f}"]["mean"] for weight in weights], dtype=float)
    ci_low = np.array([sweep[f"{weight:.1f}"]["ci_low"] for weight in weights], dtype=float)
    ci_high = np.array([sweep[f"{weight:.1f}"]["ci_high"] for weight in weights], dtype=float)
    yerr = np.vstack([means - ci_low, ci_high - means])

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.errorbar(
        weights,
        means,
        yerr=yerr,
        marker="o",
        linewidth=1.9,
        color=SCORER_COLORS["lightning_trained_plus_recency"],
    )
    ax.axhline(
        recency_mean,
        color=SCORER_COLORS["recency"],
        linestyle="--",
        linewidth=1.4,
    )
    ax.axhline(
        trained_mean,
        color=SCORER_COLORS["lightning_trained"],
        linestyle="--",
        linewidth=1.4,
    )
    best_idx = int(np.argmax(means))
    ax.scatter(
        [weights[best_idx]],
        [means[best_idx]],
        s=58,
        color=SCORER_COLORS["lightning_trained_plus_recency"],
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
    )
    ax.text(
        weights[best_idx] + 0.025,
        means[best_idx] + 0.002,
        f"best tested: {means[best_idx]:.3f}",
        fontsize=8,
        color=TITLE_COLOR,
    )
    ax.text(0.915, recency_mean + 0.002, "Recency", fontsize=8, color=SCORER_COLORS["recency"])
    ax.text(
        0.915,
        trained_mean + 0.002,
        "Trained Lightning",
        fontsize=8,
        color=SCORER_COLORS["lightning_trained"],
    )
    ax.set_xlabel("Hybrid Lightning weight", fontsize=9.2)
    ax.set_ylabel("Top-8 recall, layers 4-8", fontsize=9.2)
    ax.set_title("Hybrid Weight Ablation", color=TITLE_COLOR, loc="left", fontsize=11.2)
    ax.set_xticks(weights)
    ax.set_xlim(0.06, 0.98)
    ax.set_ylim(
        min(trained_mean, float(ci_low.min())) - 0.015,
        max(recency_mean, float(ci_high.max())) + 0.015,
    )
    fig.subplots_adjust(left=0.13, right=0.97, top=0.9, bottom=0.16)
    _save(fig, path)


def save_recall_by_text_type(rows: Sequence[dict[str, object]], path: Path) -> None:
    """Save heatmap for recall by text type."""
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="text_type", columns="scorer", values="recall")
    ordered_columns = [name for name in SCORER_LABELS if name in pivot.columns]
    pivot = pivot[ordered_columns]
    values = pivot.to_numpy(dtype=float)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "recall_heat",
        ["#F7FAFC", "#D7E7F3", "#8CBBD7", "#3C7FA8", "#174E75"],
    )
    fig, ax = plt.subplots(figsize=(8.9, 3.9))
    im = ax.imshow(values, cmap=cmap, vmin=0.0, vmax=max(0.8, float(np.nanmax(values))))
    ax.set_xticks(
        np.arange(len(ordered_columns)), [COMPACT_SCORER_LABELS[name] for name in ordered_columns]
    )
    ax.set_yticks(np.arange(len(pivot.index)), [str(item).title() for item in pivot.index])
    ax.tick_params(axis="x", rotation=30, labelsize=8.2)
    ax.tick_params(axis="y", labelsize=8.8)
    ax.set_title("Recall by Text Category", color=TITLE_COLOR, loc="left", fontsize=11.2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            color = "white" if value >= 0.55 else "#1F2933"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.8,
                color=color,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Top-8 recall", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.12, right=0.93, top=0.9, bottom=0.23)
    _save(fig, path)


def save_marginal_over_recency(marginal: Mapping[str, Sequence[float]], path: Path) -> None:
    """Save conditional recall over recency baseline."""
    names = list(marginal.keys())
    means = np.array([np.mean(marginal[name]) for name in names], dtype=float)
    cis = np.array(
        [1.96 * np.std(marginal[name], ddof=1) / np.sqrt(len(marginal[name])) for name in names],
        dtype=float,
    )
    order = np.argsort(means)
    ordered_names = [names[idx] for idx in order]
    ordered_means = means[order]
    ordered_cis = cis[order]
    y = np.arange(len(ordered_names))
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.barh(
        y,
        ordered_means,
        xerr=ordered_cis,
        capsize=3,
        color=[SCORER_COLORS[name] for name in ordered_names],
        alpha=0.92,
        height=0.62,
    )
    ax.set_yticks(y, [SCORER_LABELS[name] for name in ordered_names])
    ax.set_xlabel("Conditional recall on recency misses", fontsize=9.2)
    ax.set_title(
        "Marginal Predictive Power Beyond Recency", color=TITLE_COLOR, loc="left", fontsize=11.2
    )
    ax.set_xlim(0.0, max(0.76, float((ordered_means + ordered_cis).max()) + 0.04))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for idx, value in enumerate(ordered_means):
        ax.text(value + 0.012, idx, f"{value:.2f}", va="center", fontsize=8, color=SUBTLE_TEXT)
    fig.subplots_adjust(left=0.27, right=0.96, top=0.86, bottom=0.18)
    _save(fig, path)


def save_training_curve(
    losses: Sequence[float],
    best_step: int | None,
    best_loss: float | None,
    path: Path,
) -> None:
    """Save the KL distillation training curve with best-loss marker."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(
        np.arange(len(losses)),
        losses,
        color=SCORER_COLORS["lightning_trained"],
        linewidth=1.0,
        alpha=0.5,
        label="Per-step loss",
    )
    if len(losses) >= 20:
        window = max(10, len(losses) // 40)
        smooth = pd.Series(losses).rolling(window, min_periods=1).mean()
        ax.plot(
            smooth.index, smooth.values, color="#2b1b4f", linewidth=2.4, label=f"{window}-step mean"
        )
    if best_step is not None and best_loss is not None and best_step >= 0:
        ax.axvline(best_step, color="#888888", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.scatter(
            [best_step],
            [best_loss],
            color="#d62728",
            s=80,
            zorder=5,
            label=f"Best (step {best_step}, loss {best_loss:.3f})",
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel("KL(oracle || indexer)")
    ax.set_title("Lightning Indexer Distillation Loss")
    ax.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def save_trained_vs_untrained(results: Mapping[str, np.ndarray], path: Path) -> None:
    """Save trained-vs-untrained Lightning comparison by layer."""
    layers = np.arange(12)
    untrained_by_seed = results["lightning_untrained"].mean(axis=2)
    untrained = untrained_by_seed.mean(axis=0)
    untrained_low = untrained_by_seed.min(axis=0)
    untrained_high = untrained_by_seed.max(axis=0)
    trained = results["lightning_trained"].mean(axis=(0, 2))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    _middle_layer_band(ax)
    ax.fill_between(
        layers,
        untrained_low,
        untrained_high,
        color=SCORER_COLORS["lightning_untrained"],
        alpha=0.16,
        label="Untrained seed range",
    )
    ax.plot(
        layers,
        untrained,
        marker="o",
        linewidth=1.9,
        color=SCORER_COLORS["lightning_untrained"],
        label="Untrained mean",
    )
    ax.plot(
        layers,
        trained,
        marker="o",
        linewidth=2.15,
        color=SCORER_COLORS["lightning_trained"],
        label=SCORER_LABELS["lightning_trained"],
    )
    ax.set_xlabel("GPT-2 layer", fontsize=9.2)
    ax.set_ylabel("Top-8 recall", fontsize=9.2)
    ax.set_title(
        "Training Lifts Lightning Indexer Recall", color=TITLE_COLOR, loc="left", fontsize=11.2
    )
    ax.set_xticks(layers)
    ax.set_ylim(0.0, max(0.85, float(trained.max()) + 0.06))
    ax.legend(loc="upper right", fontsize=8.2)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.14)
    _save(fig, path)

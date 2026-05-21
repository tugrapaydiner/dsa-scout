from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dsa_scout.plots import (
    save_hybrid_weight_sweep,
    save_marginal_over_recency,
    save_oracle_heatmap,
    save_recall_by_layer,
    save_recall_by_text_type,
    save_recall_vs_k,
    save_scorer_distributions,
    save_trained_vs_untrained,
    save_training_curve,
)


def test_plot_helpers_write_files(tmp_png: Path) -> None:
    arr = np.ones((5, 12, 3), dtype=float) * 0.5
    results = {
        "random": arr * 0.2,
        "recency": arr * 0.8,
        "window_sink": arr * 0.9,
        "linear": arr * 0.7,
        "preview_attn": arr * 0.75,
        "lightning_untrained": arr * 0.3,
        "lightning_trained": arr * 0.6,
        "lightning_trained_plus_recency": arr * 0.82,
    }
    save_recall_by_layer(results, list(results), tmp_png)
    assert tmp_png.exists()
    curve_path = tmp_png.with_name("curve.png")
    save_training_curve([3.0, 2.0, 1.0], 2, 1.0, curve_path)
    assert curve_path.exists()

    scores = {name: torch.randn(32, 8) for name in results}
    dist_path = tmp_png.with_name("dist.png")
    save_scorer_distributions(scores, dist_path)
    assert dist_path.exists() and dist_path.stat().st_size > 5000

    recall_vs_k = {
        name: {4: [0.1, 0.2], 8: [0.3, 0.4], 16: [0.4, 0.5], 32: [0.5, 0.6]} for name in results
    }
    save_recall_vs_k(recall_vs_k, list(results), [4, 8, 16, 32], tmp_png.with_name("k.png"))
    rows = [
        {"scorer": name, "text_type": "prose", "recall": 0.5 + idx * 0.01}
        for idx, name in enumerate(results)
    ]
    save_recall_by_text_type(rows, tmp_png.with_name("text_type.png"))
    marginal = {name: [0.1, 0.2, 0.3] for name in list(results)[2:]}
    save_marginal_over_recency(marginal, tmp_png.with_name("marginal.png"))
    save_trained_vs_untrained(results, tmp_png.with_name("trained.png"))
    save_hybrid_weight_sweep(
        {
            "0.1": {"mean": 0.6, "ci_low": 0.55, "ci_high": 0.65},
            "0.3": {"mean": 0.58, "ci_low": 0.53, "ci_high": 0.63},
        },
        0.62,
        0.18,
        tmp_png.with_name("hybrid.png"),
    )


def test_oracle_heatmap_has_visible_structure(tmp_path: Path) -> None:
    seq = 128
    attn = torch.zeros(12, seq, seq)
    for i in range(seq):
        for j in range(i + 1):
            attn[5, i, j] = (1.0 / float(i - j + 1)) + (0.5 if j == 0 else 0.0)
        attn[5, i] = attn[5, i] / attn[5, i].sum()

    out = tmp_path / "heatmap.png"
    save_oracle_heatmap(attn, seq, out)
    assert out.exists() and out.stat().st_size > 5000

    image = Image.open(out)
    colors = image.getcolors(maxcolors=10_000_000)
    assert colors is not None
    assert len(colors) > 100

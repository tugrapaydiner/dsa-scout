from __future__ import annotations

from types import SimpleNamespace

import torch

from dsa_scout.indexer import LightningIndexer
from dsa_scout.scorers import (
    score_lightning_trained,
    score_lightning_trained_plus_recency,
    score_lightning_untrained,
    score_linear,
    score_preview_attn,
    score_random,
    score_recency,
    score_window_sink,
)


def fake_model() -> SimpleNamespace:
    c_attn = SimpleNamespace(
        weight=torch.randn(768, 2304),
        bias=torch.randn(2304),
    )
    attn = SimpleNamespace(c_attn=c_attn)
    block = SimpleNamespace(attn=attn)
    return SimpleNamespace(
        transformer=SimpleNamespace(h=[block] * 12), config=SimpleNamespace(n_head=12)
    )


def test_basic_scorers_shape(small_hidden: torch.Tensor) -> None:
    for scorer in (score_random, score_recency, score_window_sink, score_linear):
        assert scorer(small_hidden).shape == (32, 8)


def test_preview_attn_shape(small_hidden: torch.Tensor) -> None:
    assert score_preview_attn(small_hidden, fake_model(), layer_idx=0).shape == (32, 8)


def test_lightning_scorers_shape(small_hidden: torch.Tensor) -> None:
    indexer = LightningIndexer().eval()
    assert score_lightning_untrained(small_hidden, indexer).shape == (32, 8)
    assert score_lightning_trained(small_hidden, indexer).shape == (32, 8)


def test_recency_prefers_recent_complete_block(small_hidden: torch.Tensor) -> None:
    scores = score_recency(small_hidden)
    assert scores[31, 7] > scores[31, 0]


def test_window_sink_promotes_first_block(small_hidden: torch.Tensor) -> None:
    scores = score_window_sink(small_hidden)
    assert int(scores[31].argmax().item()) == 0


def test_hybrid_scorer_recovers_recency_at_weight_zero(small_hidden: torch.Tensor) -> None:
    indexer = LightningIndexer().eval()
    hybrid = score_lightning_trained_plus_recency(small_hidden, indexer, weight=0.0)
    recency = score_recency(small_hidden)
    for k in (1, 2, 4):
        hybrid_top = hybrid.topk(k=k, dim=-1).indices
        ref_top = recency.topk(k=k, dim=-1).indices
        assert torch.equal(hybrid_top.sort(dim=-1).values, ref_top.sort(dim=-1).values)


def test_hybrid_scorer_recovers_lightning_at_weight_one(small_hidden: torch.Tensor) -> None:
    indexer = LightningIndexer().eval()
    hybrid = score_lightning_trained_plus_recency(small_hidden, indexer, weight=1.0)
    lightning = score_lightning_trained(small_hidden, indexer)
    for k in (1, 2, 4):
        hybrid_top = hybrid.topk(k=k, dim=-1).indices
        ref_top = lightning.topk(k=k, dim=-1).indices
        assert torch.equal(hybrid_top.sort(dim=-1).values, ref_top.sort(dim=-1).values)

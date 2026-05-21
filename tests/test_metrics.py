from __future__ import annotations

import pytest
import torch

from dsa_scout.metrics import (
    causal_block_mask,
    conditional_recall,
    conditional_recall_loop,
    topk_recall,
    topk_recall_loop,
)


def test_topk_recall_perfect() -> None:
    scores = torch.arange(64, dtype=torch.float32).view(8, 8)
    assert topk_recall(scores, scores.clone(), k=1, m=1) == pytest.approx(1.0)


def test_topk_recall_anti_correlated() -> None:
    oracle = torch.arange(64, dtype=torch.float32).view(8, 8)
    scores = -oracle
    assert topk_recall(scores, oracle, k=1, m=1) == pytest.approx(0.0)


def test_causal_block_mask_no_partial_block_leakage() -> None:
    mask = causal_block_mask(seq=10, n_comp=3, device=torch.device("cpu"), m=4)
    assert not bool(mask[2, 0])
    assert bool(mask[3, 0])
    assert not bool(mask[6, 1])
    assert bool(mask[7, 1])


def test_vectorized_topk_matches_loop() -> None:
    torch.manual_seed(0)
    scores = torch.randn(24, 6)
    oracle = torch.randn(24, 6)
    assert topk_recall(scores, oracle, k=3, m=4) == pytest.approx(
        topk_recall_loop(scores, oracle, k=3, m=4)
    )


def test_vectorized_conditional_matches_loop() -> None:
    torch.manual_seed(1)
    scores = torch.randn(24, 6)
    oracle = torch.randn(24, 6)
    recency = torch.randn(24, 6)
    assert conditional_recall(scores, oracle, 3, recency, m=4) == pytest.approx(
        conditional_recall_loop(scores, oracle, 3, recency, m=4)
    )

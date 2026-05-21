from __future__ import annotations

import torch

from dsa_scout.indexer import LightningIndexer, compress_mean


def test_indexer_shape(small_hidden: torch.Tensor) -> None:
    ix = LightningIndexer().eval()
    out = ix(small_hidden)
    assert out.shape == (32, 8)


def test_indexer_deterministic(small_hidden: torch.Tensor) -> None:
    ix = LightningIndexer().eval()
    with torch.no_grad():
        first = ix(small_hidden)
        second = ix(small_hidden)
    assert torch.allclose(first, second)


def test_compress_mean_drops_partial_block() -> None:
    x = torch.arange(15, dtype=torch.float32).view(15, 1)
    out = compress_mean(x, m=4)
    assert out.shape == (3, 1)
    assert torch.allclose(out[:, 0], torch.tensor([1.5, 5.5, 9.5]))

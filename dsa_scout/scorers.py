"""Sparse-attention block scorers."""

from __future__ import annotations

from typing import Any, cast

import torch

from dsa_scout.indexer import LightningIndexer, compress_mean


def score_random(hidden: torch.Tensor, m: int = 4, seed: int = 42) -> torch.Tensor:
    """Return deterministic random scores.

    Args:
        hidden: Tensor with shape ``[seq, 768]``.
        m: Compression rate.
        seed: Random seed.

    Returns:
        Score tensor with shape ``[seq, seq // m]``.
    """
    seq, _ = hidden.shape
    n_comp = seq // m
    gen = torch.Generator(device=hidden.device).manual_seed(seed)
    return torch.rand(seq, n_comp, generator=gen, device=hidden.device)


def score_recency(hidden: torch.Tensor, m: int = 4) -> torch.Tensor:
    """Score blocks by recency using each block's final token index.

    Args:
        hidden: Tensor with shape ``[seq, 768]``.
        m: Compression rate.

    Returns:
        Score tensor with shape ``[seq, seq // m]``.
    """
    seq, _ = hidden.shape
    n_comp = seq // m
    block_end = ((torch.arange(n_comp, device=hidden.device).float() + 1.0) * m) - 1.0
    query_pos = torch.arange(seq, device=hidden.device).float()
    return -(query_pos.unsqueeze(1) - block_end.unsqueeze(0)).abs()


def score_window_sink(hidden: torch.Tensor, m: int = 4, n_sinks: int = 1) -> torch.Tensor:
    """Score blocks with a window plus attention-sink ranking.

    Args:
        hidden: Tensor with shape ``[seq, 768]``.
        m: Compression rate.
        n_sinks: Number of first blocks to promote as sinks.

    Returns:
        Score tensor with shape ``[seq, seq // m]``.
    """
    recency = score_recency(hidden, m=m)
    n_comp = recency.shape[1]
    sink_mask = torch.zeros(n_comp, device=hidden.device, dtype=torch.bool)
    sink_mask[:n_sinks] = True
    sink_bonus = torch.where(
        sink_mask.unsqueeze(0),
        torch.full_like(recency, 1_000_000.0),
        torch.zeros_like(recency),
    )
    return recency + sink_bonus


def score_linear(hidden: torch.Tensor, m: int = 4) -> torch.Tensor:
    """Score blocks by normalized hidden-state dot product.

    Args:
        hidden: Tensor with shape ``[seq, 768]``.
        m: Compression rate.

    Returns:
        Score tensor with shape ``[seq, seq // m]``.
    """
    _, d_model = hidden.shape
    compressed = compress_mean(hidden, m)
    scores: torch.Tensor = (hidden @ compressed.T) / (float(d_model) ** 0.5)
    return scores


def get_layer_head_qk(
    model: Any,
    layer_idx: int,
    head_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract one GPT-2 attention head's Q/K projection weights.

    Args:
        model: GPT-2 model.
        layer_idx: Layer index.
        head_idx: Head index.

    Returns:
        Q weight, Q bias, K weight, K bias. Weights have shape ``[768, 64]``.
    """
    layer = model.transformer.h[layer_idx].attn
    weight = cast(torch.Tensor, layer.c_attn.weight.detach())
    bias = cast(torch.Tensor, layer.c_attn.bias.detach())
    d_model = weight.shape[0]
    head_dim = d_model // int(model.config.n_head)
    start = head_idx * head_dim
    end = start + head_dim
    return (
        weight[:, :d_model][:, start:end],
        bias[:d_model][start:end],
        weight[:, d_model : 2 * d_model][:, start:end],
        bias[d_model : 2 * d_model][start:end],
    )


def get_layer0_head0_qk(
    model: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract layer-0 head-0 GPT-2 Q/K projection weights."""
    return get_layer_head_qk(model, layer_idx=0, head_idx=0)


def score_preview_attn(
    hidden: torch.Tensor,
    model: Any,
    layer_idx: int = 0,
    head_idx: int = 0,
    m: int = 4,
) -> torch.Tensor:
    """Score compressed blocks with one GPT-2 attention head's Q/K projections.

    Args:
        hidden: Tensor with shape ``[seq, 768]``.
        model: GPT-2 model.
        layer_idx: Target layer index.
        head_idx: Target head index.
        m: Compression rate.

    Returns:
        Score tensor with shape ``[seq, seq // m]``.
    """
    w_q, b_q, w_k, b_k = get_layer_head_qk(model, layer_idx=layer_idx, head_idx=head_idx)
    q = hidden @ w_q + b_q
    k_full = hidden @ w_k + b_k
    k_comp = compress_mean(k_full, m)
    scores: torch.Tensor = (q @ k_comp.T) / (float(q.shape[-1]) ** 0.5)
    return scores


def score_lightning_untrained(hidden: torch.Tensor, indexer: LightningIndexer) -> torch.Tensor:
    """Score blocks with an untrained Lightning Indexer."""
    scores: torch.Tensor = indexer(hidden)
    return scores


def score_lightning_trained(
    hidden: torch.Tensor, trained_indexer: LightningIndexer
) -> torch.Tensor:
    """Score blocks with a trained Lightning Indexer."""
    scores: torch.Tensor = trained_indexer(hidden)
    return scores


def score_lightning_trained_plus_recency(
    hidden: torch.Tensor,
    trained_indexer: LightningIndexer,
    m: int = 4,
    weight: float = 0.5,
) -> torch.Tensor:
    """Score blocks with a trained Lightning Indexer plus recency prior.

    Args:
        hidden: Tensor with shape ``[seq, 768]``.
        trained_indexer: Trained Lightning Indexer instance.
        m: Compression rate.
        weight: Blend in ``[0, 1]``. ``0`` ranks exactly as recency; ``1``
            ranks exactly as trained Lightning.

    Returns:
        Score tensor with shape ``[seq, seq // m]``.
    """
    if not 0.0 <= weight <= 1.0:
        msg = f"weight must be in [0, 1], got {weight}"
        raise ValueError(msg)
    lightning = score_lightning_trained(hidden, trained_indexer)
    recency = score_recency(hidden, m=m)
    return (weight * _row_zscore(lightning)) + ((1.0 - weight) * _row_zscore(recency))


def _row_zscore(scores: torch.Tensor) -> torch.Tensor:
    mean = scores.mean(dim=-1, keepdim=True)
    std = scores.std(dim=-1, keepdim=True).clamp_min(1e-6)
    normalized: torch.Tensor = (scores - mean) / std
    return normalized

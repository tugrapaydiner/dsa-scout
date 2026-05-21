"""Vectorized metrics for sparse block selection."""

from __future__ import annotations

import torch


def aggregate_oracle_to_blocks(oracle_attn: torch.Tensor, m: int = 4) -> torch.Tensor:
    """Aggregate token-level oracle attention into compressed key blocks.

    Args:
        oracle_attn: Attention tensor with shape ``[seq, seq]``.
        m: Compression rate.

    Returns:
        Tensor with shape ``[seq, seq // m]`` containing total attention mass per block.
    """
    seq = oracle_attn.shape[0]
    n_comp = seq // m
    seq_trim = n_comp * m
    return oracle_attn[:, :seq_trim].reshape(seq, n_comp, m).sum(dim=-1)


def causal_block_mask(seq: int, n_comp: int, device: torch.device, m: int = 4) -> torch.Tensor:
    """Build a leakage-safe block causal mask.

    Args:
        seq: Query sequence length.
        n_comp: Number of compressed key blocks.
        device: Torch device.
        m: Compression rate.

    Returns:
        Boolean tensor with shape ``[seq, n_comp]``. A block is valid only if its final
        token index is less than or equal to the query index.
    """
    block_idx = torch.arange(n_comp, device=device).unsqueeze(0)
    query_pos = torch.arange(seq, device=device).unsqueeze(1)
    block_end = ((block_idx + 1) * m) - 1
    return block_end <= query_pos


def topk_recall(scores: torch.Tensor, oracle_blocked: torch.Tensor, k: int, m: int = 4) -> float:
    """Compute vectorized mean top-k recall against oracle block attention.

    Args:
        scores: Scorer output with shape ``[seq, seq // m]``.
        oracle_blocked: Blocked oracle attention with shape ``[seq, seq // m]``.
        k: Number of compressed blocks selected.
        m: Compression rate.

    Returns:
        Mean fraction of oracle top-k blocks recovered by the scorer.
    """
    seq, n_comp = scores.shape
    if k > n_comp:
        return 0.0
    causal = causal_block_mask(seq, n_comp, scores.device, m)
    keep = causal.sum(dim=1) >= (k + 1)
    if not bool(keep.any()):
        return 0.0
    scores_m = scores.masked_fill(~causal, float("-inf"))
    oracle_m = oracle_blocked.masked_fill(~causal, float("-inf"))
    scorer_top = scores_m.topk(k, dim=1).indices
    oracle_top = oracle_m.topk(k, dim=1).indices
    scorer_mask = torch.zeros(seq, n_comp, dtype=torch.bool, device=scores.device)
    oracle_mask = torch.zeros_like(scorer_mask)
    scorer_mask.scatter_(1, scorer_top, True)
    oracle_mask.scatter_(1, oracle_top, True)
    overlap = (scorer_mask & oracle_mask).sum(dim=1).to(torch.float32) / float(k)
    return float(overlap[keep].mean().item())


def conditional_recall(
    scores: torch.Tensor,
    oracle_blocked: torch.Tensor,
    k: int,
    recency_scores: torch.Tensor,
    m: int = 4,
) -> float:
    """Measure oracle blocks missed by recency and recovered by another scorer.

    Args:
        scores: Candidate scorer output with shape ``[seq, seq // m]``.
        oracle_blocked: Blocked oracle attention with shape ``[seq, seq // m]``.
        k: Number of compressed blocks selected.
        recency_scores: Recency baseline scores with shape ``[seq, seq // m]``.
        m: Compression rate.

    Returns:
        Mean fraction of recency-missed oracle top-k blocks recovered by ``scores``.
    """
    seq, n_comp = scores.shape
    if k > n_comp:
        return 0.0
    causal = causal_block_mask(seq, n_comp, scores.device, m)
    valid_rows = causal.sum(dim=1) >= (k + 1)
    if not bool(valid_rows.any()):
        return 0.0
    scores_m = scores.masked_fill(~causal, float("-inf"))
    oracle_m = oracle_blocked.masked_fill(~causal, float("-inf"))
    recency_m = recency_scores.masked_fill(~causal, float("-inf"))

    scorer_top = scores_m.topk(k, dim=1).indices
    oracle_top = oracle_m.topk(k, dim=1).indices
    recency_top = recency_m.topk(k, dim=1).indices

    scorer_mask = torch.zeros(seq, n_comp, dtype=torch.bool, device=scores.device)
    oracle_mask = torch.zeros_like(scorer_mask)
    recency_mask = torch.zeros_like(scorer_mask)
    scorer_mask.scatter_(1, scorer_top, True)
    oracle_mask.scatter_(1, oracle_top, True)
    recency_mask.scatter_(1, recency_top, True)

    missed = oracle_mask & ~recency_mask
    denom = missed.sum(dim=1)
    keep = valid_rows & (denom > 0)
    if not bool(keep.any()):
        return 0.0
    recovered = (scorer_mask & missed).sum(dim=1).to(torch.float32)
    values = recovered[keep] / denom[keep].to(torch.float32)
    return float(values.mean().item())


def topk_recall_loop(
    scores: torch.Tensor, oracle_blocked: torch.Tensor, k: int, m: int = 4
) -> float:
    """Reference loop implementation used by tests.

    Args:
        scores: Scorer output with shape ``[seq, seq // m]``.
        oracle_blocked: Blocked oracle attention with shape ``[seq, seq // m]``.
        k: Number of compressed blocks selected.
        m: Compression rate.

    Returns:
        Mean top-k recall.
    """
    seq, n_comp = scores.shape
    causal = causal_block_mask(seq, n_comp, scores.device, m)
    scores_m = scores.masked_fill(~causal, float("-inf"))
    oracle_m = oracle_blocked.masked_fill(~causal, float("-inf"))
    recalls: list[float] = []
    for t in range(seq):
        if int(causal[t].sum().item()) < k + 1:
            continue
        scorer_top = set(scores_m[t].topk(k).indices.tolist())
        oracle_top = set(oracle_m[t].topk(k).indices.tolist())
        recalls.append(len(scorer_top & oracle_top) / k)
    if not recalls:
        return 0.0
    return float(torch.tensor(recalls, dtype=torch.float32).mean().item())


def conditional_recall_loop(
    scores: torch.Tensor,
    oracle_blocked: torch.Tensor,
    k: int,
    recency_scores: torch.Tensor,
    m: int = 4,
) -> float:
    """Reference loop implementation for conditional recall tests."""
    seq, n_comp = scores.shape
    causal = causal_block_mask(seq, n_comp, scores.device, m)
    scores_m = scores.masked_fill(~causal, float("-inf"))
    oracle_m = oracle_blocked.masked_fill(~causal, float("-inf"))
    recency_m = recency_scores.masked_fill(~causal, float("-inf"))
    captured: list[float] = []
    for t in range(seq):
        if int(causal[t].sum().item()) < k + 1:
            continue
        oracle_top = set(oracle_m[t].topk(k).indices.tolist())
        recency_top = set(recency_m[t].topk(k).indices.tolist())
        scorer_top = set(scores_m[t].topk(k).indices.tolist())
        missed = oracle_top - recency_top
        if missed:
            captured.append(len(scorer_top & missed) / len(missed))
    if not captured:
        return 0.0
    return float(torch.tensor(captured, dtype=torch.float32).mean().item())

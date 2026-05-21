"""Lightning Indexer implementation used by DSA-Scout."""

from __future__ import annotations

import torch
from torch import nn


def compress_mean(x: torch.Tensor, m: int = 4) -> torch.Tensor:
    """Mean-pool contiguous token blocks.

    Args:
        x: Tensor with shape ``[seq, d_model]``.
        m: Compression rate.

    Returns:
        Tensor with shape ``[seq // m, d_model]``. Final partial blocks are dropped.
    """
    seq, d_model = x.shape
    seq_trim = (seq // m) * m
    return x[:seq_trim].reshape(seq // m, m, d_model).mean(dim=1)


class LightningIndexer(nn.Module):
    """Scaled Lightning Indexer block scorer for GPT-2 small.

    Args:
        d_model: GPT-2 hidden dimension.
        n_heads: Number of indexer heads.
        c_i: Per-head indexer dimension.
        d_c: Compressed query latent dimension.
        m: Compression rate.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        c_i: int = 32,
        d_c: int = 128,
        m: int = 4,
    ) -> None:
        super().__init__()
        self.m = m
        self.n_heads = n_heads
        self.c_i = c_i
        self.w_dq = nn.Linear(d_model, d_c, bias=False)
        self.w_iuq = nn.Linear(d_c, n_heads * c_i, bias=False)
        self.w_w = nn.Linear(d_model, n_heads, bias=False)
        self.w_k = nn.Linear(d_model, c_i, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Score compressed key blocks for every query token.

        Args:
            hidden: GPT-2 hidden states with shape ``[seq, 768]``.

        Returns:
            Score tensor with shape ``[seq, seq // m]``.
        """
        seq, _ = hidden.shape
        c_q = self.w_dq(hidden)
        q_i = self.w_iuq(c_q).view(seq, self.n_heads, self.c_i)
        compressed = compress_mean(hidden, self.m)
        k_i_comp = self.w_k(compressed)
        w_i = self.w_w(hidden)
        per_head = torch.einsum("thd,sd->ths", q_i, k_i_comp)
        per_head = torch.relu(per_head)
        scores: torch.Tensor = (per_head * w_i.unsqueeze(-1)).sum(dim=1)
        return scores

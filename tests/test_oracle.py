from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from dsa_scout.oracle import get_oracle_and_hiddens


class FakeBatch(dict[str, torch.Tensor]):
    def to(self, _device: torch.device) -> FakeBatch:
        return self


class FakeTokenizer:
    def __call__(self, _text: str, **_kwargs: Any) -> FakeBatch:
        return FakeBatch({"input_ids": torch.tensor([[0, 1, 2, 3]])})

    def decode(self, ids: list[int]) -> str:
        return f"tok{ids[0]}"


class FakeModel:
    def __call__(self, **_kwargs: Any) -> SimpleNamespace:
        base = torch.tril(torch.ones(4, 4))
        base = base / base.sum(dim=-1, keepdim=True)
        attentions = tuple(base.view(1, 1, 4, 4).repeat(1, 2, 1, 1) for _ in range(12))
        hiddens = tuple(torch.zeros(1, 4, 768) for _ in range(13))
        return SimpleNamespace(attentions=attentions, hidden_states=hiddens)


def test_oracle_shapes_and_tokens() -> None:
    attn, hiddens, tokens = get_oracle_and_hiddens(
        "x",
        FakeModel(),
        FakeTokenizer(),
        torch.device("cpu"),
        max_length=4,
    )
    assert attn.shape == (12, 4, 4)
    assert hiddens.shape == (13, 4, 768)
    assert tokens == ["tok0", "tok1", "tok2", "tok3"]
    assert torch.allclose(attn[0].sum(dim=-1), torch.ones(4))

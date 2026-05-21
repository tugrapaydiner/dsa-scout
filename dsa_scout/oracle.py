"""GPT-2 oracle attention extraction."""

from __future__ import annotations

from typing import Any, cast

import torch


@torch.no_grad()
def get_oracle_and_hiddens(
    text: str,
    model: Any,
    tok: Any,
    device: torch.device,
    max_length: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Extract head-averaged attention and hidden states.

    Args:
        text: Input text.
        model: GPT-2 model.
        tok: GPT-2 tokenizer.
        device: Inference device.
        max_length: Maximum GPT-2 tokens.

    Returns:
        ``attn`` with shape ``[num_layers, seq, seq]``, ``hiddens`` with shape
        ``[num_layers + 1, seq, 768]``, and decoded token strings.
    """
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
    attentions = cast(tuple[torch.Tensor, ...], outputs.attentions)
    hidden_states = cast(tuple[torch.Tensor, ...], outputs.hidden_states)
    attn = torch.stack([layer_attn.mean(dim=1).squeeze(0) for layer_attn in attentions])
    hiddens = torch.stack(hidden_states).squeeze(1)
    input_ids = cast(torch.Tensor, inputs["input_ids"])[0]
    tokens = [str(tok.decode([int(tid)])) for tid in input_ids]
    return attn, hiddens, tokens

"""KL distillation training for the Lightning Indexer."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch.optim import Adam

from dsa_scout.corpus import CorpusSample, load_training_texts
from dsa_scout.indexer import LightningIndexer
from dsa_scout.logging import get_logger
from dsa_scout.metrics import aggregate_oracle_to_blocks, causal_block_mask

log = get_logger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration for KL distillation."""

    steps: int = 2000
    lr: float = 1e-3
    target_layers: tuple[int, ...] = (4, 5, 6, 7, 8)
    log_every: int = 100
    seed: int = 0
    max_length: int = 1024
    oracle_scale: float = 32.0
    early_stop_patience: int = 200
    early_stop_min_delta: float = 0.01
    warmup_steps: int = 50
    min_lr: float = 1e-5
    grad_clip: float = 1.0


@dataclass(frozen=True)
class TrainingExample:
    """Precomputed distillation target.

    Attributes:
        hidden: Target-layer input states with shape ``[seq, 768]``.
        oracle_blocked: Oracle block mass with shape ``[seq, seq // 4]``.
    """

    hidden: torch.Tensor
    oracle_blocked: torch.Tensor


def train_indexer(
    indexer: LightningIndexer,
    training_texts: list[str],
    model: Any,
    tok: Any,
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, object]:
    """KL-distill the indexer against head-averaged oracle attention.

    Args:
        indexer: Lightning Indexer to train.
        training_texts: Held-out training texts disjoint from eval corpus.
        model: GPT-2 model.
        tok: GPT-2 tokenizer.
        device: Training device.
        config: Training hyperparameters.

    Returns:
        JSON-serializable training log with per-step losses.
    """
    torch.manual_seed(config.seed)
    examples = precompute_training_examples(indexer, training_texts, model, tok, device, config)
    return train_indexer_on_examples(indexer, examples, device, config)


@torch.no_grad()
def precompute_training_examples(
    indexer: LightningIndexer,
    training_texts: list[str],
    model: Any,
    tok: Any,
    device: torch.device,
    config: TrainingConfig,
) -> list[TrainingExample]:
    """Precompute GPT-2 hiddens and oracle block targets for training."""
    examples: list[TrainingExample] = []
    for text_idx, text in enumerate(training_texts):
        inputs = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_length,
        ).to(device)
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
        for target_layer in config.target_layers:
            attn = outputs.attentions[target_layer].mean(dim=1).squeeze(0)
            hidden = outputs.hidden_states[target_layer].squeeze(0)
            oracle_blocked = aggregate_oracle_to_blocks(attn, m=indexer.m)
            examples.append(
                TrainingExample(hidden=hidden.detach(), oracle_blocked=oracle_blocked.detach())
            )
        log.info("precomputed training text %s/%s", text_idx + 1, len(training_texts))
    return examples


def train_indexer_on_examples(
    indexer: LightningIndexer,
    examples: list[TrainingExample],
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, object]:
    """Train an indexer from precomputed examples.

    Args:
        indexer: Lightning Indexer.
        examples: Precomputed training examples.
        device: Training device.
        config: Training hyperparameters.

    Returns:
        JSON-serializable training log.
    """
    if not examples:
        msg = "training requires at least one precomputed example"
        raise ValueError(msg)
    torch.manual_seed(config.seed)
    indexer.train()
    opt = Adam(indexer.parameters(), lr=config.lr)
    scheduler = build_lr_schedule(opt, config)
    losses: list[float] = []
    gen = torch.Generator().manual_seed(config.seed)
    perm = torch.empty(0, dtype=torch.long)
    best_loss = float("inf")
    best_step = -1
    best_state_dict: dict[str, torch.Tensor] | None = None
    fallback_loss = float("inf")
    fallback_step = -1
    fallback_state_dict: dict[str, torch.Tensor] | None = None
    best_window_indices: list[int] = []
    fallback_window_indices: list[int] = []
    step_example_indices: list[int] = []
    steps_since_improvement = 0
    actual_steps = 0
    for step in range(config.steps):
        position = step % len(examples)
        if position == 0:
            perm = torch.randperm(len(examples), generator=gen)
        example_idx = int(perm[position].item())
        step_example_indices.append(example_idx)
        example = examples[example_idx]
        hidden = example.hidden.to(device)
        oracle_blocked = example.oracle_blocked.to(device)
        scores = indexer(hidden)
        loss = kl_distillation_loss(
            scores,
            oracle_blocked,
            m=indexer.m,
            oracle_scale=config.oracle_scale,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(indexer.parameters(), max_norm=config.grad_clip)
        opt.step()
        scheduler.step()
        current_loss = float(loss.detach().cpu().item())
        losses.append(current_loss)
        actual_steps += 1

        if current_loss < fallback_loss:
            fallback_loss = current_loss
            fallback_step = step
            fallback_window_indices = [example_idx]
            fallback_state_dict = {
                name: tensor.detach().clone().cpu() for name, tensor in indexer.state_dict().items()
            }

        if len(losses) >= 50:
            rolling = sum(losses[-50:]) / 50.0
            if rolling < best_loss - config.early_stop_min_delta:
                best_loss = rolling
                best_step = step
                best_state_dict = {
                    name: tensor.detach().clone().cpu()
                    for name, tensor in indexer.state_dict().items()
                }
                best_window_indices = step_example_indices[-50:]
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1

            if steps_since_improvement >= config.early_stop_patience:
                log.info(
                    "early stop at step %s (best rolling %.4f)",
                    step + 1,
                    best_loss,
                )
                break

        if step % config.log_every == 0:
            log.info("train step %s/%s loss=%.4f", step, config.steps, losses[-1])
    if best_state_dict is None:
        best_loss = fallback_loss
        best_step = fallback_step
        best_state_dict = fallback_state_dict
        best_window_indices = fallback_window_indices
    if best_state_dict is not None:
        indexer.load_state_dict(
            {name: tensor.to(device) for name, tensor in best_state_dict.items()}
        )
    indexer.eval()
    best_rolling_loss = min(losses) if best_loss == float("inf") else best_loss
    checkpoint_eval_loss = _mean_loss_for_examples(
        indexer,
        examples,
        best_window_indices,
        device,
        config,
    )
    return {
        "losses": losses,
        "initial_loss": losses[0],
        "last_loss": losses[-1],
        "final_loss": best_rolling_loss,
        "best_loss": best_rolling_loss,
        "best_checkpoint_eval_loss": checkpoint_eval_loss,
        "best_step": best_step,
        "loss_delta": losses[0] - best_rolling_loss,
        "actual_steps": actual_steps,
        "best_rolling_loss": best_rolling_loss,
        "early_stopped": actual_steps < config.steps,
        "config": asdict(config),
        "num_examples": len(examples),
    }


@torch.no_grad()
def _mean_loss_for_examples(
    indexer: LightningIndexer,
    examples: list[TrainingExample],
    indices: list[int],
    device: torch.device,
    config: TrainingConfig,
) -> float:
    if not indices:
        return 0.0
    losses: list[float] = []
    for example_idx in indices:
        example = examples[example_idx]
        scores = indexer(example.hidden.to(device))
        loss = kl_distillation_loss(
            scores,
            example.oracle_blocked.to(device),
            m=indexer.m,
            oracle_scale=config.oracle_scale,
        )
        losses.append(float(loss.detach().cpu().item()))
    return sum(losses) / len(losses)


def build_lr_schedule(opt: Adam, config: TrainingConfig) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup, then cosine decay to ``config.min_lr``."""

    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return float(step + 1) / float(max(1, config.warmup_steps))
        progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = config.min_lr / config.lr
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)


def kl_distillation_loss(
    scores: torch.Tensor,
    oracle_blocked: torch.Tensor,
    m: int = 4,
    oracle_scale: float = 32.0,
) -> torch.Tensor:
    """Compute KL divergence from oracle block distribution to predicted scores."""
    seq, n_comp = scores.shape
    causal = causal_block_mask(seq, n_comp, scores.device, m)
    valid_rows = causal.any(dim=1)
    if not bool(valid_rows.any()):
        return scores.sum() * 0.0
    scores_m = scores.masked_fill(~causal, float("-inf"))
    oracle_m = (oracle_blocked * oracle_scale).masked_fill(~causal, float("-inf"))
    row_mask = causal[valid_rows]
    log_probs_pred = functional.log_softmax(scores_m[valid_rows], dim=-1)
    probs_oracle = functional.softmax(oracle_m[valid_rows], dim=-1)
    log_probs_pred = torch.where(row_mask, log_probs_pred, torch.zeros_like(log_probs_pred))
    probs_oracle = torch.where(row_mask, probs_oracle, torch.zeros_like(probs_oracle))
    return functional.kl_div(log_probs_pred, probs_oracle, reduction="batchmean")


def load_training_corpus(
    tok: Any,
    eval_samples: list[CorpusSample],
    n_texts: int = 50,
    cache_dir: Path = Path("corpus_cache"),
) -> list[str]:
    """Load held-out training texts disjoint from evaluation samples."""
    return load_training_texts(
        tok,
        eval_samples=eval_samples,
        n_texts=n_texts,
        cache_dir=cache_dir,
    )


def save_training_artifacts(
    indexer: LightningIndexer,
    training_log: dict[str, object],
    checkpoint_path: Path,
    log_path: Path,
) -> None:
    """Save trained indexer checkpoint and JSON training log."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": indexer.state_dict()}, checkpoint_path)
    log_path.write_text(json.dumps(training_log, indent=2), encoding="utf-8")


def load_trained_indexer(checkpoint_path: Path, device: torch.device) -> LightningIndexer:
    """Load a trained indexer checkpoint."""
    indexer = LightningIndexer().to(device).eval()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, dict):
        msg = f"Invalid checkpoint state_dict in {checkpoint_path}"
        raise TypeError(msg)
    indexer.load_state_dict(state_dict)
    return indexer

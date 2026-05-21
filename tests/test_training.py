from __future__ import annotations

from pathlib import Path
from typing import cast

import torch

from dsa_scout.indexer import LightningIndexer
from dsa_scout.scorers import score_recency
from dsa_scout.training import (
    TrainingConfig,
    TrainingExample,
    build_lr_schedule,
    load_trained_indexer,
    save_training_artifacts,
    train_indexer_on_examples,
)


def test_training_decreases_loss(device: torch.device) -> None:
    torch.manual_seed(0)
    hidden = torch.randn(32, 768, device=device)
    oracle = torch.softmax(score_recency(hidden), dim=-1)
    indexer = LightningIndexer().to(device)
    config = TrainingConfig(steps=40, lr=1e-3, log_every=100, max_length=32)
    log = train_indexer_on_examples(
        indexer,
        [TrainingExample(hidden=hidden, oracle_blocked=oracle)],
        device,
        config,
    )
    assert cast(float, log["final_loss"]) < cast(float, log["initial_loss"])


def test_training_is_deterministic_under_seed(device: torch.device) -> None:
    torch.manual_seed(99)
    examples = [
        TrainingExample(
            hidden=torch.randn(16, 768, device=device),
            oracle_blocked=torch.softmax(torch.randn(16, 4, device=device), dim=-1),
        )
        for _ in range(3)
    ]
    config = TrainingConfig(steps=9, lr=1e-3, log_every=100, max_length=16, seed=123)

    torch.manual_seed(7)
    first_indexer = LightningIndexer().to(device)
    first = train_indexer_on_examples(first_indexer, examples, device, config)

    torch.manual_seed(7)
    second_indexer = LightningIndexer().to(device)
    second = train_indexer_on_examples(second_indexer, examples, device, config)

    assert cast(list[float], first["losses"]) == cast(list[float], second["losses"])


def test_kl_loss_log_contains_config(device: torch.device) -> None:
    hidden = torch.randn(16, 768, device=device)
    oracle = torch.ones(16, 4, device=device)
    indexer = LightningIndexer().to(device)
    config = TrainingConfig(steps=2, lr=1e-3, log_every=100, max_length=16)
    log = train_indexer_on_examples(
        indexer,
        [TrainingExample(hidden=hidden, oracle_blocked=oracle)],
        device,
        config,
    )
    assert "config" in log
    assert len(cast(list[float], log["losses"])) == 2
    assert cast(int, log["best_step"]) >= 0
    assert "best_checkpoint_eval_loss" in log
    assert cast(float, log["final_loss"]) == cast(float, log["best_loss"])
    config_payload = cast(dict[str, object], log["config"])
    assert config_payload["warmup_steps"] == config.warmup_steps
    assert config_payload["grad_clip"] == config.grad_clip


def test_training_early_stop_records_actual_steps(device: torch.device) -> None:
    hidden = torch.randn(16, 768, device=device)
    oracle = torch.ones(16, 4, device=device)
    indexer = LightningIndexer().to(device)
    config = TrainingConfig(
        steps=80,
        lr=1e-3,
        log_every=100,
        max_length=16,
        early_stop_patience=5,
        early_stop_min_delta=1_000_000.0,
    )
    log = train_indexer_on_examples(
        indexer,
        [TrainingExample(hidden=hidden, oracle_blocked=oracle)],
        device,
        config,
    )
    assert bool(log["early_stopped"])
    assert cast(int, log["actual_steps"]) < config.steps
    assert cast(float, log["best_rolling_loss"]) < float("inf")


def test_lr_schedule_warms_up_and_decays(device: torch.device) -> None:
    indexer = LightningIndexer().to(device)
    config = TrainingConfig(steps=100, lr=1e-3, warmup_steps=10, min_lr=1e-5)
    opt = torch.optim.Adam(indexer.parameters(), lr=config.lr)
    scheduler = build_lr_schedule(opt, config)
    rates: list[float] = []
    for _ in range(config.steps):
        opt.step()
        scheduler.step()
        rates.append(float(opt.param_groups[0]["lr"]))
    assert rates[0] < rates[9]
    assert rates[-1] < rates[9]
    assert abs(rates[-1] - config.min_lr) < 2e-6


def test_save_and_load_training_artifacts(tmp_path: Path, device: torch.device) -> None:
    indexer = LightningIndexer().to(device).eval()
    checkpoint = tmp_path / "trained_indexer.pt"
    log_path = tmp_path / "training_log.json"
    training_log: dict[str, object] = {"initial_loss": 2.0, "final_loss": 1.0}
    save_training_artifacts(indexer, training_log, checkpoint, log_path)
    loaded = load_trained_indexer(checkpoint, device)
    hidden = torch.randn(16, 768, device=device)
    assert checkpoint.exists()
    assert log_path.exists()
    assert torch.allclose(indexer(hidden), loaded(hidden))

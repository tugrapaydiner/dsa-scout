from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from dsa_scout import cli, experiment
from dsa_scout.corpus import CorpusSample, TrainingCorpusRecord
from dsa_scout.experiment import SCORER_NAMES, summarize_results
from dsa_scout.indexer import LightningIndexer
from dsa_scout.training import TrainingConfig


def _sample(name: str = "sample", category: str = "prose") -> CorpusSample:
    return CorpusSample(
        category=category,
        name=name,
        text="synthetic text",
        token_count=40,
        unique_tokens=40,
        source="fixture",
        top_bigram_ratio=0.0,
        content_hash=name,
    )


def test_build_artifact_manifest_hashes_existing_artifacts(tmp_path: Path) -> None:
    plot = tmp_path / "plots" / "01_oracle_heatmap.png"
    plot.parent.mkdir()
    plot.write_text("png", encoding="utf-8")
    metadata = {
        "python": "3.12",
        "platform": "test",
        "device": "cpu",
        "versions": {},
        "corpus": {"content_hash": "abc"},
    }
    summary = {
        "middle_layer_top8": {
            "lightning_trained": {"mean": 0.5},
            "recency": {"mean": 0.6},
            "lightning_trained_plus_recency": {"mean": 0.55},
        }
    }

    manifest = experiment.build_artifact_manifest(tmp_path, metadata, summary)
    artifacts = cast(dict[str, dict[str, object]], manifest["artifacts"])

    assert artifacts["plots/01_oracle_heatmap.png"]["bytes"] == len("png")
    assert manifest["corpus_content_hash"] == "abc"
    assert isinstance(manifest["git_dirty"], bool)
    assert manifest["headline_metrics"]["trained_lightning_top8"] == 0.5


def test_summarize_results_includes_trained_delta(tmp_path: Path) -> None:
    recall = {name: np.ones((5, 12, 2), dtype=float) * 0.2 for name in SCORER_NAMES}
    recall["lightning_trained"][:] = 0.5
    recall["lightning_untrained"][:] = 0.3
    recall["lightning_trained_plus_recency"][:] = 0.7
    recall_vs_k = {name: {4: [0.1, 0.2], 8: [0.2], 16: [0.3], 32: [0.4]} for name in SCORER_NAMES}
    summary = summarize_results(recall, recall_vs_k, [], {}, tmp_path)
    trained_delta = cast(dict[str, float], summary["trained_vs_untrained_delta"])
    hybrid_deltas = cast(dict[str, dict[str, float]], summary["hybrid_deltas"])
    assert trained_delta["mean"] > 0
    assert hybrid_deltas["minus_lightning_trained"]["mean"] > 0
    assert hybrid_deltas["minus_recency"]["mean"] > 0


def test_retry_path_not_taken_for_normal_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[TrainingConfig] = []

    def fake_load_training_records(
        tok: Any,
        eval_samples: list[Any],
        n_texts: int,
        target_tokens: int,
        cache_dir: Path,
    ) -> list[TrainingCorpusRecord]:
        del tok, eval_samples, n_texts, target_tokens, cache_dir
        return [
            TrainingCorpusRecord(
                name="train",
                text="training text",
                token_count=16,
                unique_tokens=16,
                source="fixture",
                top_bigram_ratio=0.0,
                content_hash="abc",
            )
        ]

    def fake_train_indexer(
        indexer: LightningIndexer,
        training_texts: list[str],
        model: Any,
        tok: Any,
        device: torch.device,
        config: TrainingConfig,
    ) -> dict[str, object]:
        del indexer, training_texts, model, tok, device
        calls.append(config)
        return {
            "losses": [10.0, 11.0],
            "initial_loss": 10.0,
            "final_loss": 11.0,
            "loss_delta": -1.0,
            "actual_steps": 2,
            "best_rolling_loss": 2.0,
            "early_stopped": False,
            "config": {},
            "num_examples": 1,
        }

    monkeypatch.setattr(experiment, "load_training_records", fake_load_training_records)
    monkeypatch.setattr(experiment, "train_indexer", fake_train_indexer)

    _, log = experiment.train_or_load_indexer(
        model=object(),
        tok=object(),
        device=torch.device("cpu"),
        eval_samples=[],
        results_dir=tmp_path,
        cache_dir=tmp_path,
        force_train=True,
        config=TrainingConfig(steps=2, max_length=16),
    )

    assert len(calls) == 1
    assert cast(float, log["best_rolling_loss"]) == 2.0


def test_run_evaluation_with_synthetic_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seq = 40
    n_comp = seq // experiment.M
    torch.manual_seed(0)
    attn = torch.tril(torch.rand(12, seq, seq))
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    hiddens = torch.randn(12, seq, 768)

    def fake_oracle(
        text: str,
        model: Any,
        tok: Any,
        device: torch.device,
        max_length: int = 1024,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        del text, model, tok, device, max_length
        return attn, hiddens, [str(idx) for idx in range(seq)]

    def fake_score_layer(
        hidden: torch.Tensor,
        model: Any,
        layer_idx: int,
        untrained_indexers: dict[int, LightningIndexer],
        trained_indexer: LightningIndexer,
    ) -> dict[str, dict[int, torch.Tensor]]:
        del hidden, model, trained_indexer
        base = torch.arange(seq * n_comp, dtype=torch.float32).reshape(seq, n_comp)
        score_map: dict[str, dict[int, torch.Tensor]] = {}
        for scorer_idx, name in enumerate(experiment.SCORER_NAMES):
            score_map[name] = {
                seed: base + float(scorer_idx + layer_idx) + float(seed) * 0.01
                for seed in untrained_indexers
            }
        return score_map

    monkeypatch.setattr(experiment, "get_oracle_and_hiddens", fake_oracle)
    monkeypatch.setattr(experiment, "score_layer", fake_score_layer)
    (tmp_path / "plots").mkdir()
    (tmp_path / "results").mkdir()

    recall_by_layer, recall_vs_k, rows, marginal = experiment.run_evaluation(
        samples=[_sample()],
        model=object(),
        tok=object(),
        device=torch.device("cpu"),
        trained_indexer=LightningIndexer().eval(),
        plots_dir=tmp_path / "plots",
        results_dir=tmp_path / "results",
        max_length=seq,
    )

    assert recall_by_layer["recency"].shape == (5, 12, 1)
    assert recall_vs_k["recency"][8]
    assert rows
    assert marginal["lightning_trained"]
    assert (tmp_path / "plots" / "03_recall_by_layer.png").exists()
    assert (tmp_path / "results" / "hybrid_weight_sweep.json").exists()


def test_run_full_study_wires_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recall = {name: np.ones((5, 12, 1), dtype=float) * 0.2 for name in SCORER_NAMES}
    recall["lightning_trained"][:] = 0.4
    recall_vs_k = {name: {4: [0.1], 8: [0.2], 16: [0.3], 32: [0.4]} for name in SCORER_NAMES}
    training_log: dict[str, object] = {
        "losses": [3.0, 2.0],
        "initial_loss": 3.0,
        "last_loss": 2.0,
        "final_loss": 2.0,
        "best_loss": 2.0,
        "best_step": 1,
        "loss_delta": 1.0,
        "actual_steps": 2,
        "best_rolling_loss": 2.0,
        "early_stopped": False,
        "config": {},
        "num_examples": 1,
    }

    monkeypatch.setattr(experiment, "load_gpt2", lambda device: (object(), object()))
    monkeypatch.setattr(experiment, "run_hello_world", lambda model, tok, device: {"layers": 12})
    monkeypatch.setattr(experiment, "run_indexer_tests", lambda device: None)
    monkeypatch.setattr(
        experiment, "load_corpus", lambda tok, cache_dir, target_tokens: [_sample()]
    )
    monkeypatch.setattr(
        experiment,
        "train_or_load_indexer",
        lambda model, tok, device, eval_samples, results_dir, cache_dir, force_train, config: (
            LightningIndexer().eval(),
            training_log,
        ),
    )
    monkeypatch.setattr(
        experiment,
        "run_evaluation",
        lambda samples, model, tok, device, trained_indexer, plots_dir, results_dir, max_length: (
            recall,
            recall_vs_k,
            [{"scorer": "recency", "text_type": "prose", "recall": 0.2}],
            {"recency": [0.1], "lightning_trained": [0.2]},
        ),
    )
    result = experiment.run_full_study(base_dir=tmp_path, training_steps=2, max_length=40)

    assert "summary_stats" in result
    assert (tmp_path / "results" / "metadata.json").exists()
    assert (tmp_path / "results" / "manifest.json").exists()
    assert (tmp_path / "plots" / "07_training_curve.png").exists()


def test_run_smoke_with_synthetic_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seq = 16
    attn = torch.tril(torch.ones(12, seq, seq))
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    hiddens = torch.randn(12, seq, 768)
    monkeypatch.setattr(experiment, "load_gpt2", lambda device: (object(), object()))
    monkeypatch.setattr(experiment, "run_hello_world", lambda model, tok, device: {"layers": 12})
    monkeypatch.setattr(
        experiment,
        "get_oracle_and_hiddens",
        lambda text, model, tok, device, max_length: (
            attn,
            hiddens,
            [str(idx) for idx in range(seq)],
        ),
    )

    result = experiment.run_smoke(base_dir=tmp_path)

    assert result["hello_world"] == {"layers": 12}
    assert (tmp_path / "plots" / "smoke_oracle_heatmap.png").exists()


def test_cli_reproduce_and_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = CliRunner()
    monkeypatch.setattr(
        experiment,
        "run_full_study",
        lambda base_dir, force_train, training_steps, max_length: {
            "summary_stats": {"force_train": force_train, "steps": training_steps}
        },
    )
    monkeypatch.setattr(
        experiment,
        "run_smoke",
        lambda base_dir: {"elapsed_seconds": 0.1, "base_dir": str(base_dir)},
    )

    reproduce = runner.invoke(
        cli.app,
        [
            "reproduce",
            "--base-dir",
            str(tmp_path),
            "--steps",
            "3",
            "--max-length",
            "40",
            "--skip-training",
        ],
    )
    smoke = runner.invoke(cli.app, ["smoke", "--base-dir", str(tmp_path)])

    assert reproduce.exit_code == 0
    assert '"force_train": false' in reproduce.output
    assert smoke.exit_code == 0
    assert "elapsed_seconds" in smoke.output

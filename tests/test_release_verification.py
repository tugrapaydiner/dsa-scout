from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from scripts import verify_release


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _minimal_release_tree(root: Path) -> None:
    scorers = sorted(verify_release.EXPECTED_SCORERS)
    recall: dict[str, object] = {}
    for scorer_idx, scorer in enumerate(scorers):
        arr = np.full((5, 12, 15), 0.1 + scorer_idx * 0.05, dtype=float)
        recall[scorer] = arr.tolist()
    recall["lightning_trained"] = np.full((5, 12, 15), 0.5, dtype=float).tolist()
    recall["lightning_untrained"] = np.full((5, 12, 15), 0.2, dtype=float).tolist()
    recall["lightning_trained_plus_recency"] = np.full((5, 12, 15), 0.6, dtype=float).tolist()
    recall["recency"] = np.full((5, 12, 15), 0.55, dtype=float).tolist()

    middle = {}
    for scorer, values in recall.items():
        arr = np.asarray(values, dtype=float)
        middle[scorer] = {
            "mean": float(verify_release.values_for_summary(scorer, arr).mean()),
            "ci_low": 0.0,
            "ci_high": 1.0,
        }
    summary = {
        "middle_layer_top8": middle,
        "trained_vs_untrained_delta": {"mean": 0.3, "ci_low": 0.0, "ci_high": 1.0},
        "hybrid_deltas": {
            "minus_lightning_trained": {"mean": 0.1, "ci_low": 0.0, "ci_high": 1.0},
            "minus_recency": {"mean": 0.05, "ci_low": 0.0, "ci_high": 1.0},
        },
    }
    hashes = [f"eval-{idx}" for idx in range(15)]
    metadata = {
        "hyperparameters": copy.deepcopy(verify_release.EXPECTED_HYPERPARAMETERS),
        "corpus": {
            "num_samples": 15,
            "samples_per_category": 3,
            "token_counts": {f"sample-{idx}": 1024 for idx in range(15)},
            "sample_hashes": {f"sample-{idx}": hashes[idx] for idx in range(15)},
            "diversity_stats": {
                "min_unique_tokens": 200,
                "max_top_bigram_ratio": 0.3,
            },
        },
        "training": {
            "num_examples": 250,
            "training_corpus": {
                "num_texts": 50,
                "hashes": [f"train-{idx}" for idx in range(50)],
                "min_unique_tokens": 200,
                "max_top_bigram_ratio": 0.3,
            },
        },
    }
    training_log = {
        "losses": [2.0, 1.0],
        "actual_steps": 2,
        "final_loss": 1.0,
        "best_loss": 1.0,
        "best_rolling_loss": 1.0,
    }
    hybrid_sweep = {
        "0.1": {"mean": 0.56, "ci_low": 0.0, "ci_high": 1.0},
        "0.3": {"mean": 0.54, "ci_low": 0.0, "ci_high": 1.0},
        "0.5": {"mean": 0.53, "ci_low": 0.0, "ci_high": 1.0},
        "0.7": {"mean": 0.52, "ci_low": 0.0, "ci_high": 1.0},
        "0.9": {"mean": 0.51, "ci_low": 0.0, "ci_high": 1.0},
    }
    _write_json(root / "results" / "recall_by_layer.json", recall)
    _write_json(root / "results" / "summary_stats.json", summary)
    _write_json(root / "results" / "metadata.json", metadata)
    _write_json(root / "results" / "training_log.json", training_log)
    _write_json(root / "results" / "hybrid_weight_sweep.json", hybrid_sweep)


def test_check_result_consistency_accepts_valid_tree(tmp_path: Path) -> None:
    _minimal_release_tree(tmp_path)

    assert verify_release.check_result_consistency(tmp_path) == []


def test_check_result_consistency_catches_stale_summary(tmp_path: Path) -> None:
    _minimal_release_tree(tmp_path)
    summary_path = tmp_path / "results" / "summary_stats.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["middle_layer_top8"]["recency"]["mean"] = 0.123
    _write_json(summary_path, summary)

    failures = verify_release.check_result_consistency(tmp_path)

    assert any("summary mean mismatch for recency" in failure for failure in failures)


def test_check_result_consistency_catches_train_eval_leakage(tmp_path: Path) -> None:
    _minimal_release_tree(tmp_path)
    metadata_path = tmp_path / "results" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    leaked = metadata["corpus"]["sample_hashes"]["sample-0"]
    metadata["training"]["training_corpus"]["hashes"][0] = leaked
    _write_json(metadata_path, metadata)

    failures = verify_release.check_result_consistency(tmp_path)

    assert "training/evaluation content hashes are not disjoint" in failures

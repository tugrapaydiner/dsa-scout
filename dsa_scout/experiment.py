"""End-to-end DSA-Scout experiment orchestration."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import torch

from dsa_scout import __version__
from dsa_scout.corpus import (
    CorpusSample,
    corpus_metadata,
    load_corpus,
    load_training_records,
    training_corpus_metadata,
)
from dsa_scout.indexer import LightningIndexer
from dsa_scout.logging import get_logger
from dsa_scout.metrics import aggregate_oracle_to_blocks, conditional_recall, topk_recall
from dsa_scout.oracle import get_oracle_and_hiddens
from dsa_scout.plots import (
    save_hybrid_weight_sweep,
    save_marginal_over_recency,
    save_oracle_heatmap,
    save_recall_by_layer,
    save_recall_by_text_type,
    save_recall_vs_k,
    save_scorer_distributions,
    save_trained_vs_untrained,
    save_training_curve,
)
from dsa_scout.scorers import (
    score_lightning_trained,
    score_lightning_trained_plus_recency,
    score_lightning_untrained,
    score_linear,
    score_preview_attn,
    score_random,
    score_recency,
    score_window_sink,
)
from dsa_scout.training import (
    TrainingConfig,
    load_trained_indexer,
    save_training_artifacts,
    train_indexer,
)

log = get_logger(__name__)

FloatArray: TypeAlias = npt.NDArray[np.float64]

M = 4
INDEXER_HEADS = 8
INDEXER_HEAD_DIM = 32
INDEXER_LATENT_DIM = 128
K = 8
K_VALUES = [4, 8, 16, 32]
MID_LAYERS = [4, 5, 6, 7, 8]
INDEXER_SEEDS = [0, 1, 2, 3, 4]
HYBRID_WEIGHT = 0.5
HYBRID_WEIGHTS = (0.1, 0.3, 0.5, 0.7, 0.9)
SCORER_NAMES = [
    "random",
    "recency",
    "window_sink",
    "linear",
    "preview_attn",
    "lightning_untrained",
    "lightning_trained",
    "lightning_trained_plus_recency",
]
MARGINAL_SCORERS = [
    "window_sink",
    "linear",
    "preview_attn",
    "lightning_untrained",
    "lightning_trained",
    "lightning_trained_plus_recency",
]
EXPECTED_PLOTS = [
    "plots/01_oracle_heatmap.png",
    "plots/02_scorer_distributions.png",
    "plots/03_recall_by_layer.png",
    "plots/04_recall_vs_k.png",
    "plots/05_recall_by_text_type.png",
    "plots/06_marginal_over_recency.png",
    "plots/07_training_curve.png",
    "plots/08_trained_vs_untrained.png",
    "plots/09_hybrid_weight_sweep.png",
]
EXPECTED_RESULTS = [
    "results/hybrid_weight_sweep.json",
    "results/marginal_over_recency.json",
    "results/metadata.json",
    "results/recall_by_layer.json",
    "results/recall_by_text_type.json",
    "results/recall_vs_k.json",
    "results/summary_stats.json",
    "results/trained_indexer.pt",
    "results/training_log.json",
]
EXPECTED_CORPUS_CACHE = [
    "corpus_cache/corpus_v3.json",
    "corpus_cache/corpus_v3_1024_dc217f36e626b65f.json",
    "corpus_cache/training_corpus_v3.json",
]


def get_device() -> torch.device:
    """Select CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_reproducibility(seed: int = 0) -> None:
    """Configure deterministic random seeds where feasible."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_gpt2(device: torch.device) -> tuple[Any, Any]:
    """Load GPT-2 small and tokenizer."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    try:
        model = GPT2LMHeadModel.from_pretrained(
            "gpt2",
            output_attentions=True,
            attn_implementation="eager",
            local_files_only=True,
        )
    except TypeError:
        model = GPT2LMHeadModel.from_pretrained(
            "gpt2",
            output_attentions=True,
            local_files_only=True,
        )
    except OSError:
        try:
            model = GPT2LMHeadModel.from_pretrained(
                "gpt2",
                output_attentions=True,
                attn_implementation="eager",
            )
        except TypeError:
            model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True)
    try:
        tok = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
    except OSError:
        tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model_any = cast(Any, model)
    loaded_model = cast(Any, model_any.to(device))
    return cast(Any, loaded_model.eval()), cast(Any, tok)


@torch.no_grad()
def run_hello_world(model: Any, tok: Any, device: torch.device) -> dict[str, object]:
    """Run the GPT-2 attention smoke check."""
    inputs = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to(device)
    outputs = model(**inputs, output_attentions=True)
    attentions = cast(tuple[torch.Tensor, ...], outputs.attentions)
    layer_count = len(attentions)
    attention_shape = list(attentions[0].shape)
    if layer_count != 12 or attention_shape != [1, 12, 10, 10]:
        msg = f"unexpected GPT-2 attention shape: layers={layer_count}, shape={attention_shape}"
        raise AssertionError(msg)
    return {"layers": layer_count, "attention_shape": attention_shape}


def make_indexer(device: torch.device, seed: int = 0) -> LightningIndexer:
    """Construct a deterministic Lightning Indexer."""
    torch.manual_seed(seed)
    return (
        LightningIndexer(
            n_heads=INDEXER_HEADS,
            c_i=INDEXER_HEAD_DIM,
            d_c=INDEXER_LATENT_DIM,
            m=M,
        )
        .to(device)
        .eval()
    )


def run_indexer_tests(device: torch.device) -> None:
    """Verify indexer shape and deterministic evaluation behavior."""
    indexer = make_indexer(device, seed=0)
    torch.manual_seed(0)
    hidden = torch.randn(64, 768, device=device)
    scores = indexer(hidden)
    if scores.shape != (64, 16):
        msg = f"unexpected indexer score shape {tuple(scores.shape)}"
        raise AssertionError(msg)
    with torch.no_grad():
        first = indexer(hidden)
        second = indexer(hidden)
    if not torch.allclose(first, second):
        msg = "indexer output is not deterministic in eval mode"
        raise AssertionError(msg)


def package_version(name: str) -> str:
    """Return an installed package version."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def save_json(path: Path, payload: object) -> None:
    """Write indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    """Compute a SHA256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(base_path: Path) -> str:
    """Return the current Git commit hash, or ``unknown`` outside Git."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=base_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def git_dirty(base_path: Path) -> bool:
    """Return whether tracked or untracked files differ from HEAD."""
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=base_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return True
    return bool(result.stdout.strip())


def build_artifact_manifest(
    base_path: Path,
    metadata_payload: Mapping[str, object],
    summary: Mapping[str, object],
) -> dict[str, object]:
    """Build a reproducibility manifest with hashes for release artifacts."""
    expected = [*EXPECTED_PLOTS, *EXPECTED_RESULTS, *EXPECTED_CORPUS_CACHE]
    artifacts: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for relative in expected:
        path = base_path / relative
        if not path.exists():
            missing.append(relative)
            continue
        artifacts[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}

    corpus = cast(Mapping[str, object], metadata_payload.get("corpus", {}))
    corpus_content_hash = corpus.get("content_hash")
    corpus_cache_path = base_path / "corpus_cache" / "corpus_v3.json"
    if corpus_content_hash is None and corpus_cache_path.exists():
        corpus_payload = json.loads(corpus_cache_path.read_text(encoding="utf-8"))
        corpus_content_hash = cast(Mapping[str, object], corpus_payload).get("content_hash")
    middle = cast(Mapping[str, object], summary.get("middle_layer_top8", {}))
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(base_path),
        "git_dirty": git_dirty(base_path),
        "command": "dsa-scout reproduce",
        "package": {"name": "dsa-scout", "version": __version__},
        "model": "gpt2",
        "corpus_content_hash": corpus_content_hash or "unknown",
        "artifact_count": len(artifacts),
        "missing_artifacts": missing,
        "artifacts": artifacts,
        "headline_metrics": {
            "trained_lightning_top8": cast(
                Mapping[str, object], middle.get("lightning_trained", {})
            ).get("mean"),
            "recency_top8": cast(Mapping[str, object], middle.get("recency", {})).get("mean"),
            "hybrid_weight_0_5_top8": cast(
                Mapping[str, object],
                middle.get("lightning_trained_plus_recency", {}),
            ).get("mean"),
        },
        "environment": {
            "python": metadata_payload.get("python"),
            "platform": metadata_payload.get("platform"),
            "device": metadata_payload.get("device"),
            "versions": metadata_payload.get("versions"),
        },
    }


def mean_ci(values: npt.ArrayLike, seed: int = 0, n_boot: int = 5000) -> dict[str, float]:
    """Compute deterministic bootstrap mean and confidence interval."""
    flat = np.asarray(values, dtype=float).reshape(-1)
    if flat.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(flat, size=(n_boot, flat.size), replace=True).mean(axis=1)
    return {
        "mean": float(flat.mean()),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
    }


def train_or_load_indexer(
    model: Any,
    tok: Any,
    device: torch.device,
    eval_samples: list[CorpusSample],
    results_dir: Path,
    cache_dir: Path,
    force_train: bool = True,
    config: TrainingConfig | None = None,
) -> tuple[LightningIndexer, dict[str, object]]:
    """Train the Lightning Indexer and persist checkpoint/log artifacts."""
    checkpoint_path = results_dir / "trained_indexer.pt"
    log_path = results_dir / "training_log.json"
    if checkpoint_path.exists() and log_path.exists() and not force_train:
        return load_trained_indexer(checkpoint_path, device), json.loads(
            log_path.read_text(encoding="utf-8")
        )

    config = config or TrainingConfig()
    training_records = load_training_records(
        tok,
        eval_samples=eval_samples,
        n_texts=50,
        target_tokens=config.max_length,
        cache_dir=cache_dir,
    )
    training_texts = [record.text for record in training_records]
    indexer = make_indexer(device, seed=config.seed)
    training_log = train_indexer(indexer, training_texts, model, tok, device, config)
    training_log["training_corpus"] = training_corpus_metadata(training_records)
    final_loss = float(cast(float, training_log["final_loss"]))
    initial_loss = float(cast(float, training_log["initial_loss"]))
    best_rolling = float(cast(float, training_log.get("best_rolling_loss", final_loss)))
    if best_rolling >= initial_loss * 0.9:
        log.info("initial training did not improve rolling loss enough; retrying with lr=5e-4")
        retry_config = TrainingConfig(
            steps=max(config.steps, 800),
            lr=5e-4,
            target_layers=config.target_layers,
            log_every=config.log_every,
            seed=config.seed,
            max_length=config.max_length,
            oracle_scale=config.oracle_scale,
            early_stop_patience=config.early_stop_patience,
            early_stop_min_delta=config.early_stop_min_delta,
            warmup_steps=config.warmup_steps,
            min_lr=config.min_lr,
            grad_clip=config.grad_clip,
        )
        indexer = make_indexer(device, seed=retry_config.seed)
        training_log = train_indexer(indexer, training_texts, model, tok, device, retry_config)
        training_log["training_corpus"] = training_corpus_metadata(training_records)
    save_training_artifacts(indexer, training_log, checkpoint_path, log_path)
    return indexer.eval(), training_log


@torch.no_grad()
def score_layer(
    hidden: torch.Tensor,
    model: Any,
    layer_idx: int,
    untrained_indexers: Mapping[int, LightningIndexer],
    trained_indexer: LightningIndexer,
) -> dict[str, dict[int, torch.Tensor]]:
    """Run all scorers for one hidden-state layer.

    Returns:
        Mapping from scorer name to seed-indexed score tensors. Non-seeded
        scorers are repeated across ``INDEXER_SEEDS`` for a uniform result shape.
    """
    invariant = {
        "random": score_random(hidden, m=M, seed=42),
        "recency": score_recency(hidden, m=M),
        "window_sink": score_window_sink(hidden, m=M, n_sinks=1),
        "linear": score_linear(hidden, m=M),
        "preview_attn": score_preview_attn(hidden, model, layer_idx=layer_idx, head_idx=0, m=M),
        "lightning_trained": score_lightning_trained(hidden, trained_indexer),
        "lightning_trained_plus_recency": score_lightning_trained_plus_recency(
            hidden,
            trained_indexer,
            m=M,
            weight=HYBRID_WEIGHT,
        ),
    }
    result: dict[str, dict[int, torch.Tensor]] = {}
    for name, scores in invariant.items():
        result[name] = dict.fromkeys(INDEXER_SEEDS, scores)
    result["lightning_untrained"] = {
        seed: score_lightning_untrained(hidden, indexer)
        for seed, indexer in untrained_indexers.items()
    }
    return result


@torch.no_grad()
def _score_sample_layer(
    sample: CorpusSample,
    hidden: torch.Tensor,
    model: Any,
    layer_idx: int,
    untrained_indexers: Mapping[int, LightningIndexer],
    trained_indexer: LightningIndexer,
) -> dict[str, dict[int, torch.Tensor]]:
    """Compute all scorers for one sample/layer pair."""
    del sample
    return score_layer(hidden, model, layer_idx, untrained_indexers, trained_indexer)


def _accumulate_recall(
    score_map: Mapping[str, Mapping[int, torch.Tensor]],
    oracle_blocked: torch.Tensor,
    text_idx: int,
    layer_idx: int,
    recall_by_layer: dict[str, FloatArray],
    recall_vs_k: dict[str, dict[int, list[float]]],
    marginal: dict[str, list[float]],
) -> None:
    """Update recall, k-sweep, and marginal accumulators."""
    recency_scores = score_map["recency"][INDEXER_SEEDS[0]]
    for scorer_name in SCORER_NAMES:
        for seed_idx, seed in enumerate(INDEXER_SEEDS):
            scores = score_map[scorer_name][seed]
            recall_by_layer[scorer_name][seed_idx, layer_idx, text_idx] = topk_recall(
                scores,
                oracle_blocked,
                k=K,
                m=M,
            )
            if layer_idx in MID_LAYERS:
                for k_val in K_VALUES:
                    recall_vs_k[scorer_name][k_val].append(
                        topk_recall(scores, oracle_blocked, k=k_val, m=M)
                    )
                if scorer_name in marginal:
                    marginal[scorer_name].append(
                        conditional_recall(
                            scores,
                            oracle_blocked,
                            k=K,
                            recency_scores=recency_scores,
                            m=M,
                        )
                    )


def _build_text_type_rows(
    samples: Sequence[CorpusSample],
    recall_by_layer: Mapping[str, FloatArray],
) -> list[dict[str, object]]:
    """Build per-scorer text-category summary rows."""
    rows: list[dict[str, object]] = []
    categories = sorted({sample.category for sample in samples})
    for category in categories:
        indices = [idx for idx, sample in enumerate(samples) if sample.category == category]
        for scorer_name in SCORER_NAMES:
            values = recall_by_layer[scorer_name][:, MID_LAYERS, :][:, :, indices]
            if scorer_name != "lightning_untrained":
                values = values[:1]
            rows.append(
                {"scorer": scorer_name, "text_type": category, "recall": float(values.mean())}
            )
    return rows


def values_for_summary(name: str, arr: FloatArray) -> FloatArray:
    """Return independent values for scorer-level summaries."""
    middle = arr[:, MID_LAYERS, :]
    if name == "lightning_untrained":
        return cast(FloatArray, middle.reshape(-1))
    return cast(FloatArray, middle[0].reshape(-1))


def build_metadata(
    device: torch.device,
    samples: Sequence[CorpusSample],
    training_log: Mapping[str, object],
) -> dict[str, object]:
    """Build run metadata."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "versions": {
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "datasets": package_version("datasets"),
            "numpy": np.__version__,
            "matplotlib": package_version("matplotlib"),
            "pandas": package_version("pandas"),
        },
        "seeds": {"global": 0, "untrained_indexer": INDEXER_SEEDS, "trained_indexer": 0},
        "hyperparameters": {
            "m": M,
            "n_I_h": INDEXER_HEADS,
            "c_I": INDEXER_HEAD_DIM,
            "d_c": INDEXER_LATENT_DIM,
            "K": K,
            "K_VALUES": K_VALUES,
            "MID_LAYERS": MID_LAYERS,
            "target_tokens": 1024,
        },
        "corpus": {
            **corpus_metadata(samples),
        },
        "training": {
            "initial_loss": training_log.get("initial_loss"),
            "last_loss": training_log.get("last_loss"),
            "final_loss": training_log.get("final_loss"),
            "best_loss": training_log.get("best_loss"),
            "best_checkpoint_eval_loss": training_log.get("best_checkpoint_eval_loss"),
            "best_step": training_log.get("best_step"),
            "loss_delta": training_log.get("loss_delta"),
            "actual_steps": training_log.get("actual_steps"),
            "early_stopped": training_log.get("early_stopped"),
            "best_rolling_loss": training_log.get("best_rolling_loss"),
            "num_examples": training_log.get("num_examples"),
            "config": training_log.get("config"),
            "training_corpus": training_log.get("training_corpus"),
            "hybrid_weight": HYBRID_WEIGHT,
        },
    }


def summarize_results(
    recall_by_layer: Mapping[str, FloatArray],
    recall_vs_k: Mapping[str, Mapping[int, list[float]]],
    recall_by_text_type: list[dict[str, object]],
    marginal: Mapping[str, list[float]],
    results_dir: Path,
) -> dict[str, object]:
    """Summarize results for README and final report."""
    middle = {name: mean_ci(values_for_summary(name, arr)) for name, arr in recall_by_layer.items()}
    layer_means = {
        name: [float(x) for x in arr.mean(axis=(0, 2))] for name, arr in recall_by_layer.items()
    }
    untrained_seed_means = {
        str(seed): float(recall_by_layer["lightning_untrained"][idx, MID_LAYERS, :].mean())
        for idx, seed in enumerate(INDEXER_SEEDS)
    }
    trained_values = recall_by_layer["lightning_trained"][0, MID_LAYERS, :].reshape(-1)
    untrained_seed_avg = (
        recall_by_layer["lightning_untrained"][:, MID_LAYERS, :].mean(axis=0).reshape(-1)
    )
    trained_delta = mean_ci(trained_values - untrained_seed_avg)
    hybrid_deltas: dict[str, dict[str, float]] = {}
    paired_deltas: dict[str, dict[str, dict[str, float]]] = {}
    random_values = recall_by_layer["random"][0, MID_LAYERS, :].reshape(-1)
    recency_values = recall_by_layer["recency"][0, MID_LAYERS, :].reshape(-1)
    for name, arr in recall_by_layer.items():
        values = values_for_summary(name, arr)
        if name == "lightning_untrained":
            baseline_random = np.tile(random_values, len(INDEXER_SEEDS))
            baseline_recency = np.tile(recency_values, len(INDEXER_SEEDS))
        else:
            baseline_random = random_values
            baseline_recency = recency_values
        paired_deltas[name] = {
            "minus_random": mean_ci(values - baseline_random),
            "minus_recency": mean_ci(values - baseline_recency),
        }
    if "lightning_trained_plus_recency" in recall_by_layer:
        hybrid_values = recall_by_layer["lightning_trained_plus_recency"][0, MID_LAYERS, :].reshape(
            -1
        )
        hybrid_deltas = {
            "minus_lightning_trained": mean_ci(hybrid_values - trained_values),
            "minus_recency": mean_ci(hybrid_values - recency_values),
        }
    summary = {
        "middle_layer_top8": middle,
        "layer_means": layer_means,
        "untrained_seed_means": untrained_seed_means,
        "trained_vs_untrained_delta": trained_delta,
        "hybrid_deltas": hybrid_deltas,
        "paired_deltas": paired_deltas,
        "recall_vs_k": {
            name: {str(k_val): mean_ci(np.array(values)) for k_val, values in scorer.items()}
            for name, scorer in recall_vs_k.items()
        },
        "marginal_over_recency": {
            name: mean_ci(np.array(values)) for name, values in marginal.items()
        },
        "recall_by_text_type": recall_by_text_type,
    }
    save_json(results_dir / "summary_stats.json", summary)
    return summary


@torch.no_grad()
def evaluate_hybrid_weight_sweep(
    cached_pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    trained_indexer: LightningIndexer,
    device: torch.device,
) -> tuple[dict[str, dict[str, float]], float, float]:
    """Evaluate post-hoc hybrid weights from cached hidden/oracle tensors.

    Args:
        cached_pairs: Sequence of ``(hidden, oracle_blocked)`` tensors for
            middle-layer evaluation points. Hidden tensors have shape
            ``[seq, 768]`` and oracle tensors have shape ``[seq, seq // 4]``.
        trained_indexer: Trained Lightning Indexer.
        device: Device for scorer execution.

    Returns:
        Weight-sweep confidence intervals, pure-recency mean, and pure-trained
        Lightning mean over the same cached points.
    """
    values_by_weight: dict[float, list[float]] = {weight: [] for weight in HYBRID_WEIGHTS}
    recency_values: list[float] = []
    trained_values: list[float] = []
    for hidden_cpu, oracle_cpu in cached_pairs:
        hidden = hidden_cpu.to(device)
        oracle_blocked = oracle_cpu.to(device)
        recency_scores = score_recency(hidden, m=M)
        trained_scores = score_lightning_trained(hidden, trained_indexer)
        recency_values.append(topk_recall(recency_scores, oracle_blocked, k=K, m=M))
        trained_values.append(topk_recall(trained_scores, oracle_blocked, k=K, m=M))
        for weight in HYBRID_WEIGHTS:
            hybrid_scores = score_lightning_trained_plus_recency(
                hidden,
                trained_indexer,
                m=M,
                weight=weight,
            )
            values_by_weight[weight].append(topk_recall(hybrid_scores, oracle_blocked, k=K, m=M))

    sweep = {
        f"{weight:.1f}": mean_ci(np.array(values, dtype=float), seed=int(weight * 1000))
        for weight, values in values_by_weight.items()
    }
    return (
        sweep,
        float(np.mean(recency_values)) if recency_values else 0.0,
        float(np.mean(trained_values)) if trained_values else 0.0,
    )


def _empty_recall_accumulators(
    text_count: int,
    seed_count: int,
) -> tuple[
    dict[str, FloatArray],
    dict[str, dict[int, list[float]]],
    dict[str, list[float]],
]:
    recall_by_layer: dict[str, FloatArray] = {
        name: np.zeros((seed_count, 12, text_count), dtype=float) for name in SCORER_NAMES
    }
    recall_vs_k: dict[str, dict[int, list[float]]] = {
        name: {k_val: [] for k_val in K_VALUES} for name in SCORER_NAMES
    }
    marginal: dict[str, list[float]] = {name: [] for name in MARGINAL_SCORERS}
    return recall_by_layer, recall_vs_k, marginal


@torch.no_grad()
def _evaluate_samples(
    samples: Sequence[CorpusSample],
    model: Any,
    tok: Any,
    device: torch.device,
    trained_indexer: LightningIndexer,
    plots_dir: Path,
    max_length: int,
    recall_by_layer: dict[str, FloatArray],
    recall_vs_k: dict[str, dict[int, list[float]]],
    marginal: dict[str, list[float]],
) -> tuple[dict[str, torch.Tensor] | None, list[tuple[torch.Tensor, torch.Tensor]]]:
    distribution_scores: dict[str, torch.Tensor] | None = None
    hybrid_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
    untrained_indexers = {seed: make_indexer(device, seed=seed) for seed in INDEXER_SEEDS}
    for text_idx, sample in enumerate(samples):
        log.info("evaluating sample %s/%s (%s)", text_idx + 1, len(samples), sample.name)
        attn, hiddens, tokens = get_oracle_and_hiddens(
            sample.text,
            model,
            tok,
            device,
            max_length=max_length,
        )
        if text_idx == 0:
            save_oracle_heatmap(attn, len(tokens), plots_dir / "01_oracle_heatmap.png")
        for layer_idx in range(12):
            hidden = hiddens[layer_idx]
            oracle_blocked = aggregate_oracle_to_blocks(attn[layer_idx], m=M)
            if layer_idx in MID_LAYERS:
                hybrid_cache.append((hidden.detach().cpu(), oracle_blocked.detach().cpu()))
            score_map = _score_sample_layer(
                sample,
                hidden,
                model,
                layer_idx,
                untrained_indexers,
                trained_indexer,
            )
            if text_idx == 0 and layer_idx == 5:
                distribution_scores = {
                    name: score_map[name][INDEXER_SEEDS[0]] for name in SCORER_NAMES
                }
            _accumulate_recall(
                score_map,
                oracle_blocked,
                text_idx,
                layer_idx,
                recall_by_layer,
                recall_vs_k,
                marginal,
            )
    return distribution_scores, hybrid_cache


def _save_evaluation_artifacts(
    recall_by_layer: Mapping[str, FloatArray],
    recall_vs_k: Mapping[str, Mapping[int, list[float]]],
    rows: list[dict[str, object]],
    marginal: Mapping[str, list[float]],
    hybrid_sweep: Mapping[str, Mapping[str, float]],
    recency_ref: float,
    trained_ref: float,
    distribution_scores: Mapping[str, torch.Tensor],
    plots_dir: Path,
    results_dir: Path,
) -> None:
    save_scorer_distributions(distribution_scores, plots_dir / "02_scorer_distributions.png")
    save_json(
        results_dir / "recall_by_layer.json", {k: v.tolist() for k, v in recall_by_layer.items()}
    )
    save_json(
        results_dir / "recall_vs_k.json",
        {
            name: {str(k_val): vals for k_val, vals in data.items()}
            for name, data in recall_vs_k.items()
        },
    )
    save_json(results_dir / "recall_by_text_type.json", rows)
    save_json(results_dir / "marginal_over_recency.json", marginal)
    save_json(results_dir / "hybrid_weight_sweep.json", hybrid_sweep)
    save_recall_by_layer(recall_by_layer, SCORER_NAMES, plots_dir / "03_recall_by_layer.png")
    save_recall_vs_k(recall_vs_k, SCORER_NAMES, K_VALUES, plots_dir / "04_recall_vs_k.png")
    save_recall_by_text_type(rows, plots_dir / "05_recall_by_text_type.png")
    save_marginal_over_recency(marginal, plots_dir / "06_marginal_over_recency.png")
    save_trained_vs_untrained(recall_by_layer, plots_dir / "08_trained_vs_untrained.png")
    save_hybrid_weight_sweep(
        hybrid_sweep,
        recency_ref,
        trained_ref,
        plots_dir / "09_hybrid_weight_sweep.png",
    )


@torch.no_grad()
def run_evaluation(
    samples: list[CorpusSample],
    model: Any,
    tok: Any,
    device: torch.device,
    trained_indexer: LightningIndexer,
    plots_dir: Path,
    results_dir: Path,
    max_length: int = 1024,
) -> tuple[
    dict[str, FloatArray],
    dict[str, dict[int, list[float]]],
    list[dict[str, object]],
    dict[str, list[float]],
]:
    """Run all recall experiments."""
    recall_by_layer, recall_vs_k, marginal = _empty_recall_accumulators(
        len(samples),
        len(INDEXER_SEEDS),
    )
    distribution_scores, hybrid_cache = _evaluate_samples(
        samples,
        model,
        tok,
        device,
        trained_indexer,
        plots_dir,
        max_length,
        recall_by_layer,
        recall_vs_k,
        marginal,
    )
    if distribution_scores is None:
        msg = "distribution scores were not collected"
        raise RuntimeError(msg)
    rows = _build_text_type_rows(samples, recall_by_layer)
    hybrid_sweep, recency_ref, trained_ref = evaluate_hybrid_weight_sweep(
        hybrid_cache,
        trained_indexer,
        device,
    )
    _save_evaluation_artifacts(
        recall_by_layer,
        recall_vs_k,
        rows,
        marginal,
        hybrid_sweep,
        recency_ref,
        trained_ref,
        distribution_scores,
        plots_dir,
        results_dir,
    )
    return recall_by_layer, recall_vs_k, rows, marginal


def run_full_study(
    base_dir: Path | str = Path("."),
    force_train: bool = True,
    training_steps: int = 2000,
    max_length: int = 1024,
) -> dict[str, object]:
    """Run the full study and regenerate all artifacts."""
    start = time.perf_counter()
    base_path = Path(base_dir)
    plots_dir = base_path / "plots"
    results_dir = base_path / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    configure_reproducibility(seed=0)
    device = get_device()
    model, tok = load_gpt2(device)
    hello = run_hello_world(model, tok, device)
    run_indexer_tests(device)
    samples = load_corpus(tok, cache_dir=base_path / "corpus_cache", target_tokens=max_length)
    config = TrainingConfig(steps=training_steps, lr=1e-3, max_length=max_length)
    trained_indexer, training_log = train_or_load_indexer(
        model,
        tok,
        device,
        samples,
        results_dir,
        cache_dir=base_path / "corpus_cache",
        force_train=force_train,
        config=config,
    )
    save_training_curve(
        cast(list[float], training_log["losses"]),
        cast(int | None, training_log.get("best_step")),
        cast(float | None, training_log.get("best_loss")),
        plots_dir / "07_training_curve.png",
    )
    recall_by_layer, recall_vs_k, rows, marginal = run_evaluation(
        samples,
        model,
        tok,
        device,
        trained_indexer,
        plots_dir,
        results_dir,
        max_length=max_length,
    )
    metadata_payload = build_metadata(device, samples, training_log)
    save_json(results_dir / "metadata.json", metadata_payload)
    summary = summarize_results(recall_by_layer, recall_vs_k, rows, marginal, results_dir)
    manifest = build_artifact_manifest(base_path, metadata_payload, summary)
    save_json(results_dir / "manifest.json", manifest)
    elapsed = time.perf_counter() - start
    log.info("study completed in %.1fs", elapsed)
    return {
        "hello_world": hello,
        "metadata": metadata_payload,
        "manifest": manifest,
        "summary_stats": summary,
        "elapsed_seconds": elapsed,
    }


def run_smoke(base_dir: Path | str = Path(".")) -> dict[str, object]:
    """Run a sub-60-second smoke check."""
    start = time.perf_counter()
    base_path = Path(base_dir)
    plots_dir = base_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    configure_reproducibility(seed=0)
    device = get_device()
    model, tok = load_gpt2(device)
    hello = run_hello_world(model, tok, device)
    attn, _, tokens = get_oracle_and_hiddens(
        "The quick brown fox jumps over the lazy dog. Sparse attention smoke test.",
        model,
        tok,
        device,
        max_length=32,
    )
    save_oracle_heatmap(attn, len(tokens), plots_dir / "smoke_oracle_heatmap.png")
    elapsed = time.perf_counter() - start
    return {"hello_world": hello, "elapsed_seconds": elapsed}

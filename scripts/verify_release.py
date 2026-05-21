from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np

EXPECTED_ARTIFACTS = [
    "plots/01_oracle_heatmap.png",
    "plots/02_scorer_distributions.png",
    "plots/03_recall_by_layer.png",
    "plots/04_recall_vs_k.png",
    "plots/05_recall_by_text_type.png",
    "plots/06_marginal_over_recency.png",
    "plots/07_training_curve.png",
    "plots/08_trained_vs_untrained.png",
    "plots/09_hybrid_weight_sweep.png",
    "results/hybrid_weight_sweep.json",
    "results/manifest.json",
    "results/marginal_over_recency.json",
    "results/metadata.json",
    "results/recall_by_layer.json",
    "results/recall_by_text_type.json",
    "results/recall_vs_k.json",
    "results/summary_stats.json",
    "results/trained_indexer.pt",
    "results/training_log.json",
    "corpus_cache/corpus_v3.json",
    "corpus_cache/corpus_v3_1024_dc217f36e626b65f.json",
    "corpus_cache/training_corpus_v3.json",
]
EXPECTED_SCORERS = {
    "random",
    "recency",
    "window_sink",
    "linear",
    "preview_attn",
    "lightning_untrained",
    "lightning_trained",
    "lightning_trained_plus_recency",
}
MID_LAYERS = [4, 5, 6, 7, 8]
INDEXER_SEEDS = [0, 1, 2, 3, 4]
EXPECTED_HYPERPARAMETERS = {
    "m": 4,
    "n_I_h": 8,
    "c_I": 32,
    "d_c": 128,
    "K": 8,
    "K_VALUES": [4, 8, 16, 32],
    "MID_LAYERS": MID_LAYERS,
    "target_tokens": 1024,
}
EXPECTED_HYBRID_WEIGHTS = {"0.1", "0.3", "0.5", "0.7", "0.9"}
FLOAT_TOLERANCE = 1e-10
STALE_PATTERNS = [
    "docs/history/*.md",
    "diagrams/indexer.d2",
    "diagrams/indexer.dot",
    "dsa_scout/diagram.py",
    "scripts/render_diagram.py",
    "make.cmd",
    ".coverage",
    "corpus_cache/corpus.json",
    "corpus_cache/corpus_1024_*.json",
    "corpus_cache/corpus_128_*.json",
    "plots/00_indexer_architecture.svg",
    "plots/smoke_oracle_heatmap.png",
]
SKIP_DIRS = {
    ".git",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".pytest_tmp",
    ".ruff_cache",
    "__pycache__",
}
SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN (RSA |EC |OPENSSH |DSA )?"
    r"PRIVATE KEY-----)"
)
LOCAL_PATH_RE = re.compile(
    "|".join(
        [
            r"C:" + r"\\Dev",
            "C:" + "/Dev",
            r"C:" + r"\\Users",
            "C:" + "/Users",
            "Sandbox" + "Offline",
            "App" + "Data",
        ]
    )
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_release_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS for part in relative_parts):
            continue
        yield path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def word_count(path: Path) -> int:
    text = re.sub(r"```.*?```", "", read_text(path), flags=re.DOTALL)
    return len(re.findall(r"\b\S+\b", text))


def pyproject_version(path: Path) -> str:
    match = re.search(r'^version\s*=\s*"([^"]+)"', read_text(path), flags=re.MULTILINE)
    return match.group(1) if match else "missing"


def package_version(init_path: Path) -> str:
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', read_text(init_path), flags=re.MULTILINE)
    return match.group(1) if match else "missing"


def values_for_summary(name: str, arr: np.ndarray) -> np.ndarray:
    middle = arr[:, MID_LAYERS, :]
    if name == "lightning_untrained":
        return cast(np.ndarray, middle.reshape(-1))
    return cast(np.ndarray, middle[0].reshape(-1))


def close_enough(observed: float, expected: float) -> bool:
    return abs(observed - expected) <= FLOAT_TOLERANCE


def check_result_consistency(root: Path) -> list[str]:
    failures: list[str] = []
    summary = read_json(root / "results" / "summary_stats.json")
    metadata = read_json(root / "results" / "metadata.json")
    training_log = read_json(root / "results" / "training_log.json")
    recall_raw = read_json(root / "results" / "recall_by_layer.json")
    hybrid_sweep = read_json(root / "results" / "hybrid_weight_sweep.json")

    recall_by_layer = {
        scorer: np.asarray(values, dtype=float) for scorer, values in recall_raw.items()
    }
    middle = cast(dict[str, dict[str, float]], summary.get("middle_layer_top8", {}))
    for scorer in EXPECTED_SCORERS:
        if scorer not in recall_by_layer:
            failures.append(f"raw recall_by_layer missing scorer: {scorer}")
            continue
        if recall_by_layer[scorer].shape != (len(INDEXER_SEEDS), 12, 15):
            failures.append(
                f"unexpected recall_by_layer shape for {scorer}: {recall_by_layer[scorer].shape}"
            )
            continue
        observed = float(middle.get(scorer, {}).get("mean", float("nan")))
        expected = float(values_for_summary(scorer, recall_by_layer[scorer]).mean())
        if not close_enough(observed, expected):
            failures.append(
                f"summary mean mismatch for {scorer}: summary={observed}, raw={expected}"
            )

    if {"lightning_trained", "lightning_untrained"}.issubset(recall_by_layer):
        trained = recall_by_layer["lightning_trained"][0, MID_LAYERS, :].reshape(-1)
        untrained_avg = recall_by_layer["lightning_untrained"][:, MID_LAYERS, :].mean(axis=0)
        expected_delta = float((trained - untrained_avg.reshape(-1)).mean())
        observed_delta = float(
            cast(dict[str, float], summary["trained_vs_untrained_delta"])["mean"]
        )
        if not close_enough(observed_delta, expected_delta):
            failures.append(
                "trained-vs-untrained delta mismatch: "
                f"summary={observed_delta}, raw={expected_delta}"
            )

    if {
        "lightning_trained_plus_recency",
        "lightning_trained",
        "recency",
    }.issubset(recall_by_layer):
        hybrid = recall_by_layer["lightning_trained_plus_recency"][0, MID_LAYERS, :].reshape(-1)
        trained = recall_by_layer["lightning_trained"][0, MID_LAYERS, :].reshape(-1)
        recency = recall_by_layer["recency"][0, MID_LAYERS, :].reshape(-1)
        hybrid_deltas = cast(dict[str, dict[str, float]], summary["hybrid_deltas"])
        checks = {
            "minus_lightning_trained": float((hybrid - trained).mean()),
            "minus_recency": float((hybrid - recency).mean()),
        }
        for name, expected in checks.items():
            observed = float(hybrid_deltas[name]["mean"])
            if not close_enough(observed, expected):
                failures.append(
                    f"hybrid delta mismatch for {name}: summary={observed}, raw={expected}"
                )

    hyperparameters = cast(dict[str, object], metadata.get("hyperparameters", {}))
    if hyperparameters != EXPECTED_HYPERPARAMETERS:
        failures.append(f"locked hyperparameter mismatch: {hyperparameters}")

    corpus = cast(dict[str, Any], metadata.get("corpus", {}))
    if corpus.get("num_samples") != 15 or corpus.get("samples_per_category") != 3:
        failures.append("corpus cardinality mismatch")
    token_counts = cast(dict[str, int], corpus.get("token_counts", {}))
    if len(token_counts) != 15 or any(count != 1024 for count in token_counts.values()):
        failures.append("corpus token counts are not exactly 15 x 1024")
    sample_hashes = cast(dict[str, str], corpus.get("sample_hashes", {}))
    if len(set(sample_hashes.values())) != 15:
        failures.append("evaluation sample hashes are missing or not unique")
    diversity = cast(dict[str, float], corpus.get("diversity_stats", {}))
    if diversity.get("min_unique_tokens", 0) < 200:
        failures.append("evaluation corpus violates min unique token threshold")
    if diversity.get("max_top_bigram_ratio", 1.0) > 0.30:
        failures.append("evaluation corpus violates top bigram ratio threshold")

    training = cast(dict[str, Any], metadata.get("training", {}))
    training_corpus = cast(dict[str, Any], training.get("training_corpus", {}))
    train_hashes = cast(list[str], training_corpus.get("hashes", []))
    if training_corpus.get("num_texts") != 50 or len(set(train_hashes)) != 50:
        failures.append("training corpus must contain 50 unique texts")
    if set(train_hashes).intersection(set(sample_hashes.values())):
        failures.append("training/evaluation content hashes are not disjoint")
    if training_corpus.get("min_unique_tokens", 0) < 200:
        failures.append("training corpus violates min unique token threshold")
    if training_corpus.get("max_top_bigram_ratio", 1.0) > 0.30:
        failures.append("training corpus violates top bigram ratio threshold")
    if training.get("num_examples") != 250:
        failures.append("training example count should be 50 texts x 5 target layers")

    actual_steps = int(cast(int | float, training_log.get("actual_steps", -1)))
    losses = cast(list[float], training_log.get("losses", []))
    if actual_steps != len(losses):
        failures.append("training log actual_steps does not match loss count")
    final_loss = float(cast(float, training_log.get("final_loss", float("inf"))))
    best_rolling = float(cast(float, training_log.get("best_rolling_loss", float("inf"))))
    best_loss = float(cast(float, training_log.get("best_loss", float("inf"))))
    if final_loss > best_rolling + FLOAT_TOLERANCE:
        failures.append("final loss is worse than best rolling loss; checkpoint may be stale")
    if not close_enough(final_loss, best_loss):
        failures.append("final loss and best checkpoint loss diverge")

    if set(hybrid_sweep) != EXPECTED_HYBRID_WEIGHTS:
        failures.append(f"hybrid sweep weight set mismatch: {sorted(hybrid_sweep)}")
    else:
        best_weight = max(
            hybrid_sweep,
            key=lambda weight: float(cast(dict[str, float], hybrid_sweep[weight])["mean"]),
        )
        if best_weight != "0.1":
            failures.append(f"unexpected best hybrid weight: {best_weight}")
        hybrid_best = float(cast(dict[str, float], hybrid_sweep[best_weight])["mean"])
        recency_mean = float(middle["recency"]["mean"])
        if hybrid_best + FLOAT_TOLERANCE < recency_mean:
            failures.append("best hybrid sweep mean is below pure recency")

    return failures


def check_static(root: Path) -> list[str]:
    failures: list[str] = []
    for relative in EXPECTED_ARTIFACTS:
        if not (root / relative).exists():
            failures.append(f"missing artifact: {relative}")

    for pattern in STALE_PATTERNS:
        for path in root.glob(pattern):
            failures.append(f"stale release artifact present: {path.relative_to(root)}")

    for path in iter_release_files(root):
        text = read_text(path)
        if SECRET_RE.search(text):
            failures.append(f"high-confidence secret pattern found: {path.relative_to(root)}")
        if LOCAL_PATH_RE.search(text):
            failures.append(f"local machine path found: {path.relative_to(root)}")

    notebook = root / "dsa_scout.ipynb"
    if notebook.exists():
        notebook_text = read_text(notebook)
        if re.search(r"DSA-Scout V[0-9]+ Demo|V[0-9]+ artifacts|V[0-9]+ writeup", notebook_text):
            failures.append("stale versioned notebook label present")

    readme_words = word_count(root / "README.md")
    if not 600 <= readme_words <= 1000:
        failures.append(f"README word count out of range: {readme_words}")

    project_version = pyproject_version(root / "pyproject.toml")
    init_version = package_version(root / "dsa_scout" / "__init__.py")
    if project_version != init_version:
        failures.append(f"version mismatch: pyproject={project_version}, package={init_version}")

    summary = read_json(root / "results" / "summary_stats.json")
    middle = summary.get("middle_layer_top8", {})
    missing_scorers = EXPECTED_SCORERS.difference(middle)
    if missing_scorers:
        failures.append(f"summary missing scorers: {sorted(missing_scorers)}")

    manifest_path = root / "results" / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("missing_artifacts"):
            failures.append(f"manifest records missing artifacts: {manifest['missing_artifacts']}")
        manifest_version = manifest.get("package", {}).get("version")
        current_version = package_version(root / "dsa_scout" / "__init__.py")
        if manifest_version != current_version:
            failures.append(
                f"manifest version mismatch: manifest={manifest_version}, package={current_version}"
            )
        for relative, payload in manifest.get("artifacts", {}).items():
            path = root / relative
            if not path.exists():
                failures.append(f"manifest-listed artifact missing: {relative}")
                continue
            if sha256_file(path) != payload.get("sha256"):
                failures.append(f"manifest hash mismatch: {relative}")

    failures.extend(check_result_consistency(root))
    return failures


def run_quality_gates(root: Path) -> list[str]:
    commands = [
        [sys.executable, "-m", "ruff", "check", "dsa_scout/", "tests/"],
        [sys.executable, "-m", "ruff", "format", "--check", "dsa_scout/", "tests/"],
        [sys.executable, "-m", "mypy", "--strict", "dsa_scout/"],
        [
            sys.executable,
            "-m",
            "pytest",
            "--cov=dsa_scout",
            "--cov-report=term",
            "--cov-fail-under=80",
        ],
    ]
    failures: list[str] = []
    for command in commands:
        result = subprocess.run(command, cwd=root, check=False)
        if result.returncode != 0:
            failures.append(f"quality gate failed: {' '.join(command)}")
    return failures


def check_git_clean(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ["git status failed"]
    if result.stdout.strip():
        return ["git working tree is not clean"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify DSA-Scout release artifacts.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-gates", action="store_true", help="Run lint, type, and tests.")
    parser.add_argument("--strict-git", action="store_true", help="Require a clean Git tree.")
    args = parser.parse_args()
    root = args.root.resolve()
    failures = check_static(root)
    if args.run_gates:
        failures.extend(run_quality_gates(root))
    if args.strict_git:
        failures.extend(check_git_clean(root))
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("release check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

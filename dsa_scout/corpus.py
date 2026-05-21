"""Real-corpus construction for DSA-Scout."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from dsa_scout.logging import get_logger

log = get_logger(__name__)

TARGET_TOKEN_LENGTH = 1024
SAMPLES_PER_CATEGORY = 3
CATEGORIES = ("prose", "news", "code", "story", "structured")
CACHE_DIR = Path("corpus_cache")
MIN_UNIQUE_TOKENS = 200
MAX_NGRAM_REPEAT_RATIO = 0.30
MAX_ATTEMPTS_PER_SOURCE = 1000


@dataclass(frozen=True)
class CorpusSample:
    """A validated evaluation sample.

    Attributes:
        category: One of ``prose``, ``news``, ``code``, ``story``, or ``structured``.
        name: Stable sample identifier derived from the source stream.
        text: Sample text trimmed to the requested tokenizer length.
        token_count: Tokenizer length after trimming.
        unique_tokens: Number of unique token ids in the trimmed prefix.
        source: Dataset source label.
        top_bigram_ratio: Share of all bigrams occupied by the most common bigram.
        content_hash: SHA-256 hash of the trimmed text.
    """

    category: str
    name: str
    text: str
    token_count: int
    unique_tokens: int
    source: str
    top_bigram_ratio: float
    content_hash: str


@dataclass(frozen=True)
class TrainingCorpusRecord:
    """A validated training text with provenance."""

    name: str
    text: str
    token_count: int
    unique_tokens: int
    source: str
    top_bigram_ratio: float
    content_hash: str


@dataclass(frozen=True)
class RawDocument:
    """Raw source document yielded by a dataset stream."""

    source: str
    name: str
    text: str


@dataclass(frozen=True)
class SourceSpec:
    """A dataset source and its candidate generator."""

    source: str
    documents: Callable[[], Iterator[RawDocument]]
    fallback_for: str | None = None


def load_corpus(
    tok: Any,
    cache_dir: Path = CACHE_DIR,
    target_tokens: int = TARGET_TOKEN_LENGTH,
) -> list[CorpusSample]:
    """Load 15 real, diverse evaluation texts.

    Args:
        tok: Tokenizer with ``encode`` and ``decode`` methods.
        cache_dir: Directory for corpus cache files.
        target_tokens: Required tokenizer length for each sample.

    Returns:
        Three validated samples per category.

    Raises:
        RuntimeError: If the configured datasets and hard fallback cannot supply
            enough non-repetitive long documents.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "corpus_v3.json"
    if cache_path.exists() and not _force_rebuild():
        cached = _load_cached_samples(cache_path)
        if cached and all(sample.token_count == target_tokens for sample in cached):
            return cached

    samples: list[CorpusSample] = []
    for category in CATEGORIES:
        samples.extend(_load_category(tok, category, SAMPLES_PER_CATEGORY, target_tokens))

    expected = SAMPLES_PER_CATEGORY * len(CATEGORIES)
    if len(samples) != expected:
        msg = f"Only assembled {len(samples)} of {expected} evaluation samples"
        raise RuntimeError(msg)

    _save_eval_cache(cache_dir, samples, target_tokens)
    return samples


def load_training_texts(
    tok: Any,
    eval_samples: Iterable[CorpusSample],
    n_texts: int = 50,
    target_tokens: int = TARGET_TOKEN_LENGTH,
    cache_dir: Path = CACHE_DIR,
) -> list[str]:
    """Load held-out, hash-disjoint training texts.

    Args:
        tok: Tokenizer with ``encode`` and ``decode`` methods.
        eval_samples: Evaluation samples whose exact text hashes are excluded.
        n_texts: Minimum number of validated training texts.
        target_tokens: Required tokenizer length for each text.
        cache_dir: Directory for training corpus cache files.

    Returns:
        Validated training texts.
    """
    return [
        record.text
        for record in load_training_records(tok, eval_samples, n_texts, target_tokens, cache_dir)
    ]


def load_training_records(
    tok: Any,
    eval_samples: Iterable[CorpusSample],
    n_texts: int = 50,
    target_tokens: int = TARGET_TOKEN_LENGTH,
    cache_dir: Path = CACHE_DIR,
) -> list[TrainingCorpusRecord]:
    """Load held-out training records with source metadata."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "training_corpus_v3.json"
    eval_hashes = {_text_hash(sample.text) for sample in eval_samples}
    if cache_path.exists() and not _force_rebuild():
        cached_records = _load_cached_training(cache_path)
        cached_hashes = {record.content_hash for record in cached_records}
        if (
            len(cached_records) >= n_texts
            and eval_hashes.isdisjoint(cached_hashes)
            and all(record.token_count == target_tokens for record in cached_records[:n_texts])
        ):
            return cached_records[:n_texts]

    records: list[TrainingCorpusRecord] = []
    seen_hashes = set(eval_hashes)
    source_specs = _training_source_specs()
    per_source_target = max(1, n_texts // max(len(source_specs), 1))
    for index, spec in enumerate(source_specs):
        remaining_sources = max(len(source_specs) - index, 1)
        target_for_source = max(per_source_target, (n_texts - len(records)) // remaining_sources)
        records.extend(
            _collect_training_records(
                tok,
                spec,
                target_for_source,
                target_tokens,
                seen_hashes,
            )
        )
        seen_hashes.update(record.content_hash for record in records)
        if len(records) >= n_texts:
            break

    if len(records) < n_texts:
        for spec in source_specs:
            records.extend(
                _collect_training_records(
                    tok,
                    spec,
                    n_texts - len(records),
                    target_tokens,
                    seen_hashes,
                )
            )
            seen_hashes.update(record.content_hash for record in records)
            if len(records) >= n_texts:
                break

    if len(records) < n_texts:
        msg = (
            f"Only assembled {len(records)}/{n_texts} training texts. "
            "Corpus assembly requires either HuggingFace dataset access or nltk gutenberg."
        )
        raise RuntimeError(msg)

    _save_training_cache(cache_path, records[:n_texts], target_tokens)
    return records[:n_texts]


def corpus_metadata(samples: Sequence[CorpusSample]) -> dict[str, object]:
    """Summarize evaluation corpus provenance and diversity."""
    sources = Counter(sample.source for sample in samples)
    substitutions = {
        sample.category: sample.source
        for sample in samples
        if _primary_source_for_category(sample.category) not in sample.source
    }
    return {
        "num_samples": len(samples),
        "samples_per_category": SAMPLES_PER_CATEGORY,
        "target_tokens": TARGET_TOKEN_LENGTH,
        "sources": dict(sorted(sources.items())),
        "token_counts": {sample.name: sample.token_count for sample in samples},
        "sample_hashes": {sample.name: sample.content_hash for sample in samples},
        "diversity_stats": {
            "min_unique_tokens": min(sample.unique_tokens for sample in samples),
            "median_unique_tokens": float(
                sorted(sample.unique_tokens for sample in samples)[len(samples) // 2]
            ),
            "max_top_bigram_ratio": max(sample.top_bigram_ratio for sample in samples),
        },
        "substitutions": dict(sorted(substitutions.items())),
    }


def training_corpus_metadata(records: Sequence[TrainingCorpusRecord]) -> dict[str, object]:
    """Summarize training corpus provenance and diversity."""
    sources = Counter(record.source for record in records)
    return {
        "num_texts": len(records),
        "sources": dict(sorted(sources.items())),
        "min_unique_tokens": min(record.unique_tokens for record in records),
        "max_top_bigram_ratio": max(record.top_bigram_ratio for record in records),
        "hashes": [record.content_hash for record in records],
    }


def _load_category(
    tok: Any,
    category: str,
    needed: int,
    target_tokens: int,
) -> list[CorpusSample]:
    samples: list[CorpusSample] = []
    seen_hashes: set[str] = set()
    errors: list[str] = []
    for spec in _category_source_specs(category):
        try:
            samples.extend(
                _collect_eval_samples(
                    tok, category, spec, needed - len(samples), target_tokens, seen_hashes
                )
            )
            seen_hashes.update(sample.content_hash for sample in samples)
        except Exception as exc:
            detail = f"{spec.source}: {type(exc).__name__}: {exc}"
            errors.append(detail)
            log.warning("dataset source failed for %s: %s", category, detail)
        if len(samples) >= needed:
            return samples[:needed]

    try:
        fallback = SourceSpec(
            source="nltk.gutenberg",
            documents=lambda: _hard_fallback_documents(category),
            fallback_for="all_huggingface_sources",
        )
        samples.extend(
            _collect_eval_samples(
                tok, category, fallback, needed - len(samples), target_tokens, seen_hashes
            )
        )
    except Exception as exc:
        errors.append(f"nltk.gutenberg: {type(exc).__name__}: {exc}")

    if len(samples) < needed:
        msg = (
            f"Corpus assembly could not assemble {needed} valid {category} samples. "
            f"Tried sources: {'; '.join(errors) if errors else 'no source errors, only rejections'}. "
            "Corpus assembly requires either HuggingFace datasets access or nltk gutenberg."
        )
        raise RuntimeError(msg)
    return samples[:needed]


def _collect_eval_samples(
    tok: Any,
    category: str,
    spec: SourceSpec,
    needed: int,
    target_tokens: int,
    seen_hashes: set[str],
) -> list[CorpusSample]:
    if needed <= 0:
        return []
    accepted: list[CorpusSample] = []
    attempts = 0
    for raw in spec.documents():
        attempts += 1
        if attempts > MAX_ATTEMPTS_PER_SOURCE:
            break
        trimmed_result = _trim_or_skip(tok, raw.text, target_tokens)
        if trimmed_result is None:
            log.info("reject %s/%s/%s: too short", category, spec.source, raw.name)
            continue
        trimmed, ids = trimmed_result
        valid, reason = _validate_sample(ids, target_tokens)
        if not valid:
            log.info("reject %s/%s/%s: %s", category, spec.source, raw.name, reason)
            continue
        digest = _text_hash(trimmed)
        if digest in seen_hashes:
            log.info("reject %s/%s/%s: duplicate content hash", category, spec.source, raw.name)
            continue
        stats = _sample_stats(ids)
        accepted.append(
            CorpusSample(
                category=category,
                name=f"{category}_{len(accepted) + 1}_{_slug(raw.name)}",
                text=trimmed,
                token_count=target_tokens,
                unique_tokens=stats[0],
                source=raw.source,
                top_bigram_ratio=stats[1],
                content_hash=digest,
            )
        )
        seen_hashes.add(digest)
        log.info(
            "accepted %s sample %s from %s (%s unique tokens)",
            category,
            raw.name,
            raw.source,
            stats[0],
        )
        if len(accepted) >= needed:
            break
    return accepted


def _collect_training_records(
    tok: Any,
    spec: SourceSpec,
    needed: int,
    target_tokens: int,
    seen_hashes: set[str],
) -> list[TrainingCorpusRecord]:
    if needed <= 0:
        return []
    accepted: list[TrainingCorpusRecord] = []
    attempts = 0
    try:
        documents = spec.documents()
        for raw in documents:
            attempts += 1
            if attempts > MAX_ATTEMPTS_PER_SOURCE:
                break
            trimmed_result = _trim_or_skip(tok, raw.text, target_tokens)
            if trimmed_result is None:
                log.info("reject training/%s/%s: too short", spec.source, raw.name)
                continue
            trimmed, ids = trimmed_result
            valid, reason = _validate_sample(ids, target_tokens)
            if not valid:
                log.info("reject training/%s/%s: %s", spec.source, raw.name, reason)
                continue
            digest = _text_hash(trimmed)
            if digest in seen_hashes:
                log.info("reject training/%s/%s: duplicate content hash", spec.source, raw.name)
                continue
            stats = _sample_stats(ids)
            accepted.append(
                TrainingCorpusRecord(
                    name=f"train_{len(accepted) + 1}_{_slug(raw.name)}",
                    text=trimmed,
                    token_count=target_tokens,
                    unique_tokens=stats[0],
                    source=raw.source,
                    top_bigram_ratio=stats[1],
                    content_hash=digest,
                )
            )
            seen_hashes.add(digest)
            if len(accepted) >= needed:
                break
    except Exception as exc:
        log.warning("training source failed for %s: %s: %s", spec.source, type(exc).__name__, exc)
    return accepted


def _validate_sample(
    ids: Sequence[int],
    target_tokens: int = TARGET_TOKEN_LENGTH,
) -> tuple[bool, str]:
    """Reject samples that do not meet diversity standards."""
    if len(ids) < target_tokens:
        return False, f"too short: {len(ids)} < {target_tokens}"
    unique, top_bigram_ratio = _sample_stats(ids[:target_tokens])
    if unique < MIN_UNIQUE_TOKENS:
        return False, f"only {unique} unique tokens (min {MIN_UNIQUE_TOKENS})"
    if top_bigram_ratio > MAX_NGRAM_REPEAT_RATIO:
        return (
            False,
            f"top bigram occupies {top_bigram_ratio:.1%} of text "
            f"(max {MAX_NGRAM_REPEAT_RATIO:.0%})",
        )
    return True, "ok"


def _trim_or_skip(tok: Any, raw_text: str, target_tokens: int) -> tuple[str, list[int]] | None:
    """Trim to ``target_tokens`` and return ``None`` for short sources."""
    ids = cast(list[int], tok.encode(raw_text))
    if len(ids) < target_tokens:
        return None
    truncated = ids[:target_tokens]
    return str(tok.decode(truncated)), truncated


def _category_source_specs(category: str) -> tuple[SourceSpec, ...]:
    if category == "prose":
        return (
            SourceSpec(
                "wikitext-103-raw-v1",
                lambda: _wikitext_documents("wikitext-103-raw-v1", "test", "wikitext-103-raw-v1"),
            ),
            SourceSpec(
                "wikitext-2-raw-v1",
                lambda: _wikitext_documents("wikitext-2-raw-v1", "test", "wikitext-2-raw-v1"),
                fallback_for="wikitext-103-raw-v1",
            ),
        )
    if category == "news":
        return (
            SourceSpec(
                "cnn_dailymail",
                lambda: _field_documents(
                    "cnn_dailymail",
                    "3.0.0",
                    "test",
                    "article",
                    "cnn_dailymail",
                ),
            ),
            SourceSpec(
                "ag_news",
                lambda: _joined_field_documents("ag_news", None, "test", "text", "ag_news", 12),
                fallback_for="cnn_dailymail",
            ),
        )
    if category == "code":
        return (
            SourceSpec(
                "codeparrot/github-code-clean",
                lambda: _field_documents(
                    "codeparrot/github-code-clean",
                    None,
                    "train",
                    "code",
                    "codeparrot/github-code-clean",
                ),
            ),
            SourceSpec(
                "code_search_net",
                lambda: _code_search_net_documents("test"),
                fallback_for="codeparrot/github-code-clean",
            ),
        )
    if category == "story":
        return (
            SourceSpec(
                "pg19",
                lambda: _field_documents("pg19", None, "test", "text", "pg19"),
            ),
            SourceSpec(
                "roneneldan/TinyStories",
                lambda: _joined_field_documents(
                    "roneneldan/TinyStories",
                    None,
                    "validation",
                    "text",
                    "roneneldan/TinyStories",
                    18,
                ),
                fallback_for="pg19",
            ),
        )
    if category == "structured":
        return (
            SourceSpec(
                "sentence-transformers/eli5",
                lambda: _eli5_documents("train"),
                fallback_for="eli5_category",
            ),
            SourceSpec(
                "wikitext-103-raw-v1",
                lambda: _structured_wikitext_documents(skip=30),
                fallback_for="eli5_category",
            ),
        )
    msg = f"unknown corpus category: {category}"
    raise ValueError(msg)


def _training_source_specs() -> tuple[SourceSpec, ...]:
    return (
        SourceSpec(
            "wikitext-103-raw-v1/train",
            lambda: _wikitext_documents(
                "wikitext-103-raw-v1", "train", "wikitext-103-raw-v1/train"
            ),
        ),
        SourceSpec(
            "cnn_dailymail/train",
            lambda: _field_documents(
                "cnn_dailymail", "3.0.0", "train", "article", "cnn_dailymail/train"
            ),
        ),
        SourceSpec(
            "roneneldan/TinyStories/train",
            lambda: _joined_field_documents(
                "roneneldan/TinyStories",
                None,
                "train",
                "text",
                "roneneldan/TinyStories/train",
                18,
            ),
            fallback_for="pg19/train",
        ),
        SourceSpec(
            "code_search_net/validation",
            lambda: _code_search_net_documents("validation"),
            fallback_for="codeparrot/github-code-clean/train",
        ),
    )


def _load_dataset_rows(
    dataset_name: str,
    config_name: str | None,
    split: str,
    streaming: bool = True,
) -> Iterator[Mapping[str, Any]]:
    from datasets import load_dataset

    kwargs: dict[str, object] = {"split": split, "streaming": streaming}
    if config_name is None:
        dataset = load_dataset(dataset_name, **kwargs)
    else:
        dataset = load_dataset(dataset_name, config_name, **kwargs)
    yield from cast(Iterable[Mapping[str, Any]], dataset)


def _wikitext_documents(config_name: str, split: str, source: str) -> Iterator[RawDocument]:
    current: list[str] = []
    doc_idx = 0
    for row in _load_dataset_rows("wikitext", config_name, split):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        heading = text.startswith("=") and text.endswith("=")
        if heading and current:
            doc_idx += 1
            yield RawDocument(source=source, name=f"{split}_{doc_idx}", text="\n".join(current))
            current = [text]
        else:
            current.append(text)
    if current:
        doc_idx += 1
        yield RawDocument(source=source, name=f"{split}_{doc_idx}", text="\n".join(current))


def _structured_wikitext_documents(skip: int) -> Iterator[RawDocument]:
    for idx, raw in enumerate(
        _wikitext_documents("wikitext-103-raw-v1", "test", "wikitext-103-raw-v1")
    ):
        if idx < skip:
            continue
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw.text) if len(s.strip()) > 30]
        if len(sentences) < 12:
            continue
        lines = [f"Document: {raw.name}", "Key points:"]
        lines.extend(f"- {sentence}" for sentence in sentences)
        lines.append("Review checklist:")
        lines.extend(
            f"{number}. Confirm evidence for item {number}: {sentence}"
            for number, sentence in enumerate(sentences[:12], start=1)
        )
        yield RawDocument(
            source="wikitext-103-raw-v1/structured",
            name=f"structured_{raw.name}",
            text="\n".join(lines),
        )


def _field_documents(
    dataset_name: str,
    config_name: str | None,
    split: str,
    field: str,
    source: str,
) -> Iterator[RawDocument]:
    for idx, row in enumerate(_load_dataset_rows(dataset_name, config_name, split)):
        text = str(row.get(field, "")).strip()
        if text:
            yield RawDocument(source=source, name=f"{split}_{idx}", text=text)


def _joined_field_documents(
    dataset_name: str,
    config_name: str | None,
    split: str,
    field: str,
    source: str,
    rows_per_doc: int,
) -> Iterator[RawDocument]:
    chunk: list[str] = []
    chunk_idx = 0
    for row in _load_dataset_rows(dataset_name, config_name, split):
        text = str(row.get(field, "")).strip()
        if not text:
            continue
        chunk.append(text)
        if len(chunk) >= rows_per_doc:
            yield RawDocument(source=source, name=f"{split}_{chunk_idx}", text="\n\n".join(chunk))
            chunk_idx += 1
            chunk = []
    if chunk:
        yield RawDocument(source=source, name=f"{split}_{chunk_idx}", text="\n\n".join(chunk))


def _code_search_net_documents(split: str) -> Iterator[RawDocument]:
    chunk: list[str] = []
    chunk_idx = 0
    for row in _load_dataset_rows("code_search_net", "python", split):
        code = str(row.get("whole_func_string") or row.get("func_code_string") or "").strip()
        if not code:
            continue
        repo = str(row.get("repository_name", "unknown"))
        path = str(row.get("func_path_in_repository", "unknown.py"))
        chunk.append(f"# {repo}/{path}\n{code}")
        if len(chunk) >= 20 or sum(len(part) for part in chunk) >= 12_000:
            yield RawDocument(
                source=f"code_search_net/{split}",
                name=f"{split}_{chunk_idx}",
                text="\n\n".join(chunk),
            )
            chunk_idx += 1
            chunk = []
    if chunk:
        yield RawDocument(
            source=f"code_search_net/{split}",
            name=f"{split}_{chunk_idx}",
            text="\n\n".join(chunk),
        )


def _eli5_documents(split: str) -> Iterator[RawDocument]:
    chunk: list[str] = []
    chunk_idx = 0
    for row in _load_dataset_rows("sentence-transformers/eli5", None, split):
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue
        answer_bullets = answer.replace(". ", ".\n- ")
        chunk.append(f"Question: {question}\nAnswer bullets:\n- {answer_bullets}")
        if len(chunk) >= 10:
            yield RawDocument(
                source="sentence-transformers/eli5",
                name=f"{split}_{chunk_idx}",
                text="\n\n".join(chunk),
            )
            chunk_idx += 1
            chunk = []
    if chunk:
        yield RawDocument(
            source="sentence-transformers/eli5",
            name=f"{split}_{chunk_idx}",
            text="\n\n".join(chunk),
        )


def _hard_fallback_available() -> bool:
    """Return whether nltk Gutenberg can supply real long-form text."""
    try:
        import nltk
        from nltk.corpus import gutenberg

        nltk.download("gutenberg", quiet=True)
        return len(gutenberg.fileids()) >= 5
    except Exception:
        return False


def _hard_fallback_documents(category: str) -> Iterator[RawDocument]:
    if not _hard_fallback_available():
        msg = "Corpus assembly requires either HuggingFace datasets access or nltk gutenberg"
        raise RuntimeError(msg)
    from nltk.corpus import gutenberg

    fileids = sorted(gutenberg.fileids())
    category_offset = CATEGORIES.index(category) * 3
    for idx, fileid in enumerate(fileids[category_offset:] + fileids[:category_offset]):
        text = str(gutenberg.raw(fileid))
        yield RawDocument(source="nltk.gutenberg", name=f"{idx}_{fileid}", text=text)


def _load_cached_samples(path: Path) -> list[CorpusSample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = cast(list[dict[str, object]], raw["samples"] if isinstance(raw, dict) else raw)
    samples: list[CorpusSample] = []
    for record in records:
        text = str(record["text"])
        samples.append(
            CorpusSample(
                category=str(record["category"]),
                name=str(record["name"]),
                text=text,
                token_count=int(cast(int | str, record["token_count"])),
                unique_tokens=int(cast(int | str, record.get("unique_tokens", 0))),
                source=str(record.get("source", "unknown")),
                top_bigram_ratio=float(
                    cast(float | int | str, record.get("top_bigram_ratio", 0.0))
                ),
                content_hash=str(record.get("content_hash", _text_hash(text))),
            )
        )
    return samples


def _load_cached_training(path: Path) -> list[TrainingCorpusRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = cast(list[dict[str, object]], raw["samples"] if isinstance(raw, dict) else raw)
    return [
        TrainingCorpusRecord(
            name=str(record["name"]),
            text=str(record["text"]),
            token_count=int(cast(int | str, record["token_count"])),
            unique_tokens=int(cast(int | str, record["unique_tokens"])),
            source=str(record["source"]),
            top_bigram_ratio=float(cast(float | int | str, record["top_bigram_ratio"])),
            content_hash=str(record["content_hash"]),
        )
        for record in records
    ]


def _save_eval_cache(cache_dir: Path, samples: list[CorpusSample], target_tokens: int) -> None:
    digest = _content_hash(sample.content_hash for sample in samples)
    payload = {
        "version": 3,
        "target_tokens": target_tokens,
        "content_hash": digest,
        "samples": [asdict(sample) for sample in samples],
    }
    text = json.dumps(payload, indent=2)
    (cache_dir / "corpus_v3.json").write_text(text, encoding="utf-8")
    (cache_dir / f"corpus_v3_{target_tokens}_{digest[:16]}.json").write_text(text, encoding="utf-8")


def _save_training_cache(
    path: Path,
    records: list[TrainingCorpusRecord],
    target_tokens: int,
) -> None:
    digest = _content_hash(record.content_hash for record in records)
    payload = {
        "version": 3,
        "target_tokens": target_tokens,
        "content_hash": digest,
        "samples": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sample_stats(ids: Sequence[int]) -> tuple[int, float]:
    return _sample_stats_from_ids(ids)


def _sample_stats_from_ids(ids: Sequence[int]) -> tuple[int, float]:
    unique = len(set(ids))
    bigrams = list(pairwise(ids))
    if not bigrams:
        return unique, 1.0
    top_count = Counter(bigrams).most_common(1)[0][1]
    return unique, top_count / len(bigrams)


def _content_hash(parts: Iterable[str]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
    return h.hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _force_rebuild() -> bool:
    return os.environ.get("DSA_SCOUT_REBUILD_CORPUS") == "1"


def _primary_source_for_category(category: str) -> str:
    primary = {
        "prose": "wikitext-103-raw-v1",
        "news": "cnn_dailymail",
        "code": "codeparrot/github-code-clean",
        "story": "pg19",
        "structured": "eli5_category",
    }
    return primary[category]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug[:48] or "sample"

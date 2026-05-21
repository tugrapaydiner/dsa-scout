from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from dsa_scout import corpus
from dsa_scout.corpus import (
    CATEGORIES,
    RawDocument,
    SourceSpec,
    _trim_or_skip,
    _validate_sample,
    load_corpus,
    load_training_records,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            if word not in self._vocab:
                self._vocab[word] = len(self._vocab) + 1
            ids.append(self._vocab[word])
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(f"tok{i}" for i in ids)


def diverse_text(prefix: str, length: int = 360) -> str:
    return " ".join(f"{prefix}_{idx}" for idx in range(length))


def fake_category_specs(category: str) -> tuple[SourceSpec, ...]:
    def docs() -> Iterator[RawDocument]:
        for idx in range(4):
            yield RawDocument(
                source=f"fixture/{category}",
                name=f"{category}_{idx}",
                text=diverse_text(f"{category}_{idx}"),
            )

    return (SourceSpec(source=f"fixture/{category}", documents=docs),)


def fake_training_specs() -> tuple[SourceSpec, ...]:
    def docs() -> Iterator[RawDocument]:
        for idx in range(8):
            yield RawDocument(
                source="fixture/train",
                name=f"train_{idx}",
                text=diverse_text(f"train_{idx}"),
            )

    return (SourceSpec(source="fixture/train", documents=docs),)


def test_validate_sample_rejects_repeated_padding() -> None:
    tok = FakeTokenizer()
    repeated = "alpha beta gamma " * 120
    ok, reason = _validate_sample(tok.encode(repeated), target_tokens=256)
    assert not ok
    assert "unique tokens" in reason or "top bigram" in reason


def test_trim_or_skip_never_pads_short_text() -> None:
    tok = FakeTokenizer()
    assert _trim_or_skip(tok, "short text", 10) is None
    trimmed = _trim_or_skip(tok, diverse_text("long", 20), 10)
    assert trimmed is not None
    text, ids = trimmed
    assert len(tok.encode(text)) == 10
    assert len(ids) == 10


def test_load_corpus_has_15_diverse_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(corpus, "_category_source_specs", fake_category_specs)
    samples = load_corpus(FakeTokenizer(), cache_dir=tmp_path, target_tokens=256)
    assert len(samples) == 15
    assert {sample.category for sample in samples} == set(CATEGORIES)
    assert all(sample.token_count == 256 for sample in samples)
    assert all(sample.unique_tokens >= corpus.MIN_UNIQUE_TOKENS for sample in samples)
    assert all(sample.top_bigram_ratio <= corpus.MAX_NGRAM_REPEAT_RATIO for sample in samples)
    assert (tmp_path / "corpus_v3.json").exists()


def test_training_records_are_disjoint_and_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(corpus, "_category_source_specs", fake_category_specs)
    monkeypatch.setattr(corpus, "_training_source_specs", fake_training_specs)
    tok = FakeTokenizer()
    samples = load_corpus(tok, cache_dir=tmp_path, target_tokens=256)
    records = load_training_records(tok, samples, n_texts=5, target_tokens=256, cache_dir=tmp_path)
    cached = load_training_records(tok, samples, n_texts=5, target_tokens=256, cache_dir=tmp_path)
    assert len(records) == 5
    assert cached == records
    assert {sample.content_hash for sample in samples}.isdisjoint(
        {record.content_hash for record in records}
    )
    assert (tmp_path / "training_corpus_v3.json").exists()


def test_source_generators_parse_real_dataset_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_rows(
        dataset_name: str,
        config_name: str | None,
        split: str,
        streaming: bool = True,
    ) -> Iterator[dict[str, Any]]:
        del config_name, split, streaming
        if dataset_name == "wikitext":
            yield {"text": "= First ="}
            yield {
                "text": ". ".join(
                    f"Sentence {idx} has enough words for parsing" for idx in range(20)
                )
            }
            yield {"text": "= Second ="}
            yield {"text": diverse_text("wiki", 420)}
            return
        if dataset_name == "cnn_dailymail":
            yield {"article": diverse_text("article", 420)}
            return
        if dataset_name == "ag_news":
            for idx in range(3):
                yield {"text": diverse_text(f"ag_{idx}", 140)}
            return
        if dataset_name == "code_search_net":
            for idx in range(21):
                yield {
                    "repository_name": "repo",
                    "func_path_in_repository": f"module_{idx}.py",
                    "whole_func_string": f"def function_{idx}():\n    return {idx}",
                }
            return
        if dataset_name == "sentence-transformers/eli5":
            for idx in range(10):
                yield {
                    "question": f"why does example {idx} happen?",
                    "answer": " ".join(f"answer_{idx}_{word}." for word in range(80)),
                }

    monkeypatch.setattr(corpus, "_load_dataset_rows", fake_rows)
    assert next(corpus._wikitext_documents("wikitext-103-raw-v1", "test", "wiki")).source == "wiki"
    assert next(corpus._structured_wikitext_documents(skip=0)).source.endswith("structured")
    assert (
        next(corpus._field_documents("cnn_dailymail", "3.0.0", "test", "article", "cnn")).source
        == "cnn"
    )
    assert (
        next(corpus._joined_field_documents("ag_news", None, "test", "text", "ag", 2)).source
        == "ag"
    )
    assert next(corpus._code_search_net_documents("test")).source == "code_search_net/test"
    assert next(corpus._eli5_documents("train")).source == "sentence-transformers/eli5"


def test_load_corpus_uses_loud_hard_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_specs(category: str) -> tuple[SourceSpec, ...]:
        def boom() -> Iterator[RawDocument]:
            raise RuntimeError(f"{category} unavailable")
            yield RawDocument(source="never", name="never", text="")

        return (SourceSpec(source=f"broken/{category}", documents=boom),)

    def fake_hard(category: str) -> Iterator[RawDocument]:
        for idx in range(3):
            yield RawDocument(
                source="nltk.gutenberg",
                name=f"{category}_{idx}",
                text=diverse_text(f"hard_{category}_{idx}"),
            )

    monkeypatch.setattr(corpus, "_category_source_specs", failing_specs)
    monkeypatch.setattr(corpus, "_hard_fallback_documents", fake_hard)
    samples = load_corpus(FakeTokenizer(), cache_dir=tmp_path, target_tokens=256)
    assert len(samples) == 15
    assert {sample.source for sample in samples} == {"nltk.gutenberg"}


def test_category_source_specs_are_defined_for_all_categories() -> None:
    for category in CATEGORIES:
        specs = corpus._category_source_specs(category)
        assert specs
        assert all(spec.source for spec in specs)

    training_specs = corpus._training_source_specs()
    assert len(training_specs) >= 3

    with pytest.raises(ValueError, match="unknown corpus category"):
        corpus._category_source_specs("unknown")

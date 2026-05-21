# Data Provenance

DSA-Scout uses public text datasets through HuggingFace `datasets` when
available. The evaluation corpus is intentionally small but diverse: 15 held-out
samples, each trimmed to 1024 GPT-2 tokens, with three samples per text family.
The training corpus contains 50 hash-disjoint texts used only for KL
distillation of the Lightning Indexer.

The current cache records the following sources in `results/metadata.json`:

- `wikitext-103-raw-v1`
- `cnn_dailymail`
- `code_search_net`
- `roneneldan/TinyStories`
- `sentence-transformers/eli5`

The corpus loader rejects short or repetitive samples rather than padding them.
A sample must have at least 200 unique token ids and a top n-gram repeat ratio
no greater than 0.30. Evaluation and training samples are separated by SHA256
content hashes; the metadata file records those hashes for audit.

## Cached Text

`corpus_cache/` stores the selected text excerpts so that a reproduced run does
not silently change when upstream dataset rows shift. These cache files contain
third-party public dataset text. They are not credentials or private project
data, but they do carry the licensing and provenance constraints of their source
datasets.

For a source-only release, omit `corpus_cache/*.json` and regenerate it with:

```bash
dsa-scout reproduce
```

For a fully reproducible artifact release, keep the three corpus cache files and
include `results/metadata.json` and `results/manifest.json` alongside them.

## What Is Not Included

The repository should not include local virtual environments, Python caches,
coverage files, downloaded model cache directories, API tokens, or private
datasets. Run `python scripts/verify_release.py` before publishing to check the
expected release boundary.

# Methods

This note defines the DSA-Scout measurement surface. It is meant to make the
README claims auditable without expanding the README into a paper.

## Model And Tokens

All experiments use GPT-2 small. Each evaluation sample is tokenized with the
GPT-2 tokenizer and trimmed to 1024 tokens. The analysis reads the model's
causal self-attention tensors for all 12 transformer layers.

## Oracle Blocks

For a layer with attention tensor `A` of shape `[heads, seq, seq]`, DSA-Scout
first averages attention across heads. Keys are then grouped into compressed
blocks of size `m = 4`. The oracle block score for query position `i` and block
`b` is the sum of averaged attention mass from query `i` to all key positions in
that block. Causally masked positions are excluded from scoring and plotting.

This produces an oracle matrix with shape `[seq, seq / 4]`. The top-k oracle
blocks are the compressed blocks with the highest attention mass for each query.

## Candidate Scorers

The study compares deterministic baselines, untrained Lightning Indexer seeds,
and the trained Lightning Indexer:

- random
- recency
- window + sink
- linear decay
- one-head preview attention
- Lightning Indexer, untrained
- Lightning Indexer, KL-distilled
- trained Lightning + recency hybrid

The locked Lightning Indexer hyperparameters are `m = 4`, `n_I_h = 8`,
`c_I = 32`, and `d_c = 128`.

## Training

Training uses precomputed hidden states and oracle block targets from 50
hash-disjoint training texts. The loss is KL distillation from the oracle block
distribution to the indexer score distribution. The training loop saves the best
50-step rolling-loss checkpoint, not the last checkpoint. The optimizer uses
cosine decay with 50 warmup steps and gradient clipping at max norm 1.0.

## Metrics

Top-k recall is computed per query row as the overlap between the candidate
top-k block set and the oracle top-k block set, divided by `k`, then averaged
over valid rows. The headline number is top-8 recall averaged over middle layers
4 through 8.

Confidence intervals are deterministic bootstrap intervals over the recorded
per-sample/per-layer values. Paired deltas use the same evaluation points for
both systems, which is why the trained-vs-untrained and hybrid-vs-recency
comparisons are more informative than unpaired differences in raw means.

## Supported Claims

The results support three narrow claims: the trained Lightning Indexer learns a
real proxy for oracle attention blocks, recency remains a hard baseline to beat
under a fixed top-8 budget, and a small trained-Lightning term can be blended
with recency without hurting recall.

The results do not claim production DeepSeek Sparse Attention performance,
downstream task gains, larger-model generalization, or that learned routing
alone beats locality.

# ColBERT and SPLADE

Do not start truncation until full-budget baselines are verified.

ColBERT must use its token-level late-interaction pipeline; do not mean-pool it into one vector.

SPLADE must use its sparse vocabulary-space representation; do not replace it with a dense mean-pooled vector.

Record full-budget NDCG@10 for both before their reduction experiments.

# 100-epoch neural retrieval fine-tuning

## What has changed

A new script, `scripts/train_neural_retriever.py`, adds a gradient-based fine-tuning loop for Contriever and MedCPT. SPLADE now has a dedicated pairwise ranking trainer, `scripts/train_splade.py`, so its sparse-specific loss and token masking are explicit.

Both training scripts default to **100 epochs**. They log training/validation losses and save best/last checkpoints and JSON histories. The SPLADE trainer uses a positive-versus-negative pairwise ranking loss and a sparsity regularizer; its negative sampler excludes all known positive documents for each query.

Examples:

```bash
python scripts/train_neural_retriever.py --model contriever --size limit-small --epochs 100
python scripts/train_neural_retriever.py --model medcpt --size limit-small --epochs 100
python scripts/train_splade.py --size limit-small --epochs 100 --batch-size 2
```

The expected input layout is `data/<size>/queries.jsonl`, `corpus.jsonl`, and `qrels.jsonl`. The qrels JSONL rows need `query-id`, `corpus-id`, and `score`; only positive judgments are used as positive pairs. The script expects IDs to match the corresponding JSONL `_id` values.

## Critical benchmark caveat

If `qrels.jsonl` contains the official test judgments (as it does for some existing LIMIT data layouts), the script's 80/10/10 query split is an **exploratory split of those judgments**, not an official benchmark training split. Do not report scores on that derived split as official test results. For a valid benchmark experiment, train using an official training qrels file (or another documented training dataset) and evaluate only once on the untouched official test qrels. Query IDs are split as whole groups so a query cannot occur in multiple partitions.

## What should not be given epochs

BM25 is a non-parametric lexical retrieval baseline. PCA truncation is a post-hoc representation compression experiment. Data preparation and evaluation scripts also do not train models. Adding an `epochs` variable to these scripts would not train anything and would misrepresent the experiment, so they remain separate from neural fine-tuning.

## ColBERT status

The existing `run_colbert_baseline.py` performs inference/ranking, not training. ColBERT v2 uses its own checkpoint/training pipeline; it should not be claimed to have completed 100 epochs by adding a flag to the baseline. The new generic script does not fine-tune ColBERT. A separate ColBERT-native training run must be implemented and tested against the installed `colbert-ai` version before reporting ColBERT as trained for 100 epochs.

## Reproducibility

Set `--seed` for deterministic split/shuffle initialization. The scripts save `query_splits.json` and `training_history.json` under `checkpoints/<model>/<size>/`. SPLADE's default output is `checkpoints/splade/<size>/`. Full reproducibility can still vary by GPU, CUDA kernels, package versions, and model revision. Inspect the validation-loss curve and verify checkpoint loading before using the trained weights in evaluation.

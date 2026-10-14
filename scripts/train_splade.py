#!/usr/bin/env python3
"""Fine-tune a SPLADE sparse retriever with pairwise ranking loss.

Expected data files under data/<size>/:
  queries.jsonl: {"_id": "...", "text": "..."}
  corpus.jsonl:  {"_id": "...", "text": "..."}
  qrels.jsonl:   {"query-id": "...", "corpus-id": "...", "score": 1}

Example:
  python scripts/train_splade.py --size limit-small --epochs 100 --batch-size 2

IMPORTANT: if qrels.jsonl contains official test judgments, this script's
80/10/10 split is exploratory only, not an official benchmark training setup.
Use a proper training qrels file for publishable benchmark evaluation.
"""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_data(data_dir, seed):
    queries = {str(row["_id"]): str(row["text"]) for row in read_jsonl(data_dir / "queries.jsonl")}
    corpus_rows = read_jsonl(data_dir / "corpus.jsonl")
    corpus = {str(row["_id"]): str(row["text"]) for row in corpus_rows}
    qrels = {}
    for row in read_jsonl(data_dir / "qrels.jsonl"):
        qid, did, rel = str(row["query-id"]), str(row["corpus-id"]), int(row["score"])
        if qid in queries and did in corpus and rel > 0:
            qrels.setdefault(qid, set()).add(did)

    eligible = sorted(qid for qid, rel_docs in qrels.items() if rel_docs)
    if len(eligible) < 10:
        raise ValueError(f"Need >=10 queries with positive judgments; found {len(eligible)}")
    rng = random.Random(seed)
    rng.shuffle(eligible)
    n = len(eligible)
    n_train = max(1, int(n * 0.8))
    n_val = max(1, int(n * 0.1))
    if n_train + n_val >= n:
        n_train, n_val = n - 2, 1
    splits = {
        "train": eligible[:n_train],
        "validation": eligible[n_train:n_train + n_val],
        "test": eligible[n_train + n_val:],
    }
    return queries, corpus, qrels, splits


class QuerySampler(Dataset):
    """One query per sample; choose one judged positive and one unjudged negative."""
    def __init__(self, qids, queries, corpus, qrels, seed):
        self.qids = list(qids)
        self.queries = queries
        self.corpus = corpus
        self.qrels = qrels
        self.rng = random.Random(seed)
        self.all_doc_ids = list(corpus)
        self.samples = []
        for qid in self.qids:
            positives = sorted(qrels[qid])
            negative_pool_exists = any(did not in qrels[qid] for did in self.all_doc_ids)
            if positives and negative_pool_exists:
                self.samples.append(qid)
        if not self.samples:
            raise ValueError("No usable query has both a positive and an unjudged negative candidate.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        qid = self.samples[idx]
        positives = sorted(self.qrels[qid])
        positive_id = self.rng.choice(positives)
        # Random negatives are simple negatives, not hard negatives. Exclude every known
        # relevant document for this query to avoid false negatives in the available qrels.
        for _ in range(50):
            negative_id = self.rng.choice(self.all_doc_ids)
            if negative_id not in self.qrels[qid]:
                break
        else:
            negative_id = next(did for did in self.all_doc_ids if did not in self.qrels[qid])
        return self.queries[qid], self.corpus[positive_id], self.corpus[negative_id]


def collate(batch):
    q, pos, neg = zip(*batch)
    return {"queries": list(q), "positive_docs": list(pos), "negative_docs": list(neg)}


def sparse_encode(texts, tokenizer, model, device, max_length):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    special_mask = encoded.pop("special_tokens_mask").to(device).bool()
    encoded = {key: value.to(device) for key, value in encoded.items()}
    outputs = model(**encoded)
    # SPLADE term weights: log(1 + ReLU(MLM logits)), max pooled across
    # non-padding, non-special token positions.
    activations = torch.log1p(torch.relu(outputs.logits))
    valid_tokens = encoded["attention_mask"].bool() & ~special_mask
    activations = activations.masked_fill(~valid_tokens.unsqueeze(-1), 0.0)
    return activations.max(dim=1).values


def pairwise_loss(model, tokenizer, batch, device, max_query_length, max_doc_length, sparsity_weight):
    q_rep = sparse_encode(batch["queries"], tokenizer, model, device, max_query_length)
    p_rep = sparse_encode(batch["positive_docs"], tokenizer, model, device, max_doc_length)
    n_rep = sparse_encode(batch["negative_docs"], tokenizer, model, device, max_doc_length)

    positive_scores = (q_rep * p_rep).sum(dim=1)
    negative_scores = (q_rep * n_rep).sum(dim=1)
    ranking_loss = F.softplus(negative_scores - positive_scores).mean()

    # L1 regularization discourages diffuse vocabulary representations.
    sparsity_loss = q_rep.mean(dim=0).sum() + p_rep.mean(dim=0).sum() + n_rep.mean(dim=0).sum()
    return ranking_loss + sparsity_weight * sparsity_loss, ranking_loss.detach(), sparsity_loss.detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="limit-small")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-query-length", type=int, default=64)
    parser.add_argument("--max-doc-length", type=int, default=256)
    parser.add_argument("--sparsity-weight", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = ROOT / "data" / args.size
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "checkpoints" / "splade" / args.size
    out_dir.mkdir(parents=True, exist_ok=True)
    queries, corpus, qrels, splits = load_data(data_dir, args.seed)
    print("Query split sizes:", {name: len(ids) for name, ids in splits.items()})
    print("WARNING: if the input qrels are official test judgments, this is an exploratory split, not official benchmark training.")

    train_data = QuerySampler(splits["train"], queries, corpus, qrels, args.seed)
    val_data = QuerySampler(splits["validation"], queries, corpus, qrels, args.seed + 1)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model_name} on {device}; target epochs={args.epochs}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = rank_sum = sparse_sum = 0.0
        train_steps = 0
        for batch in tqdm(train_loader, desc=f"train {epoch}/{args.epochs}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            loss, rank_loss, sparse_loss = pairwise_loss(
                model, tokenizer, batch, device,
                args.max_query_length, args.max_doc_length, args.sparsity_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_sum += float(loss.detach().cpu())
            rank_sum += float(rank_loss.cpu())
            sparse_sum += float(sparse_loss.cpu())
            train_steps += 1

        model.eval()
        val_sum = val_rank_sum = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                loss, rank_loss, _ = pairwise_loss(
                    model, tokenizer, batch, device,
                    args.max_query_length, args.max_doc_length, args.sparsity_weight
                )
                val_sum += float(loss.cpu())
                val_rank_sum += float(rank_loss.cpu())
                val_steps += 1
        train_loss = train_sum / max(train_steps, 1)
        train_rank = rank_sum / max(train_steps, 1)
        train_sparse = sparse_sum / max(train_steps, 1)
        val_loss = val_sum / max(val_steps, 1)
        val_rank = val_rank_sum / max(val_steps, 1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ranking_loss": train_rank,
            "train_sparsity_loss": train_sparse,
            "validation_loss": val_loss,
            "validation_ranking_loss": val_rank,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d}/{args.epochs} train={train_loss:.6f} "
            f"(rank={train_rank:.6f}, sparse={train_sparse:.3f}) "
            f"validation={val_loss:.6f} (rank={val_rank:.6f})"
        )

        # Persist each epoch so a long run leaves a usable checkpoint even if interrupted.
        last_dir = out_dir / "last"
        last_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(last_dir)
        tokenizer.save_pretrained(last_dir)
        torch.save({"epoch": epoch, "optimizer": optimizer.state_dict()}, out_dir / "optimizer.pt")
        with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        if val_loss < best_val:
            best_val = val_loss
            best_dir = out_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            with open(best_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2)

    with open(out_dir / "query_splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"Finished {args.epochs} epochs. Best validation loss={best_val:.6f}")
    print(f"Best checkpoint: {out_dir / 'best'}")
    print(f"Training history: {out_dir / 'training_history.json'}")


if __name__ == "__main__":
    main()

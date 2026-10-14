#!/usr/bin/env python3
"""Fine-tune dense (Contriever/MedCPT) or SPLADE retrieval models.

IMPORTANT DATA NOTE
------------------
This script splits the query IDs in the qrels file into train/validation/test
(80/10/10). If the input qrels are the benchmark's official *test* qrels, this
creates an exploratory split of test judgments; it is NOT an official held-out
benchmark evaluation. For publishable benchmark results, provide training
qrels and keep the official test qrels untouched.

Expected input files under data/<size>/:
  queries.jsonl: {"_id": "...", "text": "..."}
  corpus.jsonl:  {"_id": "...", "text": "..."}
  qrels.jsonl:   {"query-id": "...", "corpus-id": "...", "score": 1}

Run:
  python scripts/train_neural_retriever.py --model contriever --size limit-small --epochs 100
  python scripts/train_neural_retriever.py --model medcpt --size limit-small --epochs 100
  python scripts/train_neural_retriever.py --model splade --size limit-small --epochs 100
"""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = {
    "contriever": ("facebook/contriever", "facebook/contriever"),
    "medcpt": ("ncbi/MedCPT-Query-Encoder", "ncbi/MedCPT-Article-Encoder"),
    "splade": ("naver/splade-cocondenser-ensembledistil", "naver/splade-cocondenser-ensembledistil"),
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_training_pairs(data_dir, seed):
    queries = {str(x["_id"]): str(x["text"]) for x in read_jsonl(data_dir / "queries.jsonl")}
    docs = {str(x["_id"]): str(x["text"]) for x in read_jsonl(data_dir / "corpus.jsonl")}
    qrels = {}
    for row in read_jsonl(data_dir / "qrels.jsonl"):
        qid, did, rel = str(row["query-id"]), str(row["corpus-id"]), int(row["score"])
        if qid in queries and did in docs and rel > 0:
            qrels.setdefault(qid, []).append(did)
    eligible = sorted(qid for qid, positives in qrels.items() if positives)
    if len(eligible) < 10:
        raise ValueError(f"Need at least 10 queries with positive qrels; found {len(eligible)}")
    rng = random.Random(seed)
    rng.shuffle(eligible)
    n = len(eligible)
    n_train = max(1, int(n * 0.8))
    n_val = max(1, int(n * 0.1))
    if n_train + n_val >= n:
        n_train, n_val = n - 2, 1
    split = {
        "train": eligible[:n_train],
        "validation": eligible[n_train:n_train + n_val],
        "test": eligible[n_train + n_val:],
    }
    pairs = {}
    for part, qids in split.items():
        part_pairs = []
        for qid in qids:
            for did in qrels[qid]:
                if did in docs:
                    part_pairs.append((queries[qid], docs[did], qid, did))
        pairs[part] = part_pairs
    return queries, docs, qrels, split, pairs


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        return self.pairs[idx]


def collate(batch):
    return {
        "queries": [x[0] for x in batch],
        "docs": [x[1] for x in batch],
    }


def dense_pool(hidden, mask):
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)


def splade_pool(logits, attention_mask):
    # Standard SPLADE-style log(1 + ReLU(logits)) max pooling.
    acts = torch.log1p(torch.relu(logits))
    # Exclude padding and special tokens from term activation.
    mask = attention_mask.bool()
    acts = acts.masked_fill(~mask.unsqueeze(-1), 0.0)
    return acts.max(dim=1).values


def encode(tokenizer, model, texts, device, max_length, is_splade):
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    outputs = model(**tokens)
    if is_splade:
        reps = splade_pool(outputs.logits, tokens["attention_mask"])
    else:
        reps = dense_pool(outputs.last_hidden_state, tokens["attention_mask"])
        reps = F.normalize(reps, p=2, dim=1)
    return reps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_NAMES), required=True)
    parser.add_argument("--size", default="limit-small")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-query-length", type=int, default=64)
    parser.add_argument("--max-doc-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sparsity-weight", type=float, default=1e-4)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("--epochs must be >= 1 and --batch-size must be >= 2")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = ROOT / "data" / args.size
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "checkpoints" / args.model / args.size
    out_dir.mkdir(parents=True, exist_ok=True)
    _, _, _, split, pairs = load_training_pairs(data_dir, args.seed)
    if not pairs["train"]:
        raise ValueError("No positive training query-document pairs were found.")
    if len(pairs["train"]) < args.batch_size:
        print(f"Warning: only {len(pairs['train'])} positive pairs; reducing batch size.")
        args.batch_size = max(2, min(args.batch_size, len(pairs["train"])))

    qname, dname = MODEL_NAMES[args.model]
    is_splade = args.model == "splade"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model={args.model}; device={device}; epochs={args.epochs}")
    print("Query split sizes:", {k: len(v) for k, v in split.items()})
    print("Positive pair counts:", {k: len(v) for k, v in pairs.items()})
    print("WARNING: split is only as independent as the qrels file supplied.")

    qtokenizer = AutoTokenizer.from_pretrained(qname)
    dtokenizer = qtokenizer if qname == dname else AutoTokenizer.from_pretrained(dname)
    if is_splade:
        qmodel = AutoModelForMaskedLM.from_pretrained(qname).to(device)
        dmodel = qmodel
    else:
        qmodel = AutoModel.from_pretrained(qname).to(device)
        dmodel = qmodel if qname == dname else AutoModel.from_pretrained(dname).to(device)

    params = list(qmodel.parameters())
    if dmodel is not qmodel:
        params += list(dmodel.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)

    train_loader = DataLoader(PairDataset(pairs["train"]), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(PairDataset(pairs["validation"]), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        qmodel.train()
        if dmodel is not qmodel:
            dmodel.train()
        running, steps = 0.0, 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            qvec = encode(qtokenizer, qmodel, batch["queries"], device, args.max_query_length, is_splade)
            dvec = encode(dtokenizer, dmodel, batch["docs"], device, args.max_doc_length, is_splade)
            scores = qvec @ dvec.transpose(0, 1)
            labels = torch.arange(scores.shape[0], device=device)
            loss = (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2
            if is_splade:
                loss = loss + args.sparsity_weight * (qvec.mean() + dvec.mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            running += float(loss.detach().cpu())
            steps += 1

        qmodel.eval()
        if dmodel is not qmodel:
            dmodel.eval()
        val_total, val_steps = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                qvec = encode(qtokenizer, qmodel, batch["queries"], device, args.max_query_length, is_splade)
                dvec = encode(dtokenizer, dmodel, batch["docs"], device, args.max_doc_length, is_splade)
                scores = qvec @ dvec.transpose(0, 1)
                labels = torch.arange(scores.shape[0], device=device)
                loss = (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2
                if is_splade:
                    loss = loss + args.sparsity_weight * (qvec.mean() + dvec.mean())
                val_total += float(loss.cpu())
                val_steps += 1
        val_loss = val_total / max(val_steps, 1)
        train_loss = running / max(steps, 1)
        record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss}
        history.append(record)
        print(f"epoch={epoch:03d}/{args.epochs} train_loss={train_loss:.6f} validation_loss={val_loss:.6f}")

        # Save last checkpoint every epoch so interrupted long runs can be resumed manually.
        epoch_dir = out_dir / "last"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        qmodel.save_pretrained(epoch_dir / "query_encoder")
        qtokenizer.save_pretrained(epoch_dir / "query_encoder")
        if dmodel is not qmodel:
            dmodel.save_pretrained(epoch_dir / "document_encoder")
            dtokenizer.save_pretrained(epoch_dir / "document_encoder")
        optimizer_state = out_dir / "optimizer.pt"
        torch.save({"epoch": epoch, "optimizer": optimizer.state_dict(), "history": history}, optimizer_state)
        with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        if val_loss < best_val:
            best_val = val_loss
            best_dir = out_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            qmodel.save_pretrained(best_dir / "query_encoder")
            qtokenizer.save_pretrained(best_dir / "query_encoder")
            if dmodel is not qmodel:
                dmodel.save_pretrained(best_dir / "document_encoder")
                dtokenizer.save_pretrained(best_dir / "document_encoder")
            with open(best_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)

    # Save split membership for auditability; query IDs are never mixed across splits.
    with open(out_dir / "query_splits.json", "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    print(f"Training complete. Best validation loss={best_val:.6f}")
    print(f"Checkpoints and logs: {out_dir}")


if __name__ == "__main__":
    main()

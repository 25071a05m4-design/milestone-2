import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM


MODEL_NAME = "naver/splade-cocondenser-ensembledistil"


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def splade_encode(texts, tokenizer, model, device, batch_size=8):
    """
    Convert text into SPLADE sparse representations.

    SPLADE representation:
        log(1 + ReLU(MLM logits))
        followed by max pooling over tokens.
    """

    all_embeddings = []

    model.eval()

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Encoding"
    ):
        batch_texts = texts[start:start + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

            logits = outputs.logits

            # SPLADE activation
            activations = torch.log1p(
                torch.relu(logits)
            )

            # Ignore padding tokens
            attention_mask = inputs["attention_mask"].unsqueeze(-1)

            activations = activations * attention_mask

            # Max pooling over tokens
            sparse_representation = torch.max(
                activations,
                dim=1
            ).values

        all_embeddings.append(
            sparse_representation.cpu()
        )

    return torch.cat(all_embeddings, dim=0)


def dcg_at_k(relevances, k):
    relevances = relevances[:k]

    if len(relevances) == 0:
        return 0.0

    discounts = np.log2(
        np.arange(2, len(relevances) + 2)
    )

    gains = (
        (2 ** np.array(relevances)) - 1
    )

    return float(
        np.sum(gains / discounts)
    )


def ndcg_at_k(ranked_doc_ids, relevant_docs, k=10):
    relevance = [
        1 if doc_id in relevant_docs else 0
        for doc_id in ranked_doc_ids[:k]
    ]

    dcg = dcg_at_k(relevance, k)

    ideal_relevance = sorted(
        [1] * len(relevant_docs),
        reverse=True
    )

    ideal_dcg = dcg_at_k(
        ideal_relevance,
        k
    )

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


def load_qrels(path):
    qrels = {}

    records = load_jsonl(path)

    for record in records:
        query_id = record["query-id"]
        doc_id = record["corpus-id"]
        score = int(record["score"])

        if score > 0:
            qrels.setdefault(query_id, set()).add(doc_id)

    return qrels


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--size",
        default="limit-small"
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    data_dir = root / "data" / args.size

    embeddings_dir = (
        root
        / "embeddings"
        / "splade"
        / args.size
    )

    results_dir = root / "results"
    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    embeddings_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    corpus_file = data_dir / "corpus.jsonl"
    queries_file = data_dir / "queries.jsonl"
    qrels_file = data_dir / "qrels.jsonl"

    print("=" * 60)
    print("SPLADE BASELINE")
    print("=" * 60)

    print()
    print("Dataset:", args.size)
    print("Model:", MODEL_NAME)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    print()
    print("Loading LIMIT data...")

    corpus = load_jsonl(corpus_file)
    queries = load_jsonl(queries_file)
    qrels = load_qrels(qrels_file)

    corpus_ids = [
        item["_id"]
        for item in corpus
    ]

    corpus_texts = [
        item["text"]
        for item in corpus
    ]

    query_ids = [
        item["_id"]
        for item in queries
    ]

    query_texts = [
        item["text"]
        for item in queries
    ]

    print("Queries :", len(query_texts))
    print("Corpus  :", len(corpus_texts))
    print("Qrels   :", len(qrels))

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    print()
    print("Loading SPLADE model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    print("Model loaded.")

    # ---------------------------------------------------------
    # Encode queries
    # ---------------------------------------------------------

    print()
    print("Encoding queries...")

    query_embeddings = splade_encode(
        query_texts,
        tokenizer,
        model,
        device,
        batch_size=8
    )

    print(
        "Query representation shape:",
        tuple(query_embeddings.shape)
    )

    # ---------------------------------------------------------
    # Encode corpus
    # ---------------------------------------------------------

    print()
    print("Encoding documents...")

    corpus_embeddings = splade_encode(
        corpus_texts,
        tokenizer,
        model,
        device,
        batch_size=8
    )

    print(
        "Corpus representation shape:",
        tuple(corpus_embeddings.shape)
    )

    # ---------------------------------------------------------
    # Save embeddings
    # ---------------------------------------------------------

    np.save(
        embeddings_dir / "queries.npy",
        query_embeddings.numpy()
    )

    np.save(
        embeddings_dir / "corpus.npy",
        corpus_embeddings.numpy()
    )

    (embeddings_dir / "query_ids.txt").write_text(
        "\n".join(query_ids),
        encoding="utf-8"
    )

    (embeddings_dir / "corpus_ids.txt").write_text(
        "\n".join(corpus_ids),
        encoding="utf-8"
    )

    print()
    print("Embeddings saved to:")
    print(embeddings_dir)

    # ---------------------------------------------------------
    # Ranking
    # ---------------------------------------------------------

    print()
    print("Computing SPLADE sparse dot-product rankings...")

    # Query x Document similarity
    #
    # SPLADE representations are sparse in concept,
    # but stored here as dense tensors for this small
    # 46-document LIMIT-small experiment.
    similarity = (
        query_embeddings
        @ corpus_embeddings.T
    )

    print(
        "Similarity matrix:",
        tuple(similarity.shape)
    )

    # ---------------------------------------------------------
    # Evaluate NDCG@10
    # ---------------------------------------------------------

    ndcg_scores = []

    run_file = (
        results_dir
        / f"splade_{args.size}.txt"
    )

    with open(
        run_file,
        "w",
        encoding="utf-8"
    ) as f:

        for query_index, query_id in enumerate(query_ids):

            scores = similarity[query_index]

            ranked_indices = torch.argsort(
                scores,
                descending=True
            ).tolist()

            ranked_doc_ids = [
                corpus_ids[index]
                for index in ranked_indices
            ]

            relevant_docs = qrels.get(
                query_id,
                set()
            )

            score = ndcg_at_k(
                ranked_doc_ids,
                relevant_docs,
                k=10
            )

            ndcg_scores.append(score)

            # Write top 10 results
            for rank, doc_index in enumerate(
                ranked_indices[:10],
                start=1
            ):
                doc_id = corpus_ids[doc_index]
                doc_score = float(
                    scores[doc_index]
                )

                f.write(
                    f"{query_id} Q0 "
                    f"{doc_id} {rank} "
                    f"{doc_score:.6f} SPLADE\n"
                )

            if (query_index + 1) % 50 == 0:
                print(
                    f"Processed queries: "
                    f"{query_index + 1}/{len(query_ids)}"
                )

    mean_ndcg = float(
        np.mean(ndcg_scores)
    )

    print()
    print("=" * 60)
    print("SPLADE LIMIT-SMALL BASELINE")
    print("=" * 60)
    print(
        f"Queries : {len(query_ids)}"
    )
    print(
        f"Corpus  : {len(corpus_ids)}"
    )
    print(
        f"NDCG@10 : {mean_ndcg:.6f}"
    )
    print()
    print("Run file:")
    print(run_file)
    print("=" * 60)


if __name__ == "__main__":
    main()
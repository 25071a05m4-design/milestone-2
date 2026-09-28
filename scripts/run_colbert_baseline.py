import json
import math
from pathlib import Path

import torch
from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "limit-small"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


# ============================================================
# LOAD LIMIT DATA
# ============================================================

def load_data():

    queries_raw = load_jsonl(
        DATA_DIR / "queries.jsonl"
    )

    corpus_raw = load_jsonl(
        DATA_DIR / "corpus.jsonl"
    )

    qrels_raw = load_jsonl(
        DATA_DIR / "qrels.jsonl"
    )

    queries = [
        (
            x["_id"],
            x["text"]
        )
        for x in queries_raw
    ]

    corpus = [
        (
            x["_id"],
            x["text"]
        )
        for x in corpus_raw
    ]

    qrels = {}

    for row in qrels_raw:

        qid = row["query-id"]
        docid = row["corpus-id"]
        score = int(row["score"])

        if score > 0:

            qrels.setdefault(
                qid,
                set()
            ).add(docid)

    return (
        queries,
        corpus,
        qrels
    )


# ============================================================
# NDCG
# ============================================================

def dcg_at_k(
    relevances,
    k=10
):

    score = 0.0

    for rank, rel in enumerate(
        relevances[:k],
        start=1
    ):

        score += (
            (2 ** rel - 1)
            /
            math.log2(rank + 1)
        )

    return score


def ndcg_at_10(
    ranking,
    relevant_docs
):

    relevances = [
        1 if docid in relevant_docs else 0
        for docid in ranking[:10]
    ]

    actual = dcg_at_k(
        relevances,
        10
    )

    ideal_count = min(
        len(relevant_docs),
        10
    )

    ideal = dcg_at_k(
        [1] * ideal_count,
        10
    )

    if ideal == 0:

        return 0.0

    return actual / ideal


# ============================================================
# COLBERT MAXSIM
# ============================================================

def maxsim_score(
    query_embedding,
    document_embedding
):

    """
    ColBERT late interaction:

        similarity = Q x D^T

        MaxSim:
        for every query token,
        take the maximum similarity
        over all document tokens.

        Final score:
        sum of those maximum values.
    """

    similarity = torch.matmul(
        query_embedding,
        document_embedding.transpose(
            0,
            1
        )
    )

    max_similarity = similarity.max(
        dim=1
    ).values

    return max_similarity.sum().item()


# ============================================================
# ENCODE QUERY
# ============================================================

def encode_query(
    checkpoint,
    text
):

    tokens = checkpoint.raw_tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )

    input_ids = tokens["input_ids"]

    attention_mask = tokens[
        "attention_mask"
    ]

    with torch.no_grad():

        Q = checkpoint.query(
            input_ids,
            attention_mask
        )

    return Q[0].cpu()


# ============================================================
# ENCODE DOCUMENT
# ============================================================

def encode_document(
    checkpoint,
    text
):

    tokens = checkpoint.raw_tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=180,
        return_tensors="pt"
    )

    input_ids = tokens["input_ids"]

    attention_mask = tokens[
        "attention_mask"
    ]

    with torch.no_grad():

        D = checkpoint.doc(
            input_ids,
            attention_mask,
            keep_dims=True,
            to_cpu=True
        )

    # Remove batch dimension
    if D.dim() == 3:

        D = D[0]

    # Remove padded token vectors.
    #
    # ColBERT's document mask tells us which
    # tokens are actual document tokens.

    mask = attention_mask[0].bool()

    D = D[mask]

    return D.cpu()


# ============================================================
# MAIN
# ============================================================

def main():

    queries, corpus, qrels = load_data()

    print(
        "LIMIT-small ColBERT baseline"
    )

    print(
        "--------------------------------"
    )

    print(
        "Queries :",
        len(queries)
    )

    print(
        "Corpus  :",
        len(corpus)
    )

    print()

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print(
        "Loading ColBERT v2.0..."
    )

    config = ColBERTConfig(
        query_maxlen=32,
        doc_maxlen=180,
        dim=128,
        similarity="cosine"
    )

    checkpoint = Checkpoint(
        "colbert-ir/colbertv2.0",
        colbert_config=config
    )

    print(
        "Model loaded."
    )

    print()

    # --------------------------------------------------------
    # ENCODE QUERIES
    # --------------------------------------------------------

    print(
        "Encoding queries..."
    )

    query_embeddings = []

    for i, (qid, text) in enumerate(
        queries
    ):

        Q = encode_query(
            checkpoint,
            text
        )

        query_embeddings.append(Q)

        if (i + 1) % 50 == 0:

            print(
                f"Encoded queries: "
                f"{i + 1}/{len(queries)}"
            )

    print()

    print(
        "Query embeddings:",
        len(query_embeddings)
    )

    print(
        "First query shape:",
        tuple(
            query_embeddings[0].shape
        )
    )

    print()

    # --------------------------------------------------------
    # ENCODE DOCUMENTS
    # --------------------------------------------------------

    print(
        "Encoding documents..."
    )

    document_embeddings = []

    for i, (docid, text) in enumerate(
        corpus
    ):

        D = encode_document(
            checkpoint,
            text
        )

        document_embeddings.append(D)

        print(
            f"Encoded document "
            f"{i + 1}/{len(corpus)}: "
            f"{docid}"
        )

    print()

    print(
        "Document embeddings:",
        len(document_embeddings)
    )

    print(
        "First document shape:",
        tuple(
            document_embeddings[0].shape
        )
    )

    print()

    # --------------------------------------------------------
    # RANK DOCUMENTS
    # --------------------------------------------------------

    print(
        "Computing ColBERT MaxSim rankings..."
    )

    run_path = (
        RESULTS_DIR
        /
        "colbert_limit-small.txt"
    )

    ndcg_scores = []

    with open(
        run_path,
        "w",
        encoding="utf-8"
    ) as run_file:

        for qi, (qid, _) in enumerate(
            queries
        ):

            query_embedding = (
                query_embeddings[qi]
            )

            scored_docs = []

            for di, (docid, _) in enumerate(
                corpus
            ):

                score = maxsim_score(
                    query_embedding,
                    document_embeddings[di]
                )

                scored_docs.append(
                    (
                        docid,
                        score
                    )
                )

            # Highest score first
            scored_docs.sort(
                key=lambda x: x[1],
                reverse=True
            )

            ranking = [
                docid
                for docid, score
                in scored_docs
            ]

            ndcg = ndcg_at_10(
                ranking,
                qrels.get(
                    qid,
                    set()
                )
            )

            ndcg_scores.append(
                ndcg
            )

            # Save top 10
            for rank, (
                docid,
                score
            ) in enumerate(
                scored_docs[:10],
                start=1
            ):

                run_file.write(
                    f"{qid} Q0 "
                    f"{docid} "
                    f"{rank} "
                    f"{score:.6f} "
                    f"ColBERT\n"
                )

            if (qi + 1) % 50 == 0:

                print(
                    f"Processed queries: "
                    f"{qi + 1}/{len(queries)}"
                )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    mean_ndcg = (
        sum(ndcg_scores)
        /
        len(ndcg_scores)
    )

    print()

    print(
        "========================================"
    )

    print(
        "COLBERT LIMIT-SMALL BASELINE"
    )

    print(
        "========================================"
    )

    print(
        "Queries :",
        len(queries)
    )

    print(
        "Corpus  :",
        len(corpus)
    )

    print(
        f"NDCG@10 : {mean_ndcg:.6f}"
    )

    print()

    print(
        "Run file:",
        run_path
    )


if __name__ == "__main__":

    main()
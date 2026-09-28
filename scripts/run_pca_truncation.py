import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


DIMENSIONS = [768, 384, 192, 96, 48, 24, 12]


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def load_qrels(path):
    qrels = {}

    for record in load_jsonl(path):
        query_id = record["query-id"]
        doc_id = record["corpus-id"]
        score = int(record["score"])

        if score > 0:
            qrels.setdefault(query_id, set()).add(doc_id)

    return qrels


def dcg_at_k(relevances, k):
    relevances = relevances[:k]

    if len(relevances) == 0:
        return 0.0

    discounts = np.log2(
        np.arange(2, len(relevances) + 2)
    )

    gains = (2 ** np.asarray(relevances)) - 1

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


def evaluate_dimension(
    query_embeddings,
    corpus_embeddings,
    query_ids,
    corpus_ids,
    qrels
):
    similarity = (
        query_embeddings
        @ corpus_embeddings.T
    )

    scores = []

    for query_index, query_id in enumerate(query_ids):

        ranked_indices = np.argsort(
            -similarity[query_index]
        )

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

        scores.append(score)

    return float(np.mean(scores))


def run_model(
    model_name,
    root,
    size,
    dimensions,
    query_ids,
    corpus_ids,
    qrels
):
    print()
    print("=" * 60)
    print(f"MODEL: {model_name.upper()}")
    print("=" * 60)

    embedding_dir = (
        root
        / "embeddings"
        / model_name
        / size
    )

    query_file = embedding_dir / "queries.npy"
    corpus_file = embedding_dir / "corpus.npy"

    if not query_file.exists():
        raise FileNotFoundError(
            f"Missing: {query_file}"
        )

    if not corpus_file.exists():
        raise FileNotFoundError(
            f"Missing: {corpus_file}"
        )

    queries = np.load(query_file)
    corpus = np.load(corpus_file)

    print(
        "Original query shape :",
        queries.shape
    )

    print(
        "Original corpus shape:",
        corpus.shape
    )

    # ---------------------------------------------------------
    # Normalize before PCA
    # ---------------------------------------------------------
    #
    # Each embedding is normalized independently.
    # This makes the PCA experiment less sensitive to
    # differences in vector magnitude.
    #
    query_norms = np.linalg.norm(
        queries,
        axis=1,
        keepdims=True
    )

    corpus_norms = np.linalg.norm(
        corpus,
        axis=1,
        keepdims=True
    )

    query_norms[
        query_norms == 0
    ] = 1

    corpus_norms[
        corpus_norms == 0
    ] = 1

    queries_normalized = (
        queries / query_norms
    )

    corpus_normalized = (
        corpus / corpus_norms
    )

    # ---------------------------------------------------------
    # Fit PCA ONLY on corpus
    # ---------------------------------------------------------

    max_dimension = max(dimensions)

    print()
    print(
        f"Fitting PCA with {max_dimension} components..."
    )

    pca = PCA(
        n_components=max_dimension,
        random_state=42
    )

    corpus_pca = pca.fit_transform(
        corpus_normalized
    )

    query_pca = pca.transform(
        queries_normalized
    )

    print("PCA complete.")

    results = []

    # ---------------------------------------------------------
    # Evaluate every dimensionality
    # ---------------------------------------------------------

    for dimension in dimensions:

        print()
        print(
            f"Evaluating dimension: {dimension}"
        )

        if dimension == max_dimension:
            corpus_reduced = corpus_pca
            query_reduced = query_pca
        else:
            corpus_reduced = corpus_pca[
                :, :dimension
            ]

            query_reduced = query_pca[
                :, :dimension
            ]

        # Normalize after projection
        corpus_reduced_norm = np.linalg.norm(
            corpus_reduced,
            axis=1,
            keepdims=True
        )

        query_reduced_norm = np.linalg.norm(
            query_reduced,
            axis=1,
            keepdims=True
        )

        corpus_reduced_norm[
            corpus_reduced_norm == 0
        ] = 1

        query_reduced_norm[
            query_reduced_norm == 0
        ] = 1

        corpus_reduced = (
            corpus_reduced
            / corpus_reduced_norm
        )

        query_reduced = (
            query_reduced
            / query_reduced_norm
        )

        ndcg = evaluate_dimension(
            query_reduced,
            corpus_reduced,
            query_ids,
            corpus_ids,
            qrels
        )

        print(
            f"Dimension {dimension:>4} "
            f"→ NDCG@10 = {ndcg:.6f}"
        )

        results.append(
            {
                "model": model_name,
                "dimension": dimension,
                "ndcg@10": ndcg
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--size",
        default="limit-small"
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    data_dir = (
        root
        / "data"
        / args.size
    )

    results_dir = root / "results"

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Load IDs and qrels
    # ---------------------------------------------------------

    queries = load_jsonl(
        data_dir / "queries.jsonl"
    )

    corpus = load_jsonl(
        data_dir / "corpus.jsonl"
    )

    qrels = load_qrels(
        data_dir / "qrels.jsonl"
    )

    query_ids = [
        item["_id"]
        for item in queries
    ]

    corpus_ids = [
        item["_id"]
        for item in corpus
    ]

    print("=" * 60)
    print("PCA TRUNCATION EXPERIMENT")
    print("=" * 60)

    print(
        "Dataset:",
        args.size
    )

    print(
        "Queries:",
        len(query_ids)
    )

    print(
        "Corpus:",
        len(corpus_ids)
    )

    print(
        "Dimensions:",
        DIMENSIONS
    )

    all_results = []

    # ---------------------------------------------------------
    # Run Contriever
    # ---------------------------------------------------------

    all_results.extend(
        run_model(
            "contriever",
            root,
            args.size,
            DIMENSIONS,
            query_ids,
            corpus_ids,
            qrels
        )
    )

    # ---------------------------------------------------------
    # Run MedCPT
    # ---------------------------------------------------------

    all_results.extend(
        run_model(
            "medcpt",
            root,
            args.size,
            DIMENSIONS,
            query_ids,
            corpus_ids,
            qrels
        )
    )

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    output_file = (
        results_dir
        / f"pca_truncation_{args.size}.csv"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "model,dimension,ndcg@10\n"
        )

        for result in all_results:

            f.write(
                f"{result['model']},"
                f"{result['dimension']},"
                f"{result['ndcg@10']:.6f}\n"
            )

    # ---------------------------------------------------------
    # Print final table
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL PCA TRUNCATION RESULTS")
    print("=" * 60)

    print(
        f"{'Model':<15}"
        f"{'Dimension':>12}"
        f"{'NDCG@10':>15}"
    )

    print("-" * 42)

    for result in all_results:

        print(
            f"{result['model']:<15}"
            f"{result['dimension']:>12}"
            f"{result['ndcg@10']:>15.6f}"
        )

    print()
    print("Results saved to:")
    print(output_file)
    print("=" * 60)


if __name__ == "__main__":
    main()
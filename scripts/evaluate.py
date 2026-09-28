import argparse
import math
from pathlib import Path


def load_qrels(path):
    qrels = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.strip().split()

            # Format:
            # query_id  0  document_id_with_spaces  relevance
            qid = parts[0]
            relevance = int(parts[-1])
            docid = " ".join(parts[2:-1])

            if relevance > 0:
                qrels.setdefault(qid, set()).add(docid)

    return qrels


def load_run(path):
    runs = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.strip().split()

            # Format:
            # query_id  Q0  document_id_with_spaces  rank  score  run_name
            qid = parts[0]
            rank = int(parts[-3])
            score = float(parts[-2])
            docid = " ".join(parts[2:-3])

            runs.setdefault(qid, []).append(
                (docid, rank, score)
            )

    for qid in runs:
        runs[qid].sort(key=lambda x: x[1])

    return runs


def dcg(relevances):
    total = 0.0

    for i, rel in enumerate(relevances, start=1):
        total += (2 ** rel - 1) / math.log2(i + 1)

    return total


def ndcg_at_k(qrels, run, k):
    scores = []

    for qid, relevant_docs in qrels.items():

        retrieved = run.get(qid, [])[:k]

        relevances = [
            1 if docid in relevant_docs else 0
            for docid, _, _ in retrieved
        ]

        actual_dcg = dcg(relevances)

        ideal_relevances = [
            1
        ] * min(len(relevant_docs), k)

        ideal_dcg = dcg(ideal_relevances)

        if ideal_dcg > 0:
            scores.append(actual_dcg / ideal_dcg)

    return sum(scores) / len(scores)


def recall_at_k(qrels, run, k):
    scores = []

    for qid, relevant_docs in qrels.items():

        retrieved = run.get(qid, [])[:k]

        retrieved_docs = {
            docid
            for docid, _, _ in retrieved
        }

        found = len(
            relevant_docs.intersection(retrieved_docs)
        )

        recall = found / len(relevant_docs)

        scores.append(recall)

    return sum(scores) / len(scores)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--size",
        choices=["limit-small", "limit"],
        required=True
    )

    parser.add_argument(
        "--run",
        required=True
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent

    qrels_path = (
        base_dir
        / "work"
        / args.size
        / "qrels.txt"
    )

    run_path = Path(args.run)

    if not run_path.is_absolute():
        run_path = base_dir / run_path

    qrels = load_qrels(qrels_path)
    run = load_run(run_path)

    ndcg10 = ndcg_at_k(qrels, run, 10)
    recall10 = recall_at_k(qrels, run, 10)
    recall100 = recall_at_k(qrels, run, 100)

    print("=" * 50)
    print("LIMIT BASELINE EVALUATION")
    print("=" * 50)
    print(f"Dataset    : {args.size}")
    print(f"Queries    : {len(qrels)}")
    print(f"NDCG@10    : {ndcg10:.6f}")
    print(f"Recall@10  : {recall10:.6f}")
    print(f"Recall@100 : {recall100:.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
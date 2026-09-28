import argparse
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--size", choices=["limit-small", "limit"], default="limit-small")
parser.add_argument("--run", required=True)
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]

# ============================================================
# Load qrels
# Format:
# query_id 0 document_id relevance
#
# Document IDs may contain spaces.
# ============================================================

qrels_path = root / "work" / args.size / "qrels.txt"

qrels = {}

with open(qrels_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        query_id = parts[0]
        relevance = int(parts[-1])

        # Everything between "0" and relevance is the document ID
        doc_id = " ".join(parts[2:-1])

        if query_id not in qrels:
            qrels[query_id] = {}

        qrels[query_id][doc_id] = relevance


# ============================================================
# Load BM25 run
#
# Format:
# query_id Q0 document_id rank score run_name
#
# Document IDs may contain spaces.
# ============================================================

run_path = Path(args.run)

run = {}

with open(run_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        query_id = parts[0]

        # The run format is:
        # query_id Q0 doc_id rank score run_name
        #
        # rank is the second-last field
        # score is the third-last field
        #
        # Everything between Q0 and rank is the document ID.

        rank = int(parts[-3])
        score = float(parts[-2])

        doc_id = " ".join(parts[2:-3])

        if query_id not in run:
            run[query_id] = []

        run[query_id].append((doc_id, score, rank))


# ============================================================
# Sort retrieved documents by score
# ============================================================

for query_id in run:
    run[query_id].sort(
        key=lambda x: x[1],
        reverse=True
    )


# ============================================================
# DCG
# ============================================================

def dcg(relevances):

    value = 0.0

    for rank, relevance in enumerate(relevances, start=1):
        value += relevance / math.log2(rank + 1)

    return value


# ============================================================
# NDCG@10
# ============================================================

def ndcg_at_10(query_id):

    retrieved = run.get(query_id, [])

    retrieved_top10 = retrieved[:10]

    actual_relevances = []

    for doc_id, score, rank in retrieved_top10:

        relevance = qrels.get(query_id, {}).get(doc_id, 0)

        actual_relevances.append(relevance)

    actual_dcg = dcg(actual_relevances)

    # Ideal ranking
    ideal_relevances = sorted(
        qrels.get(query_id, {}).values(),
        reverse=True
    )[:10]

    ideal_dcg = dcg(ideal_relevances)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


# ============================================================
# Calculate mean NDCG@10
# ============================================================

all_queries = sorted(qrels.keys())

scores = []

for query_id in all_queries:
    scores.append(ndcg_at_10(query_id))

mean_ndcg = (
    sum(scores) / len(scores)
    if scores
    else 0.0
)


# ============================================================
# Print result
# ============================================================

print()
print("=" * 50)
print("LIMIT BASELINE EVALUATION")
print("=" * 50)
print(f"Dataset : {args.size}")
print(f"Queries : {len(all_queries)}")
print(f"NDCG@10 : {mean_ndcg:.6f}")
print("=" * 50)
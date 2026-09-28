import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--size", default="limit-small")
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]

# Correct embedding folder:
# embeddings/<model>/<size>/
emb_dir = root / "embeddings" / args.model / args.size

results = root / "results"
results.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Load embeddings
# ------------------------------------------------------------

query_embeddings = np.load(
    emb_dir / "queries.npy"
)

doc_embeddings = np.load(
    emb_dir / "corpus.npy"
)


# ------------------------------------------------------------
# Load IDs saved during encoding
# ------------------------------------------------------------

with open(emb_dir / "query_ids.txt", "r", encoding="utf-8") as f:
    query_ids = [line.rstrip("\n") for line in f]

with open(emb_dir / "corpus_ids.txt", "r", encoding="utf-8") as f:
    doc_ids = [line.rstrip("\n") for line in f]


print("Query embeddings:", query_embeddings.shape)
print("Corpus embeddings:", doc_embeddings.shape)

print("Query IDs:", len(query_ids))
print("Document IDs:", len(doc_ids))


# ------------------------------------------------------------
# Check that embeddings and IDs match
# ------------------------------------------------------------

if len(query_ids) != query_embeddings.shape[0]:
    raise ValueError(
        f"Query count mismatch: "
        f"{len(query_ids)} IDs but "
        f"{query_embeddings.shape[0]} embeddings"
    )

if len(doc_ids) != doc_embeddings.shape[0]:
    raise ValueError(
        f"Document count mismatch: "
        f"{len(doc_ids)} IDs but "
        f"{doc_embeddings.shape[0]} embeddings"
    )


# ------------------------------------------------------------
# Normalize embeddings
# ------------------------------------------------------------

query_embeddings = query_embeddings / np.linalg.norm(
    query_embeddings,
    axis=1,
    keepdims=True
)

doc_embeddings = doc_embeddings / np.linalg.norm(
    doc_embeddings,
    axis=1,
    keepdims=True
)


# ------------------------------------------------------------
# Calculate cosine similarity
# ------------------------------------------------------------

similarities = query_embeddings @ doc_embeddings.T

print("Similarity matrix:", similarities.shape)


# ------------------------------------------------------------
# Create TREC run
# ------------------------------------------------------------

output_file = results / f"{args.model}_{args.size}.txt"

with open(output_file, "w", encoding="utf-8") as f:

    for i, query_id in enumerate(query_ids):

        scores = similarities[i]

        # Highest similarity first
        ranked_indices = np.argsort(-scores)

        # Keep top 100 documents
        ranked_indices = ranked_indices[:100]

        for rank, doc_index in enumerate(ranked_indices, start=1):

            doc_id = doc_ids[doc_index]
            score = float(scores[doc_index])

            f.write(
                f"{query_id} Q0 {doc_id} "
                f"{rank} {score:.8f} {args.model}\n"
            )

print("Dense run:", output_file)
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--size",
    choices=["limit-small", "limit"],
    default="limit-small"
)
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]

data = root / "data" / args.size
work = root / "work" / args.size

work.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# Read official LIMIT JSONL files
queries = read_jsonl(data / "queries.jsonl")
corpus = read_jsonl(data / "corpus.jsonl")
qrels = read_jsonl(data / "qrels.jsonl")


# -------------------------------------------------
# 1. Prepare corpus for Pyserini
# -------------------------------------------------

with open(work / "corpus.jsonl", "w", encoding="utf-8") as f:
    for doc in corpus:
        doc_id = str(doc["_id"])
        text = str(doc["text"])

        output = {
            "id": doc_id,
            "contents": text
        }

        f.write(json.dumps(output, ensure_ascii=False) + "\n")


# -------------------------------------------------
# 2. Prepare queries.tsv
# -------------------------------------------------

with open(work / "queries.tsv", "w", encoding="utf-8", newline="\n") as f:
    for query in queries:
        query_id = str(query["_id"])
        text = str(query["text"])

        # IMPORTANT:
        # \t and \n here are real tab/newline characters,
        # not the literal strings "\\t" and "\\n".
        f.write(query_id + "\t" + text + "\n")


# -------------------------------------------------
# 3. Prepare qrels.txt
# -------------------------------------------------

with open(work / "qrels.txt", "w", encoding="utf-8", newline="\n") as f:
    for row in qrels:
        query_id = str(row["query-id"])
        doc_id = str(row["corpus-id"])
        score = str(row["score"])

        f.write(query_id + " 0 " + doc_id + " " + score + "\n")


print(f"queries: {len(queries)}")
print(f"corpus: {len(corpus)}")
print(f"qrels: {len(qrels)}")
print(f"Prepared data in: {work}")
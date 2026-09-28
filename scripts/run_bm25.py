import argparse
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument(
    "--size",
    choices=["limit-small", "limit"],
    default="limit-small"
)
a = p.parse_args()

root = Path(__file__).resolve().parents[1]
work = root / "work" / a.size

idx = work / "bm25_index"
run = root / "results" / f"bm25_{a.size}.txt"

idx.parent.mkdir(parents=True, exist_ok=True)
run.parent.mkdir(exist_ok=True)

if not idx.exists():
    subprocess.run([
        "python",
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(work),
        "--index",
        str(idx),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        "4",
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw"
    ], check=True)

subprocess.run([
    "python",
    "-m",
    "pyserini.search.lucene",
    "--index",
    str(idx),
    "--topics",
    str(work / "queries.tsv"),
    "--output",
    str(run),
    "--output-format",
    "trec",
    "--bm25",
    "--hits",
    "100"
], check=True)

print("BM25 run:", run)
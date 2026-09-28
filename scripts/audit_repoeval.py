#!/usr/bin/env python3
"""Audit RepoEval counts and whether positive qrels stay in the query's file."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "scripts" / "code-rag-bench" / "retrieval" / "datasets"
OUTPUT = ROOT / "results" / "final" / "dataset_construct_audit.json"


def load_jsonl(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        values = [json.loads(line) for line in handle if line.strip()]
    return {value["_id"]: value for value in values}


def main() -> None:
    rows = []
    for repo_dir in sorted(DATA.glob("repoeval__*")):
        corpus = load_jsonl(repo_dir / "corpus.jsonl")
        queries = load_jsonl(repo_dir / "queries.jsonl")
        positive = 0
        same_file = 0
        with (repo_dir / "qrels" / "test.tsv").open(encoding="utf-8", newline="") as handle:
            for qrel in csv.DictReader(handle, delimiter="\t"):
                if int(qrel["score"]) <= 0:
                    continue
                positive += 1
                query_path = "-".join(queries[qrel["query-id"]]["metadata"]["fpath_tuple"])
                same_file += corpus[qrel["corpus-id"]]["title"] == query_path
        rows.append({
            "repo": repo_dir.name,
            "positive_qrels": positive,
            "positive_qrels_in_query_file": same_file,
            "share_in_query_file": same_file / positive if positive else None,
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()

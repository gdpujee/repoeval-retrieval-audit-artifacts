#!/usr/bin/env python3
"""Run CodeSage-small-v2 as a strict-budget dense RepoEval baseline."""

from __future__ import annotations

import csv
import json
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np
import sentence_transformers
import torch
import transformers
from sentence_transformers import SentenceTransformer
from transformers import modeling_utils
from transformers.pytorch_utils import Conv1D

from gcc import Retriever, f1, mrr, ndcg, precision, recall


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "scripts" / "code-rag-bench" / "retrieval" / "datasets"
OUTPUT = ROOT / "results" / "dense"
MODEL_NAME = "codesage/codesage-small-v2"
MODEL_REVISION = "4844c2f24b25e181aa43ca058cc73dd2622565c1"
BUDGETS = (500, 1000, 2000, 4000, 8000, 16000)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def install_codesage_compatibility_shim() -> None:
    """Expose the legacy Conv1D import used by the pinned CodeSage revision."""
    if not hasattr(modeling_utils, "Conv1D"):
        modeling_utils.Conv1D = Conv1D


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_qrels(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if int(row["score"]) > 0:
                result[row["query-id"]].add(row["corpus-id"])
    return dict(result)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    install_codesage_compatibility_shim()
    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        device=DEVICE,
    )
    rows = []
    for repo_dir in sorted(DATA.glob("repoeval__*")):
        corpus = read_jsonl(repo_dir / "corpus.jsonl")
        queries = [q for q in read_jsonl(repo_dir / "queries.jsonl")]
        qrels = load_qrels(repo_dir / "qrels" / "test.tsv")
        queries = [query for query in queries if query["_id"] in qrels]
        retriever = Retriever(corpus)
        print(f"encoding {repo_dir.name}: {len(corpus)} docs, {len(queries)} queries")
        doc_embeddings = model.encode(
            [doc.get("text", "") for doc in corpus], batch_size=16,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True,
        )
        query_embeddings = model.encode(
            [query.get("text", "") for query in queries], batch_size=16,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True,
        )
        similarities = query_embeddings @ doc_embeddings.T
        for query_index, query in enumerate(queries):
            ranking = np.argsort(-similarities[query_index], kind="stable").tolist()
            relevant = qrels[query["_id"]]
            for budget in BUDGETS:
                selected_indices = retriever.select(ranking, budget)
                selected = [corpus[index]["_id"] for index in selected_indices]
                tokens = sum(retriever.tokens[index] for index in selected_indices)
                rows.append({
                    "repo": repo_dir.name, "query_id": query["_id"],
                    "strategy": "codesage_small_v2", "model": MODEL_NAME,
                    "budget": budget, "recall": recall(selected, relevant),
                    "precision": precision(selected, relevant), "f1": f1(selected, relevant),
                    "ndcg@10": ndcg(selected, relevant), "mrr": mrr(selected, relevant),
                    "tokens": tokens, "n_docs": len(selected),
                })
    raw_path = OUTPUT / "codesage_small_v2_query_results.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["repo"], row["budget"])].append(row)
    summaries = []
    for (repo, budget), members in sorted(groups.items()):
        summaries.append({
            "repo": repo, "strategy": "codesage_small_v2", "budget": budget,
            "n_queries": len(members),
            **{key: sum(row[key] for row in members) / len(members)
               for key in ("recall", "precision", "f1", "ndcg@10", "mrr")},
            "avg_tokens": sum(row["tokens"] for row in members) / len(members),
            "max_tokens": max(row["tokens"] for row in members),
            "avg_docs": sum(row["n_docs"] for row in members) / len(members),
        })
    with (OUTPUT / "codesage_small_v2_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in summaries:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    environment = {
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "model_max_seq_length": model.max_seq_length,
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "numpy": np.__version__,
        "device": DEVICE,
    }
    (OUTPUT / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(raw_path)


if __name__ == "__main__":
    main()

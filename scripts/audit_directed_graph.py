#!/usr/bin/env python3
"""Build and audit directed local-module dependency graphs for RepoEval repos."""

from __future__ import annotations

import json
from pathlib import Path

from gcc import Retriever


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "scripts" / "code-rag-bench" / "retrieval" / "datasets"
SOURCES = ROOT / "scripts" / "code-rag-bench" / "retrieval" / "output" / "repoeval" / "repositories" / "function_level"
OUTPUT = ROOT / "results" / "directed_graph" / "graph_audit.json"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    rows = []
    for repo_dir in sorted(DATA.glob("repoeval__*")):
        repo_key = repo_dir.name.removeprefix("repoeval__")
        source = SOURCES / repo_key
        corpus = load_jsonl(repo_dir / "corpus.jsonl")
        retriever = Retriever(corpus, repo_root=source, repo_key=repo_key)
        graph = retriever.directed_graph
        assert graph is not None
        row = {
            "repo": repo_key,
            "source_exists": source.is_dir(),
            "corpus_files": len({doc.get("title", "") for doc in corpus}),
            "mapped_python_files": len(graph.path_to_title),
            "import_statements": graph.import_statements,
            "resolved_local_import_statements": graph.resolved_import_statements,
            "resolution_rate": graph.resolved_import_statements / graph.import_statements if graph.import_statements else 0.0,
            "directed_file_edges": graph.edge_count,
            "sample_edges": [
                [source_title, target_title]
                for source_title in sorted(graph.adjacency)
                for target_title in sorted(graph.adjacency[source_title])
            ][:10],
        }
        rows.append(row)
        print(repo_key, row["directed_file_edges"], f"resolution={row['resolution_rate']:.3f}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the frozen, strict-budget RepoEval experiment and generate paper tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from gcc import RetrievalConfig, Retriever, f1, mrr, ndcg, precision, recall


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "scripts" / "code-rag-bench" / "retrieval" / "datasets"
DEFAULT_RESULTS = ROOT / "results" / "final"
SOURCE_REPOS = ROOT / "scripts" / "code-rag-bench" / "retrieval" / "output" / "repoeval" / "repositories" / "function_level"
BUDGETS = (500, 1000, 2000, 4000, 8000, 16000)
RANDOM_SEEDS = tuple(range(30))


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


def record(repo: str, strategy: str, budget: int, query_id: str,
           selected_indices: list[int], retriever: Retriever, relevant: set[str],
           seed: int | None = None) -> dict:
    selected = [retriever.corpus[index]["_id"] for index in selected_indices]
    tokens = sum(retriever.tokens[index] for index in selected_indices)
    if tokens > budget:
        raise AssertionError(f"strict budget violated: {tokens} > {budget}")
    return {
        "repo": repo,
        "query_id": query_id,
        "strategy": strategy,
        "budget": budget,
        "seed": seed,
        "recall": recall(selected, relevant),
        "precision": precision(selected, relevant),
        "f1": f1(selected, relevant),
        "ndcg@10": ndcg(selected, relevant),
        "mrr": mrr(selected, relevant),
        "tokens": tokens,
        "n_docs": len(selected),
    }


def mean(rows: list[dict], key: str) -> float:
    return sum(row[key] for row in rows) / len(rows)


def summarize(query_rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in query_rows:
        groups[(row["repo"], row["strategy"], row["budget"], row["seed"])].append(row)
    summaries = []
    for (repo, strategy, budget, seed), rows in sorted(groups.items(), key=str):
        summaries.append({
            "repo": repo, "strategy": strategy, "budget": budget, "seed": seed,
            "n_queries": len(rows),
            **{key: mean(rows, key) for key in ("recall", "precision", "f1", "ndcg@10", "mrr")},
            "avg_tokens": mean(rows, "tokens"),
            "max_tokens": max(row["tokens"] for row in rows),
            "avg_docs": mean(rows, "n_docs"),
        })
    return summaries


def aggregate_random_seed_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["repo"], row["strategy"], row["budget"])].append(row)
    result = []
    for (repo, strategy, budget), members in sorted(groups.items()):
        item = {"repo": repo, "strategy": strategy, "budget": budget, "seeds": len(members)}
        for key in ("recall", "precision", "f1", "ndcg@10", "mrr", "avg_tokens", "avg_docs"):
            values = [member[key] for member in members]
            item[key] = statistics.mean(values)
            item[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        item["max_tokens"] = max(member["max_tokens"] for member in members)
        result.append(item)
    return result


def paired_statistics(query_rows: list[dict], budget: int = 4000) -> dict:
    pairs: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in query_rows:
        if row["budget"] == budget and row["strategy"] in {"bm25", "gcc_coimport"}:
            pairs[(row["repo"], row["query_id"])][row["strategy"]] = row["recall"]
    by_repo: dict[str, list[float]] = defaultdict(list)
    for (repo, _), values in pairs.items():
        if len(values) == 2:
            by_repo[repo].append(values["gcc_coimport"] - values["bm25"])
    differences = [value for values in by_repo.values() for value in values]
    n = len(differences)
    avg = statistics.mean(differences)
    sd = statistics.stdev(differences) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    ci = [avg - 1.96 * se, avg + 1.96 * se]
    dz = avg / sd if sd else 0.0
    rng = random.Random(20260913)
    repos = sorted(by_repo)
    bootstrap = []
    for _ in range(10_000):
        sampled_repo_means = []
        for repo in rng.choices(repos, k=len(repos)):
            values = by_repo[repo]
            sampled_repo_means.append(statistics.mean(rng.choices(values, k=len(values))))
        bootstrap.append(statistics.mean(sampled_repo_means))
    bootstrap.sort()
    cluster_ci_95 = [bootstrap[249], bootstrap[9749]]
    cluster_ci_90 = [bootstrap[499], bootstrap[9499]]
    equivalence_margin = 0.01
    return {
        "comparison": "gcc_coimport - bm25", "budget": budget, "n_queries": n,
        "n_repository_clusters": len(repos),
        "query_weighted_mean_difference": avg, "sd_difference": sd, "cohens_dz": dz,
        "descriptive_95pct_ci": ci,
        "cluster_bootstrap_macro_mean_difference": statistics.mean(
            statistics.mean(values) for values in by_repo.values()
        ),
        "cluster_bootstrap_95pct_ci": cluster_ci_95,
        "cluster_bootstrap_90pct_ci": cluster_ci_90,
        "exploratory_equivalence_margin": [-equivalence_margin, equivalence_margin],
        "exploratory_equivalence_supported": (
            cluster_ci_90[0] > -equivalence_margin and cluster_ci_90[1] < equivalence_margin
        ),
        "interpretation": (
            "The two-stage bootstrap resamples repositories and then queries. "
            "With only five repository clusters, inference and the post-hoc +/-0.01 "
            "equivalence margin are exploratory, not confirmatory."
        ),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def macro(rows: list[dict], strategy: str, budget: int, key: str) -> float:
    values = [row[key] for row in rows if row["strategy"] == strategy and row["budget"] == budget]
    return statistics.mean(values)


def render_tables(rows: list[dict], random_rows: list[dict], output: Path) -> None:
    lines = ["# Auto-generated final tables", "", "> Generated only from `results/final/final_results.jsonl`.", ""]
    lines += ["## Main results at B=4,000", "", "| Strategy | Recall | Precision | F1 | NDCG@10 | MRR | Avg tokens | Max tokens |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    labels = {"bm25": "BM25", "gcc_coimport": "BM25 + co-import", "bm25_directed": "BM25 + directed imports", "sequential": "Sequential"}
    for strategy in ("bm25", "gcc_coimport", "bm25_directed", "sequential"):
        subset = [row for row in rows if row["strategy"] == strategy and row["budget"] == 4000]
        lines.append(f"| {labels[strategy]} | {statistics.mean(r['recall'] for r in subset):.3f} | {statistics.mean(r['precision'] for r in subset):.3f} | {statistics.mean(r['f1'] for r in subset):.3f} | {statistics.mean(r['ndcg@10'] for r in subset):.3f} | {statistics.mean(r['mrr'] for r in subset):.3f} | {statistics.mean(r['avg_tokens'] for r in subset):.0f} | {max(r['max_tokens'] for r in subset):.0f} |")
    subset = [row for row in random_rows if row["strategy"] == "random" and row["budget"] == 4000]
    lines.append(f"| Random (30 seeds) | {statistics.mean(r['recall'] for r in subset):.3f} ± {statistics.mean(r['recall_std'] for r in subset):.3f} | {statistics.mean(r['precision'] for r in subset):.3f} | {statistics.mean(r['f1'] for r in subset):.3f} | {statistics.mean(r['ndcg@10'] for r in subset):.3f} | {statistics.mean(r['mrr'] for r in subset):.3f} | {statistics.mean(r['avg_tokens'] for r in subset):.0f} | {max(r['max_tokens'] for r in subset):.0f} |")
    lines += ["", "## Budget sensitivity (macro-average recall)", "", "| Budget | BM25 | BM25 + co-import | BM25 + directed imports | Random (30 seeds) |", "|---:|---:|---:|---:|---:|"]
    for budget in BUDGETS:
        lines.append(f"| {budget:,} | {macro(rows, 'bm25', budget, 'recall'):.3f} | {macro(rows, 'gcc_coimport', budget, 'recall'):.3f} | {macro(rows, 'bm25_directed', budget, 'recall'):.3f} | {macro(random_rows, 'random', budget, 'recall'):.3f} |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    query_rows: list[dict] = []
    dataset_stats: list[dict] = []
    config = RetrievalConfig()
    repo_dirs = sorted(path for path in data_dir.glob("repoeval__*") if path.is_dir())
    for repo_dir in repo_dirs:
        corpus = read_jsonl(repo_dir / "corpus.jsonl")
        queries = read_jsonl(repo_dir / "queries.jsonl")
        qrels = load_qrels(repo_dir / "qrels" / "test.tsv")
        repo = repo_dir.name
        repo_key = repo.removeprefix("repoeval__")
        retriever = Retriever(corpus, repo_root=SOURCE_REPOS / repo_key, repo_key=repo_key)
        dataset_stats.append({
            "repo": repo, "documents": len(corpus), "queries": len(qrels),
            "relevant_pairs": sum(len(values) for values in qrels.values()),
            "corpus_tokens": sum(retriever.tokens), "coimport_edges": retriever.graph.edge_count,
        })
        print(f"{repo}: {len(corpus)} docs, {len(qrels)} queries")
        for query in queries:
            query_id = query["_id"]
            if query_id not in qrels:
                continue
            relevant = qrels[query_id]
            bm25_ranking = retriever.bm25_ranking(query.get("text", ""))
            sequential_ranking = list(range(len(corpus)))
            for budget in BUDGETS:
                gcc_ranking = retriever.gcc_ranking(query.get("text", ""), budget, config)
                directed_ranking = retriever.directed_ranking(query.get("text", ""), budget)
                for strategy, ranking in (("bm25", bm25_ranking), ("gcc_coimport", gcc_ranking), ("bm25_directed", directed_ranking), ("sequential", sequential_ranking)):
                    selected = retriever.select(ranking, budget)
                    query_rows.append(record(repo, strategy, budget, query_id, selected, retriever, relevant))
                for seed in RANDOM_SEEDS:
                    selected = retriever.select(retriever.random_ranking(query_id, seed), budget)
                    query_rows.append(record(repo, "random", budget, query_id, selected, retriever, relevant, seed))

    summaries = summarize(query_rows)
    deterministic = [row for row in summaries if row["strategy"] != "random"]
    random_summary = aggregate_random_seed_rows([row for row in summaries if row["strategy"] == "random"])
    write_jsonl(output_dir / "final_query_results.jsonl", query_rows)
    write_jsonl(output_dir / "final_results.jsonl", deterministic)
    write_jsonl(output_dir / "random_30seed_results.jsonl", random_summary)
    (output_dir / "dataset_stats.json").write_text(json.dumps(dataset_stats, indent=2) + "\n")
    (output_dir / "statistics.json").write_text(json.dumps(paired_statistics(query_rows), indent=2) + "\n")
    render_tables(deterministic, random_summary, output_dir / "paper_tables.md")
    print(f"wrote frozen results to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)

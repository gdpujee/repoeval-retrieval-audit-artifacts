#!/usr/bin/env python3
"""Paired, repository-clustered analysis for final deterministic strategies."""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "final" / "final_query_results.jsonl"
DENSE_RAW = ROOT / "results" / "dense" / "codesage_small_v2_query_results.jsonl"
OUTPUT = ROOT / "results" / "final" / "statistics_extended.json"


def compare(rows: list[dict], comparator: str, budget: int = 4000) -> dict:
    pairs: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["budget"] == budget and row["strategy"] in {"bm25", comparator}:
            pairs[(row["repo"], row["query_id"])][row["strategy"]] = row["recall"]
    by_repo: dict[str, list[float]] = defaultdict(list)
    for (repo, _), values in pairs.items():
        if "bm25" in values and comparator in values:
            by_repo[repo].append(values[comparator] - values["bm25"])
    rng = random.Random(20260913)
    repos = sorted(by_repo)
    samples = []
    for _ in range(10_000):
        sampled_repo_means = []
        for repo in rng.choices(repos, k=len(repos)):
            diffs = by_repo[repo]
            sampled_repo_means.append(statistics.mean(rng.choices(diffs, k=len(diffs))))
        samples.append(statistics.mean(sampled_repo_means))
    samples.sort()
    all_diffs = [value for values in by_repo.values() for value in values]
    sd = statistics.stdev(all_diffs)
    return {
        "comparison": f"{comparator} - bm25", "budget": budget,
        "n_queries": len(all_diffs), "n_repository_clusters": len(repos),
        "macro_mean_difference": statistics.mean(statistics.mean(v) for v in by_repo.values()),
        "cluster_bootstrap_95pct_ci": [samples[249], samples[9749]],
        "cluster_bootstrap_90pct_ci": [samples[499], samples[9499]],
        "cohens_dz_query_level_descriptive": statistics.mean(all_diffs) / sd if sd else 0.0,
        "per_repository_mean_differences": {repo: statistics.mean(values) for repo, values in by_repo.items()},
        "caution": "Exploratory inference with only five repository clusters.",
    }


def main() -> None:
    with RAW.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if DENSE_RAW.exists():
        with DENSE_RAW.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    comparisons = [compare(rows, "gcc_coimport"), compare(rows, "bm25_directed")]
    if DENSE_RAW.exists():
        comparisons.append(compare(rows, "codesage_small_v2"))
    result = {
        "method": "10,000-iteration two-stage repository/query bootstrap; seed 20260913",
        "comparisons": comparisons,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()

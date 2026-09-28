#!/usr/bin/env python3
"""Independent checks over the frozen query-level result artifact."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "final" / "final_query_results.jsonl"
OUTPUT = ROOT / "results" / "final" / "verification.json"


def main() -> None:
    with RAW.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    violations = [row for row in rows if row["tokens"] > row["budget"]]
    by_repo_strategy: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["budget"] == 4000 and row["strategy"] in {"bm25", "gcc_coimport"}:
            by_repo_strategy[(row["repo"], row["strategy"])].append(row["recall"])
    macro = {}
    for strategy in ("bm25", "gcc_coimport"):
        repo_means = [
            statistics.mean(values)
            for (repo, name), values in by_repo_strategy.items()
            if name == strategy
        ]
        macro[strategy] = statistics.mean(repo_means)
    report = {
        "query_rows": len(rows),
        "budget_violations": len(violations),
        "max_tokens_minus_budget": max(row["tokens"] - row["budget"] for row in rows),
        "independent_macro_recall_at_4000": macro,
        "passed": len(rows) == 69564 and not violations,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("verification failed")
    print(OUTPUT)


if __name__ == "__main__":
    main()

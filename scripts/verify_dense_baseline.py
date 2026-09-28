#!/usr/bin/env python3
"""Independent integrity checks for CodeSage-small-v2 results."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "dense" / "codesage_small_v2_query_results.jsonl"
ENV = ROOT / "results" / "dense" / "environment.json"
OUTPUT = ROOT / "results" / "dense" / "verification.json"


def main() -> None:
    with RAW.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    environment = json.loads(ENV.read_text(encoding="utf-8"))
    task_budget_pairs = {(row["query_id"], row["budget"]) for row in rows}
    budgets = sorted({row["budget"] for row in rows})
    repositories = sorted({row["repo"] for row in rows})
    report = {
        "rows": len(rows),
        "unique_task_budget_pairs": len(task_budget_pairs),
        "budget_violations": sum(row["tokens"] > row["budget"] for row in rows),
        "model": environment.get("model"),
        "model_revision": environment.get("model_revision"),
        "embedding_dimension": environment.get("embedding_dimension"),
        "budgets": budgets,
        "repositories": repositories,
        "passed": (
            len(rows) == 2046 and len(task_budget_pairs) == 2046
            and not any(row["tokens"] > row["budget"] for row in rows)
            and environment.get("model") == "codesage/codesage-small-v2"
            and environment.get("model_revision") == "4844c2f24b25e181aa43ca058cc73dd2622565c1"
            and budgets == [500, 1000, 2000, 4000, 8000, 16000]
            and len(repositories) == 5
        ),
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("dense baseline verification failed")
    print(OUTPUT)


if __name__ == "__main__":
    main()

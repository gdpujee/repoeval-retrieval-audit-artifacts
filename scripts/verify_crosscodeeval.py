#!/usr/bin/env python3
"""Independent structural checks for the CrossCodeEval context audit."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "crosscodeeval" / "context_support_query_results.jsonl"
OUTPUT = ROOT / "results" / "crosscodeeval" / "verification.json"


def main() -> None:
    with RAW.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    task_sets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        task_sets[row["method"]].add(row["task_id"])
    unique_sets = {frozenset(tasks) for tasks in task_sets.values()}
    report = {
        "rows": len(rows),
        "methods": sorted(task_sets),
        "tasks_per_method": {method: len(tasks) for method, tasks in task_sets.items()},
        "aligned_task_sets": len(unique_sets) == 1,
        "invalid_identifier_recall_rows": sum(
            row["identifier_recall"] is not None and not 0 <= row["identifier_recall"] <= 1
            for row in rows
        ),
        "passed": len(rows) == 15990 and len(unique_sets) == 1,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("CrossCodeEval verification failed")
    print(OUTPUT)


if __name__ == "__main__":
    main()

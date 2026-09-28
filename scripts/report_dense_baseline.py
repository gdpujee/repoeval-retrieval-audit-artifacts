#!/usr/bin/env python3
"""Generate manuscript-ready CodeSage tables from verified dense results."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "dense" / "codesage_small_v2_results.jsonl"
OUTPUT = ROOT / "results" / "dense" / "paper_tables.md"


def main() -> None:
    rows = [json.loads(line) for line in INPUT.read_text(encoding="utf-8").splitlines()]
    lines = [
        "# Auto-generated CodeSage dense-baseline tables",
        "",
        "> Repository-macro averages generated only from the verified result file.",
        "",
        "| Budget | Recall | Precision | F1 | NDCG@10 | MRR | Avg tokens | Max tokens |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in sorted({row["budget"] for row in rows}):
        group = [row for row in rows if row["budget"] == budget]
        mean = lambda key: statistics.mean(row[key] for row in group)
        lines.append(
            f"| {budget:,} | {mean('recall'):.3f} | {mean('precision'):.3f} | "
            f"{mean('f1'):.3f} | {mean('ndcg@10'):.3f} | {mean('mrr'):.3f} | "
            f"{mean('avg_tokens'):.0f} | {max(row['max_tokens'] for row in group)} |"
        )
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()

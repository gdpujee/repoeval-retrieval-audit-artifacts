#!/usr/bin/env python3
"""Audit identifier support in official CrossCodeEval retrieved contexts.

This does not rerun the original retrievers because the public artifact does
not contain raw repositories. It evaluates the pre-generated contexts shipped
at official cceval commit 40c68d2.
"""

from __future__ import annotations

import json
import keyword
import re
import statistics
from collections import defaultdict
from pathlib import Path

import tiktoken


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "external" / "cceval" / "data" / "crosscodeeval_data" / "python"
OUTPUT = ROOT / "results" / "crosscodeeval"
FILES = {
    "bm25": "line_completion_rg1_bm25.jsonl",
    "unixcoder": "line_completion_rg1_unixcoder_cosine_sim.jsonl",
    "openai_embedding": "line_completion_rg1_openai_cosine_sim.jsonl",
    "oracle_bm25": "line_completion_oracle_bm25.jsonl",
    "oracle_unixcoder": "line_completion_oracle_unixcoder_cosine_sim.jsonl",
    "oracle_openai_embedding": "line_completion_oracle_openai_cosine_sim.jsonl",
}
ENCODER = tiktoken.get_encoding("cl100k_base")


def identifiers(text: str) -> set[str]:
    return {
        token for token in re.findall(r"\b[A-Za-z_]\w*\b", text)
        if not keyword.iskeyword(token) and token not in {"True", "False", "None"}
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    by_method = {}
    task_set = None
    for method, filename in FILES.items():
        with (DATA / filename).open(encoding="utf-8") as handle:
            examples = [json.loads(line) for line in handle if line.strip()]
        ids = [example["metadata"]["task_id"] for example in examples]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"duplicate task ids for {method}")
        if task_set is None:
            task_set = set(ids)
        elif set(ids) != task_set:
            raise AssertionError(f"task sets differ for {method}")
        by_method[method] = examples

    rows = []
    for method, examples in by_method.items():
        for example in examples:
            groundtruth = example["groundtruth"]
            context = example["crossfile_context"]["text"]
            gt_ids = identifiers(groundtruth)
            ctx_ids = identifiers(context)
            rows.append({
                "method": method,
                "task_id": example["metadata"]["task_id"],
                "repository": example["metadata"]["repository"],
                "context_tokens": len(ENCODER.encode(context, disallowed_special=())),
                "groundtruth_identifier_count": len(gt_ids),
                "identifier_recall": len(gt_ids & ctx_ids) / len(gt_ids) if gt_ids else None,
                "all_identifiers_supported": gt_ids <= ctx_ids if gt_ids else None,
                "exact_groundtruth_in_context": groundtruth.strip() in context,
            })
    with (OUTPUT / "context_support_query_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = []
    for method in FILES:
        members = [row for row in rows if row["method"] == method]
        id_members = [row for row in members if row["identifier_recall"] is not None]
        repository_values: dict[str, list[float]] = defaultdict(list)
        for row in id_members:
            repository_values[row["repository"]].append(row["identifier_recall"])
        summary.append({
            "method": method,
            "tasks": len(members),
            "repositories": len({row["repository"] for row in members}),
            "mean_context_tokens": statistics.mean(row["context_tokens"] for row in members),
            "query_weighted_identifier_recall": statistics.mean(row["identifier_recall"] for row in id_members),
            "repository_macro_identifier_recall": statistics.mean(
                statistics.mean(values) for values in repository_values.values()
            ),
            "all_identifiers_supported_rate": statistics.mean(
                int(row["all_identifiers_supported"]) for row in id_members
            ),
            "exact_groundtruth_in_context_rate": statistics.mean(
                int(row["exact_groundtruth_in_context"]) for row in members
            ),
        })
    (OUTPUT / "context_support_summary.json").write_text(json.dumps({
        "source": "amazon-science/cceval commit 40c68d2",
        "scope": "Official pre-generated Python contexts; retrievers were not rerun",
        "metric_note": "Identifier support is a retrieval-context diagnostic, not generation correctness",
        "results": summary,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT / "context_support_summary.json")


if __name__ == "__main__":
    main()

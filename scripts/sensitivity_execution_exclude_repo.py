#!/usr/bin/env python3
"""Recompute execution rates after excluding the partially evaluable repo.

This sensitivity analysis reuses only the saved V8 execution worker rows. It
does not rerun generation or tests. The repository/query bootstrap and macro
calculation are shared with score_execution.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

from score_execution import paired_bootstrap, repo_macro


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INDIR = os.path.join(ROOT, "results", "execution", "v8")
DEFAULT_OUT = os.path.join(
    DEFAULT_INDIR, "sensitivity_excluding_lightweight_mmm.json")
MODELS = {"deepseek-v4.1-flash", "muse-spark-1.3-contributor"}
STRATEGIES = ("bm25", "coimport", "directed")
ALL_REPOS = {
    "deepmind_tracr",
    "facebookresearch_omnivore",
    "google_lightweight_mmm",
    "lucidrains_imagen-pytorch",
    "maxhumber_redframes",
}


def load_rows(indir: str) -> dict[tuple[str, str, str], dict]:
    seen: dict[tuple[str, str, str], dict] = {}
    conflicts = []
    for filename in glob.glob(os.path.join(indir, "eval_*.jsonl")):
        with open(filename) as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                key = (row["task_id"], row["strategy"], row["model"])
                if key not in seen:
                    seen[key] = row
                elif (seen[key].get("success") != row.get("success")
                      or seen[key].get("files") != row.get("files")):
                    conflicts.append((key, filename))
    if conflicts:
        raise SystemExit(f"Conflicting execution rows; example: {conflicts[0]}")
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default=DEFAULT_INDIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--exclude-repo", default="google_lightweight_mmm")
    args = parser.parse_args()

    seen = load_rows(args.indir)
    if {key[2] for key in seen} != MODELS:
        raise SystemExit("Expected execution rows for both reported generators")
    task_sets = {}
    for model in sorted(MODELS):
        for strategy in STRATEGIES:
            ids = {task_id for task_id, strat, mod in seen
                   if strat == strategy and mod == model}
            if len(ids) != 341:
                raise SystemExit(
                    f"Incomplete source matrix for {model}/{strategy}: {len(ids)}")
            task_sets[model, strategy] = ids
        if not (task_sets[model, "bm25"] == task_sets[model, "coimport"]
                == task_sets[model, "directed"]):
            raise SystemExit(f"Strategy task sets differ for {model}")
    if task_sets["deepseek-v4.1-flash", "bm25"] != task_sets[
            "muse-spark-1.3-contributor", "bm25"]:
        raise SystemExit("Generator task sets differ")

    excluded_repo = "repoeval__" + args.exclude_repo
    rows_by_model = defaultdict(list)
    for row in seen.values():
        if not row.get("files"):
            continue
        repo = "repoeval__" + row["task_id"].split("/")[0]
        if repo != excluded_repo:
            rows_by_model[row["model"]].append({
                "task_id": row["task_id"],
                "repo": repo,
                "strategy": row["strategy"],
                "model": row["model"],
                "success": bool(row["success"]),
            })

    repos = {row["repo"] for rows in rows_by_model.values() for row in rows}
    expected_repos = {"repoeval__" + repo for repo in ALL_REPOS
                      if repo != args.exclude_repo}
    if repos != expected_repos:
        raise SystemExit(f"Unexpected retained repositories: {sorted(repos)}")

    result = {
        "analysis": "execution sensitivity excluding the partially evaluable repository",
        "excluded_repo": excluded_repo,
        "retained_repositories": sorted(repos),
        "bootstrap": {
            "method": "10,000-iteration paired two-stage repository/query bootstrap",
            "seed": 20260913,
            "interval": "percentile 95%",
            "note": "Exploratory with four repository clusters; uses saved execution rows only.",
        },
        "models": {},
    }

    for model, rows in sorted(rows_by_model.items()):
        tasks = {row["task_id"] for row in rows}
        if len(tasks) != 277:
            raise SystemExit(f"Expected 277 retained tasks for {model}, got {len(tasks)}")
        by_strategy = {
            strategy: [row for row in rows if row["strategy"] == strategy]
            for strategy in STRATEGIES
        }
        strategy_tasks = {
            strategy: {row["task_id"] for row in strategy_rows}
            for strategy, strategy_rows in by_strategy.items()
        }
        if not (strategy_tasks["bm25"] == strategy_tasks["coimport"]
                == strategy_tasks["directed"] == tasks):
            raise SystemExit(f"Retained task sets differ for {model}")

        model_result = {"evaluable_tasks": len(tasks), "strategies": {}}
        for strategy, strategy_rows in by_strategy.items():
            rate, per_repo = repo_macro(strategy_rows)
            model_result["strategies"][strategy] = {
                "pass_rate": rate,
                "per_repo": per_repo,
            }
        for strategy, label in (("coimport", "coimport-bm25"),
                                ("directed", "directed-bm25")):
            a = by_strategy[strategy]
            b = by_strategy["bm25"]
            rate_a, _ = repo_macro(a)
            rate_b, _ = repo_macro(b)
            lo, hi = paired_bootstrap(a, b)
            model_result[label] = {
                "macro_diff": rate_a - rate_b,
                "ci95": [lo, hi],
            }
        result["models"][model] = model_result

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

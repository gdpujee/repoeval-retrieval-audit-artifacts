#!/usr/bin/env python3
"""Score execution evaluation: merge worker files, compute repo-macro
test-pass rates per strategy/model with two-stage bootstrap CIs for paired
differences, and report discriminative coverage.

Success (per task) = every discriminative mapped test file passes after
patching the model's completion (see run_execution_eval.py for the
SKIP_FILES/DESELECT rules and their rationale).

Writes results/execution/summary.json.
"""

import glob
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDIR = os.path.join(ROOT, "results", "execution")
OUT = os.path.join(INDIR, "summary.json")


def repo_macro(rows):
    by_repo = {}
    for r in rows:
        by_repo.setdefault(r["repo"], []).append(r["success"])
    means = {k: sum(v) / len(v) for k, v in by_repo.items()}
    return sum(means.values()) / len(means), means


def paired_bootstrap(rows_a, rows_b, iters=10000, seed=20260913):
    ra = {r["task_id"]: r for r in rows_a}
    rb = {r["task_id"]: r for r in rows_b}
    tids = sorted(set(ra) & set(rb))
    by_repo = {}
    for t in tids:
        by_repo.setdefault(ra[t]["repo"], []).append(t)
    repos = sorted(by_repo)
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        tot = 0.0
        for _ in repos:
            rp = rng.choice(repos)
            qs = [rng.choice(by_repo[rp]) for _ in by_repo[rp]]
            tot += sum((1.0 if ra[q]["success"] else 0.0) -
                       (1.0 if rb[q]["success"] else 0.0)
                       for q in qs) / len(qs)
        diffs.append(tot / len(repos))
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=None,
                    help="execution worker-file dir (default results/execution)")
    ap.add_argument("--out", default=None,
                    help="summary output path (default results/execution/summary.json)")
    a = ap.parse_args()
    indir = a.indir or INDIR
    out = a.out or OUT
    seen = {}
    conflicts = []
    for fn in glob.glob(os.path.join(indir, "eval_*.jsonl")):
        for line in open(fn):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            k = (d["task_id"], d["strategy"], d["model"])
            if k not in seen:
                seen[k] = d
            elif (seen[k].get("success") != d.get("success")
                    or seen[k].get("files") != d.get("files")):
                # Same key, different outcome: provenance must be adjudicated
                # explicitly (R-12). Fail loudly instead of first-wins.
                conflicts.append((k, fn))
    if conflicts:
        raise SystemExit(
            f"CONFLICT: {len(conflicts)} same-key execution rows differ, "
            f"e.g. {conflicts[0][0]} in {conflicts[0][1]}. Resolve the "
            f"retry provenance before scoring.")
    expected_models = {"deepseek-v4.1-flash", "muse-spark-1.3-contributor"}
    strategies = ("bm25", "coimport", "directed")
    if {k[2] for k in seen} != expected_models:
        raise SystemExit("Incomplete execution models; expected both generators")
    model_tasks = {}
    for model in sorted(expected_models):
        task_sets = [
            {task_id for task_id, strategy, m in seen
             if strategy == s and m == model}
            for s in strategies
        ]
        if any(len(ids) != 341 for ids in task_sets) or not all(
                ids == task_sets[0] for ids in task_sets[1:]):
            raise SystemExit(
                f"Incomplete execution matrix for {model}: "
                f"{[len(ids) for ids in task_sets]} tasks by strategy")
        model_tasks[model] = task_sets[0]
    if len({frozenset(ids) for ids in model_tasks.values()}) != 1:
        raise SystemExit("Execution task sets differ between generators")
    rows = []
    for d in seen.values():
        if not d.get("files"):
            # No discriminative mapped file in this env (all skipped for
            # reference drift): unevaluable, excluded from rates AND diffs
            # (structural zeros would attenuate paired differences).
            continue
        rows.append({"task_id": d["task_id"],
                     "repo": "repoeval__" + d["task_id"].split("/")[0],
                     "strategy": d["strategy"], "model": d["model"],
                     "success": bool(d["success"]),
                     "n_files": len(d.get("files", {}))})
    # Completeness gate: every model needs 1023 unique keys.
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    summary = {"models": {}, "coverage": {}}
    for model, mr in sorted(by_model.items()):
        tasks = sorted({r["task_id"] for r in mr})
        summary["coverage"][model] = {
            "evaluable_tasks": len(tasks), "total_tasks": 341}
        ms = summary["models"][model] = {"n": len(mr), "strategies": {}}
        gs = {}
        for r in mr:
            gs.setdefault(r["strategy"], []).append(r)
        for s in ["bm25", "coimport", "directed"]:
            rate, means = repo_macro(gs[s])
            ms["strategies"][s] = {
                "pass_rate": rate, "n": len(gs[s]),
                "per_repo": means}
        for a, b, name in [("coimport", "bm25", "coimport-bm25"),
                           ("directed", "bm25", "directed-bm25")]:
            ea, _ = repo_macro(gs[a])
            eb, _ = repo_macro(gs[b])
            lo, hi = paired_bootstrap(gs[a], gs[b])
            ms[name] = {"macro_diff": ea - eb,
                        "ci95": [lo, hi]}
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

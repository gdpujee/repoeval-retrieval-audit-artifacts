#!/usr/bin/env python3
"""Leave-one-task-out sensitivity for the downstream paired EM differences.

Motivation (R-18-V8): the leakage audit finds one task whose target span
survives masking in whitespace-normalised form (``google_lightweight_mmm/22``,
Appendix E). If that single task drove a reported difference, the difference
would be an artifact of the residual overlap rather than of retrieval. This
script recomputes every reported paired EM difference with that task removed,
using exactly the same two-stage repo/query bootstrap as
``scripts/score_downstream.py`` (10,000 iterations, seed 20260913).

Reads ``results/downstream/<version>/scores_*.jsonl`` (``--kind downstream``,
default) or ``results/execution/<version>/eval_*.jsonl``
(``--kind execution``) and writes
``<indir>/sensitivity_<slug>.json``.

Usage:
  python3 scripts/sensitivity_exclude_task.py \
      --indir results/downstream/v8 --exclude google_lightweight_mmm/22
  python3 scripts/sensitivity_exclude_task.py --kind execution \
      --indir results/execution/v8 --exclude google_lightweight_mmm/22
"""

import argparse
import glob
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPARISONS = [("coimport", "bm25", "coimport-bm25"),
               ("directed", "bm25", "directed-bm25")]


def load_execution(indir):
    """Aggregate eval_*.jsonl into one binary row per (task, strategy, model).

    Mirrors scripts/score_execution.py: a task is evaluable only if at least
    one reference-passing mapped file exists, and passes only if every such
    file passes. Rows with no mapped files are excluded from rates.
    """
    seen = {}
    for path in sorted(glob.glob(os.path.join(indir, "eval_*.jsonl"))):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                seen[(d["task_id"], d["strategy"], d["model"])] = d
    rows = []
    for (task_id, strategy, model), d in seen.items():
        if not d.get("files"):
            continue
        rows.append({"task_id": task_id,
                     "repo": "repoeval__" + task_id.split("/")[0],
                     "strategy": strategy, "model": model,
                     "em": bool(d["success"])})
    return rows


def repo_macro(rows, metric="em"):
    by_repo = {}
    for r in rows:
        by_repo.setdefault(r["repo"], []).append(r[metric])
    means = [sum(v) / len(v) for v in by_repo.values()]
    return sum(means) / len(means)


def paired_bootstrap(rows_a, rows_b, iters=10000, seed=20260913):
    """Two-stage repo/query bootstrap; identical to score_downstream.py."""
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
            tot += sum((1.0 if ra[q]["em"] else 0.0) -
                       (1.0 if rb[q]["em"] else 0.0) for q in qs) / len(qs)
        diffs.append(tot / len(repos))
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


def compare(rows, exclude):
    kept = [r for r in rows if r["task_id"] != exclude]
    out = {}
    for a, b, name in COMPARISONS:
        ra = [r for r in kept if r["strategy"] == a]
        rb = [r for r in kept if r["strategy"] == b]
        diff = repo_macro(ra) - repo_macro(rb)
        lo, hi = paired_bootstrap(ra, rb)
        out[name] = {
            "macro_em_diff": round(diff, 5),
            "ci95": [round(lo, 5), round(hi, 5)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_tasks": len({r["task_id"] for r in ra}),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=os.path.join(
        ROOT, "results", "downstream", "v9"))
    ap.add_argument("--exclude", default="google_lightweight_mmm/22")
    ap.add_argument("--kind", choices=["downstream", "execution"],
                    default="downstream")
    a = ap.parse_args()

    if a.kind == "execution":
        rows = load_execution(a.indir)
        if not rows:
            raise SystemExit(f"no eval_*.jsonl rows under {a.indir}")
    else:
        files = sorted(glob.glob(os.path.join(a.indir, "scores_*.jsonl")))
        if not files:
            raise SystemExit(f"no scores_*.jsonl under {a.indir}")
        rows = []
        for path in files:
            with open(path) as f:
                rows.extend(json.loads(line) for line in f if line.strip())

    report = {
        "excluded_task": a.exclude,
        "kind": a.kind,
        "indir": os.path.relpath(a.indir, ROOT),
        "models": {},
    }
    for model in sorted({r["model"] for r in rows}):
        mr = [r for r in rows if r["model"] == model]
        present = any(r["task_id"] == a.exclude for r in mr)
        report["models"][model] = {
            "task_present": present,
            "with_task": compare(mr, None),
            "without_task": compare(mr, a.exclude) if present else None,
        }
    slug = a.exclude.replace("/", "_").replace(".", "")
    outpath = os.path.join(a.indir, f"sensitivity_{slug}.json")
    with open(outpath, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("wrote", outpath)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Downstream failure analysis: where does directed-import context change
generation outcomes vs BM25, and what do those cases look like?

Reads results/downstream/scores_<model>.jsonl (written by score_downstream.py).
Writes results/downstream/failure_analysis.json:
  - per-repo EM/ES by strategy;
  - discordant pairs (BM25 EM=1 & directed EM=0 and vice versa) with refs;
  - low-ES directed cases where BM25 succeeds (budget-competition evidence).
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IN = os.path.join(ROOT, "results", "downstream")
DEFAULT_OUT = os.path.join(DEFAULT_IN, "failure_analysis.json")


def load(model, indir):
    rows = []
    with open(os.path.join(indir, f"scores_{model}.jsonl")) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main(models, indir=None, outpath=None):
    indir = indir or DEFAULT_IN
    outpath = outpath or DEFAULT_OUT
    out = {}
    for model in models:
        rows = load(model, indir)
        by_key = {(r["task_id"], r["strategy"]): r for r in rows}
        tids = sorted({r["task_id"] for r in rows})
        per_repo = {}
        for r in rows:
            per_repo.setdefault(r["repo"], {}).setdefault(
                r["strategy"], []).append(r["em"])
        per_repo_em = {repo: {s: sum(v) / len(v) for s, v in sm.items()}
                       for repo, sm in per_repo.items()}
        b_wins, d_wins, both_ok, both_bad = [], [], 0, 0
        for t in tids:
            b = by_key[(t, "bm25")]["em"]
            d = by_key[(t, "directed")]["em"]
            if b and not d:
                b_wins.append(t)
            elif d and not b:
                d_wins.append(t)
            elif b and d:
                both_ok += 1
            else:
                both_bad += 1
        # Sample discordant cases with short refs for the manuscript.
        samples = []
        for t in (b_wins[:5] + d_wins[:5]):
            samples.append({
                "task_id": t,
                "bm25_em": by_key[(t, "bm25")]["em"],
                "directed_em": by_key[(t, "directed")]["em"],
                "bm25_es": round(by_key[(t, "bm25")]["es"], 3),
                "directed_es": round(by_key[(t, "directed")]["es"], 3),
            })
        out[model] = {
            "n": len(tids),
            "per_repo_em": {k: {s: round(v, 4) for s, v in sm.items()}
                            for k, sm in per_repo_em.items()},
            "bm25_ok_directed_fail": len(b_wins),
            "directed_ok_bm25_fail": len(d_wins),
            "both_ok": both_ok,
            "both_fail": both_bad,
            "discordant_samples": samples,
        }
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    for model, m in out.items():
        print(f"{model}: n={m['n']} B-win={m['bm25_ok_directed_fail']} "
              f"D-win={m['directed_ok_bm25_fail']} both_ok={m['both_ok']} "
              f"both_fail={m['both_fail']}")


if __name__ == "__main__":
    import sys
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    main(a.models, indir=a.indir, outpath=a.out)

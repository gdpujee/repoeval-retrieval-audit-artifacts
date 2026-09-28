#!/usr/bin/env python3
"""Per-seed repository-macro statistics for the 30-seed random control.

Method (mirrors the manuscript's macro rule): within each (seed, budget),
average queries within each repository, then average the five repository
means with equal weight. Report mean +/- sample SD across the 30 seeds.

Reads:  results/final/final_query_results.jsonl   (raw query-level rows)
Writes: results/final/random_seed_macro_stats.json
"""

import json
import math
from collections import defaultdict

SRC = "results/final/final_query_results.jsonl"
DST = "results/final/random_seed_macro_stats.json"
METRICS = ["recall", "precision", "f1", "ndcg@10", "mrr"]
N_SEEDS = 30


def main():
    # per[(seed, budget)][repo] -> list of metric dicts
    per = defaultdict(lambda: defaultdict(list))
    n_rows = 0
    for line in open(SRC):
        d = json.loads(line)
        if d.get("strategy") != "random":
            continue
        n_rows += 1
        per[(d["seed"], d["budget"])][d["repo"]].append(d)
    assert n_rows == 341 * 6 * N_SEEDS, f"unexpected random row count: {n_rows}"

    out = {"method": ("repo-macro per seed (equal repo weight), "
                      "then mean +/- sample SD across 30 seeds"),
           "budgets": {}}
    for budget in sorted({b for (_, b) in per}):
        seeds = sorted({s for (s, b) in per if b == budget})
        assert seeds == list(range(N_SEEDS)), f"seed set wrong at {budget}: {seeds}"
        macro_per_seed = {m: [] for m in METRICS}
        for s in seeds:
            repos = per[(s, budget)]
            assert len(repos) == 5, f"repo count wrong at seed {s}: {len(repos)}"
            for m in METRICS:
                repo_means = [sum(q[m] for q in qs) / len(qs)
                              for qs in repos.values()]
                macro_per_seed[m].append(sum(repo_means) / len(repo_means))
        stats = {}
        for m in METRICS:
            xs = macro_per_seed[m]
            mean = sum(xs) / len(xs)
            var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
            stats[m] = {"mean": mean, "std": math.sqrt(var),
                        "min": min(xs), "max": max(xs)}
        out["budgets"][str(budget)] = {"n_seeds": N_SEEDS, "metrics": stats}

    with open(DST, "w") as f:
        json.dump(out, f, indent=2)
    b4 = out["budgets"]["4000"]["metrics"]
    print("B=4000 macro mean +/- SD across 30 seeds:")
    for m in METRICS:
        print(f"  {m:10s} {b4[m]['mean']:.6f} +/- {b4[m]['std']:.6f}")


if __name__ == "__main__":
    main()

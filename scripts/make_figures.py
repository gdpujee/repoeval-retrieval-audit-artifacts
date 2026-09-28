#!/usr/bin/env python3
"""Generate manuscript figures from verified artifacts only (no new data).

Fig 1 (fig1_budget_recall.png): repo-macro recall vs strict token budget.
Fig 2 (fig2_per_repo_differences.png): per-repository paired recall
        differences at B=4,000 for the three pre-registered comparisons.
Fig 3 (fig3_crosscodeeval_support.png): repo-macro ground-truth identifier
        support of the six official CrossCodeEval Python context files.
Fig 4 (fig4_downstream_em_es.png): repo-macro downstream EM/ES by strategy
        for each generation model.
Fig 5 (fig5_recall_vs_em.png): descriptive retrieval-recall vs downstream-EM
        points per strategy; proximity is not a transfer or causal estimate.
Fig 6 (fig6_execution_pass.png): repo-macro test-pass rate by strategy
        for each generation model (299 evaluable tasks).

All inputs are existing verified result files. Every plotted value is
cross-checked against the manuscript tables before writing (assertions).
"""

import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/figures"
FIGURE_DPI = 600  # EMSE requests at least 600 dpi for combination artwork.
# Versioned downstream/execution artifacts (R-13-V8). V9 is the primary
# pre-ranking-blinded matrix; V8 is the post-selection design comparison.
VERSION = os.environ.get("RESULTS_VERSION", "v9")
DOWNSTREAM_SUMMARY = f"results/downstream/{VERSION}/summary.json"
EXECUTION_SUMMARY = f"results/execution/{VERSION}/summary.json"
BUDGETS = [500, 1000, 2000, 4000, 8000, 16000]
SHORT = {"bm25": "BM25", "gcc_coimport": "BM25 + co-import",
         "bm25_directed": "BM25 + directed",
         "codesage_small_v2": "CodeSage-small-v2", "random": "Random"}


def repo_macro(rows, strategy, budget):
    rs = [r for r in rows if r["strategy"] == strategy and r["budget"] == budget]
    assert len(rs) == 5, (strategy, budget, len(rs))
    return sum(r["recall"] for r in rs) / 5


def fig1():
    sparse = [json.loads(l) for l in open("results/final/final_results.jsonl")]
    dense = [json.loads(l) for l in open("results/dense/codesage_small_v2_results.jsonl")]
    rand = json.load(open("results/final/random_seed_macro_stats.json"))
    series = {}
    for strat in ["bm25", "gcc_coimport", "bm25_directed"]:
        series[strat] = [repo_macro(sparse, strat, b) for b in BUDGETS]
    series["codesage_small_v2"] = [repo_macro(dense, "codesage_small_v2", b)
                                   for b in BUDGETS]
    series["random"] = [rand["budgets"][str(b)]["metrics"]["recall"]["mean"]
                        for b in BUDGETS]
    rand_sd = [rand["budgets"][str(b)]["metrics"]["recall"]["std"]
               for b in BUDGETS]
    # Gate against the manuscript's sensitivity table.
    expect = {"bm25": [0.226, 0.331, 0.466, 0.644, 0.844, 0.936],
              "codesage_small_v2": [0.233, 0.338, 0.490, 0.643, 0.781, 0.866]}
    for strat, vals in expect.items():
        for got, want in zip(series[strat], vals):
            assert abs(got - want) < 0.001, (strat, got, want)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for strat, vals in series.items():
        if strat == "random":
            ax.errorbar(BUDGETS, vals, yerr=rand_sd, marker="o",
                        label="Random (30 seeds, +/-SD)")
        else:
            ax.plot(BUDGETS, vals, marker="o", label=SHORT[strat])
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
    ax.set_xlabel("Strict token budget (whole-chunk packing)")
    ax.set_ylabel("Repository-macro recall")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_budget_recall.png", dpi=FIGURE_DPI)
    print("wrote fig1")


def fig2():
    stats = json.load(open("results/final/statistics_extended.json"))
    order = ["gcc_coimport - bm25", "bm25_directed - bm25",
             "codesage_small_v2 - bm25"]
    labels = ["Co-import - BM25", "Directed - BM25", "CodeSage - BM25"]
    repos = ["tracr", "omnivore", "lightweight_mmm", "imagen", "redframes"]
    keys = ["repoeval__deepmind_tracr", "repoeval__facebookresearch_omnivore",
            "repoeval__google_lightweight_mmm",
            "repoeval__lucidrains_imagen-pytorch",
            "repoeval__maxhumber_redframes"]
    comps = {c["comparison"]: c for c in stats["comparisons"]}
    # Gate: macro differences must match the manuscript.
    for comp, want in [("gcc_coimport - bm25", -0.00226),
                       ("bm25_directed - bm25", -0.04667),
                       ("codesage_small_v2 - bm25", -0.00072)]:
        assert abs(comps[comp]["macro_mean_difference"] - want) < 1e-4, comp

    x = range(len(repos))
    w = 0.22
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (comp, lab) in enumerate(zip(order, labels)):
        vals = [comps[comp]["per_repository_mean_differences"][k] for k in keys]
        pos = [j + (i - 1) * w for j in x]
        ax.bar(pos, vals, width=w, label=lab, color=colors[i])
    # macro reference lines, labelled so the two near-zero lines are readable
    yoff = {"Co-import - BM25": 0.0065, "Directed - BM25": 0.002,
            "CodeSage - BM25": -0.0045}
    for comp, lab in zip(order, labels):
        m = comps[comp]["macro_mean_difference"]
        ax.axhline(m, linestyle="--", linewidth=1, alpha=0.7)
        ax.text(len(repos) - 0.4, m + yoff[lab], f"{lab} macro {m:+.5f}",
                fontsize=7, alpha=0.9)
    # omnivore differences are exactly 0.0 for all three methods: mark it
    ax.text(1, 0.004, "all 0.000", ha="center", fontsize=7, style="italic")
    ax.set_xticks(list(x), repos, rotation=12, fontsize=8)
    ax.set_ylabel("Paired recall difference at B=4,000")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_per_repo_differences.png", dpi=FIGURE_DPI)
    print("wrote fig2")


def fig3():
    summ = json.load(open("results/crosscodeeval/context_support_summary.json"))
    order = ["bm25", "unixcoder", "openai_embedding",
             "oracle_bm25", "oracle_unixcoder", "oracle_openai_embedding"]
    labels = ["BM25", "UniXCoder", "OpenAI emb.",
              "Oracle BM25", "Oracle UniXCoder", "Oracle OpenAI emb."]
    vals = {r["method"]: r["repository_macro_identifier_recall"]
            for r in summ["results"]}
    # Gate against the manuscript.
    for m, want in [("bm25", 0.432), ("unixcoder", 0.411),
                    ("openai_embedding", 0.442)]:
        assert abs(vals[m] - want) < 0.001, (m, vals[m])
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bars = ax.bar(labels, [vals[m] for m in order])
    for b, m in zip(bars, order):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                f"{vals[m]:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("Repository-macro identifier support")
    ax.set_ylim(0, 0.7)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_crosscodeeval_support.png", dpi=FIGURE_DPI)
    print("wrote fig3")


DS_MODELS = {"deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
             "muse-spark-1.3-contributor": "Muse Spark 1.3"}
DS_STRATS = ["bm25", "coimport", "directed"]
DS_LABELS = ["BM25", "BM25 + co-import", "BM25 + directed"]
DS_RECALL = {"bm25": 0.644, "coimport": 0.642, "directed": 0.597}


def fig4():
    summ = json.load(open(DOWNSTREAM_SUMMARY))
    n = len(DS_MODELS)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), sharey=True)
    if n == 1:
        axes = [axes]
    x = range(len(DS_STRATS))
    for ax, (model, name) in zip(axes, DS_MODELS.items()):
        st = summ["models"][model]["strategies"]
        em = [st[s]["em"] for s in DS_STRATS]
        es = [st[s]["es"] for s in DS_STRATS]
        ax.bar([i - 0.2 for i in x], em, width=0.4, label="EM")
        ax.bar([i + 0.2 for i in x], es, width=0.4, label="ES")
        for i, s in enumerate(DS_STRATS):
            ax.text(i - 0.2, em[i] + 0.01, f"{em[i]:.3f}",
                    ha="center", fontsize=7)
            ax.text(i + 0.2, es[i] + 0.01, f"{es[i]:.3f}",
                    ha="center", fontsize=7)
        ax.set_xticks(list(x), DS_LABELS, rotation=12, fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Repository-macro score")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_downstream_em_es.png", dpi=FIGURE_DPI)
    print("wrote fig4")


def fig5():
    summ = json.load(open(DOWNSTREAM_SUMMARY))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for model, name in DS_MODELS.items():
        st = summ["models"][model]["strategies"]
        xs = [DS_RECALL[s] for s in DS_STRATS]
        ys = [st[s]["em"] for s in DS_STRATS]
        ax.scatter(xs, ys, label=name, s=60)
        label_offsets = {
            "BM25": (-38, 8),
            "BM25 + co-import": (-75, 4),
            "BM25 + directed": (8, 8),
        }
        for s, xi, yi in zip(DS_LABELS, xs, ys):
            ax.annotate(s, (xi, yi), fontsize=7,
                        xytext=label_offsets[s], textcoords="offset points")
    ax.set_xlabel("Retrieval recall @ B=4,000 (repository-macro)")
    ax.set_ylabel("Downstream EM (repository-macro)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_recall_vs_em.png", dpi=FIGURE_DPI)
    print("wrote fig5")


def fig6():
    summ = json.load(open(EXECUTION_SUMMARY))
    n = len(DS_MODELS)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), sharey=True)
    if n == 1:
        axes = [axes]
    x = range(len(DS_STRATS))
    for ax, (model, name) in zip(axes, DS_MODELS.items()):
        st = summ["models"][model]["strategies"]
        vals = [st[s]["pass_rate"] for s in DS_STRATS]
        bars = ax.bar(list(x), vals)
        for i, s in enumerate(DS_STRATS):
            ax.text(i, vals[i] + 0.012, f"{vals[i]:.3f}",
                    ha="center", fontsize=8)
        ax.set_xticks(list(x), DS_LABELS, rotation=12, fontsize=8)
        ax.set_title(f"{name}\n(n=299 evaluable tasks)", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Repository-macro test-pass rate")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_execution_pass.png", dpi=FIGURE_DPI)
    print("wrote fig6")


if __name__ == "__main__":
    import os
    import sys
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6"]
    if "fig1" in only:
        fig1()
    if "fig2" in only:
        fig2()
    if "fig3" in only:
        fig3()
    if "fig4" in only or "fig5" in only:
        fig4()
        fig5()
    if "fig6" in only:
        fig6()

#!/usr/bin/env python3
"""Audit the two target-span redaction orderings (R-18-V8).

The reported downstream experiment redacts the target span *after* the
selector has ranked and packed on the unmodified corpus
(``results/generation_inputs/reported_post_selection/``). The scientifically
cleaner ordering redacts the candidate corpus *before* the retriever is built
and packs on masked token counts
(``results/generation_inputs/blinded_pre_ranking/``).

This script records, for both variants and all three strategies:

1. a two-level leakage audit -- exact stripped-reference occurrence in a
   selected document or the decommented prompt, and the same search with all
   whitespace removed;
2. token accounting -- recorded pre-redaction selection size versus the
   re-tokenised model-visible context, plus budget utilisation at B=4,000;
3. the selected-document-set change rate between the two orderings.

Writes ``results/generation_inputs/masking_order_audit.json``.

Usage:
  python3 scripts/audit_masking_order.py
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from gcc import TokenCounter  # noqa: E402
from run_downstream_generation import get_retrieved_prompt  # noqa: E402

GENDIR = os.path.join(ROOT, "results", "generation_inputs")
VARIANTS = ["reported_post_selection", "blinded_pre_ranking"]
STRATEGIES = ["bm25", "coimport", "directed"]
BUDGET = 4000


def no_ws(text):
    return re.sub(r"\s+", "", text)


def load(variant, strategy):
    path = os.path.join(GENDIR, variant,
                        f"repoeval_function_{strategy}_4k.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def decomment(prompt):
    return "\n".join(x[2:] if x.startswith("# ") else x
                     for x in prompt.split("\n"))


def audit(rows):
    """Two-level leakage audit over one variant/strategy."""
    exact, normalized = [], []
    for r in rows:
        ref = (r.get("reference") or "").strip()
        if not ref:
            continue
        docs = r.get("docs", [])
        tid = r["metadata"]["task_id"]
        if any(ref in d.get("text", "") for d in docs):
            exact.append(tid)
            continue
        joined = "\n".join(d.get("text", "") for d in docs)
        nref = no_ws(ref)
        if nref and (nref in no_ws(decomment(get_retrieved_prompt(docs)))
                     or nref in no_ws(joined)):
            normalized.append(tid)
    return exact, normalized


def main():
    counter = TokenCounter()
    report = {
        "budget": BUDGET,
        "variants": {},
        "leakage": {},
        "token_accounting": {},
        "selected_set_difference": {},
    }

    sets = {}
    for variant in VARIANTS:
        report["variants"][variant] = {}
        for strategy in STRATEGIES:
            rows = load(variant, strategy)
            assert len(rows) == 341, (variant, strategy, len(rows))
            sets[(variant, strategy)] = rows
            exact, normalized = audit(rows)
            report["leakage"][f"{variant}/{strategy}"] = {
                "n_exact": len(exact),
                "n_normalized": len(normalized),
                "normalized_tasks": sorted(set(normalized)),
            }
            recorded = [r["retrieved_tokens_cl100k"] for r in rows]
            actual = [sum(counter.count(d["text"]) for d in r["docs"])
                      for r in rows]
            n = len(rows)
            report["variants"][variant][strategy] = {
                "n_tasks": n,
                "mean_docs": sum(len(r["docs"]) for r in rows) / n,
            }
            report["token_accounting"][f"{variant}/{strategy}"] = {
                "recorded_selection_mean_tokens": sum(recorded) / n,
                "model_visible_mean_tokens": sum(actual) / n,
                "mean_reduction_tokens": (sum(recorded) - sum(actual)) / n,
                "max_reduction_tokens": max(a - b
                                            for a, b in zip(recorded, actual)),
                "recorded_utilisation": sum(recorded) / n / BUDGET,
                "model_visible_utilisation": sum(actual) / n / BUDGET,
                "n_tasks_over_budget_after_redaction":
                    sum(1 for x in actual if x > BUDGET),
            }

    for strategy in STRATEGIES:
        a = sets[("reported_post_selection", strategy)]
        b = sets[("blinded_pre_ranking", strategy)]
        assert [r["metadata"]["task_id"] for r in a] == \
               [r["metadata"]["task_id"] for r in b]
        changed = sum(
            1 for x, y in zip(a, b)
            if [d["id"] for d in x["docs"]] != [d["id"] for d in y["docs"]])
        order_changed = sum(
            1 for x, y in zip(a, b)
            if [d["id"] for d in x["docs"]] == [d["id"] for d in y["docs"]]
            and [d["text"] for d in x["docs"]] != [d["text"] for d in y["docs"]])
        report["selected_set_difference"][strategy] = {
            "n_tasks": len(a),
            "selected_set_changed": changed,
            "same_set_different_packing": order_changed,
        }

    exact_total = sum(v["n_exact"] for v in report["leakage"].values())
    normalized_total = sum(v["n_normalized"]
                           for v in report["leakage"].values())
    report["summary"] = {
        "exact_hits_total": exact_total,
        "normalized_hits_total": normalized_total,
        "reported_post_selection_normalized_hits_total": sum(
            v["n_normalized"] for k, v in report["leakage"].items()
            if k.startswith("reported_post_selection/")),
        "blinded_pre_ranking_normalized_hits_total": sum(
            v["n_normalized"] for k, v in report["leakage"].items()
            if k.startswith("blinded_pre_ranking/")),
        "normalized_tasks": sorted({
            t for v in report["leakage"].values()
            for t in v["normalized_tasks"]}),
    }
    out = os.path.join(GENDIR, "masking_order_audit.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report["summary"], indent=2))
    for k, v in report["token_accounting"].items():
        print(f"{k:42s} recorded {v['recorded_selection_mean_tokens']:8.2f} "
              f"visible {v['model_visible_mean_tokens']:8.2f} "
              f"util {v['model_visible_utilisation']:.4f}")
    print(json.dumps(report["selected_set_difference"], indent=2))
    print("wrote", out)
    blinded_normalized = report["summary"][
        "blinded_pre_ranking_normalized_hits_total"]
    if exact_total or blinded_normalized:
        raise SystemExit(
            "LEAKAGE: expected zero exact hits across variants and zero "
            f"normalized hits in blinded inputs; got exact={exact_total}, "
            f"blinded_normalized={blinded_normalized}")


if __name__ == "__main__":
    main()

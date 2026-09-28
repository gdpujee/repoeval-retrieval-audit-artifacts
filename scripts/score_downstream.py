#!/usr/bin/env python3
"""Score downstream generations with the official RepoEval metric code.

Pipeline (mirrors scripts/code-rag-bench/.../repoeval.py::process_results):
  1. extract ``` code block if present;
  2. truncate at official STOP_WORDS;
  3. process_prediction (comment strip, truncate to reference line count);
  4. compute_EM / compute_ES from repoeval_ESEM.py;
  5. ast.parse check on (prompt_tail + generation) as a syntax proxy.

Writes results/downstream/scores_<model>.jsonl (one row per task/strategy)
and results/downstream/summary.json (repo-macro EM/ES/parse + two-stage
repo/query bootstrap 95% CIs for paired EM differences, seed 20260913).
"""

import ast
import importlib.util
import json
import os
import random
import re
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOP_WORDS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif",
              "\n```", "<file_sep>"]
CODE_BLOCK = r"```(\w*)\n(.*?)\n```"


def load_esem():
    esem_path = os.path.join(
        ROOT, "scripts", "code-rag-bench", "generation", "eval",
        "tasks", "custom_metrics", "repoeval_ESEM.py")
    if not os.path.exists(esem_path):
        # Cold start from the supplementary archive: the official scorer lives
        # in a git submodule that the archive does not redistribute.
        raise SystemExit(
            "Missing the official RepoEval metric code at "
            f"{os.path.relpath(esem_path, ROOT)}.\n"
            "Fetch the vendored upstream first:\n"
            "  git submodule update --init scripts/code-rag-bench\n"
            "  git -C scripts/code-rag-bench rev-parse HEAD  "
            "# expect f9e100ca9ed94b8f1983b356ae81966e30210cf4\n"
            "See README.md, section 'Vendored upstreams'.")
    spec = importlib.util.spec_from_file_location("esem", esem_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stop_at(text):
    idx = [text.find(s) for s in STOP_WORDS if text.find(s) >= 0]
    return text[:min(idx)] if idx else text


def score_file(esem, path):
    rows = []
    malformed = 0
    # Dedupe by (task, strategy, model): retry passes append new rows next
    # to old error rows; a successful row always wins over an error row.
    # Any remaining same-key conflict (two successful rows) is a hard error:
    # the scorer must fail loudly rather than silently pick one (R-12).
    seen = {}
    conflicts = []
    for line in open(path):
        try:
            d = json.loads(line)
        except ValueError:
            malformed += 1  # torn write from an interrupted run; rerun fills it
            continue
        k = (d["task_id"], d["strategy"], d["model"])
        if k not in seen or (seen[k].get("error") is not None
                             and d.get("error") is None):
            seen[k] = d
        elif (seen[k].get("error") is None and d.get("error") is None
                and seen[k].get("completion") != d.get("completion")):
            conflicts.append(k)
    if conflicts:
        raise SystemExit(
            f"CONFLICT: {len(conflicts)} same-key successful rows differ, "
            f"e.g. {conflicts[0]}. Resolve provenance before scoring.")
    for d in seen.values():
        raw = d.get("completion") or ""
        mm = re.findall(CODE_BLOCK, raw, flags=re.DOTALL)
        gen = stop_at(mm[0][1] if mm else raw)
        ref, gen_clean = esem.process_prediction(d["reference"], gen)
        # Syntax proxy: wrap the candidate as a function body. Calibrated to
        # 100% (341/341) on references; the naive fragment-alone parse only
        # reached 4.7% on prompt+reference and was discarded as invalid.
        try:
            ast.parse("def _probe():\n" + textwrap.indent(gen_clean, "    "))
            parse_ok = bool(gen_clean.strip())
        except SyntaxError:
            parse_ok = False
        rows.append({
            "task_id": d["task_id"],
            "repo": "repoeval__" + d["task_id"].split("/")[0],
            "strategy": d["strategy"],
            "model": d["model"],
            "em": bool(esem.compute_EM(ref, gen_clean)),
            "es": float(esem.compute_ES(ref, gen_clean)),
            "parse_ok": parse_ok,
            "error": d.get("error"),
            "prompt_tokens": (d.get("usage") or {}).get("prompt_tokens"),
            "completion_tokens": (d.get("usage") or {}).get(
                "completion_tokens"),
        })
    if malformed:
        print(f"WARNING {path}: {malformed} malformed lines skipped", flush=True)
    return rows


def repo_macro(rows, metric):
    by_repo = {}
    for r in rows:
        by_repo.setdefault(r["repo"], []).append(r[metric])
    means = {k: sum(v) / len(v) for k, v in by_repo.items()}
    return sum(means.values()) / len(means), means


def paired_bootstrap(rows_a, rows_b, iters=10000, seed=20260913):
    """Two-stage repo/query bootstrap of the repo-macro EM difference."""
    ra = {(r["task_id"]): r for r in rows_a}
    rb = {(r["task_id"]): r for r in rows_b}
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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None,
                    help="output dir for scores/summary (default results/downstream)")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    outdir = a.outdir or os.path.join(ROOT, "results", "downstream")
    os.makedirs(outdir, exist_ok=True)
    esem = load_esem()
    files = a.files
    all_rows = []
    summary = {"models": {}}
    for path in files:
        rows = score_file(esem, path)
        model = rows[0]["model"]
        errs = sum(1 for r in rows if r["error"])
        ok = [r for r in rows if not r["error"]]
        assert len(ok) + errs == 1023, (path, len(rows))
        with open(os.path.join(
                outdir,
                f"scores_{model}.jsonl"), "w") as f:
            for r in ok:
                f.write(json.dumps(r) + "\n")
        ms = summary["models"][model] = {"n": len(ok), "errors": errs,
                                         "strategies": {}}
        by_s = {}
        for r in ok:
            by_s.setdefault(r["strategy"], []).append(r)
        for s, rs in by_s.items():
            em, _ = repo_macro(rs, "em")
            es, _ = repo_macro(rs, "es")
            po, _ = repo_macro(rs, "parse_ok")
            # R-17: keep full precision in the artifact; rounding to print
            # precision happens only in the manuscript/LaTeX generators.
            ms["strategies"][s] = {
                "em": em, "es": es,
                "parse_ok": po, "n": len(rs)}
        if errs:
            print(f"WARNING {model}: {errs} generation errors excluded",
                  flush=True)
        all_rows.extend(ok)
    # Paired EM differences per model: coimport-bm25, directed-bm25.
    for model in summary["models"]:
        mr = [r for r in all_rows if r["model"] == model]
        g = lambda s: [r for r in mr if r["strategy"] == s]  # noqa: E731
        for a, b, name in [("coimport", "bm25", "coimport-bm25"),
                           ("directed", "bm25", "directed-bm25")]:
            ea, _ = repo_macro(g(a), "em")
            eb, _ = repo_macro(g(b), "em")
            lo, hi = paired_bootstrap(g(a), g(b))
            summary["models"][model][name] = {
                "macro_em_diff": round(ea - eb, 5),
                "ci95": [round(lo, 5), round(hi, 5)]}
    with open(os.path.join(outdir,
                           "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

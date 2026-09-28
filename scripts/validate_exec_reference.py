#!/usr/bin/env python3
"""Reference validation: run every test file mapped to our 341 tasks ONCE on
unpatched repo copies. Establishes which (task, test-file) pairs have a
working reference signal in .venv-exec before any execution eval.

Writes results/execution/reference_validation.json:
  {repo: {testfile: {returncode, passed, duration_s, note}}}
A file counts as reference-passing only if pytest returns 0.
"""

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv-exec", "bin", "python")
REF = os.path.join(ROOT, "results", "execution", "ref")
MAP = os.path.join(
    ROOT, "scripts", "code-rag-bench", "generation", "eval", "tasks",
    "custom_metrics", "repoeval_task_id2tests.json")
OUT = os.path.join(ROOT, "results", "execution", "reference_validation.json")
REPOS = ["deepmind_tracr", "facebookresearch_omnivore",
         "google_lightweight_mmm", "lucidrains_imagen-pytorch",
         "maxhumber_redframes"]
TIMEOUT = 600


def main():
    mapping = json.load(open(MAP))
    need = set()
    for line in open(os.path.join(
            ROOT, "results", "generation_inputs",
            "repoeval_function_bm25_4k.jsonl")):
        need.add(json.loads(line)["metadata"]["task_id"])
    files = {}
    for t in need:
        repo = t.split("/")[0].replace("repoeval__", "")
        for f in mapping.get(t, []):
            files.setdefault(repo, set()).add(f)
    for r in REPOS:
        files.setdefault(r, set())

    only = sys.argv[1:] or REPOS
    # R-11: never silently overwrite the five-repo record with a single-repo
    # run. Merge into any existing record instead of replacing the file.
    prev = {}
    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT))
        except ValueError:
            prev = {}
    result = {r: dict(prev.get(r, {})) for r in REPOS}
    for repo in only:
        result[repo] = {}
        cwd = os.path.join(REF, repo)
        for f in sorted(files[repo]):
            t0 = time.time()
            try:
                p = subprocess.run(
                    [PY, "-m", "pytest", f, "-q", "--no-header", "-p",
                     "no:cacheprovider"],
                    cwd=cwd, capture_output=True, timeout=TIMEOUT)
                rc = p.returncode
                tail = (p.stdout.decode("utf8", errors="replace") +
                        p.stderr.decode("utf8", errors="replace"))[-500:]
            except subprocess.TimeoutExpired:
                rc, tail = "timeout", ""
            result[repo][f] = {"returncode": rc,
                               "duration_s": round(time.time() - t0, 1),
                               "tail": tail}
            print(f"{repo} :: {f} -> {rc} "
                  f"({result[repo][f]['duration_s']}s)", flush=True)
    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=2)
    npass = sum(1 for r in result.values() for v in r.values()
                if v["returncode"] == 0)
    ntot = sum(len(r) for r in result.values())
    print(f"reference-passing files: {npass}/{ntot}")


if __name__ == "__main__":
    main()

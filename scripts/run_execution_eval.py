#!/usr/bin/env python3
"""Execution evaluation: patch each model completion into a repo copy and run
the task's mapped test files (relevant-tests-only, as in the official
repoeval_execution.eval_generation).

Key deviations from the vendored harness (all documented, none touch the
tested code):
  - No conda env: uses .venv-exec directly (same interpreter for all calls).
  - Skips test files that fail on UNPATCHED reference code for pure
    library-version drift (listed in SKIP_FILES); the official harness in its
    era env passes them. Per-test deselects in DESELECT for drifted tests
    inside otherwise-passing files.
  - postprocess_by_function is imported verbatim from the harness.

Calibration: --calibrate patches REFERENCE text for N tasks and expects
success everywhere (validates line numbers + postprocessing). Do not run the
full matrix unless calibration passes.

Writes results/execution/eval_<model>.jsonl rows:
  task_id, strategy, model, files: {testfile: rc}, success, duration_s.
Task success = every discriminative mapped file returns 0.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv-exec", "bin", "python")
REF = os.path.join(ROOT, "results", "execution", "ref")
MAP = os.path.join(
    ROOT, "scripts", "code-rag-bench", "generation", "eval", "tasks",
    "custom_metrics", "repoeval_task_id2tests.json")
OUTDIR = os.path.join(ROOT, "results", "execution")
SHORT2DIR = {"deepmind_tracr": "deepmind_tracr",
             "facebookresearch_omnivore": "facebookresearch_omnivore",
             "google_lightweight_mmm": "google_lightweight_mmm",
             "lucidrains_imagen-pytorch": "lucidrains_imagen-pytorch",
             "maxhumber_redframes": "maxhumber_redframes"}

# Files failing on reference for env drift (see reference_validation.json).
SKIP_FILES = {
    "tracr/compiler/test_cases.py",  # rc=5: helper module, collects 0 tests
    "lightweight_mmm/core/transformations/saturation_test.py",  # jnp.where kwargs in repo code
    "lightweight_mmm/models_test.py",  # same
    "lightweight_mmm/media_transforms_test.py",  # same
    "lightweight_mmm/lightweight_mmm_test.py",  # needs tensorflow
    "lightweight_mmm/optimize_media_test.py",  # needs tensorflow
    "lightweight_mmm/plot_test.py",  # needs tensorflow
    "lightweight_mmm/utils_test.py",  # needs tensorflow
}
# Drifted TESTS inside otherwise-passing files (test-code library drift).
DESELECT = {
    "tests/test_docstrings.py": ["tests/test_docstrings.py::TestDocstrings::test_memory"],
    "lightweight_mmm/core/time/trend_test.py": [
        "lightweight_mmm/core/time/trend_test.py::TrendTest::test_core_dynamic_trend_produces_correct_value"],
    "lightweight_mmm/core/transformations/lagging_test.py": [
        "lightweight_mmm/core/transformations/lagging_test.py::LaggingTest::test_adstock_zeros_stay_zeros",
        "lightweight_mmm/core/transformations/lagging_test.py::LaggingTest::test_carryover_zeros_stay_zeros"],
    "lightweight_mmm/preprocessing_test.py": [
        "lightweight_mmm/preprocessing_test.py::PreprocessingTest::test_compute_vifs_returns_expected_values_geo_data",
        "lightweight_mmm/preprocessing_test.py::PreprocessingTest::test_compute_vifs_returns_expected_values_national_data",
        "lightweight_mmm/preprocessing_test.py::PreprocessingTest::test_extreme_values_for_compute_vifs",
        "lightweight_mmm/preprocessing_test.py::PreprocessingTest::test_fit_transform_works_with_operations_in_three_dimensions_min_multiply_mean_divide"],
}
FILE_TIMEOUT = 300


def load_postprocess():
    spec = importlib.util.spec_from_file_location(
        "rexec", os.path.join(
            ROOT, "scripts", "code-rag-bench", "generation", "eval",
            "tasks", "custom_metrics", "repoeval_execution.py"))
    m = importlib.util.module_from_spec(spec)
    # Prevent module-level side effects (it loads the tests json with a
    # relative path); run with cwd=that dir is NOT needed since we only take
    # the pure function. Guard: execute and catch.
    old = os.getcwd()
    os.chdir(os.path.join(ROOT, "scripts", "code-rag-bench", "generation"))
    try:
        spec.loader.exec_module(m)
    finally:
        os.chdir(old)
    return m.postprocess_by_function


def patch_and_run(postprocess, workdir, task_id, fpath_tuple, line_no,
                  reference, generation):
    """Patch generation into workdir copy, run mapped files, restore."""
    repo = fpath_tuple[0]
    local = "/".join(fpath_tuple)
    target = reference if reference.endswith("\n") else reference
    target_n = len(target.rstrip("\n").split("\n"))
    start = line_no
    end = start + target_n
    src = os.path.join(REF, local)
    dst = os.path.join(workdir, local)
    with open(src) as f:
        lines = [l.rstrip("\n") for l in f]
    gen_lines = postprocess(generation, target)
    lines[start:end] = gen_lines
    with open(dst, "w") as f:
        f.write("\n".join(lines))
    return start, end


def run_files(workdir, repo, files):
    rcs = {}
    for f in files:
        if f in SKIP_FILES:
            continue
        cmd = [PY, "-m", "pytest", f, "-q", "--no-header", "-p",
               "no:cacheprovider", "-x"]
        for d in DESELECT.get(f, []):
            cmd += ["--deselect", d]
        try:
            p = subprocess.run(cmd, cwd=os.path.join(workdir, repo),
                               capture_output=True, timeout=FILE_TIMEOUT)
            rcs[f] = p.returncode
        except subprocess.TimeoutExpired:
            rcs[f] = "timeout"
        if rcs[f] != 0:
            break  # official stop_at_the_first_failed_test behavior
    return rcs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--inputs", default=os.path.join(
        ROOT, "results", "generation_inputs", "reported_post_selection"),
        help="generation-input variant supplying canonical task metadata/order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--calibrate", type=int, default=0,
                    help="patch REFERENCE for first N tasks instead of generations")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--worker", type=int, default=0)
    a = ap.parse_args()

    postprocess = load_postprocess()
    mapping = json.load(open(MAP))
    inputs = {}
    for line in open(os.path.join(
            a.inputs, "repoeval_function_bm25_4k.jsonl")):
        d = json.loads(line)
        inputs[d["metadata"]["task_id"]] = d["metadata"]
    # Raw generation files retain rate-limit failures and successful retries.
    # Execute each successful completion once; never evaluate an error row as
    # an empty completion or let a later retry change worker ownership.
    successful = {}
    for line in open(a.raw):
        r = json.loads(line)
        if r.get("error") is not None:
            continue
        k = (r["task_id"], r["strategy"], r["model"])
        if k in successful and successful[k]["completion"] != r["completion"]:
            raise SystemExit(f"Conflicting successful generations for {k}")
        successful[k] = r
    task_order = {task_id: i for i, task_id in enumerate(inputs)}
    strategy_order = {name: i for i, name in enumerate(
        ["bm25", "coimport", "directed"])}
    rows = sorted(successful.values(), key=lambda r: (
        strategy_order[r["strategy"]], task_order[r["task_id"]]))
    if a.calibrate:
        tasks = sorted({r["task_id"] for r in rows})[:a.calibrate]
        by_task = {}
        for r in rows:
            by_task.setdefault(r["task_id"], r)
        rows = [dict(by_task[t], strategy="reference-self",
                     completion=by_task[t]["reference"]) for t in tasks]
    rows = rows[a.offset:]
    if a.stride > 1:
        if a.calibrate:
            rows = rows[a.worker::a.stride]
        else:
            rows = [r for r in rows if (
                strategy_order[r["strategy"]] * len(task_order)
                + task_order[r["task_id"]]) % a.stride == a.worker]
    if a.limit:
        rows = rows[:a.limit]

    if os.path.exists(a.workdir):
        shutil.rmtree(a.workdir)
    for repo in SHORT2DIR.values():
        shutil.copytree(os.path.join(REF, repo),
                        os.path.join(a.workdir, repo))

    # Resume: skip rows already completed in a previous partial run, then
    # append. The out file is rewritten atomically at the end.
    done = set()
    prev = []
    if os.path.exists(a.out):
        with open(a.out) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                done.add((d["task_id"], d["strategy"], d.get("model")))
                prev.append(line.rstrip("\n"))
    rows = [r for r in rows
            if (r["task_id"], r["strategy"], a.model) not in done]
    if done:
        print(f"resuming: {len(done)} done, {len(rows)} remaining",
              flush=True)

    ok = fail = 0
    with open(a.out, "w") as out:
        for line in prev:
            out.write(line + "\n")
        for i, r in enumerate(rows):
            t0 = time.time()
            md = inputs[r["task_id"]]
            fpath = tuple(md["fpath_tuple"])
            ref = r["reference"]
            gen = ref if a.calibrate else (r.get("completion") or "")
            patch_and_run(postprocess, a.workdir, r["task_id"], fpath,
                          md["line_no"], ref, gen)
            files = [f for f in mapping.get(r["task_id"], [])
                     if f not in SKIP_FILES]
            rcs = run_files(a.workdir, fpath[0], files)
            # restore patched file from pristine copy
            shutil.copy2(os.path.join(REF, "/".join(fpath)),
                         os.path.join(a.workdir, "/".join(fpath)))
            success = bool(files) and all(v == 0 for v in rcs.values())
            ok += success
            fail += (not success)
            out.write(json.dumps({
                "task_id": r["task_id"], "strategy": r["strategy"],
                "model": a.model, "files": rcs, "success": success,
                "duration_s": round(time.time() - t0, 1)}) + "\n")
            out.flush()
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(rows)} ok={ok} fail={fail}",
                      flush=True)
    print(f"done: {len(rows)} ok={ok} fail={fail}")


if __name__ == "__main__":
    main()

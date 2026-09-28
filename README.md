# RepoEval retrieval audit: reproducibility artifacts

This repository accompanies the manuscript *What Repository Retrieval Metrics
Reward Under Strict Token Budgets: Auditing RepoEval with Cross-File Diagnostics
and Downstream Evidence* submitted to Empirical Software Engineering. It
contains derived results, analysis scripts, generation inputs and outputs,
execution results, figures, dependency locks, and reproduction instructions.
The primary V9 experiment contains 2,046 successful generations (two models ×
three strategies × 341 tasks); the complete V8 post-selection comparison is
retained separately. The manuscript, submission documents, private reviews,
credentials, and private Git history are excluded.

For a fixed snapshot, use the tagged `v1.0.0` release. `MANIFEST.sha256`
records the SHA-256 digest of each distributed file.

Artifact directories are versioned by **experiment round**, and the manuscript
revision is versioned separately. The submitted manuscript uses the corrected V9
pre-ranking-blinded generation and execution matrix as its primary downstream
evidence (`results/downstream/v9/`, `results/execution/v9/`). The complete V8
post-selection-redaction matrix is retained as a design comparison. The V7
historical round is excluded because it is not needed for the reported results.

The two upstream projects must be fetched at the pinned revisions below.
Proprietary model snapshots are unavailable; rerunning generation requires
provider API access. See `NOTICE.md` for third-party provenance.
The saved retrieval, dense, and CrossCodeEval evidence can be verified without
making any model API calls:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/verify_final_results.py
python3 scripts/verify_dense_baseline.py
python3 scripts/verify_crosscodeeval.py
```

## Redaction order (read this before reusing the generation results)

The primary V9 downstream experiment masks the target completion span
throughout the *candidate corpus before retrieval*, then ranks and packs with
masked token counts. The complete V8 comparison redacts only after retrieval;
its recorded `retrieved_tokens_cl100k` field therefore describes selection,
not the model-visible context. The manuscript reports actual sizes for both.
Both matrices contain 2 models × 3 strategies × 341 tasks. The V9 corrected
matrix has 1,023 successful unique generations per model, with all input hashes
matching and no provider-error rows. The initial exact-only pilot had 866
successes; six whose prompts changed under the corrected masker were excluded,
and the remaining 860 matching rows were reused. Both input variants and the
reuse audit are retained:

| Variant | Directory | Role |
|---|---|---|
| Reported | `results/generation_inputs/reported_post_selection/` | inputs used by the reported experiment |
| Blinded | `results/generation_inputs/blinded_pre_ranking/` | what `scripts/prepare_generation_inputs.py` now produces |

```bash
python3 scripts/audit_masking_order.py
```

writes `results/generation_inputs/masking_order_audit.json`, which records
both variants' two-level leakage audit (zero exact hits; one normalized task
in the post-selection inputs and zero in corrected blinded inputs),
pre/post-redaction token accounting, and the selected-set change rate between
the two orderings (220/206/192 of 341 tasks for BM25/co-import/directed). The
separate candidate audit reports zero remaining full-reference matches after
masking.

## Reproduce

```bash
python3 -m pip install -r requirements.txt   # tiktoken, editdistance, numpy, matplotlib
```

## Pinned upstream repositories

Prepare both pinned upstreams before running `./run_all.sh` or the retrieval
and execution scripts. They are not Git submodules in this standalone
repository; clone them directly:

```bash
git clone https://github.com/code-rag-bench/code-rag-bench scripts/code-rag-bench
git -C scripts/code-rag-bench checkout f9e100ca9ed94b8f1983b356ae81966e30210cf4
mkdir -p external
git clone https://github.com/amazon-science/cceval external/cceval
git -C external/cceval checkout 40c68d2b7ca2a8eae95d901ac80e6a540a84a53d
```

Then run the retrieval audit pipeline:

```bash
./run_all.sh
```

The final retrieval evidence chain is written to `results/final/`. The
selector uses whole chunks and asserts that every selected context satisfies
its configured token budget. Here `run_all.sh` skips the manuscript-only author
gate, which requires the private manuscript checkout.

`results/execution/ref/` (the pristine per-repository trees the execution
harness patches) is a copy of
`scripts/code-rag-bench/retrieval/output/repoeval/repositories/function_level/`.
Rebuild it after cloning the pinned upstream if it is absent:

```bash
mkdir -p results/execution/ref
for r in deepmind_tracr facebookresearch_omnivore google_lightweight_mmm \
         lucidrains_imagen-pytorch maxhumber_redframes; do
  cp -r scripts/code-rag-bench/retrieval/output/repoeval/repositories/function_level/$r \
        results/execution/ref/
done
python3 scripts/validate_exec_reference.py   # expect 35/47 files reference-passing
```

## Dense baseline

The CodeSage-small-v2 baseline has an isolated environment because PyTorch is
substantially larger than the core dependencies:

```bash
python3.13 -m venv .venv-dense
.venv-dense/bin/python -m pip install -r requirements-dense.txt
.venv-dense/bin/python scripts/run_dense_baseline.py
.venv-dense/bin/python scripts/verify_dense_baseline.py
python3 scripts/report_dense_baseline.py
```

## CrossCodeEval

The analysis expects the official `amazon-science/cceval` repository at commit
`40c68d2b7ca2a8eae95d901ac80e6a540a84a53d`, with the Python part of `data/crosscodeeval_data.tar.xz` extracted
under `external/cceval/data/crosscodeeval_data/`. It audits official
pre-generated contexts; raw repositories are not included, so it does not
claim to rerun the retrievers.

## Downstream generation (requires API access)

`--inputs` defaults to the V8 post-selection variant; pass
`--inputs results/generation_inputs/blinded_pre_ranking` for the V9 primary
pre-ranking-blinded design. Each matrix has two generators × three strategies
× 341 tasks; billed usage applies when new model calls are needed. Successful
matching `(task, strategy)` rows are skipped and failed rows are retried.
`--outdir`/`--indir` keep rounds separated:

```bash
python3 scripts/run_downstream_generation.py --model deepseek-v4.1-flash \
    --out results/downstream/v8/raw_deepseek.jsonl --workers 4
python3 scripts/run_downstream_generation.py --model muse-spark-1.3-contributor --api responses \
    --out results/downstream/v8/raw_muse.jsonl --workers 4
python3 scripts/score_downstream.py --outdir results/downstream/v8 \
    results/downstream/v8/raw_deepseek.jsonl results/downstream/v8/raw_muse.jsonl
python3 scripts/analyze_downstream_failures.py --indir results/downstream/v8 \
    --out results/downstream/v8/failure_analysis.json \
    deepseek-v4.1-flash muse-spark-1.3-contributor
python3 scripts/sensitivity_exclude_task.py --indir results/downstream/v8 \
    --exclude google_lightweight_mmm/22
```

For V9, each active raw file contains 1,023 successful unique generations
whose input hash matches the corrected inputs. The earlier exact-only pilot
remains under `results/downstream/v9_pre_normalized/`. Reproduce the scored
table and failure analysis from the archived responses without API access:

```bash
python3 scripts/score_downstream.py --outdir results/downstream/v9 \
    results/downstream/v9/raw_deepseek.jsonl results/downstream/v9/raw_muse.jsonl
python3 scripts/analyze_downstream_failures.py --indir results/downstream/v9 \
    --out results/downstream/v9/failure_analysis.json \
    deepseek-v4.1-flash muse-spark-1.3-contributor
```

Without `--outdir` the scorer writes to `results/downstream/`, mixing output
from experiment rounds. The scorer also exits non-zero if two successful rows
share a `(task, strategy, model)` key. `results/final/clean_dir_repro.json`
records a check that re-running the scorer into an empty directory reproduces
`summary.json` and both `scores_*.jsonl` byte-for-byte.

Figures 4–6 read `results/downstream/<version>/summary.json` and
`results/execution/<version>/summary.json`, where `<version>` defaults to `v9`
and can be overridden with `RESULTS_VERSION=v8` for the post-selection
comparison:

```bash
python3 scripts/make_figures.py fig1 fig2 fig3 fig4 fig5 fig6
```

## Execution environment

```bash
python3.12 -m venv .venv-exec
.venv-exec/bin/python -m pip install -r requirements-exec.txt
python3 scripts/validate_exec_reference.py
```

For the primary V9 blinded execution round, run both generators explicitly so
each evaluator reads the matching raw file. Results below are already archived;
these commands reproduce the mapped-test run from the saved generations:

```bash
for W in 0 1 2 3 4 5 6 7; do
  .venv-exec/bin/python scripts/run_execution_eval.py \
      --model deepseek-v4.1-flash \
      --raw results/downstream/v9/raw_deepseek.jsonl \
      --inputs results/generation_inputs/blinded_pre_ranking \
      --out results/execution/v9/eval_deepseek_w$W.jsonl \
      --workdir results/execution/v9/work-deepseek-$W --stride 8 --worker $W &
  .venv-exec/bin/python scripts/run_execution_eval.py \
      --model muse-spark-1.3-contributor \
      --raw results/downstream/v9/raw_muse.jsonl \
      --inputs results/generation_inputs/blinded_pre_ranking \
      --out results/execution/v9/eval_muse_w$W.jsonl \
      --workdir results/execution/v9/work-muse-$W --stride 8 --worker $W &
done
wait
.venv-exec/bin/python scripts/score_execution.py --indir results/execution/v9 \
    --out results/execution/v9/summary.json
.venv-exec/bin/python scripts/sensitivity_execution_exclude_repo.py \
    --indir results/execution/v9 \
    --out results/execution/v9/sensitivity_exclude_lightweight_mmm.json \
    --exclude-repo google_lightweight_mmm
```

It is safe to resume an interrupted run: failed API rows are ignored,
successful keys are assigned to a stable worker, and completed execution keys
are skipped. `score_execution.py` refuses to summarize until both models have
all 341 tasks under each of the three strategies.

`results/execution/work-*/` and `logs/` are scratch space and are gitignored;
`results/execution/ref/` is tracked.

## Credentials

No credential file belongs in this repository.
Generation scripts read `OPENCODE_APIKEY` from `.env` (gitignored, never
committed). The retrieval experiment does not require an API key.

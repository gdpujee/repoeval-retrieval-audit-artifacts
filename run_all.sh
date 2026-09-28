#!/usr/bin/env bash
set -euo pipefail

python3 scripts/run_final_experiment.py
python3 scripts/audit_repoeval.py
python3 scripts/audit_directed_graph.py
python3 scripts/analyze_extended.py
python3 scripts/verify_final_results.py

if [[ -f results/dense/codesage_small_v2_query_results.jsonl ]]; then
  python3 scripts/verify_dense_baseline.py
  python3 scripts/report_dense_baseline.py
fi

if [[ -d external/cceval/data/crosscodeeval_data/python ]]; then
  python3 scripts/analyze_crosscodeeval_contexts.py
  python3 scripts/verify_crosscodeeval.py
fi

python3 scripts/prepare_generation_inputs.py
python3 scripts/audit_masking_order.py
python3 scripts/analyze_random_seeds.py
python3 scripts/make_figures.py fig1 fig2 fig3
if [[ -f latex/main.tex && -f latex/tables.tex ]]; then
  python3 scripts/verify_manuscript_v9.py
else
  echo "Skipping manuscript author gate: historical LaTeX sources are absent (expected in the public supplement ZIP)."
fi

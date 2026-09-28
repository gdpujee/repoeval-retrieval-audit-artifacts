# Auto-generated final tables

> Generated only from `results/final/final_results.jsonl`.

## Main results at B=4,000

| Strategy | Recall | Precision | F1 | NDCG@10 | MRR | Avg tokens | Max tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.644 | 0.640 | 0.516 | 0.899 | 0.987 | 3968 | 4000 |
| BM25 + co-import | 0.642 | 0.669 | 0.524 | 0.898 | 0.987 | 3930 | 4000 |
| BM25 + directed imports | 0.597 | 0.562 | 0.463 | 0.806 | 0.888 | 3967 | 4000 |
| Sequential | 0.034 | 0.019 | 0.022 | 0.012 | 0.007 | 3977 | 4000 |
| Random (30 seeds) | 0.018 ± 0.009 | 0.017 | 0.014 | 0.020 | 0.051 | 3972 | 4000 |

## Budget sensitivity (macro-average recall)

| Budget | BM25 | BM25 + co-import | BM25 + directed imports | Random (30 seeds) |
|---:|---:|---:|---:|---:|
| 500 | 0.226 | 0.216 | 0.209 | 0.005 |
| 1,000 | 0.331 | 0.321 | 0.297 | 0.007 |
| 2,000 | 0.466 | 0.463 | 0.425 | 0.011 |
| 4,000 | 0.644 | 0.642 | 0.597 | 0.018 |
| 8,000 | 0.844 | 0.844 | 0.788 | 0.033 |
| 16,000 | 0.936 | 0.935 | 0.882 | 0.063 |

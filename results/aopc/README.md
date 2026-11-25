# AOPC faithfulness evaluation (paper Table A5)

Canonical run: finetuned PhysioMI L/R model, run 3 (the published checkpoint `checkpoint/physionet_finetuned_best.ckpt`),
test split (n = 596), attributions generated with `src/run_experiment.py --attribute_only`.

| File | Content |
|---|---|
| `aopc_table_run_3.csv` / `.tex` | Table A5: mean (SD across samples) and ratio to the random baseline (booktabs LaTeX) |
| `aopc_evaluation_results.json` | Full results: per-sample AOPC values, curves, breakdown by predicted class, legacy scores |
| `aopc_experiment_summary.csv` | Same results in tabular form |
| `aopc_evaluation_methodology.txt` | Exact procedure and settings |

Reproduce (adapt paths; `<base_dir>` holds `true_binary_physionet_pretrained_evaluation/run_3/...` with the
checkpoint and `attribution_results_test.parquet`):

```bash
uv run python script/run_aopc_eval_suite.py --suite physiomi --base_dir <base_dir> \
  --aopc_perturbation_strategy global_mean --per_sample_ordering prediction --global_ordering magnitude \
  --aopc_max_steps 10 --aopc_split test --seed 0
```
Paths in the result files are relative to the repository or to `<base_dir>`.

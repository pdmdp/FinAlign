# FinAlign follow-up evaluation

[Open the notebook in Colab](https://colab.research.google.com/github/pdmdp/FinAlign/blob/main/experiments/run_colab.ipynb)

Select query-expansion and lexical-fusion weights on 500 FiQA dev queries, then evaluate five systems on 648 test queries with a frozen configuration. The runner implements the [published plan](../demo/EXPERIMENTS.md); the implementation status recorded in that plan's section 9 predates this notebook.

The completed [follow-up results](../results/fiqa-dev-test-v1/README.md) show gains from query expansion and dev-selected lexical fusion, but no statistically established additional gain from monoT5. The full dev/test run and its statistical analysis have been checked from the saved query records.

## Run in Colab

1. Open the notebook and select **Runtime → Change runtime type → T4 GPU**.
2. Select **Runtime → Run all** and authorize the Drive mount. The default run folder is `MyDrive/FinAlign/experiments/followup-v1`.
3. Generate the report in section 8, then download the results ZIP in section 9. The checkpoints, logs and report also remain in Drive.

The notebook installs an isolated Python environment and Java JDK, downloads the fixed 218 MB index and checks every source/index file. It does not regenerate passage embeddings. BGE, Qwen and monoT5 run on the Colab GPU; no inference API, HF token, Space upload or hardware change is involved. Models use temporary Colab storage; query checkpoints use Drive storage. Free GPU availability and session lengths vary ([Colab FAQ](https://research.google.com/colaboratory/faq.html)).

The first model load takes longer than subsequent queries. The two-query dev check runs before selection; its cached components are reused. Progress is printed for each query. Cached recombination timings are not reported as end-to-end search latency.

## Resume an interrupted run

Reconnect with a T4 runtime and run the same notebook with the **same run name**. The notebook restores the saved package lock, checks the index and source revisions, and reuses completed queries. It never replaces an existing frozen configuration.

Use only one runtime per run folder. If source, package versions, Python version or GPU type differ, the runner stops instead of mixing computations. Restore the original environment to resume. If that is impossible, retain the incomplete run and start a separate run, documenting the change. A corrupted checkpoint produces an explicit error; do not overwrite the freeze or turn failed queries into zero scores. `last_failure.json` identifies the most recent failed query; later successful completion does not erase this record.

If only the report fails, rerunning section 8 reads the saved rankings without repeating inference. Each stage also retains its full output in `logs/<stage>.log`, and an error includes the last lines of that log. Section 9 can be run independently to download the existing checkpoints and diagnostics even after a failed report. A downloaded ZIP is not by itself evidence of completion: verify that all 648 test records and the numerical report are present.

## What is compared

| ID | System |
| :--- | :--- |
| L | Terrier BM25, descriptive reference |
| B | BGE with the original query |
| E | BGE with keyword and hypothetical-answer expansion |
| H | E fused with BM25 using the dev-selected alpha |
| F | monoT5 reranking of H's first 50 candidates |

Expansion weights are selected from the five settings in the plan, followed by a BM25 weight in `{0, .25, .5, .75, 1}`. The endpoints preserve the exact dense or BM25 ranking. If zero lexical weight wins, H equals E and the second comparison is retained with zero differences.

The three planned comparisons are **E−B, H−E and F−H** on per-query nDCG@10. The report uses 20,000 paired bootstrap samples, 100,000 two-sided sign randomizations and Holm correction across these three tests. The bootstrap intervals are marginal 95% intervals. Recall@10 is descriptive; no extra significance test against BM25 is added.

The original 648-query test set influenced earlier design decisions. This follow-up does not make it an untouched holdout. The new results must be described as a retrospective evaluation, separately from the saved coursework scores.

## Saved artifacts

| File or folder | Contents |
| :--- | :--- |
| `context.json`, `environment-lock.txt` | Source/index hashes, model revisions, actual packages and compute settings |
| `cache/components/` | Generated variants, query seeds, top-400 passage scores and BM25 scores |
| `dev_selection.json` | All grid means, per-query dev scores and tie decisions |
| `frozen_config.json`, `frozen_config.sha256` | Selected parameters and full context saved before test evaluation |
| `test_started.json` | UTC start time and frozen-config hash |
| `cache/rerank/`, `cache/results/` | Candidate-specific monoT5 scores and completed test records |
| `report/` | System means, paired comparisons, per-query CSVs, rankings, summary JSON, Markdown report and SVG figure |
| `logs/` | Stage output retained by the current Colab notebook, including subprocess tracebacks |

Checkpoint files have identities and payload checksums. An empty valid ranking receives zero; a failed computation stops the run. Reporting requires every planned test query, and saved metrics are checked against their rankings.

## Local execution

The retained GPU run used Python 3.13.16, a Java JDK and PyTorch 2.8.0 with CUDA 12.6; local checks used Python 3.11.8. For CUDA, install the matching CUDA 12.6 wheel before the remaining dependencies, as shown in the notebook. From the repository root:

```bash
python -m pip install -r experiments/requirements.txt
export PYTHONPATH="$PWD/experiments:$PWD/demo"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m unittest discover -s experiments/tests -v
python -m followup.runner smoke --index /path/to/fiqa-index --run experiments/runs/local --device cuda
python -m followup.runner dev --index /path/to/fiqa-index --run experiments/runs/local --device cuda
python -m followup.runner test --index /path/to/fiqa-index --run experiments/runs/local --device cuda
python -m followup.runner report --index /path/to/fiqa-index --run experiments/runs/local --device cuda
```

For a published run, pass `--source-revision` with the full source commit SHA to every stage. The Colab notebook does this automatically. `--device cpu` is available for local checks but is much slower for generation. There is no test-query subset flag or parameter-grid override.

The experiment imports the existing ranking/model code without starting the application. The original notebook and every file under `demo/` are unchanged. [Validation record](validation.json) distinguishes the preflight checks, completed Colab run and subsequent verification. The source commit pinned in the notebook is unchanged, so the logging and download improvements do not invalidate existing run checkpoints.

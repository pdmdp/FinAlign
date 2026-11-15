# FinAlign follow-up evaluation

[Open the notebook in Colab](https://colab.research.google.com/github/pdmdp/FinAlign/blob/main/experiments/run_colab.ipynb)

Select query-expansion and lexical-fusion weights on 500 FiQA dev queries, then evaluate five systems on 648 test queries with a frozen configuration. The runner implements the [published plan](../demo/EXPERIMENTS.md); the implementation status recorded in that plan's section 9 predates this notebook. The full experiment has not yet been run.

## Run in Colab

1. Open the notebook and select **Runtime → Change runtime type → T4 GPU**.
2. Select **Runtime → Run all** and authorize the Drive mount. The default run folder is `MyDrive/FinAlign/experiments/followup-v1`.
3. Download the results ZIP from the final cell. The checkpoints and report also remain in Drive.

The notebook installs an isolated Python environment and Java JDK, downloads the fixed 218 MB index and checks every source/index file. It does not regenerate passage embeddings. BGE, Qwen and monoT5 run on the Colab GPU; no inference API, HF token, Space upload or hardware change is involved. Models use temporary Colab storage; query checkpoints use Drive storage. Free GPU availability and session lengths vary ([Colab FAQ](https://research.google.com/colaboratory/faq.html)).

The first model load takes longer than subsequent queries. The two-query dev check runs before selection; its cached components are reused. Progress is printed for each query. Cached recombination timings are not reported as end-to-end search latency.

## Resume an interrupted run

Reconnect with a T4 runtime and run the same notebook with the **same run name**. The notebook restores the saved package lock, checks the index and source revisions, and reuses completed queries. It never replaces an existing frozen configuration.

Use only one runtime per run folder. If source, package versions, Python version or GPU type differ, the runner stops instead of mixing computations. Restore the original environment to resume. If that is impossible, retain the incomplete run and start a separate run, documenting the change. A corrupted checkpoint produces an explicit error; do not overwrite the freeze or turn failed queries into zero scores. `last_failure.json` identifies the most recent failed query; later successful completion does not erase this record.

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

Checkpoint files have identities and payload checksums. An empty valid ranking receives zero; a failed computation stops the run. Reporting requires every planned test query, and saved metrics are checked against their rankings.

## Local execution

Use Python 3.11 or 3.12, a Java JDK and PyTorch 2.8.0. For CUDA, install the matching CUDA 12.6 wheel before the remaining dependencies, as shown in the notebook. From the repository root:

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

All additions are under `experiments/`; the original notebook and every file under `demo/` are unchanged. The experiment imports the existing ranking/model code without starting the application. [Validation record](validation.json) distinguishes local checks from the pending full Colab run.

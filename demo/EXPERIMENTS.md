# Follow-up evaluation plan

Status: completed. The full 500-dev / 648-test run and all three paired comparisons are available in the [follow-up results](../results/fiqa-dev-test-v1/README.md). The [Colab runner](../experiments/README.md) implements the protocol below.

The plan was published before the follow-up parameter search and neural comparisons. Its [pre-run version at the source commit used by the notebook](https://github.com/pdmdp/FinAlign/blob/f0c006a328694de8c31bb09b398a19136f027032/demo/EXPERIMENTS.md) remains unchanged. This is not a preregistration of the original coursework, whose test results were already known. Sections 1–8 retain the planned method and report template.

The question is whether query expansion, lexical fusion and reranking improve retrieval under a fixed protocol. The primary metric is mean **nDCG@10**. There are exactly three planned inferential comparisons; all three will be reported, including null or negative results.

## 1. Data and separation of roles

Use the FiQA/BEIR **500 development queries** to select weights and the **648 test queries** for the frozen comparison. The prepared index currently contains all 500 and 648 queries with positive relevance judgments. Verify query IDs, split disjointness and counts before a run; stop on a mismatch rather than silently dropping queries.

The corpus is shared across splits: 57,600 nonempty documents and 66,113 passages. Reuse the validated Colab index, whose archive SHA-256 is `45b9da96081deb1f2f8c946125c413bafe7bad464fcc0b3c9ccec21300036366`. Retain the model revisions in [models.lock.json](models.lock.json), the index manifest and the actual package versions used in Colab. [Index validation](validation/colab-index.json) records the existing archive checks.

The original 648-query test set influenced earlier design choices. New development-only selection does not make it an untouched holdout again. This experiment is a retrospective follow-up with a better specified selection procedure; it cannot provide a wholly independent confirmation of the original design. A later evaluation on a previously unused collection would address that separate question.

## 2. Five systems, three comparisons

| ID | System | Role |
| :--- | :--- | :--- |
| `L` | Terrier BM25 | Descriptive lexical reference |
| `B` | BGE retrieval from the original question | Neural reference |
| `E` | BGE with keyword and hypothetical-answer variants | Query expansion |
| `H` | `E` combined with BM25 | Lexical fusion |
| `F` | monoT5 reranking of the first 50 results from `H` | Reranking |

| Comparison | Per-query difference | Question |
| :--- | :--- | :--- |
| C1 | `nDCG@10(E) − nDCG@10(B)` | Does the joint query-expansion stage help? |
| C2 | `nDCG@10(H) − nDCG@10(E)` | Does lexical fusion help after expansion? |
| C3 | `nDCG@10(F) − nDCG@10(H)` | Does reranking help the selected hybrid ranking? |

All tests are two-sided. C1 combines the two expansion types; it does not identify their separate effects. C2 may change the retrieved candidate set as well as the order. C3 uses exactly the same first 50 candidates before and after reranking, so candidate selection is held fixed for that comparison.

Report BM25 and all five system means, but do not add a fourth significance test for the overall system versus BM25. Additional comparisons, if explored later, must be labelled exploratory and kept separate from this family.

## 3. Keep the retrieval method fixed

Keep the BGE retrieval instruction, 250-word non-overlapping passages, 400 passages per variant, BM25 depth of 1,000 and monoT5 depth of 50. Use the existing query cleaning, prompts and query-derived generation seed. Do not tune models, prompts, passage size, candidate depth, generation settings or seeds in this experiment.

For each variant, min-max normalize the scores within its retrieved passages. Sum weighted scores by passage, treating missing scores as zero, then take the maximum passage score for each document. For lexical fusion, normalize the dense and lexical document scores separately, using their first 1,000 documents, and combine their union. Retain deterministic document-ID ordering for score ties.

The demo currently uses fixed variant weights of 0.70/0.15/0.15 and equal lexical/dense weights. The follow-up may select different values in its benchmark configuration; it does not change the live demo defaults or interface.

## 4. Development-set parameter search

First select the expansion weights using mean nDCG@10 across all 500 dev queries. Use only these five candidates, in the order `(original, keywords, hypothetical answer)`:

| Candidate | Original | Keywords | Hypothetical answer |
| :--- | ---: | ---: | ---: |
| Historical setting | 0.70 | 0.15 | 0.15 |
| More expansion | 0.50 | 0.25 | 0.25 |
| Less expansion | 0.85 | 0.075 | 0.075 |
| More keyword weight | 0.70 | 0.20 | 0.10 |
| More hypothetical-answer weight | 0.70 | 0.10 | 0.20 |

If means differ by at most `1e-12`, prefer the historical setting if tied; otherwise use the first tied candidate in this table. This yields one expansion setting, reused by `E`, `H` and `F`.

Next select the lexical weight `alpha` from **{0, 0.25, 0.50, 0.75, 1}**, again on all 500 dev queries, using:

```text
H(document) = (1 − alpha) × normalized_dense_score
              + alpha × normalized_BM25_score
```

At `alpha = 0`, return the dense ranking exactly; at `alpha = 1`, return BM25 exactly. Do not add zero-weight-only candidates at either endpoint. Equal weighting has the same ranking as the demo's unscaled score sum. If tied within `1e-12`, choose the smallest alpha.

Do not tune the weights separately after adding monoT5: `F` reranks the same selected `H`. If alpha zero wins, C2 is an identical-ranking comparison and must still be reported as such; do not substitute a more favourable comparison. The grid is fixed before looking at new test outputs and is not expanded in response to them.

## 5. Freeze before test evaluation

Write and retain a `frozen_config.json` containing:

- The plan and benchmark source commit, UTC freeze time and selected weights.
- All dev candidate means, the tie-breaking decisions and the ordered dev/test query IDs.
- The corpus/index hashes, model revisions and package versions.
- Generation prompts/settings and query-seed rule.
- The three comparisons, primary metric, resampling settings and reporting rules below.

Save the file and its SHA-256 before running or inspecting the new test results. Reuse that file for every test configuration. If a bug requires a correction, record it and rerun every affected system under the corrected frozen implementation; do not combine incompatible partial runs.

## 6. Test execution and retained outputs

Run all five systems over all 648 test queries. Save query IDs, generated variants, component scores, final rankings and per-query nDCG@10. Keep Recall@10 as a descriptive secondary metric. Do not use MAP as another primary endpoint: returned ranking depths differ, particularly after reranking.

An empty returned ranking receives nDCG@10 zero. A runtime failure is not a zero-score observation: checkpoint it, repair or resume the failed computation, and require a complete paired set before reporting results. Do not exclude difficult or unfavourable queries.

Generate each query's two expansions once and retain them. Cache each variant's top-400 passage scores and the BM25 ranking so dev weight combinations reuse the same inputs. Cache monoT5 scores for the selected candidates. Include query text, model revision, prompt/settings and seed in cache identities. Validate cache keys on resume.

One completed result per query and system is the unit for effectiveness. Repeated timings are not additional query observations. Cache-building times may be recorded for operational use, but cached recombination is not a measurement of end-to-end neural search latency.

## 7. Paired inference

For each comparison, form the vector of 648 per-query differences in the direction shown in the table. Report the mean difference, median difference and counts of improved, tied and worsened queries. Use absolute nDCG differences as the main effect size.

**Bootstrap intervals.** Draw 20,000 bootstrap samples of query IDs with replacement, preserving each query's paired system scores. Use the same sampled query indices for all three comparisons and random seed `2026`. Report percentile 95% intervals for the mean differences. These are marginal intervals for each comparison, not simultaneous family-wide intervals.

**Paired randomization test.** Under the null of exchangeable paired system scores, swap the two system scores independently within each query. Use 100,000 random sign assignments, seed `2027`, and the absolute mean difference as the two-sided statistic. Estimate the raw p-value as `(1 + number of randomized statistics at least as extreme as observed) / 100001`. Process permutations in batches to bound memory. If all differences are zero, report p = 1 and interval [0, 0].

**Multiple comparisons.** Apply Holm's correction to the three raw p-values as one family, with family-wise alpha 0.05. Report both raw and adjusted values. Use the adjusted values for the planned significance decisions; the marginal bootstrap intervals describe effect uncertainty and do not replace that correction.

Do not change to a one-sided test after seeing a direction, choose a different test because it gives significance, or count repeat runs as independent observations. A non-significant result means this procedure did not establish a difference; it does not prove equivalence.

## 8. Report template

Publish one table with the five system means and a second with exactly these rows:

| Comparison | Mean delta | 95% paired bootstrap CI | Improved / tied / worsened | Raw p | Holm-adjusted p |
| :--- | ---: | :--- | :--- | ---: | ---: |
| C1: expansion | pending | pending | pending | pending | pending |
| C2: lexical fusion | pending | pending | pending | pending | pending |
| C3: reranking | pending | pending | pending | pending | pending |

Include selected dev weights, complete per-query results and a plot of the difference distributions. Keep the historical notebook table separate. Discuss both losses and gains, and describe conclusions as conditional on this corpus, parameter search and previously used test set.

## 9. Original execution notes

These notes describe the tooling available when the plan was written. The dedicated runner and completed execution are now documented in [experiments/README.md](../experiments/README.md); the notes below are retained as historical context.

The existing `evaluate.py` already supports `--split dev/test`, fixed component configurations and per-query output. A small existing-script check is:

```bash
python evaluate.py --index data/fiqa-index --output runs/dev-smoke \
  --split dev --device cuda --limit 20 --repeats 1 \
  --configurations bm25,bge,bge_expanded,hybrid,full
```

This is only an environment check and does not implement the complete plan above. The current engine hard-codes the historical weights, and the script does not yet provide the weight search, resumable component cache, frozen-config enforcement or statistical analysis. Those belong in a dedicated benchmark/Colab runner, checked on dev or synthetic inputs before the full run. The 20-query smoke sample must not be used as a substitute for selecting weights on all 500 dev queries.

Run the full experiment in Colab or another available CUDA environment, not in the public Space's visitor callbacks. Save checkpoints and the frozen configuration outside Colab's temporary runtime before disconnecting. The existing passage index can be reused; there is no need to rebuild 66,113 embeddings. Free GPU availability and session duration are not guaranteed. No paid inference API or change to the Space hardware is part of this plan.

## References and provenance

- [FiQA dev and test definitions](https://ir-datasets.com/beir.html#beir/fiqa/dev).
- [Original notebook](../notebook/Fin_IR_Project.ipynb) and [recorded metrics](../results/recorded_metrics.json).
- [Cawley and Talbot: selection bias in performance evaluation](https://www.jmlr.org/papers/v11/cawley10a.html).

The original package environment and sampled hypothetical expansions were not retained. Even with the same method, new package, device and floating-point choices can change rankings. The frozen follow-up is a new execution, not a bit-for-bit replay of the historical scores.

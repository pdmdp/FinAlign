# Component ablations and latency

The interactive controls inspect one query at a time. They do not establish that a component improves retrieval across FiQA. `evaluate.py` runs the same components over a query split and saves each ranking, generated variant, seed and stage timing.

## Comparisons

| Configuration | What it measures |
| :--- | :--- |
| `bm25` | Lexical baseline |
| `bge` | Passage retrieval from the original query |
| `bge_keywords` | Effect of technical keyword expansion |
| `bge_hyde` | Effect of hypothetical expansion |
| `bge_expanded` | Both query expansions |
| `hybrid` | Added lexical fusion |
| `full` | Added monoT5 reranking |
| `full_without_keywords` | Remove keyword expansion from the full pipeline |
| `full_without_hyde` | Remove hypothetical expansion from the full pipeline |
| `full_without_bm25` | Remove lexical fusion from the full pipeline |

`hybrid` is also the full system without reranking. Disabled query variants contribute zero; remaining weights are not retuned. The original 0.70/0.15/0.15 weights and candidate depths remain fixed. Removing a component can change the candidate set; this is part of its measured effect, not a controlled reranker-only comparison.

## Run

Start with a small development-set run to check the environment:

```bash
python evaluate.py --index data/fiqa-index --output runs/dev-smoke \
  --split dev --device cuda --limit 20 --repeats 1
```

For the full development set, omit `--limit`. The default uses three timed repeats after one warm-up per configuration. The original course test set influenced design choices, so it should not be described as an untouched evaluation set for the later work. Choose new parameters on development queries, freeze them, and report test performance with that limitation.

Do not run the benchmark inside a visitor callback or assume a full evaluation fits within a daily ZeroGPU allowance. Use an existing research GPU or an available free notebook session, and retain the output between sessions.

Each run writes:

- `run.json`: query IDs, model revisions, environment, index provenance and initialization time.
- `queries.jsonl`: complete returned rankings, binary FiQA metrics, query variants, seeds and per-stage timings. Repeat 0 is warm-up.
- `summary.json`: mean effectiveness over unique queries and latency p50/p95 across measured repetitions.

The metrics are nDCG@10, AP and Recall@10. AP is limited by the returned ranking: monoT5 returns at most 50 documents, BM25 up to 1,000, and dense retrieval up to 400 passages per variant before document aggregation. This matches the candidate-depth issue in the notebook; report it alongside MAP comparisons.

GPU work is synchronized before and after stage measurements. Benchmark timings exclude model loading, GPU queueing and browser transfer. The interactive page reports processing time and server elapsed time (including the wait for GPU allocation) separately; browser transfer is not measured. It includes BM25 because it displays the baseline alongside FinAlign; benchmark totals exclude the baseline from configurations that do not use lexical fusion. Indexing is offline and should be reported separately.

Repeated timings are not independent relevance observations. Compare per-query deltas, inspect losses as well as gains, and use a paired uncertainty estimate before making claims about small metric differences. No new full-benchmark improvement or state-of-the-art claim is made by adding this demo.

## Differences from the course run

The original notebook is retained unchanged. The demo extracts its ranking method into modules, uses exact NumPy dot products over the BGE passage vectors, pins model revisions and sets a query-derived seed for sampled hypothetical expansions. Terrier still provides BM25 and `pyterrier_t5` still provides monoT5 scoring.

The original package environment was not locked, and the original hypothetical expansions were not retained. This implementation is a new execution of the method, not a bit-for-bit replay of the historical scores. Package, device and floating-point differences can change rankings. Original recorded metrics remain in the root repository’s `results/` directory.

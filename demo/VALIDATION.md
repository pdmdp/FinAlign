# Validation

Checked on 15 November 2026 with Python 3.11.8 on an Apple M1 Mac. The Space targets Python 3.12.12 and ZeroGPU; its hosted run must be checked after upload and index preparation.

## Completed locally

- Eleven automated tests cover passage-level fusion before document aggregation, lexical-only candidates, constant scores, chunk boundaries, query validation, metric agreement with `ir_measures`, HTML escaping, archive extraction, checksums and resuming an interrupted index build.
- Gradio 6.32.0 starts and serves its page and configuration over HTTP. The preparation callback rejects a missing owner login in Space mode.
- The pinned BGE encoder loads and produces 768-dimensional query vectors.
- The pinned monoT5 checkpoint scores a short relevance example correctly using `pyterrier_t5` and the selected package versions. This is an execution check, not a benchmark evaluation.
- BM25 was rerun over all **648 FiQA test queries** with the complete nonempty corpus. nDCG@10 is **0.2526335752535022**, MAP is **0.21041877427079766**, and Recall@10 is **0.3097077648003574**. The first two match the recorded notebook values to the reported precision. The [run summary](validation/bm25-test.json) records the configuration and measured CPU timings.

## Colab preparation

The preparation notebook's Python cells and worker compile and pass Ruff checks. Its four source downloads were checked against their pinned SHA-256 digests. The corpus URL resolves, its local archive matches the expected digest, and the downloaded index reader verifies all corpus files and counts (57,600 documents; 66,113 passages). The complete notebook has not yet been executed on a Colab GPU.

## Remaining hosted checks

Complete all 66,113 passage embeddings and upload the archive, verify custom and benchmark queries, and run the combined Qwen/BGE/monoT5 pipeline. Check GPU allocation, free-quota errors, startup after an index upload, and the browser layout. Qwen generation and the full neural benchmark have not been run locally for this update.

The interactive controls and evaluation script are implemented. A complete ablation study, statistical comparison and a latency study on fixed GPU hardware remain experiments to run; they are not claimed as completed results.

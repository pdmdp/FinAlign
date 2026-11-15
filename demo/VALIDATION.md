# Validation

Checked on 15 November 2026 with Python 3.11.8 on an Apple M1 Mac, an index prepared on a Colab T4, and a manual browser check of the hosted ZeroGPU application running Python 3.12.12.

## Completed locally

- Eleven automated tests cover passage-level fusion before document aggregation, lexical-only candidates, constant scores, chunk boundaries, query validation, metric agreement with `ir_measures`, HTML escaping, archive extraction, checksums and resuming an interrupted index build.
- Gradio 6.32.0 starts and serves its page and configuration over HTTP. The preparation callback rejects a missing owner login in Space mode.
- The pinned BGE encoder loads and produces 768-dimensional query vectors.
- A BM25/BGE query completes through the Gradio interface with the full Colab index. Its results, query metrics and measured stages render without browser JavaScript errors; the completed index opens directly in search mode.
- The pinned monoT5 checkpoint scores a short relevance example correctly using `pyterrier_t5` and the selected package versions. This is an execution check, not a benchmark evaluation.
- BM25 was rerun over all **648 FiQA test queries** with the complete nonempty corpus. nDCG@10 is **0.2526335752535022**, MAP is **0.21041877427079766**, and Recall@10 is **0.3097077648003574**. The first two match the recorded notebook values to the reported precision. The [run summary](validation/bm25-test.json) records the configuration and measured CPU timings.

## Colab preparation

The notebook completed on a Colab T4 and produced the full index: **66,113 × 768 float32 vectors** for passages from 57,600 documents. Encoding took **1,085 seconds (18.1 minutes)**, excluding installation, downloads and packaging.

Every manifest checksum passed after downloading the archive. Corpus files, passage IDs and the model lock match the bootstrap archive; all vectors are finite and nonzero. Five passages re-encoded locally with the pinned BGE model had cosine similarity above 0.9999 with their Colab vectors. The [archive validation](validation/colab-index.json) records the archive digest, counts and package versions.

All three fixed demo queries completed against the full index. Their [results](validation/colab-retrieval.json) include a query with nDCG@10 of zero. These are execution checks on the existing examples, not a neural benchmark or a test of Qwen and monoT5.

## Hosted application

The uploaded index matches the validated archive byte for byte. After a rebuild, the application opens directly in search mode. The Space uses `default-jdk-headless`: the earlier JRE-only package left Pyjnius unable to locate `javac` during BM25 initialization.

A manual browser check completed the query **“what is a Tesla share?”** with both Qwen expansions, BM25/BGE fusion and monoT5 reranking enabled. Both rankings were displayed. The interface reported **2.10 seconds** of processing and **4.30 seconds** elapsed on the server, including GPU wait. These are timings from one request, not a latency benchmark. The [run record](validation/hosted-search.json) identifies the deployed revision and settings.

This confirms execution of the combined pipeline on a custom query. Hosted relevance metrics for the fixed FiQA examples and the behaviour when search quota is exhausted still need separate checks. Qwen generation and the full neural benchmark have not been run locally for this update.

The interactive controls and evaluation script are implemented. A complete ablation study, statistical comparison and a latency study on fixed GPU hardware remain experiments to run; they are not claimed as completed results.

---
title: FinAlign
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: 3.12.12
app_file: app.py
pinned: true
license: mit
short_description: Compare lexical and neural search on financial questions.
models:
  - BAAI/bge-base-en-v1.5
  - Qwen/Qwen2.5-1.5B-Instruct
  - castorini/monot5-base-msmarco
datasets:
  - BeIR/fiqa
tags:
  - information-retrieval
  - finance
  - query-expansion
  - reranking
startup_duration_timeout: 1h
hf_oauth: true
---

# FinAlign

[Open the demo](https://huggingface.co/spaces/pdmdp/finalign) · [Source and report](https://github.com/pdmdp/FinAlign)

Compare BM25 with BGE passage retrieval on the FiQA corpus. Turn on Qwen query expansion, lexical fusion and monoT5 reranking to inspect how the ranking and latency change.

## Try it

The [public demo](https://huggingface.co/spaces/pdmdp/finalign) is free to try in your browser, with no installation or API key. Click an example question to run it, or enter an English question and select **Compare rankings**. Use **All stages** to enable the complete pipeline, then run the comparison again. **Run details** shows generated query variants and stage timings.

Both rankings appear side by side on desktop; tap a method name to switch lists on mobile. Shared ZeroGPU queues and daily usage limits apply. If the neural search reaches its allowance, the BM25 results remain visible.

## Method

The demo follows the [FinAlign course project](https://github.com/pdmdp/FinAlign). It uses the complete nonempty corpus: 57,600 documents split into 66,113 passages. Query variants are intermediate retrieval inputs, not answers to the visitor.

Rankings and timings are computed on demand. Quality metrics appear only when a question exactly matches a uniquely identified FiQA query with relevance judgments. The historical notebook results are separate from these runs.

See [DEPLOY.md](DEPLOY.md) for setup, [EXPERIMENTS.md](EXPERIMENTS.md) for ablations and [DATA.md](DATA.md) for data and model provenance.

Original project: **Matteo Pietro Di Pilato, Lorenzo Cinquemani and Lorenzo Marchioni**. Matteo implemented the retrieval pipelines and ran the original evaluation. The interactive version is a later extension.

Code is MIT licensed. The bundled FiQA collection and derived index are distributed under CC BY-SA 4.0; pretrained models retain their respective licenses.

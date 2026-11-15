# Data and models

The demo uses the [FiQA retrieval collection distributed by BEIR](https://huggingface.co/datasets/BeIR/fiqa), loaded through `ir_datasets` as `beir/fiqa`. Its dataset card specifies [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The original collection has 57,638 documents; filtering empty texts leaves 57,600.

The index bundle contains document texts, 250-word passage identifiers, BGE embeddings, a Terrier lexical index, and the development/test queries with their relevance judgments. These are derived artifacts of FiQA, distributed under CC BY-SA 4.0. The document IDs and text are retained; whitespace is collapsed only in passages. Each file has a SHA-256 digest in `manifest.json`.

The corpus is a historical benchmark containing public financial discussion, including posts unrelated to a visitor’s query. It is not a source of current market data or financial advice.

References:

- Maia et al., *WWW'18 Open Challenge: Financial Opinion Mining and Question Answering*, 2018.
- Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*, [arXiv:2104.08663](https://arxiv.org/abs/2104.08663), 2021.

The bundle’s `models.json` fixes the model revisions used for indexing and inference:

| Role | Model | License |
| :--- | :--- | :--- |
| Passage and query encoder | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | MIT |
| Query expansion | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Apache 2.0 |
| Reranking | [castorini/monot5-base-msmarco](https://huggingface.co/castorini/monot5-base-msmarco) | Not specified on the model card |
| monoT5 tokenizer | [google-t5/t5-base](https://huggingface.co/google-t5/t5-base) | Apache 2.0 |

Weights are downloaded from their publishers at startup and are not included in the upload package. The monoT5 scoring implementation is provided by [pyterrier_t5](https://github.com/terrierteam/pyterrier_t5), the same library used by the notebook, pinned to a commit in `requirements.txt`.

# Experiment details


1. **Lexical baselines:** TF-IDF, BM25 and BM25 with RM3 feedback.
2. **Dense retrieval:** `BAAI/bge-base-en-v1.5`, with a retrieval instruction prepended to queries.
3. **Query variants:** `Qwen/Qwen2.5-1.5B-Instruct` generates technical keywords and hypothetical answers. The instruction query and the two variants receive weights of 0.70, 0.15 and 0.15 after per-query score normalization.
4. **Passage retrieval:** documents are split into non-overlapping 250-word chunks. Each query variant retrieves up to 400 chunks; scores are combined and the highest-scoring passage represents each document.
5. **Re-ranking:** the top 50 documents from the BM25/dense fusion are passed to `castorini/monot5-base-msmarco`.

The task is ranking documents. Generated answers are used to expand retrieval queries; the project does not evaluate an answer-generation system.

## Data

The notebook loads [`irds:beir/fiqa/test`](https://ir-datasets.com/beir.html#beir/fiqa/test) through PyTerrier. The source collection has 57,638 documents, 648 test queries and 1,706 relevance judgments. Empty text is filtered and judgments are restricted to retained documents and queries. The corpus and model weights are downloaded at run time and are not stored here.

## Run

The experiments were developed in Google Colab with a CUDA GPU. The dense encoder explicitly uses `device="cuda"`; CPU execution requires changing that setting and will be slower. A Java installation is also needed for Terrier; see the [PyTerrier installation guide](https://pyterrier.readthedocs.io/en/latest/installation.html).

```bash
git clone https://github.com/pdmdp/FinAlign.git
cd FinAlign
python -m pip install -r requirements.txt
python -m jupyter lab notebook/Fin_IR_Project.ipynb
```

Run the cells in order in a dedicated working directory. Indexing cells rebuild the named local index folders. The package list records the dependencies used by the source; a complete lockfile from the original Colab environment was not retained. HyDE generation uses sampling, so regenerated query variants and scores can differ. The full GPU pipeline has not been rerun for this repository update.

The README figure can be regenerated without downloading data or models:

```bash
python -m pip install matplotlib
python scripts/plot_results.py
```

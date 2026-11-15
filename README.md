# FinAlign

Financial document retrieval on the FiQA benchmark: BM25, dense retrieval with BGE, query expansion with Qwen, and monoT5 re-ranking.

The question behind the project is whether semantic retrieval and generated query variants help when a financial question uses different vocabulary from the relevant answer. The experiments compare each stage against lexical baselines.

[Notebook](notebook/Fin_IR_Project.ipynb) · [Report](report/FinalReportIRProject.pdf) · [Slides](FinAlign-Slides.pdf)

## Results

![Recorded nDCG at 10 for six retrieval configurations](assets/retrieval-results.png)

| Configuration | nDCG@10 | MAP |
| :--- | ---: | ---: |
| TF-IDF | 0.2533 | 0.2096 |
| BM25 | 0.2526 | 0.2104 |
| BM25 + RM3 | 0.2454 | 0.2065 |
| BGE chunks + query expansion | 0.4159 | **0.3569** |
| BM25 + BGE chunks + query expansion | 0.4040 | 0.3410 |
| Hybrid candidates + monoT5 | **0.4213** | 0.3561 |

These are the saved notebook results on the 648 FiQA test queries. The dense chunk pipeline improves substantially over BM25 in this run. Adding BM25 alone lowers both metrics; re-ranking the hybrid candidate set gives the highest nDCG@10, while the dense pipeline retains the highest MAP among the configurations above.

The same query set was used throughout the experiments, including comparisons that informed the final design. These scores are exploratory benchmark results; they do not constitute an independent final validation. The notebook does not establish statistical significance for every neural comparison. [Machine-readable values and provenance](results/recorded_metrics.json) are included.

## Retrieval pipeline

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

## Authors

Course project for Information Retrieval and Recommender Systems, BSc in Artificial Intelligence: **Matteo Pietro Di Pilato, Lorenzo Cinquemani and Lorenzo Marchioni**. Matteo implemented the retrieval pipelines and ran the experimental evaluation.

Code is distributed under the [MIT license](LICENSE). The datasets and pretrained models retain their own licenses.

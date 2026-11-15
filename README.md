# FinAlign

Can semantic retrieval find relevant financial answers that keyword search misses? FinAlign compares BM25 with a passage retriever, then tests how query expansion, lexical fusion and reranking change the ranking.

In the follow-up evaluation, the complete pipeline reached **nDCG@10 of 0.4294**, compared with **0.2526 for BM25**. Query expansion and a dev-selected BM25 weight improved the neural ranking; monoT5's additional gain was small and not statistically significant.

## Try the live demo

[![Try the live demo](assets/try-demo.svg)](https://huggingface.co/spaces/pdmdp/finalign)

**Public and free to try:** pick an example, or enter an English question and click **Compare rankings**; select **All stages** and rerun to try the complete pipeline.

<a href="https://huggingface.co/spaces/pdmdp/finalign">
  <img src="assets/demo-preview.png" alt="Open FinAlign: BM25 and BGE results for a FiQA question, captured from a local CPU run" width="640">
</a>

## How the search works

BM25 matches the question's terms against whole documents. The neural pipeline splits documents into 250-word passages and uses `bge-base-en-v1.5` to retrieve passages by meaning. `Qwen2.5-1.5B-Instruct` supplies two extra search inputs: financial keywords and a hypothetical answer. The original query and the two variants receive weights of 0.70, 0.15 and 0.15 after score normalization.

Each document is represented by its best passage. Its dense score can be combined with BM25, and `monot5-base-msmarco` reranks the first 50 candidates. The follow-up selected 75% dense score and 25% BM25 score after normalization. The demo retains the original equal-weight fusion. Generated text is used to search the collection; the system returns ranked documents rather than a generated answer.

[Notebook](notebook/Fin_IR_Project.ipynb) · [Report](report/FinalReportIRProject.pdf) · [Slides](FinAlign-Slides.pdf) · [Method and setup](EXPERIMENT.md)

## Results

Parameters were selected on **500 dev queries**, frozen, then evaluated on **648 test queries**. Those test queries had already influenced the original project, so this is a retrospective evaluation rather than an untouched holdout.

| Configuration | nDCG@10 | Recall@10 |
| :--- | ---: | ---: |
| BM25 | 0.2526 | 0.3097 |
| BGE, original query | 0.4094 | 0.4830 |
| BGE + query expansion | 0.4173 | 0.4896 |
| Expanded BGE + BM25, alpha 0.25 | 0.4287 | 0.5047 |
| Hybrid candidates + monoT5 | **0.4294** | **0.5073** |

The three planned paired comparisons found gains from expansion (**+0.0080 nDCG@10; Holm p = 0.0076**) and lexical fusion (**+0.0114; p = 0.0094**). Reranking added **+0.0007; p = 0.9281**: this run did not establish an additional gain. Recall@10 is descriptive.

[Results, intervals and interpretation](results/fiqa-dev-test-v1/README.md) · [Per-query scores](results/fiqa-dev-test-v1/per_query.csv) · [Run the experiment in Colab](https://colab.research.google.com/github/pdmdp/FinAlign/blob/main/experiments/run_colab.ipynb)

The [original coursework results](results/recorded_metrics.json) remain separate: the final pipeline scored 0.4213 nDCG@10, and equal-weight lexical fusion lowered the dense pipeline's score. The follow-up supports a narrower conclusion: lexical weighting matters, and the 25% setting selected on dev helped in this run. Live demo searches are separate executions.

## Contribution

**Matteo Pietro Di Pilato, Lorenzo Cinquemani and Lorenzo Marchioni** · Information Retrieval and Recommender Systems, BSc in Artificial Intelligence. Matteo implemented the retrieval pipelines and ran the evaluation. The interactive application is a later extension.

## Limits and next steps

The paired tests account for variation across queries and the three planned comparisons, but they do not remove the effect of earlier test-set use. There is one collection and one deterministic generation setting; a new collection and repeated generation runs would test whether these findings hold more broadly. The monoT5 result does not establish equivalence or rule out a gain in another setting.

The [follow-up notebook](experiments/README.md) saves query checkpoints and retains the frozen configuration. A useful next experiment would evaluate the fixed configurations on previously unused data and measure end-to-end latency, particularly the cost of reranking.

The demo searches 57,600 nonempty documents and 66,113 passages from the historical FiQA collection. Shared ZeroGPU queues and daily limits apply; BM25 remains available when the neural allowance is exhausted. These documents and generated query variants are not current financial advice.

## Run

```bash
git clone https://github.com/pdmdp/FinAlign.git
cd FinAlign
python -m pip install -r requirements.txt
python -m jupyter lab notebook/Fin_IR_Project.ipynb
```

The notebook expects CUDA and a Java JDK. See [notebook setup](EXPERIMENT.md) for the original environment and indexing behaviour, or [demo setup](demo/DEPLOY.md) to run the application. [Validation](demo/VALIDATION.md) records the demo checks. Recreate the follow-up figure with `python experiments/plot_results.py`; the coursework figure uses `python scripts/plot_results.py`. Both plotting scripts require Matplotlib.

Code: [MIT](LICENSE). Datasets and pretrained models retain their own licenses.

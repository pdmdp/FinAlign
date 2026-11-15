"""Read-only use of the existing retrieval implementation, outside Gradio."""

import time

from finalign.core import INSTRUCTION, clean_query, dense_top_k, query_seed
from finalign.index import LexicalSearch


class Backend:
    def __init__(self, index, device):
        self.index, self.device = index, device
        self.lexical = LexicalSearch(index)
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            from finalign.neural import NeuralSearch
            self._engine = NeuralSearch(self.index, self.device)
        return self._engine

    def components(self, row):
        import torch
        query = clean_query(row["query"])
        start = time.perf_counter()
        lexical, _ = self.lexical.search(query)
        engine = self.engine
        torch.manual_seed(query_seed(query))
        variants = {"original": query, "keywords": engine.expansion(query, "keywords"),
                    "hyde": engine.expansion(query, "hyde")}
        vectors = engine.encoder.encode([INSTRUCTION + q for q in variants.values()],
                                        normalize_embeddings=False, show_progress_bar=False)
        passages = {name: dense_top_k(self.index.embeddings, vector, self.index.passage_ids)
                    for name, vector in zip(variants, vectors)}
        engine.synchronize()
        return {"variants": variants, "seed": query_seed(query), "passages": passages,
                "lexical": lexical, "build_seconds": time.perf_counter() - start}

    def rerank(self, row, candidates):
        if not candidates:
            return {}
        import pandas as pd
        frame = pd.DataFrame([
            {"qid": row["qid"], "query": clean_query(row["query"]), "docno": doc,
             "score": score, "rank": position, "text": self.index.documents[doc]}
            for position, (doc, score) in enumerate(candidates)])
        output = self.engine.reranker.transform(frame)
        return {str(r.docno): float(r.score) for r in output.itertuples()}

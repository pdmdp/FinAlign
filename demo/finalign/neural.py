"""BGE, Qwen expansion and monoT5 with the notebook's ranking operations."""

import time
from contextlib import contextmanager

import pandas as pd
import torch

from .core import (
    INSTRUCTION,
    RERANK_DEPTH,
    Options,
    dense_top_k,
    fuse_passages,
    hybrid_scores,
    query_seed,
    ranked,
)


class NeuralSearch:
    def __init__(self, index, device="cpu", encoder_only=False):
        from sentence_transformers import SentenceTransformer
        self.index, self.device = index, device
        self.encoder_only = encoder_only
        model = index.models["encoder"]
        self.encoder = SentenceTransformer(model["id"], revision=model["revision"], device=device)
        self.encoder.max_seq_length = index.manifest["encoder_max_tokens"]
        self.encoder.eval()
        if encoder_only:
            return
        from huggingface_hub import snapshot_download
        from pyterrier_t5 import MonoT5ReRanker
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        expander = index.models["expander"]
        tokenizer = AutoTokenizer.from_pretrained(expander["id"], revision=expander["revision"])
        tokenizer.padding_side = "left"
        llm = AutoModelForCausalLM.from_pretrained(
            expander["id"], revision=expander["revision"],
            torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
        self.expander = pipeline("text-generation", model=llm, tokenizer=tokenizer, device=device)
        reranker = index.models["reranker"]
        t5_tokenizer = index.models["t5_tokenizer"]
        tokenizer_path = snapshot_download(t5_tokenizer["id"], revision=t5_tokenizer["revision"],
                                           allow_patterns=["*.json", "*.model", "*.txt"])
        reranker_path = snapshot_download(reranker["id"], revision=reranker["revision"],
                                          allow_patterns=["*.json", "*.safetensors", "pytorch_model.bin"])
        self.reranker = MonoT5ReRanker(tok_model=tokenizer_path, model=reranker_path,
                                       device=device, batch_size=4, verbose=False)

    def synchronize(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    @contextmanager
    def measure(self, timings, name):
        self.synchronize()
        start = time.perf_counter()
        yield
        self.synchronize()
        timings[name] = (time.perf_counter() - start) * 1000

    def expansion(self, query, kind):
        if kind == "keywords":
            system = ("You are a financial IR assistant. Expand the query with technical "
                      "synonyms, tickers, and metrics. Output ONLY keywords separated by spaces. No sentences.")
            user, length, sample = f"Expand: {query}", 50, False
        else:
            system = "You are a financial analyst. Write a sentence hypothetical answer that answers this question."
            user, length, sample = f"Question: {query}", 100, True
        settings = {"max_new_tokens": length, "do_sample": sample, "return_full_text": False}
        if sample:
            settings["temperature"] = 0.7
        output = self.expander([{"role": "system", "content": system},
                                {"role": "user", "content": user}], **settings)
        text = output[0]["generated_text"].strip()
        return f"{query} {text}" if kind == "keywords" else text

    def search(self, query, lexical, options):
        self.index.refresh_dense()
        if not self.index.ready:
            raise ValueError("Prepare the passage index before searching.")
        options = Options(**options) if isinstance(options, dict) else options
        if self.encoder_only and (options.keywords or options.hyde or options.rerank):
            raise ValueError("This local session loaded only the encoder.")
        timings, variants = {}, {"original": query}
        torch.manual_seed(query_seed(query))
        if options.keywords:
            with self.measure(timings, "Keyword expansion"):
                variants["keywords"] = self.expansion(query, "keywords")
        if options.hyde:
            with self.measure(timings, "Hypothetical expansion"):
                variants["hyde"] = self.expansion(query, "hyde")
        with self.measure(timings, "Query encoding"):
            vectors = self.encoder.encode([INSTRUCTION + q for q in variants.values()],
                                           normalize_embeddings=False, show_progress_bar=False)
        with self.measure(timings, "Passage search and fusion"):
            weights = {"original": 0.70, "keywords": 0.15, "hyde": 0.15}
            runs = [(weights[name], dense_top_k(self.index.embeddings, vector, self.index.passage_ids))
                    for name, vector in zip(variants, vectors)]
            dense, passages = fuse_passages(runs)
            scores = hybrid_scores(dense, lexical) if options.lexical else dense
        if options.rerank and scores:
            with self.measure(timings, "monoT5 reranking"):
                candidates = ranked(scores, RERANK_DEPTH)
                frame = pd.DataFrame([
                    {"qid": "interactive", "query": query, "docno": docno, "score": score,
                     "rank": position, "text": self.index.documents[docno]}
                    for position, (docno, score) in enumerate(candidates)])
                output = self.reranker.transform(frame)
                scores = {str(row.docno): float(row.score) for row in output.itertuples()}
        return {"ranking": ranked(scores), "passages": passages, "variants": variants,
                "timings_ms": timings, "options": options.to_dict(), "seed": query_seed(query)}

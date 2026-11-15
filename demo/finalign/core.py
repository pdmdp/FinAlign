"""Ranking operations shared by the demo and the evaluation script."""

import hashlib
import math
import re
from dataclasses import asdict, dataclass

import numpy as np

INSTRUCTION = "Represent this sentence for searching relevant passages: "
CHUNK_WORDS = 250
PASSAGE_DEPTH = 400
LEXICAL_DEPTH = 1000
RERANK_DEPTH = 50


@dataclass(frozen=True)
class Options:
    keywords: bool = False
    hyde: bool = False
    lexical: bool = False
    rerank: bool = False

    def to_dict(self):
        return asdict(self)


def clean_query(query):
    return re.sub(r"\s+", " ", str(query).replace(":", " ")).strip()


def validate_query(query):
    query = clean_query(query)
    if not query or not any(c.isalnum() for c in query):
        raise ValueError("Enter a question with at least one word or number.")
    if len(query) > 600:
        raise ValueError("Keep the question within 600 characters.")
    return query


def query_seed(query):
    return int.from_bytes(hashlib.sha256(query.encode()).digest()[:4], "big")


def chunks(documents):
    for docno, text in documents.items():
        words = text.split()
        for number, start in enumerate(range(0, len(words), CHUNK_WORDS)):
            yield f"{docno}%{number}", " ".join(words[start:start + CHUNK_WORDS])


def ranked(scores, limit=None):
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def min_max(scores):
    if not scores:
        return {}
    if not all(math.isfinite(float(v)) for v in scores.values()):
        raise ValueError("Non-finite retrieval score.")
    lo, hi = min(scores.values()), max(scores.values())
    return {k: (float(v) - lo) / (hi - lo) if hi > lo else 1.0
            for k, v in scores.items()}


def fuse_passages(runs):
    """Normalize each variant before combining passages, then take max per doc."""
    combined = {}
    for weight, scores in runs:
        for passage, score in min_max(scores).items():
            combined[passage] = combined.get(passage, 0.0) + weight * score
    documents, passages = {}, {}
    for passage, score in ranked(combined):
        docno = passage.rsplit("%", 1)[0]
        if docno not in documents:
            documents[docno] = score
            passages[docno] = passage
    return documents, passages


def hybrid_scores(dense, lexical):
    output = min_max(dict(ranked(dense, LEXICAL_DEPTH)))
    for docno, score in min_max(dict(ranked(lexical, LEXICAL_DEPTH))).items():
        output[docno] = output.get(docno, 0.0) + score
    return output


def dense_top_k(embeddings, vector, passage_ids, k=PASSAGE_DEPTH):
    scores = np.asarray(embeddings @ vector, dtype=np.float32)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite dense scores.")
    # Stable ordering keeps ties reproducible across repeated runs.
    order = np.argsort(-scores, kind="stable")[:k]
    return {passage_ids[i]: float(scores[i]) for i in order}


def metrics(ranking, judgments, k=10):
    """Binary FiQA judgments; AP uses the complete supplied ranking."""
    relevant = {d for d, rel in judgments.items() if rel > 0}
    if not relevant:
        return None
    hits, precision_sum = 0, 0.0
    for rank, docno in enumerate(ranking, 1):
        if docno in relevant:
            hits += 1
            precision_sum += hits / rank
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranking[:k])
              if d in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return {"nDCG@10": dcg / ideal,
            "AP": precision_sum / len(relevant),
            "Recall@10": len(set(ranking[:k]) & relevant) / len(relevant)}

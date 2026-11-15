"""Build the full FiQA index once; never run indexing inside a visitor request."""

import argparse
import hashlib
import json
import time
import zipfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from finalign.core import CHUNK_WORDS, chunks, clean_query

MODELS = {
    "encoder": "BAAI/bge-base-en-v1.5",
    "expander": "Qwen/Qwen2.5-1.5B-Instruct",
    "reranker": "castorini/monot5-base-msmarco",
    "t5_tokenizer": "google-t5/t5-base",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def build(args):
    import ir_datasets
    import numpy as np
    import pyterrier as pt
    import torch
    from sentence_transformers import SentenceTransformer

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise ValueError("This index is already complete. Choose a new output directory.")
    print("Loading the FiQA corpus", flush=True)
    source = ir_datasets.load("beir/fiqa/test")
    documents = {str(d.doc_id): d.text for d in source.docs_iter()
                 if d.text is not None and d.text.strip()}
    with (out / "documents.jsonl").open("w") as stream:
        for docno, text in documents.items():
            stream.write(json.dumps({"docno": docno, "text": text}, ensure_ascii=False) + "\n")
    passages = list(chunks(documents))
    write_json(out / "passages.json", [pid for pid, _ in passages])
    print(f"{len(documents):,} documents; {len(passages):,} passages", flush=True)

    queries = {}
    for split in ("dev", "test"):
        ds = ir_datasets.load(f"beir/fiqa/{split}")
        qrels = {}
        for row in ds.qrels_iter():
            if str(row.doc_id) in documents:
                qrels.setdefault(str(row.query_id), {})[str(row.doc_id)] = int(row.relevance)
        queries[split] = [{"qid": str(q.query_id), "query": clean_query(q.text),
                           "qrels": qrels.get(str(q.query_id), {})}
                          for q in ds.queries_iter() if q.text and clean_query(q.text)]
    write_json(out / "queries.json", queries)

    lock_path = out / "models.json"
    if lock_path.exists():
        model_lock = json.loads(lock_path.read_text())
    else:
        model_lock = json.loads(Path(__file__).with_name("models.lock.json").read_text())
        if any(model_lock[key]["id"] != name for key, name in MODELS.items()):
            raise ValueError("The model lock does not match this implementation.")
        write_json(lock_path, model_lock)

    pt.terrier.set_version("5.11")
    pt.terrier.set_helper_version("0.0.8")
    if not (out / "terrier" / "data.properties").exists():
        print("Building the Terrier BM25 index", flush=True)
        indexer = pt.IterDictIndexer(str(out / "terrier"), meta={"docno": 20},
                                    text_attrs=["text"], meta_reverse=["docno"],
                                    pretokenised=False, fields=False, threads=1)
        indexer.index({"docno": docno, "text": text} for docno, text in documents.items())

    # Java writes the local build path as a comment; it is not part of the index.
    properties = out / "terrier" / "data.properties"
    properties.write_text("\n".join(line for line in properties.read_text().splitlines()
                                    if not line.startswith("#/")) + "\n")

    dimensions = 768
    if not args.corpus_only:
        model = model_lock["encoder"]
        encoder = SentenceTransformer(model["id"], revision=model["revision"], device=args.device)
        encoder.eval()
        encoder.max_seq_length = 512
        dimensions = encoder.get_embedding_dimension()
        progress_path = out / "progress.json"
        offset = json.loads(progress_path.read_text())["completed"] if progress_path.exists() else 0
        mode = "r+" if offset else "w+"
        vectors = np.lib.format.open_memmap(out / "embeddings.npy", mode=mode, dtype="float32",
                                            shape=(len(passages), dimensions))
        start = time.perf_counter()
        for position in range(offset, len(passages), args.block_size):
            batch = [text for _, text in passages[position:position + args.block_size]]
            with torch.inference_mode():
                encoded = encoder.encode(batch, batch_size=args.batch_size,
                                         normalize_embeddings=False, show_progress_bar=False)
            if not np.isfinite(encoded).all():
                raise ValueError("The encoder produced non-finite vectors.")
            vectors[position:position + len(batch)] = encoded
            vectors.flush()
            completed = position + len(batch)
            write_json(progress_path, {"completed": completed})
            elapsed = time.perf_counter() - start
            rate = (completed - offset) / max(elapsed, 0.001)
            print(f"Embedded {completed:,}/{len(passages):,}; {rate:.1f} passages/s; "
                  f"remaining {(len(passages) - completed) / max(rate, .001) / 60:.1f} min", flush=True)
        del vectors
        progress_path.unlink(missing_ok=True)
    (out / "DATA-LICENSE.txt").write_text(
        "FiQA retrieval collection, distributed through BEIR and ir_datasets.\n"
        "Source: https://huggingface.co/datasets/BeIR/fiqa\n"
        "License: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).\n"
        "The corpus is filtered for nonempty text. Derived passages contain up to 250 words.\n"
        "Embeddings use BAAI/bge-base-en-v1.5; the model has its own MIT license.\n"
        "FiQA: Maia et al. (2018), WWW '18 Companion, DOI: 10.1145/3184558.3192301.\n"
        "BEIR: Thakur et al. (2021), arXiv:2104.08663.\n")
    files = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*"))
             if p.is_file() and p.name not in {"manifest.json", "progress.json"}
             and not (args.corpus_only and p.name == "embeddings.npy")}
    manifest = {"format": 1, "dense_ready": not args.corpus_only, "created_utc": datetime.now(UTC).isoformat(),
                "dataset": "beir/fiqa", "documents": len(documents), "passages": len(passages),
                "chunk_words": CHUNK_WORDS, "embedding_dimensions": dimensions,
                "encoder_max_tokens": 512, "device": args.device, "terrier_version": "5.11",
                "packages": {p: version(p) for p in ("torch", "numpy", "sentence-transformers",
                              "transformers", "python-terrier", "pyterrier", "ir-datasets")},
                "files": files}
    write_json(out / "manifest.json", manifest)
    archive = out.parent / ("fiqa-corpus.zip" if args.corpus_only else "fiqa-index.zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for path in sorted(out.rglob("*")):
            if str(path.relative_to(out)) in files or path.name == "manifest.json":
                z.write(path, str(path.relative_to(out)))
    print(f"Complete: {archive} ({archive.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/fiqa-index"))
    parser.add_argument("--corpus-only", action="store_true", help="Defer embeddings to the Space GPU")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=256)
    build(parser.parse_args())

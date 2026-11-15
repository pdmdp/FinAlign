"""Read the versioned corpus bundle and keep lexical search outside GPU workers."""

import hashlib
import json
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

from .core import CHUNK_WORDS, LEXICAL_DEPTH, clean_query


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unpack(archive, target):
    archive, target = Path(archive), Path(target)
    digest = file_hash(archive)
    marker = target / ".archive-sha256"
    if marker.exists() and marker.read_text().strip() == digest:
        return target
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist()) > 1_500_000_000:
            raise ValueError("Unexpectedly large index archive.")
        for info in z.infolist():
            destination = (target / info.filename).resolve()
            if not destination.is_relative_to(target.resolve()):
                raise ValueError("Unsafe archive path.")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symbolic links are not permitted in the index.")
        z.extractall(target)
    marker.write_text(digest)
    return target


class Index:
    def __init__(self, directory):
        self.path = Path(directory).resolve()
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest["format"] != 1 or self.manifest["chunk_words"] != CHUNK_WORDS:
            raise ValueError("Unsupported index format.")
        for name, expected in self.manifest["files"].items():
            path = (self.path / name).resolve()
            if not path.is_relative_to(self.path) or file_hash(path) != expected:
                raise ValueError(f"Index integrity check failed: {name}")
        self.documents = {}
        with (self.path / "documents.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                self.documents[row["docno"]] = row["text"]
        self.passage_ids = json.loads((self.path / "passages.json").read_text())
        self.embeddings = None
        self.refresh_dense()
        if len(self.documents) != self.manifest["documents"]:
            raise ValueError("The document count does not match the manifest.")
        self.models = json.loads((self.path / "models.json").read_text())
        self.queries = json.loads((self.path / "queries.json").read_text())
        self.query_lookup = {}
        for split, rows in self.queries.items():
            for row in rows:
                key = clean_query(row["query"])
                self.query_lookup.setdefault(key, []).append({**row, "split": split})

    @property
    def ready(self):
        return self.embeddings is not None

    def refresh_dense(self):
        manifest = json.loads((self.path / "manifest.json").read_text())
        if not manifest.get("dense_ready", False):
            return
        if self.embeddings is None:
            path = self.path / "embeddings.npy"
            if file_hash(path) != manifest["files"]["embeddings.npy"]:
                raise ValueError("Embedding integrity check failed.")
            self.embeddings = np.load(path, mmap_mode="r", allow_pickle=False)
            if self.embeddings.shape != (len(self.passage_ids), manifest["embedding_dimensions"]):
                raise ValueError("The passage list and embedding matrix do not match.")
        self.manifest = manifest

    def known_query(self, query):
        matches = self.query_lookup.get(clean_query(query), [])
        return matches[0] if len(matches) == 1 else None

    def snippet(self, docno, passage=None):
        words = self.documents[docno].split()
        if passage:
            offset = int(passage.rsplit("%", 1)[1]) * CHUNK_WORDS
            return " ".join(words[offset:offset + CHUNK_WORDS])
        return " ".join(words[:CHUNK_WORDS])


class LexicalSearch:
    def __init__(self, index):
        self.index = index
        self.retriever = None
        self.lock = threading.Lock()

    def search(self, query):
        # Initialize Java in the CPU callback. The ZeroGPU worker never uses JNI.
        import pyterrier as pt
        with self.lock:
            if self.retriever is None:
                pt.terrier.set_version(self.index.manifest["terrier_version"])
                pt.terrier.set_helper_version("0.0.8")
                self.retriever = pt.terrier.Retriever(
                    str(self.index.path / "terrier" / "data.properties"),
                    wmodel="BM25", num_results=LEXICAL_DEPTH)
            start = time.perf_counter()
            frame = self.retriever.search(query)
            scores = {str(row.docno): float(row.score) for row in frame.itertuples()}
            return scores, (time.perf_counter() - start) * 1000

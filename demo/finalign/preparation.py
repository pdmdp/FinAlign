"""Checkpointed, time-bounded embedding preparation for a free GPU session."""

import json
import time
import zipfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from filelock import FileLock

from .index import file_hash


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def prepare_batch(index, encoder, seconds=40, block_size=128, batch_size=32, device="cuda"):
    if index.ready:
        return {"completed": len(index.passage_ids), "total": len(index.passage_ids), "done": True}
    progress_path = index.path / "indexing-progress.json"
    with FileLock(str(index.path / ".indexing.lock"), timeout=0):
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
            "completed": 0, "seconds": 0.0, "encoder": index.models["encoder"], "device": device}
        if progress["encoder"] != index.models["encoder"] or progress["device"] != device:
            raise ValueError("This partial index was built with a different encoder or device.")
        offset, total = progress["completed"], len(index.passage_ids)
        if not 0 <= offset <= total:
            raise ValueError("Invalid indexing checkpoint.")
        vectors = np.lib.format.open_memmap(
            index.path / "embeddings.npy", mode="r+" if offset else "w+", dtype="float32",
            shape=(total, index.manifest["embedding_dimensions"]))
        start, previous = time.perf_counter(), time.perf_counter()
        for position in range(offset, total, block_size):
            ids = index.passage_ids[position:position + block_size]
            texts = [index.snippet(pid.rsplit("%", 1)[0], pid) for pid in ids]
            encoded = encoder.encode(texts, batch_size=batch_size, normalize_embeddings=False,
                                     show_progress_bar=False)
            if not np.isfinite(encoded).all():
                raise ValueError("The encoder produced non-finite vectors.")
            vectors[position:position + len(ids)] = encoded
            vectors.flush()
            now = time.perf_counter()
            progress["completed"] = position + len(ids)
            progress["seconds"] += now - previous
            previous = now
            atomic_json(progress_path, progress)
            if now - start >= seconds:
                break
        del vectors
        return {"completed": progress["completed"], "total": total,
                "seconds": progress["seconds"], "done": progress["completed"] == total}


def finish_index(index, archive):
    archive = Path(archive)
    if not index.ready:
        progress = json.loads((index.path / "indexing-progress.json").read_text())
        if progress["completed"] != len(index.passage_ids):
            raise ValueError("The dense index is incomplete.")
        manifest = {**index.manifest, "dense_ready": True, "device": progress["device"],
                    "embedding_seconds": progress["seconds"],
                    "embeddings_created_utc": datetime.now(UTC).isoformat(),
                    "embedding_packages": {p: version(p) for p in ("torch", "transformers", "sentence-transformers")}}
        manifest["files"] = {**manifest["files"], "embeddings.npy": file_hash(index.path / "embeddings.npy")}
        atomic_json(index.path / "manifest.json", manifest)
        index.refresh_dense()
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
            for name in [*index.manifest["files"], "manifest.json"]:
                z.write(index.path / name, name)
        temporary.replace(archive)
    return str(archive)

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from finalign.index import Index, file_hash, unpack
from finalign.preparation import finish_index, prepare_batch


class TinyEncoder:
    def encode(self, texts, **kwargs):
        return np.array([[len(t), t.count("a")] for t in texts], dtype=np.float32)


def corpus_fixture(root):
    (root / "documents.jsonl").write_text(
        '\n'.join(json.dumps({"docno": d, "text": t}) for d, t in [("a", "alpha"), ("b", "beta"), ("c", "gamma")]) + '\n')
    (root / "passages.json").write_text(json.dumps(["a%0", "b%0", "c%0"]))
    (root / "queries.json").write_text('{"dev": [], "test": []}')
    (root / "models.json").write_text('{"encoder": {"id": "fixture", "revision": "fixed"}}')
    files = {p.name: file_hash(p) for p in root.iterdir()}
    (root / "manifest.json").write_text(json.dumps({
        "format": 1, "dense_ready": False, "chunk_words": 250, "documents": 3,
        "passages": 3, "embedding_dimensions": 2, "files": files}))


class PreparationTests(unittest.TestCase):
    def test_resume_and_export_preserve_every_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "corpus"
            root.mkdir()
            corpus_fixture(root)
            index = Index(root)
            first = prepare_batch(index, TinyEncoder(), seconds=0, block_size=1, device="cpu")
            self.assertEqual(first["completed"], 1)
            self.assertFalse(index.ready)
            with self.assertRaises(ValueError):
                finish_index(index, Path(directory) / "index.zip")
            second = prepare_batch(index, TinyEncoder(), seconds=10, block_size=1, device="cpu")
            self.assertTrue(second["done"])
            archive = finish_index(index, Path(directory) / "index.zip")
            reloaded = Index(unpack(archive, Path(directory) / "unpacked"))
            self.assertTrue(reloaded.ready)
            np.testing.assert_array_equal(reloaded.embeddings, [[5, 2], [4, 1], [5, 2]])

    def test_bad_file_is_rejected_before_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_fixture(root)
            (root / "documents.jsonl").write_text("tampered")
            with self.assertRaises(ValueError):
                Index(root)


if __name__ == "__main__":
    unittest.main()

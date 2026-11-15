import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from finalign.core import (
    chunks,
    clean_query,
    dense_top_k,
    fuse_passages,
    hybrid_scores,
    metrics,
    min_max,
    validate_query,
)
from finalign.index import unpack
from finalign.presentation import results_html


class RankingTests(unittest.TestCase):
    def test_fusion_happens_before_document_aggregation(self):
        # A has different best chunks in each variant; combining document maxima
        # would incorrectly put A above B (1.0 vs 0.8).
        scores, passages = fuse_passages([
            (.7, {"A%0": 1., "A%1": 0., "B%0": .8}),
            (.3, {"A%0": 0., "A%1": 1., "B%0": .8})])
        self.assertAlmostEqual(scores["B"], .8)
        self.assertAlmostEqual(scores["A"], .7)
        self.assertEqual(passages["A"], "A%0")

    def test_hybrid_retains_lexical_only_documents(self):
        output = hybrid_scores({"a": 10, "b": 0}, {"b": 10, "c": 5, "d": 0})
        self.assertEqual(output, {"a": 1., "b": 1., "c": .5, "d": 0.})

    def test_constant_scores_and_missing_runs(self):
        self.assertEqual(min_max({"a": 4, "b": 4}), {"a": 1., "b": 1.})
        self.assertEqual(fuse_passages([]), ({}, {}))
        with self.assertRaises(ValueError):
            min_max({"a": float("nan")})

    def test_exact_chunk_boundaries_and_empty_text(self):
        items = list(chunks({"a": " ".join(str(i) for i in range(501)), "empty": ""}))
        self.assertEqual([p for p, _ in items], ["a%0", "a%1", "a%2"])
        self.assertEqual(items[-1][1], "500")
        self.assertEqual(len(items[1][1].split()), 250)

    def test_top_k_orders_dot_products(self):
        matrix = np.array([[1, 0], [0, 1], [.5, .5]], dtype=np.float32)
        scores = dense_top_k(matrix, np.array([0, 1], dtype=np.float32), ["a", "b", "c"], 2)
        self.assertEqual(list(scores), ["b", "c"])

    def test_metrics_match_independent_trec_evaluation(self):
        import ir_measures
        from ir_measures import AP, R, nDCG
        qrels = [ir_measures.Qrel("q", d, rel) for d, rel in {"a": 1, "b": 1, "d": 0}.items()]
        run = [ir_measures.ScoredDoc("q", d, 10 - i) for i, d in enumerate(["c", "a", "b"])]
        expected = ir_measures.calc_aggregate([AP, R@10, nDCG@10], qrels, run)
        actual = metrics(["c", "a", "b"], {"a": 1, "b": 1, "d": 0})
        for name, metric in (("AP", AP), ("Recall@10", R@10), ("nDCG@10", nDCG@10)):
            self.assertAlmostEqual(actual[name], expected[metric])
        self.assertIsNone(metrics(["c"], {}))

    def test_query_validation_preserves_notebook_normalization(self):
        self.assertEqual(clean_query("  bonds:\n interest? "), "bonds interest?")
        for query in ("", "   ", "?!", "a" * 601):
            with self.assertRaises(ValueError):
                validate_query(query)

    def test_corpus_html_is_escaped(self):
        class FakeIndex:
            def __init__(self):
                self.documents = {"a": '<script>alert(1)</script> & <img src=x onerror="alert(1)">'}

            def snippet(self, docno, passage=None):
                return self.documents[docno]
        output = results_html(FakeIndex(), [("a", 1.)], known={"qrels": {}})
        self.assertNotIn("<script>", output)
        self.assertNotIn("<img ", output)
        self.assertIn("&lt;script&gt;", output)
        # Unjudged documents carry no label; judged-relevant ones are marked.
        self.assertNotIn('class="fa-badge"', output)
        judged = results_html(FakeIndex(), [("a", 1.)], known={"qrels": {"a": 1}})
        self.assertIn('class="fa-badge"', judged)

    def test_archive_cannot_escape_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../outside.txt", "no")
            with self.assertRaises(ValueError):
                unpack(archive, root / "index")
            self.assertFalse((root / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()

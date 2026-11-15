import copy
import itertools
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from finalign.core import clean_query, hybrid_scores, query_seed, ranked
from followup.protocol import (WEIGHTS, best_candidate, blend, effectiveness, holm,
                               paired_analysis, select_parameters)
from followup.runner import Experiment, validate_splits
from followup.storage import Store, digest, read_json, write_json


def row(qid):
    return {"qid": str(qid), "query": f"question {qid}", "qrels": {"a": 1, "c": 1}}


class FakeBackend:
    def __init__(self):
        self.calls, self.reranks, self.fail = [], [], None

    def components(self, r):
        self.calls.append(r["qid"])
        if r["qid"] == self.fail:
            raise RuntimeError("Interrupted computation")
        return {"variants": {"original": clean_query(r["query"]), "keywords": "terms", "hyde": "answer"},
                "seed": query_seed(clean_query(r["query"])), "build_seconds": 0.,
                "lexical": {"b": 4., "a": 2., "c": 1.},
                "passages": {"original": {"b%0": .9, "a%0": .5, "c%0": .1},
                             "keywords": {"a%0": .9, "c%0": .7, "b%0": .2},
                             "hyde": {"c%0": .8, "a%0": .4, "b%0": .1}}}

    def rerank(self, r, candidates):
        self.reranks.append(r["qid"])
        scores = {"a": 3., "b": 1., "c": 2.}
        return {doc: scores[doc] for doc, _ in candidates}


class ProtocolTests(unittest.TestCase):
    def test_endpoints_keep_exact_rankings_and_do_not_add_documents(self):
        dense, lexical = {"d": 3., "x": 1.}, {"l": 9., "x": 2.}
        self.assertEqual(blend(dense, lexical, 0), dense)
        self.assertEqual(blend(dense, lexical, 1), lexical)
        self.assertNotIn("l", blend(dense, lexical, 0))
        self.assertNotIn("d", blend(dense, lexical, 1))

    def test_equal_weight_blend_matches_demo_order(self):
        dense, lexical = {"c": .1, "a": .8, "b": .4}, {"b": 20., "d": 11., "a": 2.}
        self.assertEqual([d for d, _ in ranked(blend(dense, lexical, .5))],
                         [d for d, _ in ranked(hybrid_scores(dense, lexical))])

    def test_selection_ties_follow_fixed_order(self):
        self.assertEqual(best_candidate([.5, .5 + 5e-13, .1]), (0, [0, 1]))
        self.assertEqual(best_candidate([.4, .5, .5]), (1, [1, 2]))
        with self.assertRaises(ValueError):
            best_candidate([float("nan")])

    def test_metrics_against_ir_measures_and_empty_ranking(self):
        import ir_measures as irm
        qrels = [irm.Qrel("q", doc, rel) for doc, rel in {"a": 1, "b": 0, "c": 1}.items()]
        run = [irm.ScoredDoc("q", "b", 3.), irm.ScoredDoc("q", "a", 2.), irm.ScoredDoc("q", "c", 1.)]
        expected = irm.calc_aggregate([irm.nDCG@10, irm.R@10], qrels, run)
        actual = effectiveness({"b": 3., "a": 2., "c": 1.}, {"a": 1, "b": 0, "c": 1})
        self.assertAlmostEqual(actual["nDCG@10"], expected[irm.nDCG@10])
        self.assertAlmostEqual(actual["Recall@10"], expected[irm.R@10])
        self.assertEqual(effectiveness({}, {"a": 1}), {"nDCG@10": 0., "Recall@10": 0.})

    def test_holm_order_and_monotonicity(self):
        np.testing.assert_allclose(holm([.03, .01, .04]), [.06, .03, .06])

    def test_randomization_agrees_with_exact_small_null(self):
        delta = np.column_stack([np.ones(4), np.zeros(4), -np.ones(4)])
        result = paired_analysis(delta, bootstrap_samples=100, permutations=20000)
        exact = sum(abs(sum(s)) >= 4 for s in itertools.product([-1, 1], repeat=4)) / 16
        self.assertAlmostEqual(result[0]["raw_p"], exact, delta=.015)
        self.assertEqual(result[0]["raw_p"], result[2]["raw_p"])
        self.assertEqual((result[0]["ci_low"], result[0]["ci_high"]), (1., 1.))
        self.assertEqual((result[1]["raw_p"], result[1]["ci_low"], result[1]["ci_high"]), (1., 0., 0.))

    def test_resampling_is_reproducible_and_paired(self):
        values = np.linspace(-.4, .6, 40)
        delta = np.column_stack([values, -values, np.zeros(40)])
        first = paired_analysis(delta, bootstrap_samples=300, permutations=500)
        second = paired_analysis(delta, bootstrap_samples=300, permutations=500)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0]["ci_low"], -first[1]["ci_high"])
        self.assertAlmostEqual(first[0]["ci_high"], -first[1]["ci_low"])

    def test_dev_grid_retains_every_query_and_candidate(self):
        rows = [row(1), row(2)]
        result = select_parameters(rows, [FakeBackend().components(r) for r in rows])
        self.assertEqual(result["query_ids"], ["1", "2"])
        self.assertEqual(np.shape(result["per_query_weight_ndcg"]), (2, 5))
        self.assertEqual(np.shape(result["per_query_alpha_ndcg"]), (2, 5))
        self.assertIn(tuple(result["weights"]), WEIGHTS)

    def test_split_counts_overlap_and_unjudged_queries(self):
        index = SimpleNamespace(queries={"dev": [row(i) for i in range(500)],
                                         "test": [row(i) for i in range(500, 1148)]})
        self.assertEqual(len(validate_splits(index)["test"]), 648)
        index.queries["test"][0]["qid"] = "0"
        with self.assertRaises(ValueError):
            validate_splits(index)
        index.queries["test"][0]["qid"] = "500"
        index.queries["dev"][0]["qrels"] = {"a": 0}
        with self.assertRaises(ValueError):
            validate_splits(index)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.context = {"source_revision": "local", "model_revision": "fixed"}
        self.store = Store(self.path, self.context)
        self.backend = FakeBackend()
        self.index = SimpleNamespace(documents={"a": "a", "b": "b", "c": "c"},
                                     passage_ids=["a%0", "b%0", "c%0"])
        self.rows = {"dev": [row(1), row(2)], "test": [row(3), row(4)]}
        self.exp = Experiment(self.index, self.backend, self.store, self.rows)

    def tearDown(self):
        self.tmp.cleanup()

    def test_test_requires_freeze_before_computation(self):
        with self.assertRaisesRegex(ValueError, "freeze"):
            self.exp.test()
        self.assertEqual(self.backend.calls, [])

    def test_freeze_is_immutable_and_dev_does_not_touch_test(self):
        self.exp.dev()
        before = (self.path / "frozen_config.json").read_bytes()
        self.exp.dev()
        self.assertEqual(before, (self.path / "frozen_config.json").read_bytes())
        self.assertEqual(self.backend.calls, ["1", "2"])
        with self.assertRaises(ValueError):
            self.store.freeze({})

    def test_partial_test_resumes_without_recomputing_completed_queries(self):
        self.exp.dev()
        self.backend.fail = "4"
        with self.assertRaises(RuntimeError):
            self.exp.test()
        self.assertFalse(read_json(self.path / "last_failure.json")["scored_as_zero"])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.exp.complete_results()
        self.backend.fail = None
        self.exp.test()
        self.assertEqual(self.backend.calls.count("3"), 1)
        self.assertEqual(self.backend.calls.count("4"), 2)
        self.assertEqual(self.backend.reranks, ["3", "4"])
        self.assertEqual(len(self.exp.complete_results()), 2)
        self.exp.test()
        self.assertEqual(self.backend.reranks, ["3", "4"])

    def test_context_change_rejects_cache_reuse(self):
        with self.assertRaisesRegex(ValueError, "context changed"):
            Store(self.path, {**self.context, "model_revision": "other"})

    def test_corruption_is_not_a_cache_hit(self):
        self.exp.components(row(1), "dev")
        path = self.store.path("components/dev", row(1))
        data = read_json(path)
        data["payload"]["lexical"]["a"] = 99.
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, "corruption"):
            self.exp.components(row(1), "dev")

    def test_freeze_or_dev_selection_edit_is_detected(self):
        self.exp.dev()
        selection = read_json(self.path / "dev_selection.json")
        selection["alpha"] = .123
        write_json(self.path / "dev_selection.json", selection)
        with self.assertRaisesRegex(ValueError, "dev selection"):
            self.exp.test()

    def test_missing_hash_recovers_only_before_test(self):
        self.exp.dev()
        (self.path / "frozen_config.sha256").unlink()
        self.store.frozen()
        self.exp.test()
        (self.path / "frozen_config.sha256").unlink()
        with self.assertRaisesRegex(ValueError, "after test"):
            self.store.frozen()

    def test_reranker_must_retain_exact_candidate_set(self):
        self.exp.dev()
        self.backend.rerank = lambda r, c: {"a": 1.}
        with self.assertRaisesRegex(ValueError, "every selected candidate"):
            self.exp.test()

    def test_empty_rankings_are_valid_but_nonfinite_scores_are_not(self):
        components = self.backend.components(row(1))
        components["lexical"] = {}
        components["passages"] = {k: {} for k in components["passages"]}
        self.backend.components = lambda r: components
        self.backend.rerank = lambda r, c: {}
        data, _ = self.exp.components(row(1), "dev")
        runs = self.exp.rankings(row(1), data, WEIGHTS[0], .5, "smoke", smoke=True)
        self.assertTrue(all(r == {} for r in runs.values()))
        with self.assertRaises(ValueError):
            self.exp.validate_scores({"a": math.nan})


if __name__ == "__main__":
    unittest.main()

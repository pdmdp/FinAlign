"""CLI stages: smoke, dev, test, report. All model computation stays in Colab."""

import argparse
import importlib.metadata
import math
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from finalign.core import RERANK_DEPTH, clean_query, fuse_passages, query_seed, ranked
from finalign.index import Index, file_hash

from .backend import Backend
from .protocol import (ALPHAS, ARCHIVE_SHA256, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED,
                       COMPARISONS, PERMUTATIONS, PERMUTATION_SEED, SYSTEMS,
                       VARIANTS, WEIGHTS, blend, effectiveness, expanded_scores,
                       select_parameters)
from .storage import Store, atomic_bytes, digest, read_json, write_json

SOURCE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    "demo/finalign/__init__.py", "demo/finalign/core.py", "demo/finalign/index.py",
    "demo/finalign/neural.py", "demo/models.lock.json", "demo/EXPERIMENTS.md",
    "experiments/requirements.txt", "experiments/followup/__init__.py",
    "experiments/followup/protocol.py", "experiments/followup/storage.py",
    "experiments/followup/backend.py", "experiments/followup/runner.py",
    "experiments/followup/report.py",
)
MANIFEST_SHA256 = "b6512b74414279d7bde915fb2e7a2fce01cc93eac8884365927f884b4d587480"


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_splits(index):
    selected = {}
    for split, count in (("dev", 500), ("test", 648)):
        rows = index.queries[split]
        ids = [r["qid"] for r in rows]
        if len(rows) != count or len(set(ids)) != count:
            raise ValueError(f"Expected {count} unique {split} queries.")
        for row in rows:
            if not isinstance(row["qid"], str) or not clean_query(row["query"]):
                raise ValueError("Invalid query ID or text.")
            if not row["qrels"] or not any(v > 0 for v in row["qrels"].values()):
                raise ValueError(f"Unjudged query: {row['qid']}")
            if any(v not in (0, 1) for v in row["qrels"].values()):
                raise ValueError("This protocol expects binary FiQA judgments.")
        selected[split] = sorted(rows, key=lambda r: r["qid"])
    if {r["qid"] for r in selected["dev"]} & {r["qid"] for r in selected["test"]}:
        raise ValueError("Dev and test query IDs overlap.")
    return selected


def build_context(index, device, source_revision):
    import torch
    if not (source_revision == "local" or re.fullmatch(r"[0-9a-f]{40}", source_revision)):
        raise ValueError("Use a full source commit SHA, or 'local' for local validation.")
    if file_hash(index.path / "manifest.json") != MANIFEST_SHA256:
        raise ValueError("Use the validated Colab index, not a rebuilt or partial index.")
    if (not index.ready or len(index.documents) != 57600 or len(index.passage_ids) != 66113
            or tuple(index.embeddings.shape) != (66113, 768)):
        raise ValueError("Unexpected corpus or embedding dimensions.")
    if index.models != read_json(SOURCE_ROOT / "demo/models.lock.json"):
        raise ValueError("Index model revisions differ from the published lock.")
    rows = validate_splits(index)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Select a Colab T4 GPU runtime before running the experiment.")
    packages = {d.metadata["Name"].lower().replace("_", "-"): d.version
                for d in importlib.metadata.distributions() if d.metadata["Name"]}
    direct = {}
    for d in importlib.metadata.distributions():
        info = d.read_text("direct_url.json")
        if info and d.metadata["Name"]:
            import json
            direct[d.metadata["Name"]] = json.loads(info)
    context = {
        "format": 1, "source_revision": source_revision,
        "source_sha256": {p: file_hash(SOURCE_ROOT / p) for p in SOURCE_PATHS},
        "archive_sha256": ARCHIVE_SHA256, "index_manifest_sha256": MANIFEST_SHA256,
        "index_manifest": index.manifest, "models": index.models,
        "query_ids": {s: [r["qid"] for r in rr] for s, rr in rows.items()},
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "machine": platform.machine(), "packages": packages, "direct_urls": direct,
                        "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                        "cuda_build": torch.version.cuda, "deterministic_algorithms": True,
                        "matmul_precision": "highest", "tf32": False},
        "method": {"weight_grid": [list(w) for w in WEIGHTS], "alpha_grid": list(ALPHAS),
                   "chunk_words": 250, "passage_depth": 400, "lexical_depth": 1000, "rerank_depth": 50,
                   "encoder_query_normalization": False, "query_cleaning": "finalign.core.clean_query",
                   "seed_rule": "first four SHA-256 bytes of cleaned query, big-endian",
                   "generation_order": ["keywords", "hyde"],
                   "generation_implementation": (SOURCE_ROOT / "demo/finalign/neural.py").read_text(),
                   "generation_defaults": "generation_config.json at the locked Qwen revision",
                   "primary_metric": "nDCG@10", "secondary_metric": "Recall@10",
                   "comparisons": [list(c) for c in COMPARISONS],
                   "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
                   "permutations": PERMUTATIONS, "permutation_seed": PERMUTATION_SEED,
                   "bootstrap_interval": "marginal percentile 95%", "alternative": "two-sided",
                   "correction": "Holm", "family_alpha": .05},
        "prior_test_use": "The 648 test queries informed the original coursework design. "
                          "This is a retrospective follow-up, not an untouched holdout.",
    }
    return context, rows


class Experiment:
    def __init__(self, index, backend, store, rows):
        self.index, self.backend, self.store, self.rows = index, backend, store, rows

    def validate_scores(self, scores, passages=False):
        if not isinstance(scores, dict) or not all(math.isfinite(float(s)) for s in scores.values()):
            raise ValueError("Invalid or non-finite scores.")
        for identifier in scores:
            doc = identifier.rsplit("%", 1)[0] if passages else identifier
            if doc not in self.index.documents:
                raise ValueError(f"Unknown document: {doc}")
        if passages and any(p not in self._passage_set for p in scores):
            raise ValueError("Unknown passage.")

    @property
    def _passage_set(self):
        if not hasattr(self, "_passage_ids"):
            self._passage_ids = set(self.index.passage_ids)
        return self._passage_ids

    def components(self, row, split):
        kind = f"components/{split}"
        data = self.store.get(kind, row)
        cached = data is not None
        if data is None:
            data = self.backend.components(row)
        if set(data["variants"]) != set(VARIANTS) or set(data["passages"]) != set(VARIANTS):
            raise ValueError("Missing query variant.")
        if data["variants"]["original"] != clean_query(row["query"]) or data["seed"] != query_seed(clean_query(row["query"])):
            raise ValueError("Component query or seed mismatch.")
        if not all(isinstance(v, str) for v in data["variants"].values()):
            raise ValueError("Invalid generated variants.")
        self.validate_scores(data["lexical"])
        if len(data["lexical"]) > 1000:
            raise ValueError("Unexpected BM25 depth.")
        for run in data["passages"].values():
            self.validate_scores(run, passages=True)
            if len(run) > 400:
                raise ValueError("Unexpected passage depth.")
        if not cached:
            self.store.put(kind, row, data)
        return data, cached

    def rankings(self, row, components, weights, alpha, freeze_hash, smoke=False):
        b = fuse_passages([(1., components["passages"]["original"])])[0]
        e = expanded_scores(components["passages"], weights)
        h = blend(e, components["lexical"], alpha)
        candidates = ranked(h, RERANK_DEPTH)
        identity = {"freeze_sha256": freeze_hash, "candidates": [list(c) for c in candidates]}
        kind = "smoke-rerank" if smoke else "rerank"
        f = self.store.get(kind, row, identity)
        if f is None:
            f = self.backend.rerank(row, candidates)
            self.validate_scores(f)
            if set(f) != {doc for doc, _ in candidates}:
                raise ValueError("Reranking must retain every selected candidate, exactly once.")
            self.store.put(kind, row, f, identity)
        self.validate_scores(f)
        if set(f) != {doc for doc, _ in candidates}:
            raise ValueError("Reranker cache candidate mismatch.")
        return {"L": components["lexical"], "B": b, "E": e, "H": h, "F": f}

    def failure(self, stage, row, exc):
        write_json(self.store.root / "last_failure.json",
                   {"utc": now(), "stage": stage, "qid": row["qid"],
                    "exception": type(exc).__name__, "message": str(exc),
                    "scored_as_zero": False})

    def smoke(self):
        records = []
        for row in self.rows["dev"][:2]:
            c, cached = self.components(row, "dev")
            runs = self.rankings(row, c, WEIGHTS[0], .5, "dev-smoke", smoke=True)
            records.append({"qid": row["qid"], "metrics": {s: effectiveness(r, row["qrels"])
                                                             for s, r in runs.items()}})
            print(f"Smoke: dev query {row['qid']} completed (components cached={cached}).", flush=True)
        write_json(self.store.root / "smoke.json", {"purpose": "Environment check, not parameter selection",
                                                   "records": records})

    def dev(self):
        if (self.store.root / "frozen_config.json").exists():
            frozen, checksum = self.store.frozen()
            print(f"Already frozen: weights={frozen['weights']}, alpha={frozen['alpha']}; SHA-256 {checksum}", flush=True)
            return frozen
        if (self.store.root / "test_started.json").exists():
            raise ValueError("Test has started; development selection cannot be changed.")
        inputs = []
        for n, row in enumerate(self.rows["dev"], 1):
            try:
                data, cached = self.components(row, "dev")
                inputs.append(data)
            except Exception as exc:
                self.failure("dev", row, exc)
                raise
            print(f"Dev components {n}/{len(self.rows['dev'])}: {row['qid']} ({'cached' if cached else 'computed'})", flush=True)
        selection = select_parameters(self.rows["dev"], inputs)
        write_json(self.store.root / "dev_selection.json", selection)
        frozen = {"format": 1, "created_utc": now(), "context_sha256": self.store.context_hash,
                  "source_revision": self.store.context["source_revision"],
                  "context": self.store.context, "weights": selection["weights"], "alpha": selection["alpha"],
                  "dev_selection_sha256": digest(selection), "dev_selection": selection}
        self.store.freeze(frozen)
        print(f"Frozen: weights={selection['weights']}, alpha={selection['alpha']}; SHA-256 {digest(frozen)}", flush=True)
        return frozen

    def test(self):
        frozen, checksum = self.store.frozen()
        started = self.store.root / "test_started.json"
        if started.exists():
            if read_json(started)["freeze_sha256"] != checksum:
                raise ValueError("Test start record refers to a different freeze.")
        else:
            write_json(started, {"utc": now(), "freeze_sha256": checksum})
        for n, row in enumerate(self.rows["test"], 1):
            try:
                result = self.store.get("results", row, checksum)
                cached = result is not None
                if result is None:
                    components, _ = self.components(row, "test")
                    runs = self.rankings(row, components, frozen["weights"], frozen["alpha"], checksum)
                    result = {"qid": row["qid"], "query": row["query"], "variants": components["variants"],
                              "seed": components["seed"], "freeze_sha256": checksum,
                              "rankings": {s: [list(x) for x in ranked(r)] for s, r in runs.items()},
                              "metrics": {s: effectiveness(r, row["qrels"]) for s, r in runs.items()}}
                    self.validate_result(row, result, checksum)
                    self.store.put("results", row, result, checksum)
                self.validate_result(row, result, checksum)
            except Exception as exc:
                self.failure("test", row, exc)
                raise
            print(f"Test {n}/{len(self.rows['test'])}: {row['qid']} ({'cached' if cached else 'computed'})", flush=True)

    def validate_result(self, row, result, checksum):
        if (result["qid"] != row["qid"] or result["query"] != row["query"]
                or result["freeze_sha256"] != checksum or set(result["rankings"]) != set(SYSTEMS)
                or set(result["metrics"]) != set(SYSTEMS)):
            raise ValueError("Result identity or system mismatch.")
        for system in SYSTEMS:
            ranking = result["rankings"][system]
            scores = dict(ranking)
            if len(scores) != len(ranking) or [list(x) for x in ranked(scores)] != ranking:
                raise ValueError("Invalid ranking or duplicate documents.")
            self.validate_scores(scores)
            if effectiveness(scores, row["qrels"]) != result["metrics"][system]:
                raise ValueError("Saved metrics do not match the ranking.")
        if set(dict(result["rankings"]["H"][:50])) != set(dict(result["rankings"]["F"])):
            raise ValueError("The reranked candidate set differs from H's first 50.")

    def complete_results(self):
        _, checksum = self.store.frozen()
        output = []
        for row in self.rows["test"]:
            result = self.store.get("results", row, checksum)
            if result is None:
                raise ValueError(f"Incomplete test set: query {row['qid']} is missing. Resume the test stage.")
            self.validate_result(row, result, checksum)
            output.append(result)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "dev", "test", "report"))
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--source-revision", default="local")
    args = parser.parse_args()
    from filelock import FileLock
    args.run.mkdir(parents=True, exist_ok=True)
    # Keep advisory locking on local disk: Drive/FUSE need not support flock.
    # One runtime may use a given Drive run folder at a time.
    lock_directory = Path(tempfile.gettempdir()) / "finalign-followup-locks"
    lock_directory.mkdir(exist_ok=True)
    with FileLock(str(lock_directory / (digest(str(args.run.resolve())) + ".lock")), timeout=0):
        print("Checking the corpus, embeddings and saved run context...", flush=True)
        index = Index(args.index)
        context, rows = build_context(index, args.device, args.source_revision)
        store = Store(args.run, context)
        lock = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"])
        lockfile = args.run / "environment-lock.txt"
        if not lockfile.exists():
            atomic_bytes(lockfile, lock)
        experiment = Experiment(index, Backend(index, args.device), store, rows)
        print(f"Ready: {len(rows['dev'])} dev queries, {len(rows['test'])} test queries; stage={args.stage}.", flush=True)
        start = time.perf_counter()
        if args.stage == "report":
            from .report import write_report
            write_report(experiment)
        else:
            getattr(experiment, args.stage)()
        print(f"{args.stage} completed in {(time.perf_counter() - start) / 60:.1f} min. Saved to {args.run}", flush=True)


if __name__ == "__main__":
    main()

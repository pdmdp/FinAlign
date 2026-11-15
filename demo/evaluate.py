"""Run fixed component ablations and record per-query effectiveness and latency."""

import argparse
import json
import platform
import random
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from finalign.core import Options, metrics, ranked
from finalign.index import Index, LexicalSearch
from finalign.neural import NeuralSearch

CONFIGURATIONS = {
    "bm25": None,
    "bge": Options(),
    "bge_keywords": Options(keywords=True),
    "bge_hyde": Options(hyde=True),
    "bge_expanded": Options(keywords=True, hyde=True),
    "hybrid": Options(keywords=True, hyde=True, lexical=True),
    "full": Options(keywords=True, hyde=True, lexical=True, rerank=True),
    "full_without_keywords": Options(hyde=True, lexical=True, rerank=True),
    "full_without_hyde": Options(keywords=True, lexical=True, rerank=True),
    "full_without_bm25": Options(keywords=True, hyde=True, rerank=True),
}


def summarize(records):
    output = {}
    for config in sorted({r["configuration"] for r in records}):
        rows = [r for r in records if r["configuration"] == config and r["repeat"] > 0]
        # Repeats measure timing, not extra independent queries.
        first = {r["qid"]: r for r in reversed(rows)}
        stages = sorted({stage for row in rows for stage in row["timings_ms"]})
        output[config] = {
            "queries": len(first), "timed_runs": len(rows),
            "effectiveness": {m: float(np.mean([r["metrics"][m] for r in first.values()]))
                              for m in ("nDCG@10", "AP", "Recall@10")},
            "latency_ms": {stage: {"p50": float(np.median([r["timings_ms"][stage] for r in rows])),
                                    "p95": float(np.percentile([r["timings_ms"][stage] for r in rows], 95))}
                           for stage in stages}}
    return output


def main(args):
    if args.repeats < 1 or args.limit < 0:
        raise ValueError("Use at least one repeat and a nonnegative query limit.")
    configurations = args.configurations.split(",")
    unknown = set(configurations) - CONFIGURATIONS.keys()
    if unknown:
        raise ValueError(f"Unknown configurations: {sorted(unknown)}")
    if args.output.exists():
        raise ValueError("The output directory already exists; use a new run name.")
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    index = Index(args.index)
    lexical = LexicalSearch(index)
    encoder_only = all(c in ("bm25", "bge") for c in configurations)
    engine = None if configurations == ["bm25"] else NeuralSearch(index, args.device, encoder_only)
    startup_seconds = time.perf_counter() - start
    rows = [r for r in index.queries[args.split] if any(rel > 0 for rel in r["qrels"].values())]
    random.Random(2026).shuffle(rows)
    if args.limit:
        rows = rows[:args.limit]
    metadata = {"created_utc": datetime.now(UTC).isoformat(), "split": args.split,
                "query_ids": [r["qid"] for r in rows], "configurations": configurations,
                "repeats": args.repeats, "startup_seconds": startup_seconds,
                "device": args.device, "platform": platform.platform(), "python": platform.python_version(),
                "model_revisions": index.models, "index_manifest": index.manifest,
                "packages": {p: version(p) for p in ("torch", "transformers", "sentence-transformers",
                                                      "pyterrier", "numpy")},
                "notes": "One warm-up per configuration is excluded. GPU synchronization is included. "
                         "Queueing and model loading are excluded from query timings. No query-result cache. "
                         "The course test split informed past design choices; dev is used for new tuning."}
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    records = []
    with (args.output / "queries.jsonl").open("w") as stream:
        # Warm-up is separate from the retained sample and does not become another observation.
        for repeat in range(args.repeats + 1):
            batch = rows[:1] if repeat == 0 else rows
            for row in batch:
                order = configurations.copy()
                random.Random(f"{row['qid']}:{repeat}").shuffle(order)
                for config in order:
                    started = time.perf_counter()
                    scores, lexical_ms = lexical.search(row["query"])
                    options = CONFIGURATIONS[config]
                    timings = {"BM25": lexical_ms}
                    result = {"ranking": ranked(scores), "variants": {}, "seed": None}
                    if options is not None:
                        if not options.lexical:
                            started = time.perf_counter()
                        result = engine.search(row["query"], scores, options)
                        timings.update(result["timings_ms"])
                        # BM25 is displayed alongside every demo run, but ablations without
                        # lexical fusion must not count it as part of the neural pipeline.
                        if not options.lexical:
                            timings.pop("BM25")
                    timings["Total"] = (time.perf_counter() - started) * 1000
                    record = {"qid": row["qid"], "query": row["query"], "configuration": config,
                              "repeat": repeat, "metrics": metrics([d for d, _ in result["ranking"]], row["qrels"]),
                              "timings_ms": timings, "variants": result["variants"], "seed": result["seed"],
                              "ranking": result["ranking"]}
                    records.append(record)
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                print(f"repeat={repeat} qid={row['qid']}", flush=True)
    (args.output / "summary.json").write_text(json.dumps(summarize(records), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates every judged query")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--configurations", default=",".join(CONFIGURATIONS))
    main(parser.parse_args())

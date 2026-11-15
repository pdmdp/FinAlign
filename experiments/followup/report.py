"""Complete paired results, tables and a figure; no additional comparisons."""

import csv
import io

import numpy as np

from .protocol import COMPARISONS, SYSTEMS, paired_analysis
from .storage import atomic_bytes, canonical, write_json


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def write_report(experiment):
    records = experiment.complete_results()
    frozen, checksum = experiment.store.frozen()
    folder = experiment.store.root / "report"
    folder.mkdir(exist_ok=True)
    deltas = np.array([[r["metrics"][left]["nDCG@10"] - r["metrics"][right]["nDCG@10"]
                       for _, left, right in COMPARISONS] for r in records])
    comparisons = paired_analysis(deltas)
    means = [{"system": s, "queries": len(records),
              **{m: float(np.mean([r["metrics"][s][m] for r in records]))
                 for m in ("nDCG@10", "Recall@10")}} for s in SYSTEMS]
    per_query = [{"qid": r["qid"], "system": s, **r["metrics"][s]} for r in records for s in SYSTEMS]
    differences = [{"qid": r["qid"], **{c[0]: float(deltas[i, j]) for j, c in enumerate(COMPARISONS)}}
                   for i, r in enumerate(records)]
    summary = {"freeze_sha256": checksum, "weights": frozen["weights"], "alpha": frozen["alpha"],
               "queries": len(records), "systems": means, "comparisons": comparisons,
               "inference": experiment.store.context["method"],
               "limitation": experiment.store.context["prior_test_use"],
               "timing_note": "Cache build times are operational measurements, not end-to-end search latency."}
    for name, rows in (("systems.csv", means), ("comparisons.csv", comparisons),
                       ("per_query.csv", per_query), ("paired_differences.csv", differences)):
        atomic_bytes(folder / name, csv_bytes(rows))
    write_json(folder / "summary.json", summary)
    atomic_bytes(folder / "rankings.jsonl", b"".join(canonical(r) for r in records))
    text = ["# FinAlign follow-up results", "", summary["limitation"], "",
            f"Complete paired set: {len(records)} test queries. Primary metric: nDCG@10.",
            f"Dev-selected weights: {frozen['weights']}; BM25 alpha: {frozen['alpha']}.",
            f"Frozen configuration SHA-256: `{checksum}`.", "",
            "| System | nDCG@10 | Recall@10 |", "| :--- | ---: | ---: |"]
    text += [f"| {r['system']} | {r['nDCG@10']:.6f} | {r['Recall@10']:.6f} |" for r in means]
    text += ["", "L: BM25; B: original-query BGE; E: expanded BGE; H: lexical fusion; F: monoT5.", "",
             "| Comparison | Mean delta | Marginal 95% bootstrap CI | Win / tie / loss | Raw p | Holm p |",
             "| :--- | ---: | :--- | :--- | ---: | ---: |"]
    text += [f"| {r['comparison']}: {r['difference']} | {r['mean_delta']:.6f} | "
             f"[{r['ci_low']:.6f}, {r['ci_high']:.6f}] | {r['improved']} / {r['tied']} / {r['worsened']} | "
             f"{r['raw_p']:.6f} | {r['holm_p']:.6f} |" for r in comparisons]
    text += ["", "Three two-sided paired randomization tests, with Holm correction as one family.",
             "The 95% bootstrap intervals are marginal, not simultaneous intervals.",
             "All comparisons, including null and negative effects, are retained. Non-significance does not establish equivalence.",
             "BM25 is a descriptive reference; no extra significance test against the complete pipeline was performed.",
             "Historical coursework scores are not combined with this run.", "",
             "![Per-query nDCG differences](paired-differences.svg)", ""]
    atomic_bytes(folder / "REPORT.md", "\n".join(text).encode())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    bins = np.linspace(-1, 1, 41)
    for j, (ax, comparison) in enumerate(zip(axes, COMPARISONS)):
        ax.hist(deltas[:, j], bins=bins, color="#3168be", edgecolor="white", linewidth=.4)
        ax.axvline(0, color="#334155", linewidth=.8)
        ax.set(title=f"{comparison[0]}: {comparison[1]} − {comparison[2]}", xlabel="Per-query Δ nDCG@10")
    axes[0].set_ylabel("Queries")
    fig.tight_layout()
    fig.savefig(folder / "paired-differences.svg")
    plt.close(fig)
    print("\n".join(text[:-3]), flush=True)

"""Plot the retained evaluation summary; no models or GPU are required."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(directory):
    summary = json.loads((directory / "summary.json").read_text())
    names = {"L": "BM25", "B": "BGE", "E": "BGE + expansion",
             "H": f"+ BM25 ({summary['alpha']:.0%})", "F": "+ monoT5"}
    fig, (scores, changes) = plt.subplots(1, 2, figsize=(11.6, 4.8),
                                        gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=.18, right=.96, top=.80, bottom=.28, wspace=.65)
    fig.suptitle("FinAlign · follow-up evaluation", fontsize=17, x=.18, ha="left", y=.96)
    systems = summary["systems"]
    values = [row["nDCG@10"] for row in systems]
    scores.barh(range(5), values, height=.62,
                color=["#94a3b8", "#94a3b8", "#2563eb", "#2563eb", "#64748b"])
    scores.set_yticks(range(5), [names[row["system"]] for row in systems])
    scores.invert_yaxis()
    scores.set_xlim(0, .50)
    scores.set_xlabel("Mean nDCG@10", labelpad=9)
    scores.set_title("Test-set effectiveness", loc="left", fontsize=12, pad=16)
    scores.set_axisbelow(True)
    scores.grid(axis="x", color="#e2e8f0", linewidth=.7)
    for i, value in enumerate(values):
        scores.text(value + .008, i, f"{value:.4f}", va="center", fontsize=10)

    labels = ["Expansion\nE − B", "BM25 fusion\nH − E", "monoT5\nF − H"]
    for i, row in enumerate(summary["comparisons"]):
        color = "#2563eb" if row["reject_at_0_05"] else "#64748b"
        mean = row["mean_delta"]
        changes.errorbar(mean, i, xerr=[[mean-row["ci_low"]], [row["ci_high"]-mean]],
                         fmt="o", color=color, capsize=4, elinewidth=2, markersize=5)
        changes.text(.024, i, f"p = {row['holm_p']:.4f}", va="center", fontsize=9, color="#334155")
    changes.axvline(0, color="#94a3b8", linestyle="--", linewidth=1)
    changes.set_yticks(range(3), labels)
    changes.set_ylim(2.6, -.6)
    changes.set_xlim(-.02, .041)
    changes.set_xticks([-.02, 0, .02], ["−0.02", "0", "+0.02"])
    changes.set_xlabel("Mean Δ nDCG@10", labelpad=9)
    changes.set_title("Changes between stages", loc="left", fontsize=12, pad=16)
    for ax in (scores, changes):
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.tick_params(axis="both", length=0, labelsize=10, pad=7)
    fig.text(.18, .11, "648 test queries · weights selected on 500 dev queries · test set used in the original project", fontsize=9, color="#475569")
    fig.text(.18, .065, "Intervals: marginal 95% paired bootstrap. p-values: two-sided paired randomization, Holm correction (3 tests).", fontsize=9, color="#475569")
    fig.savefig(directory / "results-overview.svg", facecolor="white")
    fig.savefig(directory / "results-overview.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/fiqa-dev-test-v1")
    plot(parser.parse_args().directory)

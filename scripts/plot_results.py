"""Plot saved notebook metrics; this script does not run retrieval."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
rows = json.loads((root / "results/recorded_metrics.json").read_text())["metrics"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
colors = ["#a5b6c5"] * 3 + ["#317c9c", "#317c9c", "#156b58"]
bars = ax.barh([x["label"] for x in rows], [x["ndcg10"] for x in rows], color=colors, height=.6)
ax.bar_label(bars, fmt="%.4f", padding=7, fontsize=11)
ax.invert_yaxis()
ax.set(xlim=(0, .49), xlabel="nDCG@10", title="FinAlign · recorded FiQA results")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0, pad=10)
ax.xaxis.grid(True, color="#e7ebee", linewidth=.7)
ax.set_axisbelow(True)
(root / "assets").mkdir(exist_ok=True)
fig.savefig(root / "assets/retrieval-results.png", dpi=180, facecolor="white")

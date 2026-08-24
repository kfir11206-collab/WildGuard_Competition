#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
PHASE_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4"]
BLUE = "#2a78d6"
AQUA = "#1baf7a"

res = Path(sys.argv[1])

RUNS = {
    "first boot of day\n(pre-warmed)": [156.56],
    "warm boot,\npre-warmed": [114.29, 114.19, 113.73],
    "warm boot,\ncold cache": [118.50, 110.79],
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), width_ratios=[1, 1.6])
fig.suptitle(
    "Wake-on-MHT-trigger latency — VLM+classifier module, trigger to first scene description",
    fontsize=13, fontweight="bold", color=INK,
)

def style(ax, xgrid=False):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

labels = list(RUNS)
means = [np.mean(RUNS[k]) for k in labels]
bars = ax1.bar(labels, means, width=0.55, color=BLUE)
ax1.bar_label(bars, labels=[f"{m:.0f} s" for m in means], color=INK, fontsize=11, padding=3)
for i, k in enumerate(labels):
    ax1.scatter([i] * len(RUNS[k]), RUNS[k], color=INK_2, s=18, zorder=4)
ax1.set_ylabel("wake latency (s)")
ax1.set_title("Latency by condition (dots = individual runs)")
style(ax1)

runs = ["cold_1", "cold_2", "prewarmed_3"]
names = ["cold cache #1", "cold cache #2", "pre-warmed #3"]
phase_data = {r: json.load(open(res / f"phases_{r}.json")) for r in runs}
phase_labels = [p[0] for p in phase_data[runs[0]]["phases"]]

y = np.arange(len(runs))
left = np.zeros(len(runs))
for pi, plabel in enumerate(phase_labels):
    vals = np.array([dict(phase_data[r]["phases"])[plabel] for r in runs])
    ax2.barh(y, vals, left=left, height=0.55, color=PHASE_COLORS[pi],
             edgecolor=SURFACE, linewidth=1.5, label=plabel)
    for i, (v, l) in enumerate(zip(vals, left)):
        if v > 10:
            ax2.text(l + v / 2, i, f"{v:.0f}", ha="center", va="center",
                     color="#ffffff", fontsize=9, fontweight="bold")
    left += vals

ax2.set_yticks(y, names)
ax2.invert_yaxis()
ax2.set_xlabel("seconds since MHT trigger")
ax2.set_title("Anatomy of a wake (SD reads ≤ 3.5 s of the AWQ-load phase)")
ax2.legend(frameon=False, labelcolor=INK_2, fontsize=8, loc="upper center",
           bbox_to_anchor=(0.5, -0.13), ncols=4)
style(ax2, xgrid=True)

fig.text(0.01, 0.01,
         "conditions identical except page-cache state at trigger · model set 2.8 GB on SD Express · data: " + str(res),
         color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
fig.savefig(res / "wake_latency.png", dpi=130, facecolor=SURFACE)
print("saved", res / "wake_latency.png")

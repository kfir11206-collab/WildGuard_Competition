#!/usr/bin/env python3
import json
import sys

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
AQUA = "#1baf7a"

runs = [json.loads(line) for line in open(sys.argv[1])]
out_png = sys.argv[2]

cold = {r["advice"]: r for r in runs if not r["warm"] and r["chunk_mb"] == 16 and r["bursts"]}
warm = next(r for r in runs if r["warm"])
by_chunk = {r["chunk_mb"]: r for r in runs if not r["warm"] and r["advice"] == "sequential" and r["bursts"]}
sweep = [by_chunk[k] for k in sorted(by_chunk)]

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

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
fig.suptitle(
    "mmap beta — DistilBERT classifier (256 MB safetensors) on SD Express, cold page cache",
    fontsize=13, fontweight="bold", color=INK,
)

def style(ax, xgrid=False):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

ax = axes[0][0]
order = ["normal", "sequential", "willneed", "random"]
labels = order + ["warm cache\n(any advice)"]
vals = [cold[a]["avg_mbps"] for a in order] + [warm["avg_mbps"]]
bars = ax.bar(labels, vals, width=0.55, color=BLUE)
ax.bar_label(bars, fmt="%.0f", color=INK, fontsize=10, padding=2)
ax.set_ylabel("MB/s")
ax.set_title("Full-file read throughput by madvise")
style(ax)

ax = axes[0][1]
faults = [cold[a]["major_faults"] for a in order]
bars = ax.barh(order[::-1], faults[::-1], height=0.55, color=BLUE)
ax.set_xscale("log")
ax.set_xlim(0.5, 3e5)
ax.bar_label(bars, labels=[f"{v:,}" for v in faults[::-1]], color=INK, fontsize=10, padding=3)
ax.set_xlabel("major page faults (log scale)")
ax.set_title("Major faults: readahead on vs off")
style(ax, xgrid=True)

ax = axes[1][0]
for advice, color in (("normal", BLUE), ("random", AQUA)):
    b = cold[advice]["bursts"]
    ax.plot(
        [p["offset_mb"] for p in b], [p["mbps"] for p in b],
        color=color, linewidth=2, marker="o", markersize=4,
    )
    ax.annotate(
        f"{advice}  ({cold[advice]['avg_mbps']:.0f} MB/s avg)",
        (b[-1]["offset_mb"], b[-1]["mbps"]),
        textcoords="offset points", xytext=(-4, 10 if advice == "normal" else 12),
        ha="right", color=INK_2, fontsize=10,
    )
ax.set_yscale("log")
ax.set_xlabel("file offset (MB)")
ax.set_ylabel("burst throughput, MB/s (log)")
ax.set_title("16 MB bursts across the file: normal vs random")
ax.legend(["normal", "random"], frameon=False, labelcolor=INK_2, loc="center right")
style(ax)

ax = axes[1][1]
labels = [f"{r['chunk_mb']} MB" for r in sweep]
vals = [r["avg_mbps"] for r in sweep]
bars = ax.bar(labels, vals, width=0.55, color=BLUE)
ax.bar_label(bars, fmt="%.0f", color=INK, fontsize=10, padding=2)
ax.set_ylabel("MB/s")
ax.set_xlabel("application burst (slice) size")
ax.set_ylim(0, 900)
ax.set_title("Slice size barely matters (kernel does the bursting)")
style(ax)

fig.text(0.01, 0.005, f"load avg during runs ≈ {runs[0]['load_average']:.2f} · device readahead 128 KB · data: {sys.argv[1]}",
         color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.015, 1, 1))
fig.savefig(out_png, dpi=130, facecolor=SURFACE)
print("saved", out_png)

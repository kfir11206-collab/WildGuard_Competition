#!/usr/bin/env python3
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"

runs = [json.loads(line) for line in open(sys.argv[1])]
out_png = sys.argv[2]

FILES = [
    ("model.safetensors", "typeform", "classifier\n256 MB"),
    ("model.safetensors", "vision_tower", "vision tower\n817 MB"),
    ("vila-1.5-3b-w4-g128-awq-v2.pt", "llm", "AWQ LLM\n1.6 GB"),
]

def pick(marker, advice, warm):
    return next(
        r for r in runs
        if marker in r["file"] and r["advice"] == advice and r["warm"] == warm
    )

cold = [pick(m, "normal", False) for _, m, _ in FILES]
rand = [pick(m, "random", False) for _, m, _ in FILES]
warm = [pick(m, "normal", True) for _, m, _ in FILES]
labels = [lbl for _, _, lbl in FILES]

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
    "mmap at real-model scale — classifier vs VILA vision tower vs AWQ LLM (SD Express)",
    fontsize=13, fontweight="bold", color=INK,
)

def style(ax, xgrid=False):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

x = np.arange(len(labels))

ax = axes[0][0]
b1 = ax.bar(x - 0.17, [r["total_s"] for r in cold], width=0.3, color=BLUE, label="cold (from SD)")
b2 = ax.bar(x + 0.17, [r["total_s"] for r in warm], width=0.3, color=AQUA, label="warm (page cache)")
ax.bar_label(b1, fmt="%.2f s", color=INK, fontsize=10, padding=2)
ax.bar_label(b2, fmt="%.2f s", color=INK, fontsize=10, padding=2)
ax.set_xticks(x, labels)
ax.set_ylabel("seconds")
ax.set_title("Time to read the whole file")
ax.legend(frameon=False, labelcolor=INK_2)
style(ax)

ax = axes[0][1]
bars = ax.bar(labels, [r["avg_mbps"] for r in rand], width=0.5, color=BLUE)
ax.bar_label(
    bars,
    labels=[f"{r['avg_mbps']:.0f} MB/s\n{r['total_s']:.0f} s total" for r in rand],
    color=INK, fontsize=10, padding=2,
)
ax.set_ylabel("MB/s")
ax.set_ylim(0, 100)
ax.set_title("Readahead off (MADV_RANDOM): ~70 MB/s flat")
style(ax)

ax = axes[1][0]
for r, lbl, color in zip(cold, labels, (BLUE, AQUA, YELLOW)):
    b = r["bursts"]
    ax.plot([p["offset_mb"] for p in b], [p["mbps"] for p in b], color=color, linewidth=2)
    ax.annotate(
        lbl.replace("\n", " "), (b[-1]["offset_mb"], b[-1]["mbps"]),
        textcoords="offset points", xytext=(6, 8), color=INK_2, fontsize=9,
    )
ax.set_xlabel("file offset (MB)")
ax.set_ylabel("burst throughput (MB/s)")
ax.set_ylim(0, 1100)
ax.set_title("16 MB bursts across each file (cold): no sag at 1.6 GB")
ax.legend([l.replace("\n", " ") for l in labels], frameon=False, labelcolor=INK_2, loc="lower right", fontsize=9)
style(ax)

ax = axes[1][1]
y = np.arange(len(labels))
b1 = ax.barh(y + 0.17, [r["major_faults"] for r in rand][::-1], height=0.3, color=AQUA, label="random")
b2 = ax.barh(y - 0.17, [max(r["major_faults"], 0.6) for r in cold][::-1], height=0.3, color=BLUE, label="normal")
ax.bar_label(b1, labels=[f"{r['major_faults']:,}" for r in rand][::-1], color=INK, fontsize=10, padding=3)
ax.bar_label(b2, labels=[f"{r['major_faults']:,}" for r in cold][::-1], color=INK, fontsize=10, padding=3)
ax.set_xscale("log")
ax.set_xlim(0.5, 8e6)
ax.set_yticks(y, labels[::-1])
ax.set_xlabel("major page faults (log scale)")
ax.set_title("Faults scale with size only when readahead is off")
ax.legend(frameon=False, labelcolor=INK_2, loc="lower right")
style(ax, xgrid=True)

fig.text(0.01, 0.005, f"cold cache via posix_fadvise(DONTNEED) · chunk 16 MB · load avg ≈ {runs[0]['load_average']:.2f} · data: {sys.argv[1]}",
         color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.015, 1, 1))
fig.savefig(out_png, dpi=130, facecolor=SURFACE)
print("saved", out_png)

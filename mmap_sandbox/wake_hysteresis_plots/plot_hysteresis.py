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
YELLOW = "#eda100"

events = [json.loads(l) for l in open(sys.argv[1])]
out_png = sys.argv[2]

conf = [e for e in events if e["event"] == "confidence"]
t_end = max(e["t"] for e in events)

plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})

fig, ax = plt.subplots(figsize=(14, 5.5))
fig.suptitle(
    "Full-model lifecycle with Schmitt-trigger hysteresis — noisy MHT, 2 samples/s, EMA α=0.4",
    fontsize=13, fontweight="bold", color=INK,
)

for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(length=0)

for e in (e for e in events if e["event"] == "prewarm_done"):
    ax.axvspan(e["t"] - e["seconds"], e["t"], color=AQUA, alpha=0.18, lw=0)
    ax.annotate(f"pre-warm\n{e['seconds']} s", (e["t"] - e["seconds"] / 2, 0.06),
                ha="center", color=INK_2, fontsize=9)
for e in (e for e in events if e["event"] == "keepwarm"):
    ax.axvspan(e["t"] - e["seconds"], e["t"], color=AQUA, alpha=0.35, lw=0)
    ax.annotate(f"keep-warm\n{e['seconds']} s", (e["t"], 0.06), ha="center", color=INK_2, fontsize=9)
for s, a in zip((e for e in events if e["event"] == "wake_start"),
                (e for e in events if e["event"] == "awake")):
    ax.axvspan(s["t"], a["t"], color=YELLOW, alpha=0.35, lw=0)
    ax.annotate(f"wake {a['wake_seconds']} s", (a["t"], 1.0),
                textcoords="offset points", xytext=(4, 0), color=INK_2, fontsize=10)
for e in (e for e in events if e["event"] == "slept"):
    ax.axvline(e["t"], color=BASELINE, linewidth=1)
    ax.annotate("slept", (e["t"], 0.5), textcoords="offset points", xytext=(4, 0),
                color=INK_2, fontsize=9)

ax.scatter([e["t"] for e in conf], [e["value"] for e in conf],
           s=10, color=MUTED, alpha=0.6, zorder=3)
ax.plot([e["t"] for e in conf], [e["smoothed"] for e in conf],
        color=BLUE, linewidth=2, zorder=4)
ax.annotate("raw MHT samples", (conf[6]["t"], conf[6]["value"] + 0.05), color=MUTED, fontsize=9)
ax.annotate("smoothed (EMA)", (conf[-25]["t"], conf[-25]["smoothed"] - 0.09),
            color=BLUE, fontsize=10)

for y, lbl in ((0.85, "wake 85%"), (0.60, "pre-warm 60%"), (0.40, "sleep 40%")):
    ax.axhline(y, color=BASELINE, linewidth=0.8, linestyle="--")
    ax.annotate(lbl, (t_end, y), ha="right", va="bottom", color=MUTED, fontsize=8)

ax.annotate("false spike:\nprewarm only, no wake", (conf[14]["t"], 0.93),
            ha="center", color=INK_2, fontsize=9)

ax.set_xlabel("time (s)")
ax.set_ylabel("fire confidence")
ax.set_ylim(0, 1.08)
ax.set_xlim(0, t_end + 1)

fig.text(0.01, 0.01, f"resident mode · full model set 2.8 GB · keep-warm every 5 s · data: {sys.argv[1]}",
         color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(out_png, dpi=130, facecolor=SURFACE)
print("saved", out_png)

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
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"

results_dir = Path(sys.argv[1])
out_png = sys.argv[2]

def load(name):
    return [json.loads(l) for l in open(results_dir / f"{name}.jsonl")]

def wake_s(events):
    return next(e["wake_seconds"] for e in events if e["event"] == "awake")

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), width_ratios=[1, 1.5])
fig.suptitle(
    "Fire-watcher beta — wake latency for VLM + classifier (2.8 GB) from SD Express",
    fontsize=13, fontweight="bold", color=INK,
)

def style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

x = np.arange(2)
resident = [wake_s(load("resident_prewarm")), wake_s(load("resident_cold"))]
spawn = [wake_s(load("spawn_prewarm")), wake_s(load("spawn_cold"))]
b1 = ax1.bar(x - 0.17, resident, width=0.3, color=BLUE, label="resident process")
b2 = ax1.bar(x + 0.17, spawn, width=0.3, color=AQUA, label="spawned process")
ax1.bar_label(b1, fmt="%.2f s", color=INK, fontsize=10, padding=2)
ax1.bar_label(b2, fmt="%.2f s", color=INK, fontsize=10, padding=2)
ax1.set_xticks(x, ["pre-warmed\n(confidence hit 60% first)", "cold\n(jumped straight to 85%)"])
ax1.set_ylabel("wake latency (s)")
ax1.set_title("85%-trigger to all weights in RAM")
ax1.legend(frameon=False, labelcolor=INK_2)
style(ax1)

events = load("lifecycle")
conf = [(e["t"], e["value"]) for e in events if e["event"] == "confidence"]
t_end = max(e["t"] for e in events)
ct = [t for t, _ in conf] + [t_end]
cv = [v for _, v in conf] + [conf[-1][1]]

pw = next(e for e in events if e["event"] == "prewarm_done")
wakes = [(s["t"], a["t"]) for s, a in zip(
    (e for e in events if e["event"] == "wake_start"),
    (e for e in events if e["event"] == "awake"))]
slept = [e["t"] for e in events if e["event"] == "slept"]

ax2.step(ct, cv, where="post", color=BLUE, linewidth=2)
for y, lbl in ((0.85, "wake 85%"), (0.60, "pre-warm 60%"), (0.40, "sleep 40%")):
    ax2.axhline(y, color=BASELINE, linewidth=0.8, linestyle="--")
    ax2.annotate(lbl, (t_end, y), ha="right", va="bottom", color=MUTED, fontsize=8)

ax2.axvspan(pw["t"] - pw["seconds"], pw["t"], color=AQUA, alpha=0.18, lw=0)
ax2.annotate(f"pre-warm burst\n{pw['mb']} MB in {pw['seconds']} s",
             ((2 * pw["t"] - pw["seconds"]) / 2, 0.08), ha="center", color=INK_2, fontsize=9)
for i, (t0, t1) in enumerate(wakes):
    ax2.axvspan(t0, t1, color=YELLOW, alpha=0.3, lw=0)
    ax2.annotate(f"wake {t1 - t0:.2f} s", (t1, 0.95 - 0.0 * i),
                 textcoords="offset points", xytext=(4, 0), color=INK_2, fontsize=9)
for t in slept:
    ax2.axvline(t, color=BASELINE, linewidth=1)
    ax2.annotate("slept\n(evict requested)", (t, 0.55), textcoords="offset points",
                 xytext=(4, 0), color=INK_2, fontsize=9)

ax2.set_xlabel("time (s)")
ax2.set_ylabel("MHT fire confidence")
ax2.set_ylim(0, 1.05)
ax2.set_title("Lifecycle: pre-warm → wake → sleep → re-wake (resident mode)")
style(ax2)

fig.text(0.01, 0.01, f"model set: classifier 256 MB + vision tower 817 MB + AWQ LLM 1.6 GB · data: {results_dir}",
         color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(out_png, dpi=130, facecolor=SURFACE)
print("saved", out_png)

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

res = Path(sys.argv[1])

def load_jsonl(path):
    return [json.loads(l) for l in open(path)]

def scenario(name):
    samples = load_jsonl(res / f"tegrastats_{name}.jsonl")
    t0 = json.load(open(res / f"marks_{name}.json"))["action_start_epoch"]
    for s in samples:
        s["t"] = (s["epoch"] - t0) / 60
    return samples

burst = scenario("burst")
cont = scenario("continuous")
events = load_jsonl(res / "watcher_burst.jsonl")

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

def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

PANELS = [
    ("power_vdd_in_mw", "module power VDD_IN (W)", 1e-3),
    ("temp_tj", "junction temperature Tj (°C)", 1),
    ("ram_mb", "RAM used (GB)", 1e-3),
    ("cpu_pct", "CPU utilization, all-core avg (%)", 1),
    ("gpu_pct", "GPU utilization GR3D (%)", 1),
]

fig, axes = plt.subplots(len(PANELS), 1, figsize=(13, 13), sharex=True)
fig.suptitle(
    "Burst (fire-watcher) vs continuous operation — full VLM+classifier on Jetson Orin / SD Express",
    fontsize=13, fontweight="bold", color=INK,
)

wakes = [(s["t"] / 60, a["t"] / 60) for s, a in zip(
    (e for e in events if e["event"] == "wake_start"),
    (e for e in events if e["event"] == "awake"))]
sleeps = [e["t"] / 60 for e in events if e["event"] == "slept"]
prewarms = [((e["t"] - e["seconds"]) / 60, e["t"] / 60) for e in events if e["event"] == "prewarm_done"]

for ax, (key, label, scale) in zip(axes, PANELS):
    for data, color in ((cont, AQUA), (burst, BLUE)):
        xs = [s["t"] for s in data if key in s]
        ys = [s[key] * scale for s in data if key in s]
        ax.plot(xs, ys, color=color, linewidth=1.6)
    for a, b in prewarms:
        ax.axvspan(a, b, color=AQUA, alpha=0.15, lw=0)
    for a, b in wakes:
        ax.axvspan(a, b, color=YELLOW, alpha=0.22, lw=0)
    for (a, b), sl in zip(wakes, sleeps):
        ax.axvspan(b, sl, color=BLUE, alpha=0.08, lw=0)
    ax.set_ylabel(label, fontsize=9)
    style(ax)

axes[0].legend(["continuous (always-on)", "burst (fire-watcher)"],
               frameon=False, labelcolor=INK_2, loc="upper left", ncols=2)
axes[0].annotate("shaded: pre-warm (green wash) / container wake (yellow) / model awake (blue)",
                 (0.01, 1.04), xycoords="axes fraction", color=MUTED, fontsize=9)
axes[-1].set_xlabel("time since scenario start (min); t<0 = idle baseline")

fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig(res / "perf_timeseries.png", dpi=130, facecolor=SURFACE)
print("saved", res / "perf_timeseries.png")

def window(data, lo=0, hi=11.5):
    return [s for s in data if lo <= s["t"] <= hi]

def stats(data):
    w = window(data)
    p = [s["power_vdd_in_mw"] for s in w if "power_vdd_in_mw" in s]
    return {
        "energy_wh": sum(p) / 1000 / 3600,
        "avg_power_w": np.mean(p) / 1000,
        "peak_power_w": max(p) / 1000,
        "avg_tj": np.mean([s["temp_tj"] for s in w]),
        "max_tj": max(s["temp_tj"] for s in w),
        "avg_ram_gb": np.mean([s["ram_mb"] for s in w]) / 1000,
        "avg_cpu": np.mean([s["cpu_pct"] for s in w]),
        "avg_gpu": np.mean([s["gpu_pct"] for s in w]),
    }

sb, sc = stats(burst), stats(cont)
json.dump({"burst": sb, "continuous": sc}, open(res / "summary_stats.json", "w"), indent=2)

METRICS = [
    ("energy_wh", "energy over window (Wh)", "%.2f"),
    ("avg_power_w", "average power (W)", "%.1f"),
    ("peak_power_w", "peak power (W)", "%.1f"),
    ("avg_tj", "mean Tj (°C)", "%.1f"),
    ("max_tj", "peak Tj (°C)", "%.1f"),
    ("avg_ram_gb", "mean RAM used (GB)", "%.1f"),
    ("avg_cpu", "mean CPU (%)", "%.0f"),
    ("avg_gpu", "mean GPU (%)", "%.0f"),
]

fig, axes = plt.subplots(2, 4, figsize=(13, 6))
fig.suptitle(
    "Burst vs continuous — summary over the 11.5-min active window (idle baseline excluded)",
    fontsize=13, fontweight="bold", color=INK,
)
for ax, (key, label, fmt) in zip(axes.flat, METRICS):
    vals = [sb[key], sc[key]]
    bars = ax.bar(["burst", "continuous"], vals, width=0.55, color=[BLUE, AQUA])
    ax.bar_label(bars, labels=[fmt % v for v in vals], color=INK, fontsize=10, padding=2)
    ax.set_title(label, fontsize=10)
    ax.set_ylim(0, max(vals) * 1.25)
    style(ax)

fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(res / "perf_summary.png", dpi=130, facecolor=SURFACE)
print("saved", res / "perf_summary.png")
print(json.dumps({"burst": sb, "continuous": sc}, indent=2))

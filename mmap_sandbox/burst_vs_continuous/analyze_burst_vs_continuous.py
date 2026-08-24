#!/usr/bin/env python3
"""Parse + aggregate + plot the burst-vs-continuous comparison.

Reads each run's watch.jsonl (for wake_start/awake timestamps -> latency)
and tegrastats.log (for temperature/CPU/GPU/RAM/power), time-aligns every
tegrastats sample to that run's MHT trigger instant (t=0 = wake_start), and
produces:
  - summary.json: per-run and per-mode (mean/std/max) metrics
  - latency.png: latency comparison, mean+-std with individual reps
  - timeseries.png: Tj, CPU%, GPU%, board power over time since trigger
  - summary_panel.png: avg/max small-multiples across all tracked metrics
"""
import json
import statistics
import sys
from pathlib import Path

SANDBOX = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX))
from parse_tegrastats import parse_line  # noqa: E402

RUN_ROOT = SANDBOX / "results" / "burst_vs_continuous_run1"
RUNS = {
    "burst": ["burst_0", "burst_1", "burst_2"],
    "continuous703": ["continuous_0", "continuous_1", "continuous_3"],
    "continuous350": ["continuous_lp_0", "continuous_lp_1", "continuous_lp_2"],
}
# display label + color for each mode (ordered as they should appear)
MODE_META = {
    "burst": ("burst\n(full-throttle read)", "#2a78d6"),
    "continuous703": ("continuous @703 MB/s\n(paced at native speed)", "#1baf7a"),
    "continuous350": ("continuous @350 MB/s\n(paced at half speed)", "#eda100"),
}
EXCLUDED = {"continuous_2": "hung: GPU/CPU went idle at t~180s and never produced output before the 300s ready-timeout"}

METRICS = [
    ("temp_tj", "Tj (junction) temperature", "°C"),
    ("cpu_pct", "CPU utilization", "%"),
    ("gpu_pct", "GPU utilization (GR3D)", "%"),
    ("ram_mb", "RAM used", "MB"),
    ("power_vdd_in_mw", "board power (VDD_IN)", "mW"),
]

C = {"surface": "#fcfcfb", "grid": "#e1e0d9", "ink": "#0b0b0b", "ink2": "#52514e",
     "muted": "#898781", "axis": "#c3c2b7"}
C.update({mode: color for mode, (_, color) in MODE_META.items()})
MODES = list(MODE_META)


def load_run(tag):
    run_dir = RUN_ROOT / tag
    result = json.loads((run_dir / "result.json").read_text())
    t_launch = result["t_watcher_launch_epoch"]
    trigger_epoch = t_launch + result["wake_start"]["t"]
    latency = result["awake"]["wake_seconds"]

    records = []
    for line in (run_dir / "tegrastats.log").read_text().splitlines():
        rec = parse_line(line)
        if rec:
            rec["rel_t"] = rec["epoch"] - trigger_epoch
            records.append(rec)

    load_window = [r for r in records if 0 <= r["rel_t"] <= latency]
    return {"tag": tag, "latency": latency, "records": records, "load_window": load_window}


def window_stats(load_window, key):
    vals = [r[key] for r in load_window if key in r]
    if not vals:
        return None
    return {"avg": round(statistics.mean(vals), 2), "max": round(max(vals), 2)}


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {
        "mean": round(statistics.mean(vals), 2),
        "std": round(statistics.pstdev(vals), 2) if len(vals) > 1 else 0.0,
        "n": len(vals),
        "values": [round(v, 2) for v in vals],
    }


def build_summary(runs_by_mode):
    summary = {"excluded_runs": EXCLUDED, "modes": {}}
    for mode, runs in runs_by_mode.items():
        per_run = []
        for r in runs:
            row = {"tag": r["tag"], "latency_s": r["latency"]}
            for key, _, _ in METRICS:
                row[key] = window_stats(r["load_window"], key)
            per_run.append(row)
        agg = {"latency_s": mean_std([r["latency"] for r in runs])}
        for key, _, _ in METRICS:
            agg[f"{key}_avg"] = mean_std([row[key]["avg"] if row[key] else None for row in per_run])
            agg[f"{key}_max"] = mean_std([row[key]["max"] if row[key] else None for row in per_run])
        summary["modes"][mode] = {"per_run": per_run, "aggregate": agg}
    return summary


def style_ax(ax):
    ax.set_facecolor(C["surface"])
    ax.grid(True, axis="y", color=C["grid"], linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C["axis"])
    ax.tick_params(colors=C["muted"], labelsize=8)
    ax.title.set_color(C["ink"])
    ax.xaxis.label.set_color(C["ink2"])
    ax.yaxis.label.set_color(C["ink2"])


def plot_latency(summary, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 5.6), facecolor=C["surface"])
    modes = MODES
    means = [summary["modes"][m]["aggregate"]["latency_s"]["mean"] for m in modes]
    stds = [summary["modes"][m]["aggregate"]["latency_s"]["std"] for m in modes]
    colors = [C[m] for m in modes]
    x = np.arange(len(modes))
    ax.bar(x, means, yerr=stds, capsize=6, color=colors, width=0.6, zorder=3,
           error_kw={"ecolor": C["ink2"], "linewidth": 1.2})
    for i, m in enumerate(modes):
        vals = summary["modes"][m]["aggregate"]["latency_s"]["values"]
        jitter = np.linspace(-0.08, 0.08, len(vals))
        ax.scatter(x[i] + jitter, vals, color=C["ink"], s=22, zorder=4, alpha=0.75)
    for i, v in enumerate(means):
        ax.annotate(f"{v:.1f} s", (x[i], v + stds[i] + 3), ha="center", color=C["ink2"], fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_META[m][0] for m in modes], fontsize=9)
    ax.set_ylabel("time from MHT high-probability trigger to first VLM output (s)")
    ax.set_title("Wake latency vs. SD-card read pacing\nreal fused system, 3 cold-start reps/mode (dots = individual reps)", fontsize=11, fontweight="bold")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(outdir / "latency.png", dpi=150, facecolor=C["surface"])
    plt.close(fig)


def plot_timeseries(runs_by_mode, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series_keys = [("temp_tj", "Tj (°C)"), ("cpu_pct", "CPU (%)"),
                   ("gpu_pct", "GPU (%)"), ("power_vdd_in_mw", "board power (mW)")]
    fig, axes = plt.subplots(len(series_keys), 1, figsize=(11, 10), sharex=True, facecolor=C["surface"])

    t_max = 0
    for mode, runs in runs_by_mode.items():
        for r in runs:
            t_max = max(t_max, max((rec["rel_t"] for rec in r["records"]), default=0))
    t_max = min(t_max, 175)

    for ax, (key, ylabel) in zip(axes, series_keys):
        for mode, runs in runs_by_mode.items():
            color = C[mode]
            for r in runs:
                xs = [rec["rel_t"] for rec in r["records"] if key in rec and -8 <= rec["rel_t"] <= t_max]
                ys = [rec[key] for rec in r["records"] if key in rec and -8 <= rec["rel_t"] <= t_max]
                ax.plot(xs, ys, color=color, linewidth=0.9, alpha=0.35, zorder=2)
            mean_latency = statistics.mean(r["latency"] for r in runs)
            ax.axvline(mean_latency, color=color, linewidth=1.1, linestyle="--", alpha=0.7, zorder=1)
        ax.axvline(0, color=C["ink2"], linewidth=1.2, zorder=1)
        ax.set_ylabel(ylabel, fontsize=9)
        style_ax(ax)

    axes[0].annotate("MHT trigger (t=0)", (0.3, axes[0].get_ylim()[1] * 0.9), color=C["ink2"], fontsize=8)
    axes[-1].set_xlabel("time since MHT high-probability trigger (s)")
    fig.suptitle("System behavior during wake: burst vs. continuous (paced 703 & 350 MB/s)", fontsize=13, fontweight="bold", color=C["ink"], y=0.995)
    fig.text(0.5, 0.965, "thin lines = individual reps, dashed = mean completion time per mode",
              ha="center", fontsize=9.5, color=C["ink2"])
    fig.legend(handles=[
        plt.Line2D([0], [0], color=C[m], lw=2, label=MODE_META[m][0].replace("\n", " "))
        for m in MODES
    ], loc="upper right", bbox_to_anchor=(0.995, 0.958), frameon=False, fontsize=8, labelcolor=C["ink2"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outdir / "timeseries.png", dpi=150, facecolor=C["surface"])
    plt.close(fig)


def plot_summary_panel(summary, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor=C["surface"])
    axes = axes.flatten()
    modes = MODES
    short = {"burst": "burst", "continuous703": "cont.\n703", "continuous350": "cont.\n350"}
    x = np.arange(len(modes))

    ax0 = axes[0]
    means = [summary["modes"][m]["aggregate"]["latency_s"]["mean"] for m in modes]
    stds = [summary["modes"][m]["aggregate"]["latency_s"]["std"] for m in modes]
    ax0.bar(x, means, yerr=stds, capsize=5, color=[C[m] for m in modes], width=0.62, zorder=3)
    ax0.set_xticks(x)
    ax0.set_xticklabels([short[m] for m in modes], fontsize=8.5)
    ax0.set_title("wake latency (s)", fontsize=10, fontweight="bold")
    style_ax(ax0)

    for ax, (key, label, unit) in zip(axes[1:], METRICS):
        avg_means = [summary["modes"][m]["aggregate"][f"{key}_avg"]["mean"] for m in modes]
        avg_stds = [summary["modes"][m]["aggregate"][f"{key}_avg"]["std"] for m in modes]
        max_means = [summary["modes"][m]["aggregate"][f"{key}_max"]["mean"] for m in modes]
        max_stds = [summary["modes"][m]["aggregate"][f"{key}_max"]["std"] for m in modes]
        w = 0.4
        ax.bar(x - w / 2, avg_means, yerr=avg_stds, width=w, capsize=3,
               color=[C[m] for m in modes], zorder=3, label="avg")
        ax.bar(x + w / 2, max_means, yerr=max_stds, width=w, capsize=3,
               color=[C[m] for m in modes], alpha=0.45, hatch="///", zorder=3, label="max")
        ax.set_xticks(x)
        ax.set_xticklabels([short[m] for m in modes], fontsize=8.5)
        ax.set_title(f"{label} ({unit})", fontsize=10, fontweight="bold")
        style_ax(ax)

    fig.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, facecolor=C["ink2"], label="avg (solid)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=C["ink2"], alpha=0.45, hatch="///", label="max (hatched)"),
    ], loc="upper right", frameon=False, fontsize=9, labelcolor=C["ink2"], ncol=2)
    fig.suptitle("Averages & maxima during the load window (trigger → first output), mean±std across 3 reps",
                  fontsize=12, fontweight="bold", color=C["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outdir / "summary_panel.png", dpi=150, facecolor=C["surface"])
    plt.close(fig)


def main():
    runs_by_mode = {mode: [load_run(tag) for tag in tags] for mode, tags in RUNS.items()}
    summary = build_summary(runs_by_mode)
    (RUN_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["modes"], indent=2))

    plot_latency(summary, RUN_ROOT)
    plot_timeseries(runs_by_mode, RUN_ROOT)
    plot_summary_panel(summary, RUN_ROOT)
    print(f"\nfigures + summary.json in {RUN_ROOT}")


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

C = {"seq": "#1f77b4", "rand": "#d62728", "warm": "#2ca02c", "cold": "#7f7f7f",
     "burst": "#1b9e77", "cont": "#d95f02", "accent": "#333333", "slc": "#e8b800"}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.grid": True, "grid.color": "#e6e6e6",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.edgecolor": "#cccccc",
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "legend.frameon": False, "figure.autolayout": False,
})


def loadj(p):
    return json.load(open(RES / p))


def loadl(p):
    return [json.loads(x) for x in open(RES / p)]


def headline(ax, text, y=1.14):
    ax.text(0, y, text, transform=ax.transAxes, fontsize=11.5,
            color=C["accent"], fontweight="bold", va="bottom")


def fig_seq_read():
    sr = loadj("sustained_rate/sustained_rate.json")
    ch = sr["chunks"]
    t = np.array([c["t"] for c in ch])
    mb = np.cumsum([c["mb"] for c in ch])
    v = np.array([c["mbps"] for c in ch])
    rr = loadj("seq_read_rerun/seq_read_rerun.json")
    ov = np.array([r["overall_mbps"] for r in rr["runs"]])

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6),
                               gridspec_kw={"width_ratios": [2.3, 1]})
    a.scatter(mb, v, s=10, color=C["seq"], alpha=0.5, label="16 MB chunk")
    k = 9
    sm = np.convolve(v, np.ones(k) / k, mode="valid")
    a.plot(mb[k - 1:], sm, color="#08306b", lw=2, label="rolling mean")
    a.axhline(sr["overall_mbps"], color=C["rand"], ls="--", lw=1.5,
              label=f"overall {sr['overall_mbps']:.0f} MB/s")
    a.set_xlabel("cumulative data read (MB)")
    a.set_ylabel("throughput (MB/s)")
    a.set_ylim(0, 1000)
    a.set_title("Sequential read is fast and flat across the full 2.8 GB")
    a.legend(loc="lower right")
    headline(a, "No SLC cliff on read — DRAM-less controller holds ~700 MB/s end to end")

    b.bar(range(len(ov)), ov, color=C["seq"], alpha=0.85)
    b.axhline(ov.mean(), color=C["rand"], ls="--", lw=1.5)
    b.text(0.5, ov.mean() + 8, f"mean {ov.mean():.0f} ± {ov.std():.0f} MB/s",
           transform=b.get_yaxis_transform() if False else b.transData,
           ha="left", color=C["rand"], fontsize=10, fontweight="bold")
    b.set_xticks(range(len(ov)))
    b.set_xticklabels([f"#{i+1}" for i in range(len(ov))])
    b.set_ylim(0, 900)
    b.set_xlabel("cold run")
    b.set_ylabel("overall MB/s")
    b.set_title("Run-to-run consistency (5× cold)")
    fig.tight_layout()
    fig.savefig(OUT / "F1_sequential_read.png", bbox_inches="tight")
    plt.close(fig)


def fig_write_curve():
    w = loadj("write_curve/write_curve.json")
    g = np.array([m["cum_gb"] for m in w["marks"]])
    v = np.array([m["mbps"] for m in w["marks"]])
    peak = v.max()
    tail = v[len(v) // 2:].mean()

    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(g, v, color=C["cont"], lw=2, marker="o", ms=4)
    knee = g[np.argmax(v < peak * 0.6)] if np.any(v < peak * 0.6) else g[-1]
    ax.axvspan(0, knee, color=C["slc"], alpha=0.12)
    ax.axvspan(knee, g[-1], color="#cccccc", alpha=0.18)
    ax.axhline(tail, color=C["accent"], ls="--", lw=1.3)
    ax.text(g[-1], tail + 10, f"sustained ~{tail:.0f} MB/s", ha="right",
            color=C["accent"], fontsize=10, fontweight="bold")
    ax.text(knee / 2, peak + 12, "SLC burst", ha="center", color="#8a6d00",
            fontsize=10, fontweight="bold")
    ax.text((knee + g[-1]) / 2, peak + 12, "folded to TLC (throttled)",
            ha="center", color="#666666", fontsize=10, fontweight="bold")
    ax.set_xlabel("cumulative data written (GiB)")
    ax.set_ylabel("write throughput (MB/s)")
    ax.set_ylim(0, peak * 1.25)
    ax.set_title("Sustained write: SLC-cache burst, then throttle")
    headline(ax, f"Peak {peak:.0f} MB/s → {tail:.0f} MB/s once the DRAM-less SLC cache saturates")
    fig.tight_layout()
    fig.savefig(OUT / "F2_write_curve.png", bbox_inches="tight")
    plt.close(fig)


def fig_access_pattern():
    sw = loadj("rand_read_sweep/rand_read_sweep.json")["rows"]
    rnd = sorted([r for r in sw if r["pattern"] == "random"], key=lambda r: r["block_bytes"])
    seq = sorted([r for r in sw if r["pattern"] == "sequential"], key=lambda r: r["block_bytes"])
    bs = [r["block_bytes"] / 1024 for r in rnd]
    beta = loadl("beta_madvise/runs.jsonl")
    chn = sorted([r for r in beta if r["advice"] == "sequential" and not r["warm"]],
                 key=lambda r: r["chunk_mb"])

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    a.plot(bs, [r["mbps"] for r in seq], "-o", color=C["seq"], lw=2, label="sequential")
    a.plot(bs, [r["mbps"] for r in rnd], "-o", color=C["rand"], lw=2, label="random")
    a.set_xscale("log", base=2)
    a.set_xticks(bs)
    a.set_xticklabels([f"{int(x)}K" if x < 1024 else "1M" for x in bs])
    a.set_xlabel("read block size")
    a.set_ylabel("throughput (MB/s)")
    a.set_title("Access pattern is the dominant lever")
    a.legend()
    gap = seq[0]["mbps"] / rnd[0]["mbps"]
    a.annotate(f"{gap:.0f}× gap\nat 4K", xy=(bs[0], rnd[0]["mbps"]),
               xytext=(bs[0] * 1.4, seq[0]["mbps"] * 0.6), color=C["rand"],
               fontsize=10, fontweight="bold",
               arrowprops=dict(arrowstyle="->", color=C["rand"]))
    headline(a, "Random 4K reads pay a 15× penalty; the gap closes as blocks grow")

    if chn:
        cm = [r["chunk_mb"] for r in chn]
        cv = [r["avg_mbps"] for r in chn]
        b.bar([str(c) for c in cm], cv, color=C["seq"], alpha=0.85)
        b.axhline(np.mean(cv), color=C["rand"], ls="--", lw=1.3,
                  label=f"mean {np.mean(cv):.0f} MB/s")
        b.set_ylim(0, max(cv) * 1.3)
        b.legend()
    b.set_xlabel("mmap read chunk size (MB)")
    b.set_ylabel("throughput (MB/s)")
    b.set_title("Chunk size barely matters (kernel does the bursting)")
    fig.tight_layout()
    fig.savefig(OUT / "F3_access_pattern.png", bbox_inches="tight")
    plt.close(fig)


def fig_cache():
    beta = loadl("beta_madvise/runs.jsonl")

    def pick(warm, advice="normal"):
        xs = [r for r in beta if r["warm"] == warm and r["advice"] == advice]
        return np.mean([r["avg_mbps"] for r in xs]) if xs else None

    cold = pick(False)
    warm = pick(True)
    rnd = pick(False, "random")
    labels, vals, cols = [], [], []
    if rnd is not None:
        labels.append("random\n(MADV_RANDOM)"); vals.append(rnd); cols.append(C["rand"])
    if cold is not None:
        labels.append("cold\n(page cache empty)"); vals.append(cold); cols.append(C["cold"])
    if warm is not None:
        labels.append("warm\n(pre-warmed cache)"); vals.append(warm); cols.append(C["warm"])

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color=cols, alpha=0.9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(vals) * 0.02,
                f"{val:.0f}", ha="center", fontweight="bold")
    ax.set_ylabel("throughput (MB/s)")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_title("Pre-warming the page cache turns SD reads into RAM reads")
    if warm and cold:
        headline(ax, f"Cross-process pre-warm: {warm/cold:.1f}× faster than cold; "
                     f"{cold/rnd:.0f}× the random-advice floor" if rnd else
                     f"Warm cache is {warm/cold:.1f}× cold")
    fig.tight_layout()
    fig.savefig(OUT / "F4_cache.png", bbox_inches="tight")
    plt.close(fig)


def fig_system_metrics():
    s = loadj("perf_burst_vs_continuous/summary_stats.json")
    bu, co = s["burst"], s["continuous"]
    metrics = [
        ("Energy", "energy_wh", "Wh"), ("Avg power", "avg_power_w", "W"),
        ("Avg T_j", "avg_tj", "°C"), ("Avg RAM", "avg_ram_gb", "GB"),
        ("Avg CPU", "avg_cpu", "%"), ("Avg GPU", "avg_gpu", "%"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (name, key, unit) in zip(axes.ravel(), metrics):
        bv, cv = bu[key], co[key]
        bars = ax.bar(["burst\n(gated)", "continuous\n(always-on)"], [bv, cv],
                      color=[C["burst"], C["cont"]], alpha=0.9)
        for bar, val in zip(bars, [bv, cv]):
            ax.text(bar.get_x() + bar.get_width() / 2, val,
                    f"{val:.1f}", ha="center", va="bottom", fontweight="bold")
        delta = (bv - cv) / cv * 100
        ax.set_title(f"{name}  ({delta:+.0f}%)")
        ax.set_ylabel(unit)
        ax.set_ylim(0, max(bv, cv) * 1.25)
    fig.suptitle("Gated wake vs always-on: 11.5 min window, two fire episodes (~73% active)",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "F5_system_metrics.png", bbox_inches="tight")
    plt.close(fig)


def fig_power_timeline():
    def series(tag):
        rows = loadl(f"perf_burst_vs_continuous/tegrastats_{tag}.jsonl")
        mark = loadj(f"perf_burst_vs_continuous/marks_{tag}.json")["action_start_epoch"]
        t = np.array([r["epoch"] - mark for r in rows]) / 60.0
        p = np.array([r["power_vdd_in_mw"] for r in rows]) / 1000.0
        return t, p

    tb, pb = series("burst")
    tc, pc = series("continuous")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(tc, pc, color=C["cont"], lw=1.4, alpha=0.9, label="continuous (always-on)")
    ax.plot(tb, pb, color=C["burst"], lw=1.4, alpha=0.9, label="burst (gated)")
    ax.set_xlabel("time from first trigger (min)")
    ax.set_ylabel("board power VDD_IN (W)")
    ax.set_title("Power over a two-fire window: gated wake returns to the idle floor")
    ax.legend(loc="upper right")
    headline(ax, "Between fires, burst drops to ~5.5 W idle; always-on stays ~15–18 W")
    fig.tight_layout()
    fig.savefig(OUT / "F6_power_timeline.png", bbox_inches="tight")
    plt.close(fig)


def fig_wake_anatomy():
    ph = loadj("wake_latency/phases_cold_1.json")
    sr = loadj("sustained_rate/sustained_rate.json")
    sd_s = sr["total_mb"] / sr["overall_mbps"]
    names = [p[0] for p in ph["phases"]]
    secs = [p[1] for p in ph["phases"]]
    awq_i = next(i for i, n in enumerate(names) if "AWQ" in n)
    awq_start = sum(secs[:awq_i])
    bc = loadj("burst_vs_continuous_run1/summary.json")["modes"]
    lat = {"burst": bc["burst"]["aggregate"]["latency_s"],
           "cont @703": bc["continuous703"]["aggregate"]["latency_s"],
           "cont @350": bc["continuous350"]["aggregate"]["latency_s"]}

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8),
                               gridspec_kw={"width_ratios": [1.8, 1]})
    cmap = plt.get_cmap("Blues")
    left = 0
    for i, (n, s) in enumerate(zip(names, secs)):
        a.barh(0, s, left=left, color=cmap(0.30 + 0.5 * i / len(names)),
               edgecolor="white")
        if s > 6:
            a.text(left + s / 2, 0.32, f"{s:.0f}s", ha="center", va="center",
                   color="white", fontsize=9, fontweight="bold")
        left += s
    a.barh(0, sd_s, left=awq_start, color=C["rand"], edgecolor="white", zorder=5)
    a.annotate(f"SD weight read\n~{sd_s:.0f}s (2.8 GB)", xy=(awq_start + sd_s, 0.15),
               xytext=(awq_start + 14, 0.62), color=C["rand"], fontsize=9,
               fontweight="bold", ha="left",
               arrowprops=dict(arrowstyle="->", color=C["rand"]))
    a.set_ylim(-0.5, 0.9)
    a.set_yticks([])
    a.set_xlabel("seconds")
    a.set_xlim(0, left)
    a.set_title(f"Anatomy of a {ph['total_wake_seconds']:.0f}s wake (trigger → first VLM verdict)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(0.5)),
               plt.Rectangle((0, 0), 1, 1, color=C["rand"])]
    a.legend(handles, ["framework / CUDA / dequant / inference", "actual SD read"],
             loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.18))
    headline(a, "The SD read is a ~4 s sliver of a 118 s wake — storage is never the bottleneck")

    names2 = list(lat.keys())
    means = [lat[k]["mean"] for k in names2]
    stds = [lat[k]["std"] for k in names2]
    cols = [C["burst"], C["cont"], "#f0a35e"]
    b.bar(names2, means, yerr=stds, capsize=4, color=cols, alpha=0.9)
    for i, m in enumerate(means):
        b.text(i, m + 3, f"{m:.0f}s", ha="center", fontweight="bold")
    b.set_ylabel("wake latency (s)")
    b.set_ylim(0, max(means) * 1.25)
    b.set_title("Halving read bandwidth\nadds only ~7%")
    fig.tight_layout()
    fig.savefig(OUT / "F7_wake_anatomy.png", bbox_inches="tight")
    plt.close(fig)


def main():
    for fn in [fig_seq_read, fig_write_curve, fig_access_pattern, fig_cache,
               fig_system_metrics, fig_power_timeline, fig_wake_anatomy]:
        fn()
        print("ok:", fn.__name__)


if __name__ == "__main__":
    main()

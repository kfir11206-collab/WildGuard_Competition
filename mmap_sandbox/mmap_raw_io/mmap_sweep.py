#!/usr/bin/env python3
"""Race a fixed data volume to durable storage: burst vs continuous mmap writes.

One knob separates the two: the msync interval. Each config writes --total-mb
through mmap in --chunk-mb pieces, msyncing every <burst> MB (burst == chunk
means continuous; burst 0 means no msync until the end). The clock stops after
the final fsync, so every config races to the same "all data durable" finish.

Sweeps --burst-mb, reports total time / effective throughput / peak Tj per
config, and plots the comparison.
"""

import argparse
import csv
import json
import mmap
import os
import time
from datetime import datetime
from pathlib import Path

import psutil

from mmap_bench import find_thermal_zones, read_temps


def run_config(burst_mb, path, total, chunk, zones, samples):
    """Write `total` bytes, msync every burst_mb MB. Returns wall seconds."""
    burst = burst_mb * 1024 * 1024 if burst_mb else total
    with open(path, "wb") as f:
        f.truncate(total)
    fd = os.open(path, os.O_RDWR)
    try:
        mm = mmap.mmap(fd, total, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        data = os.urandom(chunk)
        t0 = time.monotonic()
        sync_from = 0
        for off in range(0, total, chunk):
            c0 = time.monotonic()
            mm[off:off + chunk] = data
            end = off + chunk
            if end - sync_from >= burst or end == total:
                mm.flush(sync_from, end - sync_from)
                sync_from = end
            c1 = time.monotonic()
            samples.append({
                "burst_mb": burst_mb,
                "t": c1 - t0,
                "mbps": chunk / 1e6 / max(c1 - c0, 1e-9),
                "cpu_pct": psutil.cpu_percent(None),
                **{k.split("-")[0] + "_c": v for k, v in read_temps(zones).items()},
            })
        mm.close()
        os.fsync(fd)
        elapsed = time.monotonic() - t0
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    os.unlink(path)
    return elapsed


def plot(summary, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = {"surface": "#fcfcfb", "grid": "#e1e0d9", "ink": "#0b0b0b",
         "ink2": "#52514e", "muted": "#898781", "axis": "#c3c2b7",
         "s1": "#2a78d6", "s2": "#1baf7a"}
    labels = [s["label"] for s in summary]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor=C["surface"])
    for ax, key, color, title, unit in (
            (ax1, "eff_MBps", C["s1"], "effective throughput (higher = finishes first)", "MB/s"),
            (ax2, "tj_max_c", C["s2"], "peak Tj during config", "°C")):
        vals = [s[key] for s in summary]
        bars = ax.bar(labels, vals, color=color, width=0.62, zorder=3)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=8, color=C["ink2"],
                        xytext=(0, 2), textcoords="offset points")
        ax.set_title(title, fontsize=10, color=C["ink"])
        ax.set_ylabel(unit, color=C["ink2"])
        ax.set_xlabel("msync interval (MB)", color=C["ink2"])
        ax.set_facecolor(C["surface"])
        ax.grid(True, axis="y", color=C["grid"], linewidth=0.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C["axis"])
        ax.tick_params(colors=C["muted"], labelsize=8)
    ax2.set_ylim(min(s["tj_max_c"] for s in summary) - 2,
                 max(s["tj_max_c"] for s in summary) + 2)
    fig.tight_layout()
    fig.savefig(outdir / "sweep.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", required=True,
                    help="directory on the filesystem under test (e.g. SD mount)")
    ap.add_argument("--total-mb", type=int, default=10240)
    ap.add_argument("--chunk-mb", type=int, default=4)
    ap.add_argument("--burst-mb", default="4,16,64,256,1024,0",
                    help="comma list of msync intervals in MB; 0 = only at end; "
                         "the value equal to --chunk-mb is the continuous case")
    ap.add_argument("--cooldown", type=float, default=5.0,
                    help="seconds to idle between configs")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    total = args.total_mb * 1024 * 1024
    chunk = args.chunk_mb * 1024 * 1024
    bursts = [int(b) for b in args.burst_mb.split(",")]
    testfile = Path(args.target) / "mmap_sweep.bin"
    outdir = Path(args.outdir) if args.outdir else (
        Path(__file__).parent.parent / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_sweep"))
    outdir.mkdir(parents=True, exist_ok=True)

    zones = find_thermal_zones()
    psutil.cpu_percent(None)
    samples, summary = [], []
    for burst_mb in bursts:
        label = ("end-only" if burst_mb == 0 else
                 f"{burst_mb} (cont.)" if burst_mb == args.chunk_mb else str(burst_mb))
        print(f"config msync={label} MB: writing {args.total_mb} MB ...", flush=True)
        elapsed = run_config(burst_mb, testfile, total, chunk, zones, samples)
        rows = [s for s in samples if s["burst_mb"] == burst_mb]
        summary.append({
            "label": label,
            "burst_mb": burst_mb,
            "total_s": round(elapsed, 2),
            "eff_MBps": round(total / 1e6 / elapsed, 2),
            "tj_max_c": max(s.get("tj_c", 0) for s in rows),
            "cpu_avg_pct": round(sum(s["cpu_pct"] for s in rows) / len(rows), 1),
        })
        print(f"  -> {elapsed:.2f}s  ({summary[-1]['eff_MBps']} MB/s)", flush=True)
        time.sleep(args.cooldown)

    with open(outdir / "samples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for s in samples for k in s}))
        w.writeheader()
        w.writerows(samples)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    best = max(summary, key=lambda s: s["eff_MBps"])
    print(json.dumps(summary, indent=2))
    print(f"winner: msync interval {best['label']} MB "
          f"({best['total_s']}s for {args.total_mb} MB)")
    plot(summary, outdir)
    print(f"results in {outdir}")


if __name__ == "__main__":
    main()

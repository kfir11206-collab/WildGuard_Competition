#!/usr/bin/env python3
"""Benchmark mmap burst vs continuous read/write on a target filesystem (e.g. SD card).

Four sequential phases against one file in --target:
  burst_write  : write chunk + msync, back-to-back (device-limited)
  burst_read   : fault chunks back in, back-to-back (page cache evicted first)
  cont_write   : same, paced at --rate MB/s
  cont_read    : same, paced at --rate MB/s

Per chunk we record throughput, CPU%, and SoC temperatures (cpu/gpu/tj).
Outputs results.csv, summary.json and two PNG plots into --target-independent
results dir (default: mmap_sandbox/results/<timestamp>/).
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

THERMAL_TYPES = ("cpu-thermal", "gpu-thermal", "tj-thermal")


def find_thermal_zones():
    zones = {}
    for z in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            ztype = (z / "type").read_text().strip()
        except OSError:
            continue
        if ztype in THERMAL_TYPES:
            zones[ztype] = z / "temp"
    return zones


def read_temps(zones):
    out = {}
    for name, path in zones.items():
        try:
            out[name] = int(path.read_text()) / 1000.0
        except (OSError, ValueError):
            out[name] = float("nan")
    return out


def run_phase(phase, path, size, chunk, rate_bps, zones, samples, t0):
    """Run one phase; append one sample dict per chunk to samples."""
    write = phase.endswith("write")
    fd = os.open(path, os.O_RDWR)
    try:
        mm = mmap.mmap(fd, size, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        data = os.urandom(chunk) if write else None
        sink = 0
        start = time.monotonic()
        for off in range(0, size, chunk):
            c0 = time.monotonic()
            if write:
                mm[off:off + chunk] = data
                mm.flush(off, chunk)
            else:
                sink += bytes(mm[off:off + chunk])[0]
            c1 = time.monotonic()
            temps = read_temps(zones)
            samples.append({
                "phase": phase,
                "t_wall": c1 - t0,
                "t_phase": c1 - start,
                "mb": chunk / 1e6,
                "mbps": chunk / 1e6 / max(c1 - c0, 1e-9),
                "cpu_pct": psutil.cpu_percent(None),
                **{k.split("-")[0] + "_c": v for k, v in temps.items()},
            })
            if rate_bps:  # continuous: sleep until this chunk was "due"
                due = start + (off + chunk) / rate_bps
                if due > time.monotonic():
                    time.sleep(due - time.monotonic())
        mm.close()
        # flush + evict page cache so the next read phase hits the device
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def plot(samples, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = {"surface": "#fcfcfb", "grid": "#e1e0d9", "ink": "#0b0b0b",
         "ink2": "#52514e", "muted": "#898781", "axis": "#c3c2b7",
         "s1": "#2a78d6", "s2": "#1baf7a", "s3": "#eda100"}

    def style(ax):
        ax.set_facecolor(C["surface"])
        ax.grid(True, color=C["grid"], linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C["axis"])
        ax.tick_params(colors=C["muted"], labelsize=8)
        ax.margins(x=0.08)
        ax.title.set_color(C["ink"])
        ax.xaxis.label.set_color(C["ink2"])
        ax.yaxis.label.set_color(C["ink2"])

    def series(phase, key):
        rows = [s for s in samples if s["phase"] == phase]
        return [s["t_phase"] for s in rows], [s[key] for s in rows]

    # Figure 1: throughput, write vs read subplots, burst vs continuous series
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=C["surface"])
    for ax, op in zip(axes, ("write", "read")):
        for mode, color in (("burst", C["s1"]), ("cont", C["s2"])):
            x, y = series(f"{mode}_{op}", "mbps")
            label = "burst" if mode == "burst" else "continuous"
            ax.plot(x, y, color=color, linewidth=1.8, label=label)
            if x:
                ax.annotate(label, (x[-1], y[-1]), textcoords="offset points",
                            xytext=(4, 0), fontsize=8, color=C["ink2"])
        ax.set_title(f"mmap {op} throughput", fontsize=10)
        ax.set_xlabel("phase time (s)")
        ax.set_ylabel("MB/s")
        ax.legend(fontsize=8, frameon=False, labelcolor=C["ink2"])
        style(ax)
    fig.tight_layout()
    fig.savefig(outdir / "throughput.png", dpi=140)
    plt.close(fig)

    # Figure 2: temperature + CPU over the whole run, phases shaded
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True,
                                   facecolor=C["surface"])
    tw = [s["t_wall"] for s in samples]
    temp_series = [("cpu_c", "CPU", C["s1"]), ("gpu_c", "GPU", C["s2"]),
                   ("tj_c", "Tj", C["s3"])]
    for i, (key, label, color) in enumerate(temp_series):
        if any(key in s for s in samples):
            y = [s.get(key, float("nan")) for s in samples]
            ax1.plot(tw, y, color=color, linewidth=1.8, label=label)
            ax1.annotate(label, (tw[-1], y[-1]), textcoords="offset points",
                         xytext=(4, 6 - 6 * i), fontsize=8, color=C["ink2"])
    ax1.set_title("SoC temperature", fontsize=10)
    ax1.set_ylabel("°C")
    ax1.legend(fontsize=8, frameon=False, labelcolor=C["ink2"])
    ax2.plot(tw, [s["cpu_pct"] for s in samples], color=C["s1"], linewidth=1.8)
    ax2.set_title("CPU utilization", fontsize=10)
    ax2.set_ylabel("%")
    ax2.set_xlabel("run time (s)")
    for ax in (ax1, ax2):
        style(ax)
        # shade phase spans
        for i, phase in enumerate(dict.fromkeys(s["phase"] for s in samples)):
            x, _ = series(phase, "mbps")
            rows = [s["t_wall"] for s in samples if s["phase"] == phase]
            if rows and i % 2:
                ax.axvspan(rows[0], rows[-1], color=C["grid"], alpha=0.4, lw=0)
    for i, phase in enumerate(dict.fromkeys(s["phase"] for s in samples)):
        rows = [s["t_wall"] for s in samples if s["phase"] == phase]
        ax2.annotate(phase, ((rows[0] + rows[-1]) / 2, ax2.get_ylim()[0]),
                     ha="center", va="bottom", fontsize=7, color=C["muted"],
                     xytext=(0, 10 * (i % 2)), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(outdir / "system.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", required=True,
                    help="directory on the filesystem under test (e.g. SD mount)")
    ap.add_argument("--size-mb", type=int, default=256)
    ap.add_argument("--chunk-mb", type=int, default=4)
    ap.add_argument("--rate", type=float, default=4.0,
                    help="continuous-mode pace in MB/s")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    size = args.size_mb * 1024 * 1024
    chunk = args.chunk_mb * 1024 * 1024
    target = Path(args.target)
    testfile = target / "mmap_bench.bin"
    outdir = Path(args.outdir) if args.outdir else (
        Path(__file__).parent.parent / "results" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    outdir.mkdir(parents=True, exist_ok=True)

    # preallocate the file
    with open(testfile, "wb") as f:
        f.truncate(size)

    zones = find_thermal_zones()
    psutil.cpu_percent(None)  # prime the counter
    samples = []
    t0 = time.monotonic()
    phases = [("burst_write", 0), ("burst_read", 0),
              ("cont_write", args.rate * 1e6), ("cont_read", args.rate * 1e6)]
    for phase, rate in phases:
        print(f"[{time.monotonic()-t0:7.1f}s] {phase} ...", flush=True)
        run_phase(phase, testfile, size, chunk, rate, zones, samples, t0)
    testfile.unlink()

    with open(outdir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for s in samples for k in s}))
        w.writeheader()
        w.writerows(samples)

    summary = {}
    for phase, _ in phases:
        rows = [s for s in samples if s["phase"] == phase]
        mb = sum(s["mb"] for s in rows)
        dur = rows[-1]["t_phase"]
        summary[phase] = {
            "MB": round(mb, 1),
            "duration_s": round(dur, 2),
            "avg_MBps": round(mb / dur, 2),
            "chunk_MBps_min": round(min(s["mbps"] for s in rows), 2),
            "chunk_MBps_max": round(max(s["mbps"] for s in rows), 2),
            "temp_max_c": max((s.get("tj_c") or s.get("cpu_c") or 0) for s in rows),
        }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    plot(samples, outdir)
    print(f"results in {outdir}")


if __name__ == "__main__":
    main()

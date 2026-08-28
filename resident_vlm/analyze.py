#!/usr/bin/env python3
"""Turn bench.py's raw samples into per-window energy / throughput / thermal numbers."""
import argparse
import json
import statistics
from pathlib import Path

WINDOWS = [
    ("idle_asleep", "idle_start", "idle_end"),
    ("wake", "trigger", "first_verdict"),
    ("awake", "awake_start", "awake_end"),
    ("sleep_transition", "sleep_start", "sleep_done"),
    ("idle_after", "mht_restarted", "sleep_idle_end"),
]


def load(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def integrate(samples, t0, t1, key="power_vdd_in_mw"):
    """Trapezoidal integral of mW over seconds -> joules, plus mean W."""
    pts = [(s["t"], s[key]) for s in samples if t0 <= s["t"] <= t1 and key in s]
    if len(pts) < 2:
        return None, None
    joules = sum(
        (pts[i + 1][0] - pts[i][0]) * (pts[i + 1][1] + pts[i][1]) / 2.0 / 1000.0
        for i in range(len(pts) - 1)
    )
    span = pts[-1][0] - pts[0][0]
    return round(joules, 2), (round(joules / span, 2) if span > 0 else None)


def window_stats(samples, t0, t1):
    inside = [s for s in samples if t0 <= s["t"] <= t1]
    if len(inside) < 2:
        return None
    energy_j, mean_w = integrate(samples, t0, t1)
    rd = inside[-1]["disk_read_bytes"] - inside[0]["disk_read_bytes"]
    span = inside[-1]["t"] - inside[0]["t"]
    read_ms = inside[-1].get("disk_read_ms", 0) - inside[0].get("disk_read_ms", 0)
    reads = inside[-1].get("disk_reads_completed", 0) - inside[0].get("disk_reads_completed", 0)
    jumps = [
        inside[i + 1].get("disk_read_ms", 0) - inside[i].get("disk_read_ms", 0)
        for i in range(len(inside) - 1)
    ]
    tj = [s["temp_tj_c"] for s in inside if "temp_tj_c" in s]
    avail = [s["mem_available_mb"] for s in inside if "mem_available_mb" in s]
    return {
        "seconds": round(span, 3),
        "energy_j": energy_j,
        "mean_power_w": mean_w,
        "sd_read_mb": round(rd / 1e6, 1),
        "sd_read_mb_s": round(rd / 1e6 / span, 1) if span > 0 else None,
        "swap_in_mb": round((inside[-1].get("pswpin_bytes", 0)
                             - inside[0].get("pswpin_bytes", 0)) / 1e6, 1),
        "swap_out_mb": round((inside[-1].get("pswpout_bytes", 0)
                              - inside[0].get("pswpout_bytes", 0)) / 1e6, 1),
        "sd_read_ms": read_ms,
        "sd_mean_read_latency_ms": round(read_ms / reads, 3) if reads else None,
        "sd_max_stall_ms": max(jumps) if jumps else None,
        "sd_stalls_over_500ms": sum(1 for j in jumps if j > 500),
        "tj_start_c": tj[0] if tj else None,
        "tj_peak_c": max(tj) if tj else None,
        "mem_available_min_mb": min(avail) if avail else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rundir")
    args = p.parse_args()
    rundir = Path(args.rundir)

    rows = []
    for marks_path in sorted(rundir.glob("*.marks.json")):
        tag = marks_path.name[: -len(".marks.json")]
        samples_path = rundir / f"{tag}.jsonl"
        rec_path = rundir / f"{tag}.json"
        if not samples_path.exists():
            continue
        samples = load(samples_path)
        marks = {m["name"]: m["t"] for m in json.loads(marks_path.read_text())}
        rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
        row = {
            "tag": tag,
            "arm": rec.get("arm"),
            "rep": rec.get("rep"),
            "trigger_to_verdict_s": rec.get("trigger_to_verdict_seconds"),
        }
        for name, a, b in WINDOWS:
            if a in marks and b in marks:
                row[name] = window_stats(samples, marks[a], marks[b])
        rows.append(row)

    (rundir / "analysis.json").write_text(json.dumps(rows, indent=2))

    hdr = f"{'arm':<10}{'rep':<5}{'t2v_s':>8}{'wake_J':>9}{'wake_W':>8}{'SD_MB':>8}{'MB/s':>8}{'swapIn':>8}{'stallms':>9}{'sleepW':>8}{'tj_pk':>7}{'freeMB':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        w = r.get("wake") or {}
        idle = r.get("idle_asleep") or {}
        print(f"{str(r['arm']):<10}{str(r['rep']):<5}"
              f"{str(r['trigger_to_verdict_s']):>8}{str(w.get('energy_j')):>9}"
              f"{str(w.get('mean_power_w')):>8}{str(w.get('sd_read_mb')):>8}"
              f"{str(w.get('sd_read_mb_s')):>8}{str(w.get('swap_in_mb')):>8}"
              f"{str(w.get('sd_max_stall_ms')):>9}"
              f"{str(idle.get('mean_power_w')):>8}"
              f"{str(w.get('tj_peak_c')):>7}{str(idle.get('mem_available_min_mb')):>8}")

    for arm in sorted({r["arm"] for r in rows if r["arm"]}):
        vals = [r["trigger_to_verdict_s"] for r in rows
                if r["arm"] == arm and r["trigger_to_verdict_s"]]
        ej = [(r.get("wake") or {}).get("energy_j") for r in rows if r["arm"] == arm]
        ej = [e for e in ej if e]
        if vals:
            print(f"\n{arm}: trigger->verdict median {statistics.median(vals):.2f}s "
                  f"(n={len(vals)})", end="")
            if ej:
                print(f", wake energy median {statistics.median(ej):.1f} J", end="")
            print()


if __name__ == "__main__":
    main()

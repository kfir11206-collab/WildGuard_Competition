#!/usr/bin/env python3
"""Per-test table for a card_eval run: throughput, latency, board energy, card temperature, stalls."""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "resident_vlm"))
from analyze import integrate  # noqa: E402

PAIRS = [("burst_read", "continuous_read"), ("burst_read_128k_qd1", "continuous_read_128k_qd1")]


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def window(d, name):
    rows = load_jsonl(d / f"{name}.jsonl")
    mk = {x["name"]: x["t"] for x in json.loads((d / f"{name}.marks.json").read_text())}
    return [r for r in rows if mk["start"] <= r["t"] <= mk["end"]], rows, mk["start"], mk["end"]


def iostat_rows(d):
    rows, cur, hdr = [], None, None
    for line in (d / "iostat.txt").read_text().splitlines():
        if re.match(r"\d\d/\d\d/\d{4} \d\d:\d\d:\d\d [AP]M", line):
            cur = time.mktime(time.strptime(line.strip(), "%m/%d/%Y %I:%M:%S %p"))
        elif line.startswith("Device"):
            hdr = line.split()
        elif line.startswith("nvme0n1") and hdr and cur:
            rows.append({"t": cur, **dict(zip(hdr[1:], map(float, line.split()[1:])))})
    return rows


def mean_w(rows, key):
    vals = [r[key] for r in rows if key in r]
    return round(sum(vals) / len(vals) / 1000, 3) if vals else None


def cell(v, width, spec=""):
    return (format(v, spec) if v is not None else "-").rjust(width)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rundir")
    d = Path(p.parse_args().rundir)

    m = json.loads((d / "manifest.json").read_text())
    dev = json.loads((d / "device.json").read_text())
    host = json.loads((d / "host.json").read_text())
    steps = m["steps"]

    def step_of(t):
        return next((s["name"] for s in steps if s["wall_start"] <= t <= s["wall_end"] + 1), "(rest)")

    stuck = {}
    for r in iostat_rows(d):
        if r["r/s"] == 0 and r["w/s"] == 0 and r["%util"] >= 50:
            stuck[step_of(r["t"])] = stuck.get(step_of(r["t"]), 0) + 1

    tests, idle = [], {}
    for s in steps:
        inside, rows, t0, t1 = window(d, s["name"])
        energy, watts = integrate(rows, t0, t1)
        temps = [r["temp_card_c"] for r in inside if "temp_card_c" in r]
        row = {"test": s["name"], "seconds": round(t1 - t0, 1), "board_w": watts, "board_j": energy,
               "card_c_start": temps[0] if temps else s.get("start_temp_c"),
               "card_c_peak": max(temps) if temps else None,
               "cool_wait_s": s.get("cool_wait_s"), "stuck_s": stuck.get(s["name"], 0)}
        fj, gt = d / f"{s['name']}.json", d / f"{s['name']}.txt"
        if fj.exists():
            job = json.loads(fj.read_text())["jobs"][0]
            r = job["write"] if job["write"]["io_bytes"] > 0 else job["read"]
            pct = r["clat_ns"]["percentile"]
            gb = r["io_bytes"] / 1e9
            row.update(mb_s=round(r["bw_bytes"] / 1e6, 1), iops=round(r["iops"]),
                       p50_us=round(pct["50.000000"] / 1e3), p99_us=round(pct["99.000000"] / 1e3),
                       max_ms=round(r["clat_ns"]["max"] / 1e6, 1), gb=round(gb, 2),
                       j_per_gb=round(energy / gb, 2) if gb and energy else None)
        elif gt.exists():
            g = re.search(r"DataSetSize: (\d+)/.*Throughput: ([\d.]+) GiB/sec, Avg_Latency: ([\d.]+) usecs",
                          gt.read_text())
            if g:
                gb = int(g.group(1)) * 1024 / 1e9
                row.update(mb_s=round(float(g.group(2)) * 1073.741824, 1), avg_lat_us=round(float(g.group(3))),
                           gb=round(gb, 2), j_per_gb=round(energy / gb, 2) if energy else None)
        if s["name"].startswith("idle"):
            vin, cpu, soc = (mean_w(inside, k) for k in
                             ("power_vdd_in_mw", "power_vdd_cpu_gpu_cv_mw", "power_vdd_soc_mw"))
            idle[s["name"]] = {"vdd_in_w": vin, "vdd_cpu_gpu_cv_w": cpu, "vdd_soc_w": soc,
                               "rest_of_board_w": round(vin - cpu - soc, 3) if None not in (vin, cpu, soc) else None}
        tests.append(row)

    print(f"{d.name} | {dev['label']} pcie x{dev['pcie_width']} | {host['pmode']} ({host['pmode_name']})")
    print(f"steps {len(steps)} | failed: {[s['name'] for s in steps if s['rc']] or 'none'} | "
          f"drive rests at {m.get('cool_ref_temp_c', '-')} C")
    print(f"{'test':<26}{'sec':>6}{'W':>6}{'J':>6}{'card C':>10}{'cool s':>7}{'MB/s':>8}{'IOPS':>8}"
          f"{'p50us':>7}{'p99us':>7}{'max ms':>9}{'GB':>7}{'J/GB':>7}{'stuck s':>8}")
    for t in tests:
        start, peak = t["card_c_start"], t["card_c_peak"]
        temp = f"{start:.0f}->{peak:.0f}" if start is not None and peak is not None else (
            f"{start:.0f}" if start is not None else "-")
        print(f"{t['test']:<26}{cell(t['seconds'], 6, '.1f')}{cell(t['board_w'], 6, '.2f')}"
              f"{cell(t['board_j'], 6, '.0f')}{temp:>10}{cell(t['cool_wait_s'], 7, '.0f')}"
              f"{cell(t.get('mb_s'), 8, '.1f')}{cell(t.get('iops'), 8)}{cell(t.get('p50_us'), 7)}"
              f"{cell(t.get('p99_us'), 7)}{cell(t.get('max_ms'), 9, '.1f')}{cell(t.get('gb'), 7, '.2f')}"
              f"{cell(t.get('j_per_gb'), 7, '.2f')}{cell(t['stuck_s'], 8)}")

    if idle:
        print("\nidle (mean W)            VDD_IN  CPU/GPU    SoC  rest of board (incl. drive)")
        for name, v in idle.items():
            print(f"  {name:<22}{cell(v['vdd_in_w'], 7, '.3f')}{cell(v['vdd_cpu_gpu_cv_w'], 9, '.3f')}"
                  f"{cell(v['vdd_soc_w'], 7, '.3f')}{cell(v['rest_of_board_w'], 10, '.3f')}")

    by = {t["test"]: t for t in tests}
    pairs = []
    for b, c in PAIRS:
        if b in by and c in by:
            tb, tc = by[b], by[c]
            ratio = (round(tb["j_per_gb"] / tc["j_per_gb"], 3)
                     if tb.get("j_per_gb") and tc.get("j_per_gb") else None)
            pairs.append({"burst": b, "continuous": c, "j_per_gb_burst_over_continuous": ratio})
            print(f"\n{b} vs {c}\n  MB/s {tb.get('mb_s')} / {tc.get('mb_s')}   J/GB {tb.get('j_per_gb')} / "
                  f"{tc.get('j_per_gb')} (burst/continuous {ratio})   p99 us {tb.get('p99_us')} / "
                  f"{tc.get('p99_us')}   slowest ms {tb.get('max_ms')} / {tc.get('max_ms')}   "
                  f"stuck s {tb['stuck_s']} / {tc['stuck_s']}")

    sustained_window = None
    if "sustained_read" in by:
        print("\nsustained_read over time (10s buckets):")
        rows = window(d, "sustained_read")[0]
        b = 0
        while True:
            seg = [r for r in rows if b <= r["t"] - rows[0]["t"] < b + 10]
            if len(seg) < 2:
                break
            mbs = (seg[-1]["disk_read_bytes"] - seg[0]["disk_read_bytes"]) / 1e6 / (seg[-1]["t"] - seg[0]["t"])
            print(f"  {b:4d}-{b + 10:<4d}s {mbs:7.1f} MB/s  card {seg[-1].get('temp_card_c', 0):.1f} C")
            b += 10
        timed = 4 * m["args"]["runtime"]
        win = [r for r in rows if r["t"] - rows[0]["t"] <= timed]
        sustained_window = round((win[-1]["disk_read_bytes"] - win[0]["disk_read_bytes"]) / 1e6
                                 / (win[-1]["t"] - win[0]["t"]), 1)
        print(f"  timed {timed}s window: {sustained_window} MB/s  (fio's own figure also counts any "
              f"wait for the last reads after the window)")

    (d / "analysis.json").write_text(json.dumps(
        {"device": dev, "host": host, "tests": tests, "idle": idle, "burst_vs_continuous": pairs,
         "sustained_timed_window_mb_s": sustained_window, "stuck_seconds_by_step": stuck}, indent=2))


if __name__ == "__main__":
    main()

import json
import statistics
import sys
from pathlib import Path

MIN_BYTES = 1.6e9
MAX_EXCESS_S = 1.0


def rep_checks(d, rec):
    tag = f"{rec['arm']}_{rec['rep']}"
    samples = [json.loads(line) for line in open(d / f"{tag}.jsonl")]
    marks = {m["name"]: m["t"] for m in json.load(open(d / f"{tag}.marks.json"))}
    near = lambda t: min(samples, key=lambda s: abs(s["t"] - t))
    wake = [s for s in samples if marks["wake_sent"] <= s["t"] <= marks["first_verdict"]]
    bins = [(b["disk_read_ms"] - a["disk_read_ms"], b["disk_reads_completed"] - a["disk_reads_completed"])
            for a, b in zip(wake, wake[1:])]
    typical = statistics.median(ms / n for ms, n in bins if n > 50)
    pw = [(s["t"], s["power_vdd_in_mw"]) for s in wake if "power_vdd_in_mw" in s]
    wake_j = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(pw, pw[1:])) / 1000
    excess_s = max(ms - n * typical for ms, n in bins) / 1000
    read_end = marks.get("preread_end", marks["sd_read_end"])
    rd = [s for s in samples if marks["wake_sent"] <= s["t"] <= read_end]
    rd_bytes = rd[-1]["disk_read_bytes"] - rd[0]["disk_read_bytes"]
    at = lambda f: next(s["t"] for s in rd if s["disk_read_bytes"] - rd[0]["disk_read_bytes"] >= f * rd_bytes)
    w = rec["wake_reply"]
    trig = near(marks["trigger"])
    verdict = "ok"
    if w["bytes_read"] < MIN_BYTES:
        verdict = "VOID warm"
    elif excess_s > MAX_EXCESS_S:
        verdict = f"VOID stall {excess_s:.1f}s"
    return {
        "tag": tag, "arm": rec["arm"], "t2v": rec["trigger_to_verdict_seconds"],
        "wake_s": w["wake_seconds"], "read_s": w["sd_read_seconds"],
        "pre_s": w.get("preread_seconds") or w.get("prefetch_seconds"), "gb": w["bytes_read"] / 1e9,
        "read_mbs": 0.98 * rd_bytes / 1e6 / (at(0.99) - at(0.01)),
        "inflight_md": statistics.median(s["disk_io_in_flight"] for s in rd),
        "excess_s": excess_s, "wake_j": wake_j, "wake_w": wake_j / (pw[-1][0] - pw[0][0]),
        "gate_c": rec.get("card_start_c"), "trig_c": trig.get("temp_card_c"),
        "avail_mb": trig["mem_available_mb"],
        "swapout_mb": (near(marks["first_verdict"])["pswpout_bytes"] - trig["pswpout_bytes"]) / 1e6,
        "verdict": verdict,
    }


def main():
    for arg in sys.argv[1:]:
        d = Path(arg)
        label = json.load(open(d / "device.json"))["label"]
        rows = [rep_checks(d, r) for r in json.load(open(d / "summary.json")) if "wake_reply" in r]
        print(f"\n{d.name} ({label})")
        print(f"{'rep':<24}{'t2v':>7}{'wake':>7}{'read':>7}{'pre':>6}{'GB':>6}{'MB/s':>6}{'infl':>5}"
              f"{'excess':>7}{'J':>6}{'W':>5}{'gate':>6}{'trig':>6}{'availMB':>8}{'swapMB':>7}  verdict")
        for r in rows:
            pre = f"{r['pre_s']:.2f}" if r["pre_s"] else "-"
            gate = f"{r['gate_c']:.1f}" if r["gate_c"] is not None else "-"
            trig = f"{r['trig_c']:.1f}" if r["trig_c"] is not None else "-"
            print(f"{r['tag']:<24}{r['t2v']:7.2f}{r['wake_s']:7.2f}{r['read_s']:7.2f}{pre:>6}{r['gb']:6.2f}"
                  f"{r['read_mbs']:6.0f}{r['inflight_md']:5.0f}{r['excess_s']:7.2f}{r['wake_j']:6.1f}{r['wake_w']:5.1f}{gate:>6}"
                  f"{trig:>6}{r['avail_mb']:8}{r['swapout_mb']:7.0f}  {r['verdict']}")
        for arm in dict.fromkeys(r["arm"] for r in rows):
            ok = [r for r in rows if r["arm"] == arm and r["verdict"] == "ok"]
            if ok:
                t2v = [r["t2v"] for r in ok]
                print(f"  {arm:<22} valid n={len(ok)}  t2v {' / '.join(f'{x:.2f}' for x in t2v)}"
                      f"  median {statistics.median(t2v):.2f}  wake median {statistics.median(r['wake_s'] for r in ok):.2f}")
                first = ok[:3]
                print(f"  {'':<22} first 3 valid in run order: t2v {' / '.join(f'{r[chr(116)+chr(50)+chr(118)]:.2f}' for r in first)}"
                      f"  median {statistics.median(r['t2v'] for r in first):.2f}"
                      f"  J median {statistics.median(r['wake_j'] for r in first):.1f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reproduce and characterise the 1MB QD1 read stall on whichever drive is in the M.2 slot."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from card_eval import DEV, ROOT, S, preflight  # noqa: E402

RUNS = [
    ("qd1_1m_plain_a", "1M", 1, False),
    ("qd1_1m_rescue_a", "1M", 1, True),
    ("qd1_1m_plain_b", "1M", 1, False),
    ("qd1_1m_rescue_b", "1M", 1, True),
    ("qd1_128k", "128k", 1, False),
    ("qd1_256k", "256k", 1, False),
    ("qd8_128k", "128k", 8, False),
]
SLOW_KEYS = ("100", "250", "500", "750", "1000", "2000", ">=2000")


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def host_info():
    irqs = [l.split()[-1] + " " + " ".join(l.split()[-3:-1])
            for l in (read("/proc/interrupts") or "").splitlines() if "nvme" in l]
    return {
        "use_threaded_interrupts": read("/sys/module/nvme/parameters/use_threaded_interrupts"),
        "io_timeout_s": read("/sys/module/nvme_core/parameters/io_timeout"),
        "apst_ps_max_latency_us": read("/sys/module/nvme_core/parameters/default_ps_max_latency_us"),
        "max_sectors_kb": read(f"/sys/block/{DEV}/queue/max_sectors_kb"),
        "max_hw_sectors_kb": read(f"/sys/block/{DEV}/queue/max_hw_sectors_kb"),
        "cmdline": read("/proc/cmdline"),
        "nvme_irqs": irqs,
    }


def fio(testfile, size_gb, name, bs, qd, runtime, out):
    subprocess.run(["taskset", "-c", "0", "fio", f"--name={name}", f"--filename={testfile}",
                    f"--size={size_gb}G", "--direct=1", "--ioengine=libaio", "--rw=read",
                    f"--bs={bs}", f"--iodepth={qd}", "--time_based", f"--runtime={runtime}",
                    "--output-format=json", f"--output={out}"])
    job = json.loads(Path(out).read_text())["jobs"][0]
    r = job["read"]
    return {"bs": bs, "qd": qd,
            "mb_s": round(r["bw_bytes"] / 1e6, 1),
            "ios": r["total_ios"],
            "clat_max_ms": round(r["clat_ns"]["max"] / 1e6, 1),
            "ios_over_100ms_pct": round(sum(v for k, v in job["latency_ms"].items() if k in SLOW_KEYS), 2)}


def rescuer(testfile):
    return subprocess.Popen(
        ["bash", "-c", f'while :; do taskset -c 0 dd if="{testfile}" of=/dev/null bs=4k count=1 '
                       f'skip=$RANDOM iflag=direct status=none; sleep 1; done'],
        start_new_session=True)


def verdict(res):
    plain = [res[n]["clat_max_ms"] for n in ("qd1_1m_plain_a", "qd1_1m_plain_b")]
    rescue = [res[n]["clat_max_ms"] for n in ("qd1_1m_rescue_a", "qd1_1m_rescue_b")]
    if max(plain) <= 2000:
        return "NO STALLS: no 1MB QD1 read took longer than 2s"
    if max(rescue) < 1500:
        return ("STALLS, RESCUED: 1MB QD1 reads stall, but a 4K read on the same queue once a second "
                "releases them within ~1s -> the drive had already completed them; the completion "
                "interrupt was missed")
    return ("STALLS, NOT RESCUED: another I/O on the same queue does not release them -> "
            "the drive itself is holding the reads")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size-gb", type=int, default=2)
    p.add_argument("--runtime", type=int, default=30)
    p.add_argument("--rest", type=int, default=20)
    p.add_argument("--workdir", default=str(Path.home() / "card_eval_work"))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(exist_ok=True)
    problems = preflight(workdir, args.size_gb << 30)
    if problems:
        for msg in problems:
            print(f"pre-flight: {msg}")
        print("nothing was started.")
        sys.exit(1)

    stamp = S.device_stamp(DEV)
    outdir = (Path(args.out) if args.out else ROOT / "mmap_sandbox" / "results" / "card_eval"
              / f"stall_probe_{stamp['label']}_{time.strftime('%Y%m%d_%H%M%S')}")
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "device.json").write_text(json.dumps(stamp, indent=2))
    (outdir / "host.json").write_text(json.dumps(S.power_stamp(), indent=2))
    print(f"[stall_probe] drive: {stamp['label']} ({stamp['model']}) pcie x{stamp['pcie_width']}")
    print(f"[stall_probe] writing to {outdir}")
    print(f"[stall_probe] ~{(len(RUNS) * args.runtime + args.rest) / 60 + 0.5:.0f} min", flush=True)

    testfile = workdir / "stall_probe_testfile"
    res = {}
    try:
        subprocess.run(["fio", "--name=layout", f"--filename={testfile}", f"--size={args.size_gb}G",
                        "--rw=write", "--bs=1M", "--iodepth=8", "--direct=1", "--ioengine=libaio",
                        "--output-format=terse"], stdout=subprocess.DEVNULL)
        time.sleep(args.rest)
        for name, bs, qd, rescue in RUNS:
            helper = rescuer(testfile) if rescue else None
            try:
                res[name] = fio(testfile, args.size_gb, name, bs, qd, args.runtime, outdir / f"{name}.json")
            finally:
                if helper is not None:
                    os.killpg(helper.pid, signal.SIGTERM)
                    helper.wait()
            r = res[name]
            print(f"[{name:<16}] bs={bs:<5} qd={qd}  {r['mb_s']:7.1f} MB/s  "
                  f"slowest {r['clat_max_ms']:9.1f} ms  ios>100ms {r['ios_over_100ms_pct']:5.2f}%", flush=True)
    finally:
        testfile.unlink(missing_ok=True)

    v = verdict(res)
    (outdir / "summary.json").write_text(json.dumps(
        {"verdict": v, "runs": res, "host": host_info(), "args": vars(args)}, indent=2))
    print(f"\n[stall_probe] {v}")
    print(f"[stall_probe] DONE {outdir}", flush=True)


if __name__ == "__main__":
    main()

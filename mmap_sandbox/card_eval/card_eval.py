#!/usr/bin/env python3
"""fio / iostat / gdsio suite for whichever drive is in the M.2 slot, sampled for board power and card temperature."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "resident_vlm"))
import sampler as S  # noqa: E402

DEV = "nvme0n1"
GDSIO = "/usr/local/cuda/gds/tools/gdsio"
BURST_MB = 128
BURST_GAP_US = 1_000_000
COOL_MARGIN_C = 5

READS = [
    ("seq_read_1m_qd8", "read", "1M", 8),
    ("seq_read_1m_qd1", "read", "1M", 1),
    ("rand_read_4k_qd32", "randread", "4k", 32),
    ("rand_read_4k_qd1", "randread", "4k", 1),
]
WRITES = [
    ("seq_write_1m_qd8", "write", "1M", 8),
    ("seq_write_1m_qd1", "write", "1M", 1),
    ("rand_write_4k_qd32", "randwrite", "4k", 32),
    ("rand_write_4k_qd1", "randwrite", "4k", 1),
]
GDS_XFER = {0: "gdsio_x0_gpud_compat", 1: "gdsio_x1_cpu_only", 2: "gdsio_x2_cpu_gpu"}


def preflight(workdir, size_bytes):
    problems = []
    for tool in ("fio", "iostat"):
        if shutil.which(tool) is None:
            problems.append(f"{tool} not installed")
    if not Path(GDSIO).exists():
        problems.append(f"{GDSIO} missing")
    src = subprocess.run(["findmnt", "-n", "-o", "SOURCE", "-T", str(workdir)],
                         capture_output=True, text=True).stdout.strip()
    if not src.startswith(f"/dev/{DEV}"):
        problems.append(f"{workdir} is on {src}, not /dev/{DEV}")
    if shutil.disk_usage(workdir).free < size_bytes + (2 << 30):
        problems.append("not enough free space for the test file")
    if subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True).stdout.strip():
        problems.append("docker containers are running")
    if subprocess.run(["pgrep", "-f", "bench.py"], capture_output=True).returncode == 0:
        problems.append("bench.py is running")
    mode = first_line(["cat", "/var/lib/nvpmodel/status"])
    cap = first_line(["cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"])
    gpu = first_line(["cat", "/sys/class/devfreq/17000000.gpu/max_freq"])
    if mode != "pmode:0001" or cap != "1344000" or gpu != "918000000":
        problems.append(f"power mode {mode}, cpu cap {cap}, gpu cap {gpu} - expected pmode:0001 / "
                        f"1344000 / 918000000; run mmap_sandbox/card_eval/set_25w_clocks.sh")
    return problems


def first_line(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip().splitlines()[0]
    except (OSError, IndexError):
        return None


def drive_temp():
    card = next(Path(f"/sys/block/{DEV}/device").glob("hwmon*/temp1_input"), None)
    try:
        return int(card.read_text()) / 1000.0 if card else None
    except (OSError, ValueError):
        return None


def coolest(jsonl):
    temps = [json.loads(l).get("temp_card_c") for l in Path(jsonl).read_text().splitlines() if l.strip()]
    temps = [t for t in temps if t is not None]
    return min(temps) if temps else None


class Suite:
    def __init__(self, outdir, testfile, size_gb, rest, cool_max):
        self.outdir = outdir
        self.testfile = testfile
        self.size_gb = size_gb
        self.rest_s = rest
        self.cool_max = cool_max
        self.ref_temp = None
        self.gate = {}
        self.steps = []

    def rest(self, factor=1):
        time.sleep(self.rest_s * factor)

    def settle(self):
        self.rest()
        t0 = time.monotonic()
        target = None if self.ref_temp is None else self.ref_temp + COOL_MARGIN_C
        temp = drive_temp()
        while (target is not None and temp is not None and temp > target
               and time.monotonic() - t0 < self.cool_max):
            time.sleep(1)
            temp = drive_temp()
        self.gate = {"cool_target_c": target, "cool_wait_s": round(time.monotonic() - t0, 1),
                     "start_temp_c": temp}

    def step(self, name, cmd, stdout_path=None, card_temp=True):
        smp = S.Sampler(self.outdir / f"{name}.jsonl", hz=10, dev=DEV, card_temp=card_temp)
        smp.start()
        smp.mark("start")
        wall0 = time.time()
        if stdout_path is None:
            rc = subprocess.run(cmd).returncode
        else:
            with open(stdout_path, "w") as f:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        smp.mark("end")
        smp.stop()
        self.steps.append({"name": name, "cmd": cmd, "rc": rc, "wall_start": wall0,
                           "wall_end": time.time(),
                           "local_start": time.strftime("%H:%M:%S", time.localtime(wall0)),
                           **self.gate})
        self.gate = {}
        return rc

    def start_note(self):
        s = self.steps[-1]
        if s.get("start_temp_c") is None:
            return ""
        return f"  (start {s['start_temp_c']:.0f} C, cooled {s['cool_wait_s']:.0f}s)"

    def fio(self, name, rw, bs, qd, runtime=None, *extra):
        out = self.outdir / f"{name}.json"
        cmd = ["fio", f"--name={name}", f"--filename={self.testfile}", f"--size={self.size_gb}G",
               "--direct=1", "--ioengine=libaio", f"--rw={rw}", f"--bs={bs}", f"--iodepth={qd}",
               "--eta=never", "--output-format=json", f"--output={out}", *extra]
        if runtime is not None:
            cmd += ["--time_based", f"--runtime={runtime}"]
        rc = self.step(name, cmd)
        try:
            job = json.loads(out.read_text())["jobs"][0]
            r = job["write" if "write" in rw else "read"]
            p99 = r["clat_ns"]["percentile"]["99.000000"] / 1e6
            print(f"[{name}] {r['bw_bytes'] / 1e6:8.1f} MB/s  {r['iops']:9.0f} IOPS  "
                  f"p99 {p99:7.2f} ms{self.start_note()}", flush=True)
            return r
        except (OSError, ValueError, KeyError, IndexError):
            print(f"[{name}] FAILED rc={rc}", flush=True)
            return None

    def gdsio(self, xfer, runtime):
        name = GDS_XFER[xfer]
        out = self.outdir / f"{name}.txt"
        rc = self.step(name, [GDSIO, "-f", str(self.testfile), "-d", "0", "-w", "8",
                              "-s", f"{self.size_gb}G", "-i", "1M", "-x", str(xfer),
                              "-I", "0", "-T", str(runtime)], stdout_path=out)
        line = next((l for l in out.read_text().splitlines() if "Throughput" in l), None)
        print(f"[{name}] {line or f'FAILED rc={rc}'}{self.start_note()}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size-gb", type=int, default=8)
    p.add_argument("--runtime", type=int, default=30)
    p.add_argument("--rest", type=int, default=20)
    p.add_argument("--cool-max", type=int, default=120)
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
              / f"{stamp['label']}_{time.strftime('%Y%m%d_%H%M%S')}")
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "device.json").write_text(json.dumps(stamp, indent=2))
    (outdir / "host.json").write_text(json.dumps(S.power_stamp(), indent=2))

    R = args.runtime
    print(f"[card_eval] drive: {stamp['label']} ({stamp['model']}) pcie x{stamp['pcie_width']}")
    print(f"[card_eval] writing to {outdir}")
    print(f"[card_eval] ~{(27 * R + 19 * args.rest) / 60:.0f} min, plus laying out the {args.size_gb}GB "
          f"test file and cool-down pauses (at most {args.cool_max}s before each test)", flush=True)

    manifest = {
        "args": vars(args),
        "fio_version": first_line(["fio", "--version"]),
        "iostat_version": first_line(["iostat", "-V"]),
        "gdsio_version": first_line([GDSIO, "-h"]),
        "apst_ps_max_latency_us": first_line(
            ["cat", "/sys/module/nvme_core/parameters/default_ps_max_latency_us"]),
        "burst_mb": BURST_MB,
        "burst_gap_us": BURST_GAP_US,
        "cool_margin_c": COOL_MARGIN_C,
    }
    suite = Suite(outdir, workdir / "testfile", args.size_gb, args.rest, args.cool_max)
    with open(outdir / "iostat.txt", "w") as log:
        iostat = subprocess.Popen(["iostat", "-x", "-m", "-t", "-y", DEV, "1"],
                                  stdout=log, stderr=subprocess.STDOUT)
        try:
            suite.fio("layout_seq_write_1m_qd8", "write", "1M", 8)
            suite.rest(3)
            suite.step("idle", ["sleep", str(2 * R)])
            suite.step("idle_no_temp_poll", ["sleep", str(2 * R)], card_temp=False)
            suite.ref_temp = coolest(outdir / "idle.jsonl")
            manifest["cool_ref_temp_c"] = suite.ref_temp
            print(f"[idle] 2 x {2 * R}s of board idle recorded (with and without drive-temperature "
                  f"reads); drive rests at {suite.ref_temp} C", flush=True)

            for name, rw, bs, qd in READS:
                suite.settle()
                suite.fio(name, rw, bs, qd, R)

            suite.settle()
            burst = suite.fio("burst_read", "read", "1M", 8, 2 * R,
                              f"--thinktime={BURST_GAP_US}", f"--thinktime_blocks={BURST_MB}")
            if burst:
                manifest["continuous_rate_bytes"] = burst["bw_bytes"]
                suite.settle()
                suite.fio("continuous_read", "read", "1M", 8, 2 * R, f"--rate={burst['bw_bytes']}")

            suite.settle()
            burst = suite.fio("burst_read_128k_qd1", "read", "128k", 1, 2 * R,
                              f"--thinktime={BURST_GAP_US}", f"--thinktime_blocks={BURST_MB * 8}")
            if burst:
                manifest["continuous_128k_rate_bytes"] = burst["bw_bytes"]
                suite.settle()
                suite.fio("continuous_read_128k_qd1", "read", "128k", 1, 2 * R,
                          f"--rate={burst['bw_bytes']}")

            suite.settle()
            suite.fio("sustained_read", "read", "1M", 8, 4 * R)

            for xfer in GDS_XFER:
                suite.settle()
                suite.gdsio(xfer, R)

            for name, rw, bs, qd in WRITES:
                suite.settle()
                suite.fio(name, rw, bs, qd, R)
        finally:
            iostat.terminate()
            iostat.wait()
            suite.testfile.unlink(missing_ok=True)
            manifest["steps"] = suite.steps
            (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[card_eval] DONE {outdir}", flush=True)


if __name__ == "__main__":
    main()

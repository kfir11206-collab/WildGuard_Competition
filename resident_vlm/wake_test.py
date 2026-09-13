#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench as B  # noqa: E402
import sampler as S  # noqa: E402

DEV = "nvme0n1"


def mem_available_mb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def running_scripts(names):
    found = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if int(proc.name) == os.getpid() or not argv or not Path(argv[0].decode(errors="ignore")).name.startswith("python"):
            continue
        if any(Path(a.decode(errors="ignore")).name in names for a in argv[1:]):
            found.append(int(proc.name))
    return found


def preflight(min_ram):
    problems = []
    cpu = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq").read_text().strip()
    gpu = Path("/sys/class/devfreq/17000000.gpu/max_freq").read_text().strip()
    if cpu != "1344000" or gpu != "918000000":
        problems.append(f"power: cpu {cpu}, gpu {gpu} - run mmap_sandbox/card_eval/set_25w_clocks.sh")
    if subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True).stdout.strip():
        problems.append("containers are running - clear with: docker ps -q | xargs -r docker rm -f")
    if running_scripts(("bench.py", "wake_test.py")):
        problems.append("bench.py or wake_test.py is already running")
    if not Path("/dev/video0").exists():
        problems.append("camera /dev/video0 missing")
    if not B.FUSED.exists():
        problems.append(f"{B.FUSED} missing")
    avail = mem_available_mb()
    if avail < min_ram:
        problems.append(f"RAM available {avail} MB - close VS Code and the browser (need >= {min_ram})")
    return problems, avail


def run_rep(arm, rep, args, outdir, card_ref):
    tag = f"{arm}_{rep}"
    rec = {"arm": arm, "rep": rep}
    B.drop_cache([B.FUSED])
    rec["cooldown_seconds"], rec["tj_start"], rec["card_start_c"] = B.cooldown(
        args.tj_target, args.cooldown_max, card_ref)
    smp = S.Sampler(outdir / f"{tag}.jsonl", hz=args.hz, dev=DEV)
    smp.start()
    try:
        smp.mark("before_start")
        time.sleep(args.before)
        t_trigger = time.monotonic()
        smp.mark("trigger", t_trigger)
        reply = B.send(f"WAKE {arm.upper()}", timeout=args.verdict_timeout)
        t_verdict = time.monotonic()
        if "error" in reply:
            rec["error"] = reply["error"]
        else:
            rec["wake_reply"] = reply
            B.mark_wake_phases(smp, t_trigger, reply)
        smp.mark("first_verdict", t_verdict)
        rec["trigger_to_verdict_seconds"] = round(t_verdict - t_trigger, 3)
        time.sleep(args.after)
        smp.mark("after_end")
    finally:
        smp.stop()
    rec["sleep"] = B.send("SLEEP", timeout=120)
    (outdir / f"{tag}.json").write_text(json.dumps(rec, indent=2))
    status = rec.get("error") or f"trigger->verdict = {rec['trigger_to_verdict_seconds']:.3f}s"
    print(f"[{tag}] {status}  (cool-down {rec['cooldown_seconds']}s, drive {rec['card_start_c']} C)", flush=True)
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--arms", nargs="+", default=["burst", "continuous"], choices=["burst", "continuous"])
    p.add_argument("--before", type=float, default=5.0)
    p.add_argument("--after", type=float, default=5.0)
    p.add_argument("--tj-target", type=float, default=52.0)
    p.add_argument("--cooldown-max", type=float, default=240.0)
    p.add_argument("--verdict-timeout", type=float, default=120.0)
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--min-ram", type=int, default=5300)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    problems, avail = preflight(args.min_ram)
    if args.preflight:
        print(f"=== pre-flight ===\n  RAM available {avail} MB")
        for msg in problems or ["all ok"]:
            print(f"  {msg}")
        sys.exit(1 if problems else 0)
    if problems:
        for msg in problems:
            print(f"pre-flight: {msg}")
        sys.exit(1)

    outdir = Path(args.out or B.ROOT / "resident_vlm" / "results" / time.strftime("waketest_%Y%m%d_%H%M%S"))
    outdir.mkdir(parents=True, exist_ok=False)
    stamp = S.device_stamp(DEV)
    power = S.power_stamp()
    (outdir / "args.json").write_text(json.dumps({**vars(args), "mem_available_mb_at_start": avail}, indent=2))
    (outdir / "device.json").write_text(json.dumps(stamp, indent=2))
    (outdir / "host.json").write_text(json.dumps(power, indent=2))
    print(f"[waketest] writing to {outdir}", flush=True)
    print(f"[waketest] drive: {stamp['label']} ({stamp['model']}) pcie x{stamp['pcie_width']}", flush=True)
    print(f"[waketest] power: cpu_max={power['cpu_max_khz']} gpu_max={power.get('gpu_max_hz')}", flush=True)
    card_ref = S.read_card_temp()
    print(f"[waketest] drive resting at {card_ref} C; {args.reps} reps x {args.arms}", flush=True)

    recs = []
    daemon = None
    try:
        t0 = time.monotonic()
        daemon = B.start_daemon(outdir / "daemon.log")
        print(f"[waketest] model loaded in {time.monotonic() - t0:.0f}s", flush=True)
        warm = {"wake": B.send("WAKE", timeout=args.verdict_timeout), "sleep": B.send("SLEEP", timeout=120)}
        (outdir / "warmup.json").write_text(json.dumps(warm, indent=2))
        print("[waketest] warm-up wake done (not measured)", flush=True)
        for rep in range(args.reps):
            for arm in args.arms:
                recs.append(run_rep(arm, rep, args, outdir, card_ref))
    finally:
        B.stop_daemon(daemon)
        (outdir / "summary.json").write_text(json.dumps(recs, indent=2))
    print(f"[waketest] done -> {outdir}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Orchestrate the burst-vs-continuous comparison on the real fused system.

Both modes are triggered by the identical mht_sim.py trace/seed. Burst uses
the real fire_watcher.py unmodified (--wake-mode compose: on threshold cross,
`docker compose up -d` immediately). Continuous uses continuous_watcher.py
(same trigger, but paces a host-side pre-load of the model files to the
measured sustained SD throughput before `docker compose up -d`).

Each rep: docker compose down -> drop page cache -> start tegrastats ->
start watcher -> start mht_sim -> poll watch log for the 'awake' event ->
tear everything down -> cooldown -> next rep. Modes are interleaved across
reps to spread any thermal drift evenly between conditions.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FW = ROOT / "fire_watcher"
SANDBOX = Path(__file__).resolve().parent
MMAP_SANDBOX = ROOT / "mmap_sandbox"
sys.path.insert(0, str(FW))
from model_files import default_model_files  # noqa: E402


def drop_cache(files):
    for path in files:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def compose_down(compose_file):
    subprocess.run(["docker", "compose", "-f", str(compose_file), "down"],
                    capture_output=True, text=True)


def read_tj():
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            if (zone / "type").read_text().strip() == "tj-thermal":
                return int((zone / "temp").read_text()) / 1000.0
        except OSError:
            continue
    return None


def cooldown(seconds, target=None, tol=1.5):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        tj = read_tj()
        if target is not None and tj is not None and tj <= target + tol:
            break
        time.sleep(2)


def find_event(log_path, event):
    if not log_path.exists():
        return None
    for line in log_path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") == event:
            return rec
    return None


def run_one(mode, rep, args, outdir):
    tag = f"{mode}_{rep}"
    run_dir = outdir / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    sock = str(run_dir / "watcher.sock")
    watch_log = run_dir / "watch.jsonl"
    tegra_log = run_dir / "tegrastats.log"

    print(f"[{tag}] docker compose down + drop page cache", flush=True)
    compose_down(args.compose_file)
    drop_cache(args.files)
    baseline_tj = read_tj()
    time.sleep(2)

    print(f"[{tag}] tegrastats -> {tegra_log}", flush=True)
    tegra = subprocess.Popen(["tegrastats", "--interval", "200", "--logfile", str(tegra_log)])

    if mode == "burst":
        watcher_cmd = [sys.executable, str(FW / "fire_watcher.py"),
                        "--sock", sock, "--wake-mode", "compose", "--log", str(watch_log),
                        "--compose-file", str(args.compose_file), "--ready-file", str(args.ready_file),
                        "--ready-timeout", str(args.ready_timeout),
                        "--prewarm-at", str(args.wake_at), "--wake-at", str(args.wake_at),
                        "--sleep-at", str(args.sleep_at), "--ema-alpha", str(args.ema_alpha)]
    else:
        watcher_cmd = [sys.executable, str(SANDBOX / "continuous_watcher.py"),
                        "--sock", sock, "--log", str(watch_log), "--pace-mbps", str(args.pace_mbps),
                        "--compose-file", str(args.compose_file), "--ready-file", str(args.ready_file),
                        "--ready-timeout", str(args.ready_timeout),
                        "--wake-at", str(args.wake_at), "--sleep-at", str(args.sleep_at),
                        "--ema-alpha", str(args.ema_alpha)]

    print(f"[{tag}] launch watcher: {' '.join(watcher_cmd)}", flush=True)
    t_watcher_launch_epoch = time.time()
    watcher_proc = subprocess.Popen(watcher_cmd)

    t0 = time.monotonic()
    while not os.path.exists(sock):
        if time.monotonic() - t0 > 15:
            raise RuntimeError(f"[{tag}] watcher socket never appeared")
        time.sleep(0.1)

    mht_cmd = [sys.executable, str(FW / "mht_sim.py"), sock, args.trace,
               "--interval", str(args.interval), "--noise", str(args.noise), "--seed", str(args.seed)]
    print(f"[{tag}] launch mht_sim: {' '.join(mht_cmd)}", flush=True)
    mht_proc = subprocess.Popen(mht_cmd)

    print(f"[{tag}] waiting for 'awake' (timeout {args.ready_timeout + 60:.0f}s) ...", flush=True)
    awake = None
    deadline = time.monotonic() + args.ready_timeout + 60
    while time.monotonic() < deadline:
        awake = find_event(watch_log, "awake")
        if awake:
            break
        time.sleep(1)

    for proc in (mht_proc, watcher_proc):
        proc.terminate()
    for proc in (mht_proc, watcher_proc):
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    tegra.terminate()
    try:
        tegra.wait(timeout=5)
    except subprocess.TimeoutExpired:
        tegra.kill()

    compose_down(args.compose_file)

    result = {
        "mode": mode, "rep": rep,
        "t_watcher_launch_epoch": t_watcher_launch_epoch,
        "wake_start": find_event(watch_log, "wake_start"),
        "awake": awake,
        "baseline_tj_c": baseline_tj,
        "run_dir": str(run_dir),
        "timed_out": awake is None,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    status = f"latency={awake['wake_seconds']}s" if awake else "TIMED OUT"
    print(f"[{tag}] done: {status}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--pace-mbps", type=float, required=True)
    ap.add_argument("--trace", default="0.20:5,0.95:600")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wake-at", type=float, default=0.85)
    ap.add_argument("--sleep-at", type=float, default=0.40)
    ap.add_argument("--ema-alpha", type=float, default=0.4)
    ap.add_argument("--compose-file", default=str(ROOT / "docker-compose_fused_system.yaml"))
    ap.add_argument("--ready-file", default=str(ROOT / "data" / "out_txt"))
    ap.add_argument("--ready-timeout", type=float, default=300.0)
    ap.add_argument("--cooldown", type=float, default=30.0)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    args.files = default_model_files()

    outdir = Path(args.outdir) if args.outdir else (
        MMAP_SANDBOX / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_burst_vs_continuous"))
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    results = []
    for rep in range(args.reps):
        for mode in ("burst", "continuous"):
            baseline = read_tj()
            r = run_one(mode, rep, args, outdir)
            results.append(r)
            (outdir / "results.json").write_text(json.dumps(results, indent=2))
            print(f"cooldown up to {args.cooldown}s (baseline tj={baseline})...", flush=True)
            cooldown(args.cooldown, target=baseline)

    print(json.dumps(results, indent=2))
    print(f"results in {outdir}")


if __name__ == "__main__":
    main()

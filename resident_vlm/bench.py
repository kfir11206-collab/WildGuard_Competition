#!/usr/bin/env python3
"""Resident-VLM vs cold-start benchmark on the real system.

Both arms are timed over the identical span: trigger -> first smoke verdict
appended to data/out_txt. Both arms stop mht_watch before the trigger and
restart it after sleeping, so the /dev/video0 handoff is paid in both.

  baseline : docker compose up -d wildfire_classifier wildfire_detection  (cold start)
  resident : WAKE on the resident daemon's socket (weights streamed off the card)
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fire_watcher"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_files import default_model_files  # noqa: E402
import sampler as S  # noqa: E402

COMPOSE = ROOT / "docker-compose_fused_system.yaml"
OUT_TXT = ROOT / "data" / "out_txt"
SOCK = ROOT / "resident_vlm" / "resident.sock"
FUSED = ROOT / "resident_vlm" / "results" / "fused_state.pt"
STACK = ["wildfire_classifier", "wildfire_detection"]


def compose(*args, timeout=600):
    return subprocess.run(["docker", "compose", "-f", str(COMPOSE), *args],
                          capture_output=True, text=True, timeout=timeout)


def mht_up():
    return compose("--profile", "watch", "up", "-d", "mht_watch")


def mht_stop():
    return compose("stop", "-t", "10", "mht_watch")


def drop_cache(paths):
    os.sync()
    for p in paths:
        try:
            fd = os.open(str(p), os.O_RDONLY)
        except OSError:
            continue
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def cooldown(tj_target, max_seconds, poll=2.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < max_seconds:
        tj = S.read_tj()
        if tj is None or tj <= tj_target:
            break
        time.sleep(poll)
    return round(time.monotonic() - t0, 1), S.read_tj()


def out_txt_size():
    try:
        return OUT_TXT.stat().st_size
    except FileNotFoundError:
        return 0


def wait_for_verdict(baseline_size, timeout, poll=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if out_txt_size() > baseline_size:
            return time.monotonic()
        time.sleep(poll)
    return None


def send(cmd, timeout=600):
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(timeout)
    s.connect(str(SOCK))
    s.sendall(cmd.encode() + b"\n")
    line = s.makefile().readline()
    s.close()
    return json.loads(line)


def start_daemon(log_path, ready_timeout=400):
    f = open(log_path, "w")
    proc = subprocess.Popen(
        ["bash", str(ROOT / "resident_vlm" / "run_resident.sh"), "--sleep-on-start"],
        stdout=f, stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + ready_timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"daemon exited early, see {log_path}")
        if "[resident] ready" in Path(log_path).read_text(errors="ignore"):
            return proc
        time.sleep(1.0)
    raise TimeoutError(f"daemon not ready in {ready_timeout}s, see {log_path}")


def stop_daemon(proc):
    try:
        send("QUIT", timeout=30)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    subprocess.run(["docker", "rm", "-f", "resident_vlm"], capture_output=True, text=True)
    if proc is not None:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_arm(arm, rep, args, outdir):
    tag = f"{arm}_{rep}"
    rec = {"arm": arm, "rep": rep}
    daemon = None
    smp = S.Sampler(outdir / f"{tag}.jsonl", hz=args.hz)

    try:
        if arm == "resident":
            daemon = start_daemon(outdir / f"{tag}.daemon.log")
            rec["load"] = send("STATS")

        compose("stop", "-t", "20", *STACK)
        mht_up()
        time.sleep(args.mht_settle)

        drop_cache([FUSED] if arm == "resident" else default_model_files())
        rec["cooldown_seconds"], rec["tj_start"] = cooldown(args.tj_target, args.cooldown_max)

        smp.start()
        smp.mark("idle_start")
        time.sleep(args.idle_seconds)
        smp.mark("idle_end")

        size0 = out_txt_size()
        smp.mark("trigger")
        t_trigger = time.monotonic()
        mht_stop()
        smp.mark("mht_stopped")
        compose("up", "-d", "wildfire_classifier")
        if arm == "resident":
            rec["wake_reply"] = send("WAKE", timeout=args.verdict_timeout)
        else:
            compose("up", "-d", "wildfire_detection")
        t_verdict = wait_for_verdict(size0, args.verdict_timeout)
        smp.mark("first_verdict")
        rec["trigger_to_verdict_seconds"] = (
            round(t_verdict - t_trigger, 3) if t_verdict else None
        )

        smp.mark("awake_start")
        time.sleep(args.awake_seconds)
        smp.mark("awake_end")
        rec["tj_awake"] = S.read_tj()

        smp.mark("sleep_start")
        if arm == "resident":
            rec["sleep_reply"] = send("SLEEP", timeout=120)
        else:
            compose("stop", "-t", "30", *STACK)
        smp.mark("sleep_done")
        mht_up()
        smp.mark("mht_restarted")
        time.sleep(args.idle_seconds)
        smp.mark("sleep_idle_end")
    finally:
        smp.stop()
        if daemon is not None:
            stop_daemon(daemon)

    (outdir / f"{tag}.json").write_text(json.dumps(rec, indent=2))
    print(f"[{tag}] trigger->verdict = {rec.get('trigger_to_verdict_seconds')}s", flush=True)
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--arms", nargs="+", default=["baseline", "resident"])
    p.add_argument("--awake-seconds", type=float, default=60.0)
    p.add_argument("--idle-seconds", type=float, default=20.0)
    p.add_argument("--mht-settle", type=float, default=15.0)
    p.add_argument("--tj-target", type=float, default=47.0)
    p.add_argument("--cooldown-max", type=float, default=180.0)
    p.add_argument("--verdict-timeout", type=float, default=420.0)
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    outdir = Path(args.out or (ROOT / "resident_vlm" / "results" /
                               time.strftime("bench_%Y%m%d_%H%M%S")))
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"[bench] writing to {outdir}", flush=True)

    all_recs = []
    try:
        for rep in range(args.reps):
            for arm in args.arms:
                all_recs.append(run_arm(arm, rep, args, outdir))
    finally:
        compose("stop", "-t", "30", *STACK)
        subprocess.run(["docker", "rm", "-f", "resident_vlm"], capture_output=True, text=True)
        mht_up()
        (outdir / "summary.json").write_text(json.dumps(all_recs, indent=2))
    print(f"[bench] done -> {outdir}/summary.json", flush=True)


if __name__ == "__main__":
    main()

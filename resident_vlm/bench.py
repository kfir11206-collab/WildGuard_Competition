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
import threading
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
WATCHER_SOCK = ROOT / "fire_watcher" / "watcher.sock"


class MhtSink(threading.Thread):
    """Stands in for fire_watcher's listener.

    mht_live.py connects to watcher.sock at startup and exits with
    ConnectionRefusedError if nobody is listening - so without this the MHT
    container starts, reports rc=0, and dies ~3s later, leaving the benchmark
    with no camera contention and no MHT memory load. We only drain the scores;
    fire_watcher's wake logic would fight the harness for control.
    """

    def __init__(self, path):
        super().__init__(daemon=True)
        self.path = Path(path)
        self.scores = []
        self.stopping = threading.Event()
        if self.path.exists():
            self.path.unlink()
        self.srv = socket.socket(socket.AF_UNIX)
        self.srv.bind(str(self.path))
        os.chmod(self.path, 0o777)
        self.srv.listen(2)
        self.srv.settimeout(1.0)

    def run(self):
        while not self.stopping.is_set():
            try:
                conn, _ = self.srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._drain, args=(conn,), daemon=True).start()

    def _drain(self, conn):
        f = conn.makefile()
        while not self.stopping.is_set():
            line = f.readline()
            if not line:
                break
            try:
                self.scores.append((round(time.monotonic(), 3), float(line.strip())))
            except ValueError:
                pass
        conn.close()

    def stop(self):
        self.stopping.set()
        try:
            self.srv.close()
        except OSError:
            pass
        self.path.unlink(missing_ok=True)


def mht_container_running():
    return bool(compose("ps", "-q", "--status", "running", "mht_watch").stdout.strip())


def wait_for_mht(sink, timeout=90):
    """rc=0 from compose never meant the MHT survived - wait for a real score."""
    n0 = len(sink.scores)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(sink.scores) > n0:
            return True
        if not mht_container_running():
            return False
        time.sleep(0.5)
    return False


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


def run_arm(arm, rep, args, outdir, sink):
    tag = f"{arm}_{rep}"
    rec = {"arm": arm, "rep": rep}
    daemon = None
    smp = S.Sampler(outdir / f"{tag}.jsonl", hz=args.hz)

    try:
        # Clear the camera first: the previous arm ends with mht_up(), and
        # /dev/video0 is single-opener, so anything below that needs the camera
        # would fail while MHT still holds it.
        compose("stop", "-t", "20", *STACK)
        mht_stop()

        if arm == "resident":
            daemon = start_daemon(outdir / f"{tag}.daemon.log")
            rec["load"] = send("STATS")
            # Unmeasured warm-up: the camera's first open costs ~6s and every
            # later one ~0.1s. A continuously-running daemon pays that once at
            # boot, so measuring its first-ever open would inflate every wake.
            rec["warmup_wake"] = send("WAKE", timeout=args.verdict_timeout)
            rec["warmup_sleep"] = send("SLEEP", timeout=120)

        mht_up()
        rec["mht_alive"] = wait_for_mht(sink)
        if not rec["mht_alive"]:
            print(f"[{tag}] WARNING: MHT is not producing scores - "
                  f"camera contention and its memory load are ABSENT", flush=True)
        time.sleep(args.mht_settle)
        mht_scores_at_trigger = len(sink.scores)

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
            compose("stop", "-t", "30", "wildfire_classifier")
        else:
            compose("stop", "-t", "30", *STACK)
        smp.mark("sleep_done")
        mht_up()
        rec["mht_alive_after_wake_cycle"] = wait_for_mht(sink)
        if not rec["mht_alive_after_wake_cycle"]:
            print(f"[{tag}] WARNING: MHT did not come back after the wake cycle "
                  f"- likely the camera was not released", flush=True)
        smp.mark("mht_restarted")
        time.sleep(args.idle_seconds)
        smp.mark("sleep_idle_end")
        rec["mht_scores_during_arm"] = len(sink.scores) - mht_scores_at_trigger
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
    p.add_argument("--tj-target", type=float, default=52.0)
    p.add_argument("--cooldown-max", type=float, default=240.0)
    p.add_argument("--verdict-timeout", type=float, default=420.0)
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    outdir = Path(args.out or (ROOT / "resident_vlm" / "results" /
                               time.strftime("bench_%Y%m%d_%H%M%S")))
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "args.json").write_text(json.dumps(vars(args), indent=2))
    stamp = S.device_stamp()
    (outdir / "device.json").write_text(json.dumps(stamp, indent=2))
    print(f"[bench] writing to {outdir}", flush=True)
    print(f"[bench] storage device: {stamp['label']} "
          f"model={stamp['model']} sn={stamp['serial']}", flush=True)

    sink = MhtSink(WATCHER_SOCK)
    sink.start()
    print(f"[bench] listening on {WATCHER_SOCK} for MHT scores", flush=True)

    all_recs = []
    try:
        for rep in range(args.reps):
            for arm in args.arms:
                all_recs.append(run_arm(arm, rep, args, outdir, sink))
    finally:
        compose("stop", "-t", "30", *STACK)
        subprocess.run(["docker", "rm", "-f", "resident_vlm"], capture_output=True, text=True)
        compose("stop", "-t", "10", "mht_watch")
        (outdir / "summary.json").write_text(json.dumps(all_recs, indent=2))
        (outdir / "mht_scores.json").write_text(json.dumps(sink.scores))
        sink.stop()
    print(f"[bench] done -> {outdir}/summary.json", flush=True)


if __name__ == "__main__":
    main()

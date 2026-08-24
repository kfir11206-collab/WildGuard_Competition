#!/usr/bin/env python3
# 2026-08-24 16:02 IDT - kfir: runs as the `fire_watcher` compose service now.
#   - compose up/stop scoped via --stack-services, so the daemon no longer
#     stops its own container when it goes back to sleep.
#   - run_mht(check_alive=True) spawns verify_mht(), which polls --mht-service
#     for --mht-check-window seconds and logs mht_alive / mht_dead(+tail).
#     compose rc=0 only means the container was launched, not that mht_live.py
#     survived. measured: a camera-busy failure surfaces ~2.9s after rc=0, and
#     mht_live imports cv2 + loads weights before it opens the camera, so a
#     single short check reports a false alive. polled in a thread so it never
#     blocks the confidence loop.
#   - self.lock (RLock) serializes the WATCH<->ACTIVE transitions: on_confidence
#     commits state=WAKING under it, and back_to_sleep holds it across
#     stop-VLM + flip-to-SLEEPING + restart-MHT. Without it a wake firing in the
#     gap right after `slept` left MHT and the VLM both up, and MHT stole the
#     exclusive /dev/video0 from the reloading VLM (Device or resource busy).
import argparse
import json
import mmap
import os
import signal
import socket
import subprocess
import sys
import threading
import time

from model_files import default_model_files

CHUNK = 16 * 1024 * 1024


class Watcher:
    def __init__(self, files, wake_mode, stub_sock, log_path, prewarm_t, wake_t, sleep_t, ema_alpha, keepwarm_interval,
                 classifier_out=None, no_fire_timeout=60.0, classifier_poll=2.0,
                 mht_start_cmd=None, mht_stop_cmd=None):
        self.files = files
        self.wake_mode = wake_mode
        self.stub_sock = stub_sock
        self.log_file = open(log_path, "a")
        self.prewarm_t = prewarm_t
        self.wake_t = wake_t
        self.sleep_t = sleep_t
        self.ema_alpha = ema_alpha
        self.keepwarm_interval = keepwarm_interval
        self.classifier_out = classifier_out
        self.no_fire_timeout = no_fire_timeout
        self.classifier_poll = classifier_poll
        self.mht_start_cmd = mht_start_cmd
        self.mht_stop_cmd = mht_stop_cmd
        self.smoothed = None
        self.state = "SLEEPING"
        self.lock = threading.RLock()
        self.t0 = time.monotonic()

    def log(self, event, **fields):
        rec = {"t": round(time.monotonic() - self.t0, 3), "state": self.state, "event": event, **fields}
        line = json.dumps(rec)
        print(line, flush=True)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def on_confidence(self, conf):
        if self.smoothed is None:
            self.smoothed = conf
        else:
            self.smoothed = self.ema_alpha * conf + (1 - self.ema_alpha) * self.smoothed
        level = self.smoothed
        self.log("confidence", value=conf, smoothed=round(level, 3))
        with self.lock:
            if level >= self.wake_t and self.state not in ("WAKING", "AWAKE"):
                from_state = self.state
                self.state = "WAKING"
                threading.Thread(target=self.wake, args=(level, from_state), daemon=True).start()
            elif level >= self.prewarm_t and self.state == "SLEEPING":
                self.state = "PREWARMING"
                threading.Thread(target=self.prewarm, daemon=True).start()
            elif level < self.sleep_t and self.state == "PREWARMED":
                self.back_to_sleep()

    def walk_files(self, advice=None):
        total = 0
        for path in self.files:
            fd = os.open(path, os.O_RDONLY)
            mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
            if advice is not None:
                mm.madvise(advice)
            size = len(mm)
            for off in range(0, size, CHUNK):
                mm[off : min(off + CHUNK, size)]
            total += size
            mm.close()
            os.close(fd)
        return total

    def prewarm(self):
        t0 = time.monotonic()
        total = self.walk_files(mmap.MADV_WILLNEED)
        if self.state == "PREWARMING":
            self.state = "PREWARMED"
        self.log("prewarm_done", mb=round(total / 1e6), seconds=round(time.monotonic() - t0, 2))
        while self.state == "PREWARMED":
            time.sleep(self.keepwarm_interval)
            if self.state != "PREWARMED":
                break
            t0 = time.monotonic()
            self.walk_files()
            self.log("keepwarm", seconds=round(time.monotonic() - t0, 2))

    def mht_running(self):
        proc = subprocess.run(
            ["docker", "compose", "-f", self.compose_file, "ps", "-q",
             "--status", "running", self.mht_service],
            capture_output=True, text=True,
        )
        return bool(proc.stdout.strip())

    def mht_tail(self):
        proc = subprocess.run(
            ["docker", "compose", "-f", self.compose_file, "logs", "--tail", "3", self.mht_service],
            capture_output=True, text=True,
        )
        return proc.stdout.strip()[-300:]

    def verify_mht(self, t0):
        deadline = time.monotonic() + self.mht_check_window
        while time.monotonic() < deadline:
            time.sleep(0.5)
            if not self.mht_running():
                self.log("mht_dead", after=round(time.monotonic() - t0, 2), tail=self.mht_tail())
                return
        self.log("mht_alive", after=round(time.monotonic() - t0, 2))

    def run_mht(self, cmd, event, check_alive=False):
        if not cmd:
            return
        t0 = time.monotonic()
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        self.log(event, rc=proc.returncode, seconds=round(time.monotonic() - t0, 2))
        if check_alive:
            threading.Thread(target=self.verify_mht, args=(t0,), daemon=True).start()

    def last_wildfire(self, since):
        latest = since
        try:
            with open(self.classifier_out) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = rec.get("timestamp", 0)
                    if ts >= since and rec.get("label") == "WILDFIRE":
                        latest = max(latest, ts)
        except OSError:
            pass
        return latest

    def monitor_active(self, since):
        while self.state == "AWAKE":
            time.sleep(self.classifier_poll)
            if self.state != "AWAKE":
                return
            quiet = time.time() - self.last_wildfire(since)
            if quiet >= self.no_fire_timeout:
                self.log("no_fire_timeout", quiet_seconds=round(quiet, 1))
                self.back_to_sleep()
                return

    def wake_compose(self):
        ready = self.ready_file
        mtime_before = os.stat(ready).st_mtime if os.path.exists(ready) else None
        t0 = time.monotonic()
        up = subprocess.run(
            ["docker", "compose", "-f", self.compose_file, "up", "-d", *self.stack_services],
            capture_output=True, text=True,
        )
        compose_up = round(time.monotonic() - t0, 2)
        if up.returncode != 0:
            return {"error": up.stderr.strip()[-300:], "compose_up_seconds": compose_up}
        first_output = None
        while time.monotonic() - t0 < self.ready_timeout:
            if os.path.exists(ready) and os.stat(ready).st_mtime != mtime_before:
                first_output = round(time.monotonic() - t0, 2)
                break
            time.sleep(0.5)
        return {"compose_up_seconds": compose_up, "first_output_seconds": first_output}

    def wake(self, conf, from_state):
        self.log("wake_start", confidence=conf, from_state=from_state)
        t0 = time.monotonic()
        self.run_mht(self.mht_stop_cmd, "mht_stopped")
        if self.wake_mode == "compose":
            loader = self.wake_compose()
        elif self.wake_mode == "spawn":
            here = os.path.dirname(os.path.abspath(__file__))
            out = subprocess.run(
                [sys.executable, os.path.join(here, "touch_weights.py"), *self.files],
                capture_output=True, text=True,
            )
            loader = json.loads(out.stdout)
        else:
            s = socket.socket(socket.AF_UNIX)
            s.connect(self.stub_sock)
            s.sendall(b"WAKE\n")
            loader = json.loads(s.makefile().readline())
            s.close()
        self.state = "AWAKE"
        self.log("awake", wake_seconds=round(time.monotonic() - t0, 2), loader=loader)
        if self.classifier_out:
            threading.Thread(target=self.monitor_active, args=(time.time(),), daemon=True).start()

    def stop_stack(self):
        t0 = time.monotonic()
        subprocess.run(
            ["docker", "compose", "-f", self.compose_file, "stop", *self.stack_services],
            capture_output=True, text=True,
        )
        return round(time.monotonic() - t0, 2)

    def back_to_sleep(self):
        with self.lock:
            was_awake = self.state == "AWAKE"
            stop_seconds = None
            if self.wake_mode == "compose" and was_awake:
                stop_seconds = self.stop_stack()
            for path in self.files:
                fd = os.open(path, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            self.state = "SLEEPING"
            self.log("slept", compose_stop_seconds=stop_seconds)
            if was_awake:
                self.run_mht(self.mht_start_cmd, "mht_started", check_alive=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", default=None)
    parser.add_argument("--sock", required=True)
    parser.add_argument("--wake-mode", choices=["spawn", "resident", "compose"], required=True)
    parser.add_argument("--stub-sock")
    parser.add_argument("--log", required=True)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--compose-file", default=os.path.join(root, "docker-compose_fused_system.yaml"))
    parser.add_argument("--ready-file", default=os.path.join(root, "data", "out_txt"))
    parser.add_argument("--ready-timeout", type=float, default=300.0)
    parser.add_argument("--stack-services", nargs="+",
                        default=["wildfire_classifier", "wildfire_detection"])
    parser.add_argument("--mht-service", default="mht_watch")
    parser.add_argument("--mht-check-window", type=float, default=15.0)
    parser.add_argument("--classifier-out", default=os.path.join(root, "data", "classifier_out", "classifier_results.jsonl"))
    parser.add_argument("--no-fire-timeout", type=float, default=60.0)
    parser.add_argument("--classifier-poll", type=float, default=2.0)
    compose_default = os.path.join(root, "docker-compose_fused_system.yaml")
    parser.add_argument("--mht-start-cmd",
                        default=f"docker compose -f {compose_default} --profile watch up -d mht_watch")
    parser.add_argument("--mht-stop-cmd",
                        default=f"docker compose -f {compose_default} stop mht_watch")
    parser.add_argument("--prewarm-at", type=float, default=0.60)
    parser.add_argument("--wake-at", type=float, default=0.85)
    parser.add_argument("--sleep-at", type=float, default=0.40)
    parser.add_argument("--ema-alpha", type=float, default=1)
    parser.add_argument("--keepwarm-interval", type=float, default=5.0)
    args = parser.parse_args()

    watcher = Watcher(
        args.files or default_model_files(), args.wake_mode, args.stub_sock, args.log,
        args.prewarm_at, args.wake_at, args.sleep_at, args.ema_alpha, args.keepwarm_interval,
        classifier_out=args.classifier_out, no_fire_timeout=args.no_fire_timeout,
        classifier_poll=args.classifier_poll,
        mht_start_cmd=args.mht_start_cmd, mht_stop_cmd=args.mht_stop_cmd,
    )
    watcher.compose_file = args.compose_file
    watcher.ready_file = args.ready_file
    watcher.ready_timeout = args.ready_timeout
    watcher.stack_services = args.stack_services
    watcher.mht_service = args.mht_service
    watcher.mht_check_window = args.mht_check_window

    if os.path.exists(args.sock):
        os.unlink(args.sock)
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(args.sock)
    srv.listen(1)
    watcher.log("listening", sock=args.sock, wake_mode=args.wake_mode)
    watcher.run_mht(args.mht_start_cmd, "mht_started", check_alive=True)

    def shutdown(signum=None):
        watcher.log("shutdown", signal=signum)
        if watcher.wake_mode == "compose" and watcher.state in ("WAKING", "AWAKE"):
            watcher.log("stack_stopped", seconds=watcher.stop_stack())
        watcher.run_mht(watcher.mht_stop_cmd, "mht_stopped")

    def handle_signal(signum, frame):
        shutdown(signum)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    while True:
        conn, _ = srv.accept()
        for line in conn.makefile():
            line = line.strip()
            if line == "quit":
                shutdown()
                return
            watcher.on_confidence(float(line))
        conn.close()


if __name__ == "__main__":
    main()

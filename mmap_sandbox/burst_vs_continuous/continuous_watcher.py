#!/usr/bin/env python3
"""Continuous-mode counterpart to fire_watcher.py, for the burst-vs-continuous
comparison. Does NOT modify fire_watcher.py: imports and subclasses its Watcher
class, overriding only how the wake load happens.

fire_watcher.py's default ("burst") wake: on MHT threshold cross, immediately
`docker compose up -d`; the container does its own uncontrolled SD-card reads.

This script's ("continuous") wake: on the identical MHT threshold cross, first
walk the same model files in 16 MB chunks paced to --pace-mbps (measured
sustained cold-read throughput, see measure_sustained_rate.py), warming the
host page cache smoothly over time via posix page cache sharing with the
bind-mounted volumes, THEN `docker compose up -d` (which now mostly hits warm
cache instead of the SD card).

Same CLI shape as fire_watcher.py's compose mode, run against the same
mht_sim.py trace/seed, so both scripts see an identical trigger instant.
"""
import argparse
import mmap
import os
import socket
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "fire_watcher"))
from fire_watcher import Watcher, CHUNK
from model_files import default_model_files


class ContinuousWatcher(Watcher):
    def __init__(self, *args, pace_mbps, **kwargs):
        super().__init__(*args, **kwargs)
        self.pace_bps = pace_mbps * 1e6

    def paced_preload(self):
        t0 = time.monotonic()
        total = 0
        for path in self.files:
            fd = os.open(path, os.O_RDONLY)
            mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
            size = len(mm)
            for off in range(0, size, CHUNK):
                mm[off:min(off + CHUNK, size)]
                total += min(off + CHUNK, size) - off
                due = t0 + total / self.pace_bps
                now = time.monotonic()
                if due > now:
                    time.sleep(due - now)
            mm.close()
            os.close(fd)
        return total, time.monotonic() - t0

    def wake(self, conf):
        self.state = "WAKING"
        self.log("wake_start", confidence=conf, from_state="SLEEPING", mode="continuous")
        t0 = time.monotonic()
        mb, seconds = self.paced_preload()
        self.log("continuous_preload_done", mb=round(mb / 1e6), seconds=round(seconds, 2),
                  pace_mbps=round(self.pace_bps / 1e6))
        loader = self.wake_compose()
        self.state = "AWAKE"
        self.log("awake", wake_seconds=round(time.monotonic() - t0, 2), loader=loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", default=None)
    parser.add_argument("--sock", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--pace-mbps", type=float, required=True)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--compose-file", default=os.path.join(root, "docker-compose_fused_system.yaml"))
    parser.add_argument("--ready-file", default=os.path.join(root, "data", "out_txt"))
    parser.add_argument("--ready-timeout", type=float, default=300.0)
    parser.add_argument("--wake-at", type=float, default=0.85)
    parser.add_argument("--sleep-at", type=float, default=0.40)
    parser.add_argument("--ema-alpha", type=float, default=0.4)
    args = parser.parse_args()

    watcher = ContinuousWatcher(
        args.files or default_model_files(), "compose", None, args.log,
        prewarm_t=args.wake_at, wake_t=args.wake_at, sleep_t=args.sleep_at,
        ema_alpha=args.ema_alpha, keepwarm_interval=10 ** 9,
        pace_mbps=args.pace_mbps,
    )
    watcher.compose_file = args.compose_file
    watcher.ready_file = args.ready_file
    watcher.ready_timeout = args.ready_timeout

    if os.path.exists(args.sock):
        os.unlink(args.sock)
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(args.sock)
    srv.listen(1)
    watcher.log("listening", sock=args.sock, wake_mode="continuous", pace_mbps=args.pace_mbps)

    while True:
        conn, _ = srv.accept()
        for line in conn.makefile():
            line = line.strip()
            if line == "quit":
                watcher.log("shutdown")
                return
            watcher.on_confidence(float(line))
        conn.close()


if __name__ == "__main__":
    main()

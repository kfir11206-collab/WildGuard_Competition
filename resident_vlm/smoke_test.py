#!/usr/bin/env python3
"""Gate B1: can a WOKEN model process a real camera frame? Correctness only, not timing."""
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOCK = HERE / "resident.sock"


def send(cmd, timeout=900):
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(timeout)
    s.connect(str(SOCK))
    s.sendall(cmd.encode() + b"\n")
    line = s.makefile().readline()
    s.close()
    return json.loads(line)


def main():
    log = HERE / "results" / "smoke.log"
    log.parent.mkdir(exist_ok=True)
    SOCK.unlink(missing_ok=True)
    f = open(log, "w")
    proc = subprocess.Popen(["bash", str(HERE / "run_resident.sh")],
                            stdout=f, stderr=subprocess.STDOUT)
    print("[smoke] daemon starting (load ~75s + fused save ~15s, first run)...", flush=True)
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            print(f"[smoke] FAIL: daemon exited early, see {log}")
            return 1
        if "[resident] ready" in log.read_text(errors="ignore"):
            break
        time.sleep(2)
    else:
        print(f"[smoke] FAIL: not ready in time, see {log}")
        return 1

    results = {}
    try:
        results["stats_after_load"] = send("STATS")
        print("[smoke] loaded:", json.dumps(results["stats_after_load"]), flush=True)
        for cycle in range(2):
            results[f"sleep_{cycle}"] = send("SLEEP")
            print(f"[smoke] sleep {cycle}:", json.dumps(results[f'sleep_{cycle}']), flush=True)
            results[f"wake_{cycle}"] = send("WAKE")
            print(f"[smoke] wake  {cycle}:", json.dumps(results[f'wake_{cycle}']), flush=True)
        results["stats_final"] = send("STATS")
    finally:
        try:
            send("QUIT", timeout=60)
        except Exception:
            pass
        subprocess.run(["docker", "rm", "-f", "resident_vlm"], capture_output=True)
        proc.wait(timeout=60)
        (HERE / "results" / "smoke_result.json").write_text(json.dumps(results, indent=2))

    ok = True
    for cycle in range(2):
        w = results.get(f"wake_{cycle}", {})
        v = (w.get("verdict") or "").strip()
        checks = {
            "verdict non-empty": bool(v),
            "no re-fusion": w.get("fuse_calls_total") == 1,
            "no missing keys": w.get("missing_keys") == 0,
            "no params left on meta": w.get("params_still_meta") == 0,
        }
        print(f"\n--- cycle {cycle} ---")
        for name, passed in checks.items():
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
            ok &= passed
        print(f"  verdict: {v[:160]!r}")
    print("\n[smoke]", "ALL GATES PASSED" if ok else "GATE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

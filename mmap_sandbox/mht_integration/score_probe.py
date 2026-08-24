#!/usr/bin/env python3
import argparse
import json
import os
import socket
import statistics
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sock", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report-interval", type=float, default=15.0)
    args = parser.parse_args()

    if os.path.exists(args.sock):
        os.unlink(args.sock)
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(args.sock)
    srv.listen(1)
    print(json.dumps({"event": "listening", "sock": args.sock}), flush=True)

    conn, _ = srv.accept()
    t0 = time.monotonic()
    values = []
    last_report = t0
    with open(args.out, "w") as f:
        for line in conn.makefile():
            line = line.strip()
            if not line:
                continue
            v = float(line)
            t = time.monotonic() - t0
            values.append(v)
            f.write(json.dumps({"t": round(t, 3), "conf": v}) + "\n")
            f.flush()
            if time.monotonic() - last_report >= args.report_interval:
                window = values[-int(args.report_interval * 2):]
                print(json.dumps({"event": "window", "t": round(t, 1), "n": len(values),
                                  "last": v, "window_max": max(window),
                                  "window_mean": round(statistics.fmean(window), 3)}), flush=True)
                last_report = time.monotonic()

    elapsed = time.monotonic() - t0
    if values:
        print(json.dumps({"event": "summary", "seconds": round(elapsed, 1), "samples": len(values),
                          "min": min(values), "max": max(values),
                          "mean": round(statistics.fmean(values), 3),
                          "median": round(statistics.median(values), 3)}), flush=True)
    else:
        print(json.dumps({"event": "summary", "samples": 0}), flush=True)


if __name__ == "__main__":
    main()

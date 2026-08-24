#!/usr/bin/env python3
import argparse
import random
import socket
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sock")
    parser.add_argument("trace", help="comma list of base_confidence:duration_seconds phases")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    s = socket.socket(socket.AF_UNIX)
    s.connect(args.sock)
    for phase in args.trace.split(","):
        base, duration = phase.split(":")
        base, duration = float(base), float(duration)
        for _ in range(max(1, round(duration / args.interval))):
            value = min(1.0, max(0.0, rng.gauss(base, args.noise)))
            s.sendall(f"{value:.3f}\n".encode())
            time.sleep(args.interval)
    s.sendall(b"quit\n")
    s.close()


if __name__ == "__main__":
    main()

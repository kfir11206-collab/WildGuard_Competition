#!/usr/bin/env python3
import argparse
import json
import os
import signal
import socket
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MHT1 = os.path.join(ROOT, "MHT_TOP", "MHT1")
sys.path.insert(0, MHT1)

from wildfire_core import WildfirePipeline, MAX_PROC_W
from mht_tracker import MHTTracker
from motion_classifier import GroupMotionClassifier

running = True


def stop(signum, frame):
    global running
    running = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sock", required=True)
    parser.add_argument("--source", default="/dev/video0")
    parser.add_argument("--weights", default=os.path.join(MHT1, "motion_weights.json"))
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--max-proc-w", type=int, default=MAX_PROC_W)
    parser.add_argument("--min-frames", type=int, default=12)
    parser.add_argument("--stats-interval", type=float, default=30.0)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    params = json.load(open(args.weights))["trained_with_params"]
    pipe = WildfirePipeline(params, max_proc_w=args.max_proc_w)
    trk = MHTTracker()
    clf = GroupMotionClassifier(min_frames=args.min_frames)
    if not clf.load_model(args.weights):
        sys.exit(f"could not load {args.weights}")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.source}")

    s = socket.socket(socket.AF_UNIX)
    s.connect(args.sock)

    last_emit = last_stats = time.monotonic()
    frames = 0
    proc_time = 0.0
    score = 0.0

    while running:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        h0, w0 = frame.shape[:2]
        if w0 > args.max_proc_w:
            frame = cv2.resize(frame, (args.max_proc_w, int(h0 * args.max_proc_w / w0)),
                               interpolation=cv2.INTER_AREA)
        g32 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g8 = cv2.convertScaleAbs(g32)
        sv = pipe.gabor(g8, pipe.hsv(frame, pipe.iir(g32, params), params), params)
        dets = MHTTracker.detections_from_mask(sv, min_area=args.min_area)
        for d in dets:
            _, d.iso = pipe.texture_stats(g8, (d.x, d.y, d.w, d.h), params)
        trk.update(dets)
        clf.update(trk.active_tracks())
        score, _ = clf.alarm()
        proc_time += time.perf_counter() - t0
        frames += 1

        now = time.monotonic()
        if now - last_emit >= args.interval:
            s.sendall(f"{score:.3f}\n".encode())
            last_emit = now
        if args.stats_interval and now - last_stats >= args.stats_interval:
            print(json.dumps({"frames": frames, "fps": round(frames / proc_time, 1),
                              "ms_per_frame": round(1000 * proc_time / frames, 1),
                              "score": round(score, 3)}), flush=True)
            last_stats = now

    cap.release()
    s.close()


if __name__ == "__main__":
    main()

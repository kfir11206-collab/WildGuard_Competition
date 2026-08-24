#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MHT1 = os.path.join(ROOT, "MHT_TOP", "MHT1")
sys.path.insert(0, MHT1)

from wildfire_core import WildfirePipeline, MAX_PROC_W
from mht_tracker import MHTTracker
from motion_classifier import GroupMotionClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/dev/video0")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights", default=os.path.join(MHT1, "motion_weights.json"))
    args = parser.parse_args()

    params = json.load(open(args.weights))["trained_with_params"]
    pipe = WildfirePipeline(params, max_proc_w=MAX_PROC_W)
    trk = MHTTracker()
    clf = GroupMotionClassifier(min_frames=12)
    clf.load_model(args.weights)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.source}")

    t0 = time.monotonic()
    n = 0
    with open(args.out, "w") as f:
        while time.monotonic() - t0 < args.seconds:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            h0, w0 = frame.shape[:2]
            if w0 > MAX_PROC_W:
                frame = cv2.resize(frame, (MAX_PROC_W, int(h0 * MAX_PROC_W / w0)),
                                   interpolation=cv2.INTER_AREA)
            g32 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            g8 = cv2.convertScaleAbs(g32)
            motion = pipe.iir(g32, params)
            smoke = pipe.hsv(frame, motion, params)
            sv = pipe.gabor(g8, smoke, params)
            dets = MHTTracker.detections_from_mask(sv, min_area=300)
            for d in dets:
                _, d.iso = pipe.texture_stats(g8, (d.x, d.y, d.w, d.h), params)
            trk.update(dets)
            clf.update(trk.active_tracks())
            score, _ = clf.alarm()
            px = sv.size
            f.write(json.dumps({
                "t": round(time.monotonic() - t0, 3), "n": n,
                "bright": round(float(g8.mean()), 2),
                "contrast": round(float(g8.std()), 2),
                "motion_pct": round(100.0 * float(np.count_nonzero(motion)) / px, 3),
                "mask_pct": round(100.0 * float(np.count_nonzero(sv)) / px, 3),
                "dets": len(dets),
                "tracks": len(trk.active_tracks()),
                "groups": len([g for g in clf.groups if g.misses == 0]),
                "score": round(score, 3),
            }) + "\n")
    cap.release()
    print(json.dumps({"frames": n, "seconds": round(time.monotonic() - t0, 1)}))


if __name__ == "__main__":
    main()

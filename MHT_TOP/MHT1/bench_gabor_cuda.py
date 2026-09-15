#!/usr/bin/env python3
"""
bench_gabor_cuda.py
===============================================================================
Head-to-head benchmark for the Gabor filter bank -- the hot spot of the MHT
pipeline (measured at 93.7% of frame time).

Three variants are compared on the same ROIs:

    cpu-uncapped   original: 6x cv2.filter2D over the full ROI
    cpu-capped     same, but through the wildfire_core tile budget
    cupy           CuPy on the GPU, through the same tile budget

Run:
    python3 bench_gabor_cuda.py
    python3 bench_gabor_cuda.py --video ../val_videos/smoke/final.mp4 --frame 375
===============================================================================
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

MHT1 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MHT1)

from wildfire_core import (WildfirePipeline, MAX_PROC_W, _gabor_tiles,
                           bank_response_cpu)
from gabor_cupy import (CUPY_AVAILABLE, TiledGaborCuPy, GaborBankCuPy,
                        HybridGaborBank)

if CUPY_AVAILABLE:
    import cupy as cp


def bank_uncapped(roi, bank):
    """The original code path, before the ROI budget."""
    return np.array(
        [np.mean(np.abs(cv2.filter2D(roi, cv2.CV_32F, k))) for k in bank],
        np.float32,
    )


def bank_capped(roi, bank):
    """CPU path with the tile budget applied (the live wildfire_core path)."""
    return bank_response_cpu(roi, bank)


def bench(fn, arg, iters, warmup=5, sync=False):
    for _ in range(warmup):
        fn(arg)
    if sync:
        cp.cuda.Stream.null.synchronize()
    tm = cv2.TickMeter()
    tm.start()
    for _ in range(iters):
        fn(arg)
    if sync:
        cp.cuda.Stream.null.synchronize()
    tm.stop()
    ms = tm.getTimeMilli() / iters
    return ms, (1000.0 / ms if ms > 0 else float("inf"))


def collect_rois(video, frame_idx, params):
    pipe = WildfirePipeline(params, max_proc_w=MAX_PROC_W)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"cannot open {video}")

    gray_u8 = smoke = None
    for _ in range(frame_idx + 1):
        ok, frame = cap.read()
        if not ok:
            break
        h0, w0 = frame.shape[:2]
        if w0 > MAX_PROC_W:
            frame = cv2.resize(frame, (MAX_PROC_W, int(h0 * MAX_PROC_W / w0)),
                               interpolation=cv2.INTER_AREA)
        g32 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_u8 = cv2.convertScaleAbs(g32)
        smoke = pipe.hsv(frame, pipe.iir(g32, params), params)
    cap.release()

    if smoke is None:
        sys.exit("no frames read")
    pipe.gabor(gray_u8, smoke, params)

    cnts, _ = cv2.findContours(smoke, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for c in sorted((c for c in cnts if cv2.contourArea(c) >= 300),
                    key=cv2.contourArea, reverse=True):
        x, y, bw, bh = cv2.boundingRect(c)
        if bw >= 4 and bh >= 4:
            rois.append(np.ascontiguousarray(
                gray_u8[y:y + bh, x:x + bw].astype(np.float32)))
    if not rois:
        sys.exit(f"no smoke blobs >=300px at frame {frame_idx}")
    return pipe, rois


def main():
    ap = argparse.ArgumentParser(description="CPU vs CuPy Gabor bank benchmark")
    ap.add_argument("--video", default=os.path.join(MHT1, "..", "val_videos",
                                                    "smoke", "final.mp4"))
    ap.add_argument("--frame", type=int, default=375)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--weights", default=os.path.join(MHT1, "motion_weights.json"))
    args = ap.parse_args()

    params = json.load(open(args.weights))["trained_with_params"]
    pipe, rois = collect_rois(args.video, args.frame, params)
    bank = pipe._gabor_bank

    print(f"OpenCV {cv2.__version__}   CuPy: "
          f"{cp.__version__ + ' on ' + cp.cuda.runtime.getDeviceProperties(0)['name'].decode() if CUPY_AVAILABLE else 'UNAVAILABLE'}")
    print(f"{os.path.basename(args.video)} frame {args.frame}   "
          f"bank {len(bank)}x{bank[0].shape[0]}x{bank[0].shape[1]}   "
          f"{len(rois)} ROIs   iters {args.iters}")
    print()

    gpu = TiledGaborCuPy(bank) if CUPY_AVAILABLE else None
    gpu_raw = {}
    hyb = HybridGaborBank(bank)

    hdr = (f"{'ROI (px)':>11} {'tiles':>6} | {'cpu uncap':>9} {'cpu cap':>8} "
           f"| {'gpu cap':>8} {'gpu uncap':>9} | {'hybrid':>6} {'best':>9} | {'max err':>9}")
    print(hdr)
    print("-" * len(hdr))

    tot = {"uncap": 0.0, "cap": 0.0, "cupy": 0.0, "cupy_raw": 0.0, "best": 0.0}
    for roi in rois:
        n_tiles = len(_gabor_tiles(roi))
        u_ms, _ = bench(lambda r: bank_uncapped(r, bank), roi, args.iters)
        c_ms, _ = bench(lambda r: bank_capped(r, bank), roi, args.iters)
        tot["uncap"] += u_ms
        tot["cap"] += c_ms
        label = f"{roi.shape[1]}x{roi.shape[0]}"

        if CUPY_AVAILABLE:
            g_ms, _ = bench(gpu, roi, args.iters, sync=True)
            tot["cupy"] += g_ms
            raw = gpu_raw.get(roi.shape)
            if raw is None:
                raw = gpu_raw[roi.shape] = GaborBankCuPy(bank, roi.shape)
            gr_ms, _ = bench(raw, roi, args.iters, sync=True)
            tot["cupy_raw"] += gr_ms
            err = float(np.max(np.abs(bank_capped(roi, bank) - gpu(roi))))
            best = min((u_ms, "cpu-uncap"), (c_ms, "cpu-cap"),
                       (g_ms, "gpu-cap"), (gr_ms, "gpu-uncap"))[1]
            h_ms, _ = bench(hyb, roi, args.iters, sync=True)
            tot["best"] += h_ms
            print(f"{label:>11} {n_tiles:>6} | {u_ms:9.2f} {c_ms:8.2f} "
                  f"| {g_ms:8.2f} {gr_ms:9.2f} | {h_ms:6.2f} {best:>9} | {err:9.2e}")
        else:
            best = min((u_ms, "cpu-uncap"), (c_ms, "cpu-cap"))[1]
            tot["best"] += min(u_ms, c_ms)
            print(f"{label:>11} {n_tiles:>6} | {u_ms:9.2f} {c_ms:8.2f} "
                  f"| {'n/a':>8} {'n/a':>9} | {best:>10} | {'n/a':>9}")

    print("-" * len(hdr))
    u, c, b = tot["uncap"], tot["cap"], tot["best"]
    if CUPY_AVAILABLE:
        g, gr = tot["cupy"], tot["cupy_raw"]
        print(f"{'ALL ROIs':>11} {'':>6} | {u:9.2f} {c:8.2f} | {g:8.2f} {gr:9.2f} "
              f"| {b:10.2f} |")
        print(f"\nbank-only ms   cpu-uncap {u:.1f}   cpu-cap {c:.1f}   "
              f"gpu-cap {g:.1f}   gpu-uncap {gr:.1f}   hybrid {b:.1f}")
        print(f"bank-only FPS  cpu-uncap {1000/u:.1f}   cpu-cap {1000/c:.1f}   "
              f"gpu-cap {1000/g:.1f}   gpu-uncap {1000/gr:.1f}   hybrid {1000/b:.1f}")
        print(f"speedup vs original:  cpu-cap {u/c:.2f}x   gpu-cap {u/g:.2f}x   "
              f"gpu-uncap {u/gr:.2f}x   hybrid {u/b:.2f}x")
    else:
        print(f"{'ALL ROIs':>11} {'':>6} | {u:9.2f} {c:8.2f} | {'n/a':>8} {'n/a':>9} "
              f"| {b:10.2f} |")
        print(f"\nbank-only FPS  cpu-uncap {1000/u:.1f}   cpu-cap {1000/c:.1f}")
        print("CuPy unavailable -- install cupy-cuda13x to fill the GPU columns.")

    print("note: the bank runs twice per detection per frame "
          "(gabor() + texture_stats()), so per-frame cost is ~2x these rows.")


if __name__ == "__main__":
    main()

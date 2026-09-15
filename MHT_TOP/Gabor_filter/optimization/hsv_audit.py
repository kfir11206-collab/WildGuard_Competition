"""Audit the HSV mask thresholds.

Builds a 64x64 joint (S, V) histogram per clip from sampled frames, so the
coverage implied by ANY (s_high, v_low, v_high) can be evaluated instantly
without re-decoding video.

  python hsv_audit.py step    # repeat until 84/84
  python hsv_audit.py sweep   # coverage vs s_high / v_high, split by label
"""
import os, sys, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import os, sys
HERE  = os.path.dirname(os.path.abspath(__file__))   # Gabor_filter/optimization
GABOR = os.path.dirname(HERE)                        # Gabor_filter
R     = os.path.dirname(GABOR)                       # project root (holds val_videos/)
OUT   = HERE
sys.path.insert(0, GABOR)
import tempfile
TMP = tempfile.gettempdir()
PARTS = os.path.join(TMP, "hsv_hist")
NB = 64          # histogram bins per axis
NFRAMES = 8      # frames sampled per clip
STRIDE = 25
BUDGET = 20.0

from extract_cache import inventory


def work(item):
    import cv2
    cap = cv2.VideoCapture(item["path"])
    hist = np.zeros((NB, NB), np.float64)
    got = idx = 0
    while got < NFRAMES:
        if idx % STRIDE != 0:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, fr = cap.read()
        if not ok:
            break
        idx += 1
        fr = cv2.resize(fr, (320, 240), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        s = (hsv[..., 1].astype(np.int32) * NB) // 256
        v = (hsv[..., 2].astype(np.int32) * NB) // 256
        np.add.at(hist, (s.ravel(), v.ravel()), 1.0)
        got += 1
    cap.release()
    if got:
        hist /= hist.sum()
    np.save(f"{PARTS}/{item['sig']}.npy", hist.astype(np.float32))
    return item["sig"], got


def step():
    os.makedirs(PARTS, exist_ok=True)
    items = inventory()
    todo = [i for i in items if not os.path.exists(f"{PARTS}/{i['sig']}.npy")]
    print(f"[step] {len(items)-len(todo)}/{len(items)} done, {len(todo)} left")
    if not todo:
        return
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        it = iter(todo); futs = {}
        for _ in range(4):
            n = next(it, None)
            if n:
                futs[ex.submit(work, n)] = n
        while futs:
            done, _ = wait(list(futs), return_when=FIRST_COMPLETED, timeout=60)
            for f in done:
                n = futs.pop(f)
                try:
                    f.result()
                except Exception as e:
                    print(f"  FAIL {n['name']}: {e}")
                    np.save(f"{PARTS}/{n['sig']}.npy", np.zeros((NB, NB), np.float32))
                if time.time() - t0 < BUDGET:
                    nx = next(it, None)
                    if nx:
                        futs[ex.submit(work, nx)] = nx
    print(f"[step] {time.time()-t0:.1f}s")


def coverage(h, s_hi, v_lo, v_hi):
    si = int(np.ceil(s_hi * 255 / 256 * NB))
    a = int(np.floor(v_lo * 255 / 256 * NB))
    b = int(np.ceil(v_hi * 255 / 256 * NB))
    return float(h[:si, a:b].sum())


def sweep():
    items = inventory()
    H = {i["sig"]: np.load(f"{PARTS}/{i['sig']}.npy") for i in items}
    lab = np.array([i["label"] for i in items])
    print(f"[audit] {len(items)} clips, {int((lab==1).sum())} smoke / "
          f"{int((lab==0).sum())} non-smoke\n")

    print("mean mask coverage vs s_high   (v range fixed at 0.38-0.985)")
    print("  s_high |  smoke  non-smoke   gap  | empty-mask clips (cov<1%)")
    for s_hi in [0.28, 0.35, 0.45, 0.55, 0.65, 0.80, 1.00]:
        c = np.array([coverage(H[i["sig"]], s_hi, 0.38, 0.985) for i in items])
        empty = int((c < 0.01).sum())
        print(f"   {s_hi:.2f}  |  {c[lab==1].mean():.3f}    {c[lab==0].mean():.3f}   "
              f"{c[lab==1].mean()-c[lab==0].mean():+.3f}  |  {empty}")

    print("\nmean mask coverage vs v_high   (s_high fixed at 0.28)")
    print("  v_high |  smoke  non-smoke   gap  | empty-mask clips")
    for v_hi in [0.985, 0.995, 1.0]:
        c = np.array([coverage(H[i["sig"]], 0.28, 0.38, v_hi) for i in items])
        print(f"   {v_hi:.3f} |  {c[lab==1].mean():.3f}    {c[lab==0].mean():.3f}   "
              f"{c[lab==1].mean()-c[lab==0].mean():+.3f}  |  {int((c<0.01).sum())}")

    print("\nclips with an EMPTY mask under the current settings "
          "(s<=0.28, 0.38<=v<=0.985):")
    for i in items:
        c = coverage(H[i["sig"]], 0.28, 0.38, 0.985)
        if c < 0.01:
            c2 = coverage(H[i["sig"]], 0.60, 0.30, 1.0)
            print(f"   {i['cls']+'/'+i['name']:46s} cov={c:.4f} "
                  f"-> {c2:.3f} with s<=0.60, 0.30<=v<=1.0")


if __name__ == "__main__":
    (sweep if len(sys.argv) > 1 and sys.argv[1] == "sweep" else step)()

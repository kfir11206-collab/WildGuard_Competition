"""Resumable extraction of the 28-D v2 features for every unique clip.

Same clip inventory, stride, frame cap and IIR mode as extract_cache.py, so the
v1 vs v2 comparison is like-for-like -- only the feature stage differs.

  python extract_cache_v2.py step   # repeat until it reports 84/84
  python extract_cache_v2.py pack   # -> feats_cache_v2.npz
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
STRIDE, MAX_FRAMES, DIFF_MODE = 5, 50, "iir-auto"
BUDGET = 1.0

# HSV mask thresholds, overridable so the mask can be retuned without edits.
# Defaults are the paper values (s_high=0.28, v 0.38-0.985).
S_HI = float(os.environ.get("SMOKE_SHI", 0.28))
V_LO = float(os.environ.get("SMOKE_VLO", 0.38))
V_HI = float(os.environ.get("SMOKE_VHI", 0.985))
TAG = os.environ.get("SMOKE_TAG", "v2")
PARTS = os.path.join(TMP, f"parts_{TAG}")

from extract_cache import inventory  # identical clip inventory / dedup


def work(item):
    import cv2
    from smoke_features import SmokeConfig, make_iir
    from smoke_features_v2 import (build_bank_v2, frame_pair_features_v2,
                                   FEATURE_DIM_V2)
    cfg = SmokeConfig(diff_mode=DIFF_MODE, s_high=S_HI, v_low=V_LO, v_high=V_HI)
    bank = build_bank_v2(cfg)
    cap = cv2.VideoCapture(item["path"])
    feats, prev, idx, iir = [], None, 0, make_iir(cfg)
    while True:
        if idx % STRIDE != 0:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, fr = cap.read()
        if not ok:
            break
        idx += 1
        if cfg.resize is not None:
            fr = cv2.resize(fr, cfg.resize, interpolation=cv2.INTER_AREA)
        if prev is not None:
            feats.append(frame_pair_features_v2(prev, fr, bank, cfg, iir=iir))
            if len(feats) >= MAX_FRAMES:
                break
        prev = fr
    cap.release()
    a = (np.asarray(feats, np.float32) if feats
         else np.empty((0, FEATURE_DIM_V2), np.float32))
    np.save(f"{PARTS}/{item['sig']}.npy", a)
    return item["sig"], len(a)


def step():
    os.makedirs(PARTS, exist_ok=True)
    items = inventory()
    todo = [it for it in items if not os.path.exists(f"{PARTS}/{it['sig']}.npy")]
    print(f"[step] {len(items)-len(todo)}/{len(items)} done, {len(todo)} left")
    if not todo:
        return
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=3) as ex:
        it = iter(todo)
        futs = {}
        for _ in range(3):
            n = next(it, None)
            if n:
                futs[ex.submit(work, n)] = n
        while futs:
            done, _ = wait(list(futs), return_when=FIRST_COMPLETED, timeout=60)
            for f in done:
                n = futs.pop(f)
                try:
                    _s, k = f.result()
                    print(f"  ok {n['cls']}/{n['name']} -> {k}", flush=True)
                except Exception as e:
                    print(f"  FAIL {n['name']}: {e}", flush=True)
                    np.save(f"{PARTS}/{n['sig']}.npy", np.empty((0, 28), np.float32))
                if time.time() - t0 < BUDGET:
                    nx = next(it, None)
                    if nx:
                        futs[ex.submit(work, nx)] = nx
    print(f"[step] {time.time()-t0:.1f}s")


def pack():
    items = inventory()
    X, y, g, meta = [], [], [], []
    for it in items:
        fv = np.load(f"{PARTS}/{it['sig']}.npy")
        if len(fv) == 0:
            print(f"[warn] no frames: {it['name']}")
            continue
        gid = len(meta)
        X.append(fv)
        y.append(np.full(len(fv), it["label"], np.int64))
        g.append(np.full(len(fv), gid, np.int64))
        meta.append({k: v for k, v in it.items() if k != "sig"})
    Xa = np.vstack(X)
    for p in (os.path.join(OUT, f"feats_cache_{TAG}.npz"),):
        np.savez_compressed(p, X=Xa, y=np.concatenate(y),
                            groups=np.concatenate(g), meta=json.dumps(meta))
    print(f"[saved] feats_cache_{TAG}.npz  {Xa.shape} over {len(meta)} clips")


if __name__ == "__main__":
    (pack if len(sys.argv) > 1 and sys.argv[1] == "pack" else step)()

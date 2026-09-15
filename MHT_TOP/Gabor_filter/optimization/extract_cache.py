"""Resumable extraction of the 21-D Appana features for every UNIQUE clip.

Deduplicates across val_videos / test_videos / new_train_videos / new_test_videos
by (size, md5-of-first-1MB) so each physical clip is featurised exactly once.
Writes one .npy per clip into PARTS and exits after a time budget, so it can be
re-run until complete. Settings match the baseline: stride=5, max_frames=50,
diff_mode=iir-auto.

  python extract_cache.py step     # do as much as fits in the budget
  python extract_cache.py pack     # combine parts -> feats_cache.npz
"""
import os, sys, glob, hashlib, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import os, sys
HERE  = os.path.dirname(os.path.abspath(__file__))   # Gabor_filter/optimization
GABOR = os.path.dirname(HERE)                        # Gabor_filter
R     = os.path.dirname(GABOR)                       # project root (holds val_videos/)
OUT   = HERE
sys.path.insert(0, GABOR)
import tempfile
TMP = tempfile.gettempdir()
PARTS = os.path.join(TMP, "parts")

STRIDE, MAX_FRAMES, DIFF_MODE = 5, 50, "iir-auto"
ROOTS = ["val_videos", "test_videos", "new_train_videos", "new_test_videos"]
BUDGET = 26.0


def sig(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
    return f"{os.path.getsize(p)}_{h.hexdigest()}"


def inventory():
    cache = os.path.join(TMP, "uniq_inv.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    uniq = {}
    for root in ROOTS:
        for p in sorted(glob.glob(f"{R}/{root}/*/*.mp4")):
            cls = os.path.basename(os.path.dirname(p))
            s = sig(p)
            u = uniq.setdefault(s, {
                "sig": s, "path": p, "name": os.path.basename(p), "cls": cls,
                "label": 1 if cls in ("smoke", "both") else 0, "roots": [],
            })
            u["roots"].append(root)
    items = sorted(uniq.values(), key=lambda d: (d["cls"], d["name"]))
    json.dump(items, open(cache, "w"))
    return items


def fast_extract(path, bank, cfg, stride, max_frames):
    """Same output as smoke_detection.extract_video_features, but skipped frames
    are grab()'d instead of read() -- no decode for 4 of every 5 frames."""
    import cv2
    from smoke_features import frame_pair_features, make_iir
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return np.empty((0, 21))
    feats, prev, idx, iir = [], None, 0, make_iir(cfg)
    while True:
        if idx % stride != 0:
            if not cap.grab():           # advance without decoding
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if cfg.resize is not None:
            frame = cv2.resize(frame, cfg.resize, interpolation=cv2.INTER_AREA)
        if prev is not None:
            feats.append(frame_pair_features(prev, frame, bank, cfg, iir=iir))
            if max_frames is not None and len(feats) >= max_frames:
                break
        prev = frame
    cap.release()
    return np.asarray(feats, np.float64) if feats else np.empty((0, 21))


def work(item):
    from smoke_features import SmokeConfig, build_gabor_bank
    cfg = SmokeConfig(diff_mode=DIFF_MODE)
    bank = build_gabor_bank(cfg)
    fv = fast_extract(item["path"], bank, cfg, STRIDE, MAX_FRAMES)
    np.save(f"{PARTS}/{item['sig']}.npy", np.asarray(fv, dtype=np.float32))
    return item["sig"], len(fv)


def step():
    os.makedirs(PARTS, exist_ok=True)
    items = inventory()
    todo = [it for it in items if not os.path.exists(f"{PARTS}/{it['sig']}.npy")]
    print(f"[step] {len(items)-len(todo)}/{len(items)} done, {len(todo)} left")
    if not todo:
        return
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=5) as ex:
        futs = {}
        it = iter(todo)
        for _ in range(5):
            n = next(it, None)
            if n:
                futs[ex.submit(work, n)] = n
        while futs:
            from concurrent.futures import wait, FIRST_COMPLETED
            done, _p = wait(list(futs), return_when=FIRST_COMPLETED, timeout=60)
            for f in done:
                n = futs.pop(f)
                try:
                    _s, k = f.result()
                    print(f"  ok {n['cls']}/{n['name']} -> {k} frames", flush=True)
                except Exception as e:
                    print(f"  FAIL {n['name']}: {e}", flush=True)
                    np.save(f"{PARTS}/{n['sig']}.npy", np.empty((0, 21), np.float32))
                if time.time() - t0 < BUDGET:
                    nxt = next(it, None)
                    if nxt:
                        futs[ex.submit(work, nxt)] = nxt
    print(f"[step] {time.time()-t0:.1f}s")


def pack():
    items = inventory()
    X, y, g, meta = [], [], [], []
    for it in items:
        f = f"{PARTS}/{it['sig']}.npy"
        if not os.path.exists(f):
            raise SystemExit(f"missing {it['name']} - run step again")
        fv = np.load(f)
        if len(fv) == 0:
            print(f"[warn] no frames: {it['name']}")
            continue
        gid = len(meta)
        X.append(fv)
        y.append(np.full(len(fv), it["label"], np.int64))
        g.append(np.full(len(fv), gid, np.int64))
        meta.append({k: v for k, v in it.items() if k != "sig"})
    np.savez_compressed(os.path.join(OUT, "feats_cache.npz"), X=np.vstack(X),
                        y=np.concatenate(y), groups=np.concatenate(g),
                        meta=json.dumps(meta))
    print(f"[saved] feats_cache.npz  {np.vstack(X).shape} over {len(meta)} clips")


if __name__ == "__main__":
    (pack if len(sys.argv) > 1 and sys.argv[1] == "pack" else step)()

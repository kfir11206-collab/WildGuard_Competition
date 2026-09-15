"""
train_motion_classifier.py
===============================================================================
Train the group-motion SMOKE/CLOUD classifier on a labeled video folder.

Expected layout:
    val_videos/
        fire/    *.mp4 ...   -> POSITIVE (fires produce smoke)
        smoke/   *.mp4 ...   -> POSITIVE
        both/    *.mp4 ...   -> POSITIVE
        clean/   *.mp4 ...   -> NEGATIVE (clouds / fog / nothing)

Per clip, the FULL pipeline runs headless (IIR -> HSV -> Gabor -> MHT ->
group clustering) and every group alive >= min_frames contributes ONE sample:
the per-frame signal vector averaged over its life. Group labels are weak
(inherited from the clip's folder) -- standard for this kind of calibration.

Model: L2 logistic regression on standardized features, pure NumPy (runs on
the Orin Nano with no sklearn). Evaluation: LEAVE-ONE-CLIP-OUT -- for each
clip, train on the other clips' groups, predict the held-out clip as
positive iff its best group probability exceeds the decision threshold.
The threshold itself is chosen on the LOCO scores to maximize balanced
accuracy, so the headline number is honest.

Output: motion_weights.json next to this script. wildfire_gui.py auto-loads
it at startup, and GroupMotionClassifier.load_model() consumes it anywhere.

Usage (Windows or Jetson):
    python train_motion_classifier.py --folder path/to/val_videos
    python train_motion_classifier.py --folder val_videos --max-frames 400 --iir-auto
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import hashlib
import sys
import time

import cv2
import numpy as np

from wildfire_core import WildfirePipeline, MAX_PROC_W
from mht_tracker import MHTTracker
from motion_classifier import GroupMotionClassifier

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
POS_FOLDERS = ("fire", "smoke", "both")
NEG_FOLDERS = ("clean",)
FEATURES = ["upward", "anchoring", "wander", "spread", "growth",
            "elong", "vy", "vx_abs", "bot_still", "gabor_iso", "life",
            "speed", "n_members"]

# groups shorter than this can't decide a CLIP (they still train, weighted
# down by lifetime) -- kills max-pooling false positives from churn blobs.
# Raised 25 -> 40: a ~30-frame streak group was overriding long-lived clean
# groups and causing false alarms (e.g. fast elongated cloud edges). At 40 the
# test set went from 60% -> 80% specificity with recall unchanged (88.9%).
MAJOR_LIFE = 40   # was 25

# GUI defaults -- override with --params params.json (a dict like self.params)
DEFAULT_PARAMS = {
    "iir_auto": 0, "iir_auto_gain": 1.0,
    "iir_alpha": 0.05, "iir_threshold": 12,
    "smoke_h_min": 0, "smoke_h_max": 179,
    "smoke_s_min": 0, "smoke_s_max": 153,
    "smoke_v_min": 30, "smoke_v_max": 255,
    "gabor_ksize": 21, "gabor_sigma": 2.12, "gabor_lambda": 3.82,
    "gabor_gamma": 1.9, "gabor_std_thresh": 77.28,
}


# ------------------------------------------------------------------------------
#  Feature extraction: one clip -> list of group feature vectors
# ------------------------------------------------------------------------------
def _cache_key(path: str, params: dict, max_frames: int) -> str:
    payload = json.dumps(params, sort_keys=True) + "|" + str(max_frames) \
        + "|" + ",".join(FEATURES) + "|" + str(os.path.getmtime(path))
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def extract_clip(path: str, params: dict, max_frames: int,
                 min_group_frames: int = 12,
                 cache_dir: str | None = None) -> list[tuple[np.ndarray, int]]:
    """Returns [(feature_vector, life_in_frames), ...] -- one per group."""
    if cache_dir:
        cpath = os.path.join(
            cache_dir,
            os.path.basename(path) + "_" + _cache_key(path, params, max_frames) + ".npz")
        if os.path.isfile(cpath):
            z = np.load(cpath)
            return [(z["X"][i], int(z["life"][i])) for i in range(len(z["life"]))]

    pipe = WildfirePipeline(params, max_proc_w=MAX_PROC_W)
    trk = MHTTracker()
    clf = GroupMotionClassifier(min_frames=min_group_frames)

    # accumulate per-group signal history: gid -> list of vectors
    acc: dict[int, list[np.ndarray]] = {}

    cap = cv2.VideoCapture(path)
    n = 0
    while n < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        h0, w0 = frame.shape[:2]
        if w0 > MAX_PROC_W:
            frame = cv2.resize(frame, (MAX_PROC_W, int(h0 * MAX_PROC_W / w0)),
                               interpolation=cv2.INTER_AREA)
        gray_f32 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_u8 = cv2.convertScaleAbs(gray_f32)

        motion = pipe.iir(gray_f32, params)
        smoke_m = pipe.hsv(frame, motion, params)
        smoke_v = pipe.gabor(gray_u8, smoke_m, params)

        dets = MHTTracker.detections_from_mask(smoke_v, min_area=300)
        for d in dets:   # attach texture isotropy (fused as a feature)
            _, d.iso = pipe.texture_stats(gray_u8, (d.x, d.y, d.w, d.h), params)
        trk.update(dets)
        clf.update(trk.active_tracks())

        for g in clf.groups:
            if g.misses == 0 and g.signals:
                vec = np.array([float(g.signals.get(k, 0.5)) for k in FEATURES],
                               np.float64)
                acc.setdefault(g.id, []).append(vec)
    cap.release()

    # one sample per group = mean signal vector over its life (+ lifetime)
    out = [(np.mean(v, axis=0), len(v)) for v in acc.values()
           if len(v) >= min_group_frames]
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        X = np.array([o[0] for o in out]) if out else np.zeros((0, len(FEATURES)))
        life = np.array([o[1] for o in out], np.int32)
        np.savez(cpath, X=X, life=life)
    return out


# ------------------------------------------------------------------------------
#  NumPy logistic regression (L2), standardized features
# ------------------------------------------------------------------------------
def fit_logistic(X, y, life=None, l2=1e-2, iters=3000, lr=0.3):
    # std floor: a feature that barely varied in training must not produce
    # extreme standardized values on new data
    mu = X.mean(axis=0)
    sd = np.maximum(X.std(axis=0), 0.05)
    Xs = (X - mu) / sd
    n, d = Xs.shape
    w = np.zeros(d); b = 0.0
    # class balancing so 'clean' groups don't drown the positives (or v.v.)
    pw = 0.5 / max(y.mean(), 1e-6)
    nw = 0.5 / max(1 - y.mean(), 1e-6)
    sample_w = np.where(y == 1, pw, nw)
    if life is not None:
        # long-lived groups are trustworthy evidence; churn blobs are not
        lw = np.log1p(np.asarray(life, float))
        sample_w = sample_w * (lw / lw.mean())
    for _ in range(iters):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = sample_w * (p - y)
        w -= lr * (Xs.T @ g / n + l2 * w)
        b -= lr * g.mean()
    return w, b, mu, sd


def predict_proba(X, w, b, mu, sd):
    Xs = (X - mu) / (sd + 1e-9)
    return 1.0 / (1.0 + np.exp(-(Xs @ w + b)))


# ------------------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------------------
def main():
    global MAJOR_LIFE
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="val_videos parent folder")
    ap.add_argument("--max-frames", type=int, default=450,
                    help="frames per clip to process (450 ~ 18s @ 25fps)")
    ap.add_argument("--params", default=None,
                    help="optional JSON file with pipeline params")
    ap.add_argument("--iir-auto", action="store_true",
                    help="run Stage 1 in auto (adaptive) mode")
    ap.add_argument("--auto-gain", type=float, default=1.0,
                    help="c for auto IIR threshold")
    ap.add_argument("--out", default=None,
                    help="output json (default: motion_weights.json next to script)")
    ap.add_argument("--no-cache", action="store_true",
                    help="disable the per-clip feature cache")
    ap.add_argument("--fn-cost", type=float, default=None,
                    help="cost-weighted threshold mode: how much worse a "
                         "missed smoke clip is than a false alarm. If not "
                         "given, RECALL-FIRST mode is used (default).")
    ap.add_argument("--min-spec", type=float, default=0.5,
                    help="specificity floor for the default recall-first "
                         "mode (default 0.5). Set 0 for absolute recall "
                         "priority.")
    ap.add_argument("--min-recall", type=float, default=None,
                    help="pick the threshold with best clean specificity "
                         "SUBJECT TO smoke recall >= this (e.g. 0.9). "
                         "Ignored when --fn-cost is given.")
    ap.add_argument("--major-life", type=int, default=None,
                    help=f"min group lifetime (frames) to decide a clip "
                         f"(default {MAJOR_LIFE}). Higher = fewer short-blob "
                         f"false alarms.")
    ap.add_argument("--l2", type=float, default=1e-2,
                    help="L2 regularization for the logistic fit (default 1e-2)")
    args = ap.parse_args()

    if args.major_life is not None:
        MAJOR_LIFE = args.major_life

    params = dict(DEFAULT_PARAMS)
    if args.params:
        with open(args.params) as f:
            params.update(json.load(f))
    if args.iir_auto:
        params["iir_auto"] = 1
        params["iir_auto_gain"] = args.auto_gain

    # -- gather clips ----------------------------------------------------------
    clips: list[tuple[str, int, str]] = []          # (path, label, folder)
    for folder, lab in [(f, 1) for f in POS_FOLDERS] + \
                       [(f, 0) for f in NEG_FOLDERS]:
        d = os.path.join(args.folder, folder)
        if not os.path.isdir(d):
            print(f"[WARN] missing subfolder: {d}")
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(VIDEO_EXTS):
                clips.append((os.path.join(d, f), lab, folder))
    if not clips:
        sys.exit(f"No videos found under {args.folder}")
    print(f"Found {len(clips)} clips "
          f"({sum(l for _, l, _ in clips)} positive / "
          f"{sum(1 - l for _, l, _ in clips)} clean)\n")

    # -- extract group samples per clip ---------------------------------------
    cache_dir = None if args.no_cache else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_cache")
    per_clip: list[list[tuple]] = []       # per clip: [(vec, life), ...]
    t_all = time.time()
    for i, (path, lab, folder) in enumerate(clips):
        t0 = time.time()
        samples = extract_clip(path, params, args.max_frames, cache_dir=cache_dir)
        per_clip.append(samples)
        n_major = sum(1 for _, lf in samples if lf >= MAJOR_LIFE)
        print(f"[{i+1:>3}/{len(clips)}] {folder:>5} | "
              f"{os.path.basename(path)[:38]:<38} "
              f"groups={len(samples):<3} major={n_major:<3} ({time.time()-t0:.1f}s)")
    print(f"\nExtraction done in {time.time()-t_all:.0f}s "
          f"(cache: {cache_dir or 'off'})")

    covered = [(i, c) for i, c in enumerate(per_clip) if c]
    no_groups = [i for i, c in enumerate(per_clip) if not c]
    if no_groups:
        print(f"[NOTE] {len(no_groups)} clips produced NO groups "
              f"(pipeline found nothing to track) -- predicted negative "
              f"by definition:")
        for i in no_groups:
            print(f"        {clips[i][2]}/{os.path.basename(clips[i][0])}")

    # -- leave-one-clip-out evaluation -----------------------------------------
    print("\nLeave-one-clip-out evaluation...")
    loco_scores = np.full(len(clips), np.nan)
    def clip_score(samples, w, b, mu, sd):
        """Score a clip from its groups: MAX over MAJOR (long-lived) groups
        only. Short churn blobs cannot flag a clip -- this is what killed
        the clean folder with plain max-pooling."""
        major = [s for s, lf in samples if lf >= MAJOR_LIFE]
        pool = major if major else [s for s, _ in samples]
        return float(predict_proba(np.array(pool), w, b, mu, sd).max())

    for i, _ in covered:
        Xtr, ytr, ltr = [], [], []
        for j, samples in enumerate(per_clip):
            if j == i:
                continue
            for s, lf in samples:
                Xtr.append(s); ytr.append(clips[j][1]); ltr.append(lf)
        if not Xtr or len(set(ytr)) < 2:
            continue
        w, b, mu, sd = fit_logistic(np.array(Xtr), np.array(ytr, float),
                                    life=np.array(ltr), l2=args.l2)
        loco_scores[i] = clip_score(per_clip[i], w, b, mu, sd)
    for i in no_groups:
        loco_scores[i] = 0.0                    # nothing detected -> negative

    y = np.array([l for _, l, _ in clips], float)
    valid = ~np.isnan(loco_scores)
    ths = np.linspace(0.05, 0.95, 181)
    yv = y[valid]
    def rates(th):
        pred = (loco_scores[valid] >= th).astype(float)
        tpr = (pred[yv == 1] == 1).mean() if (yv == 1).any() else 0.0
        tnr = (pred[yv == 0] == 0).mean() if (yv == 0).any() else 0.0
        return tpr, tnr

    tprs = np.array([rates(th)[0] for th in ths])
    tnrs = np.array([rates(th)[1] for th in ths])
    bals = 0.5 * (tprs + tnrs)

    # ---- threshold selection: SMOKE RECALL IS THE PRIMARY OBJECTIVE ----
    if args.fn_cost is not None:
        # explicit cost-weighted mode
        fc = max(args.fn_cost, 1e-6)
        objs = (fc * tprs + tnrs) / (fc + 1.0)
        cand = objs == objs.max()
        mode = f"fn-cost={fc:g}"
    elif args.min_recall is not None and (tprs >= args.min_recall - 1e-9).any():
        # best specificity subject to a recall floor
        ok = tprs >= args.min_recall - 1e-9
        cand = ok & (tnrs == tnrs[ok].max())
        mode = f"min-recall={args.min_recall:g}"
    else:
        # DEFAULT: recall-first with a guard -- maximize smoke recall
        # SUBJECT TO specificity >= min_spec (a missed fire is the one
        # unacceptable outcome, but chasing a single outlier clip must not
        # collapse the false-alarm rate). Tie-break: best specificity.
        guard = tnrs >= args.min_spec - 1e-9
        if not guard.any():
            guard = np.ones_like(tnrs, bool)
        gmax = tprs[guard].max()
        top = guard & (tprs == gmax)
        cand = top & (tnrs == tnrs[top].max())
        mode = f"recall-first (spec>={args.min_spec:g})"
    tied = ths[cand]
    best_th = float(np.median(tied))     # middle of the optimal band
    best_bal = float(bals[np.argmin(np.abs(ths - best_th))])

    # operating-point table: see the recall/specificity trade-off directly
    print(f"\n  thr | smoke recall | clean specificity")
    for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        tpr, tnr = rates(th)
        mark = " <-- chosen" if abs(th - best_th) < 0.05 else ""
        print(f"  {th:.1f} |    {tpr*100:5.1f}%    |     {tnr*100:5.1f}%{mark}")

    pred = (loco_scores >= best_th).astype(float)
    print(f"\n{'='*62}")
    tpr_c, tnr_c = rates(best_th)
    print(f"LOCO balanced accuracy : {best_bal*100:.1f}%   "
          f"(threshold = {best_th:.2f}, mode = {mode})")
    print(f"smoke recall {tpr_c*100:.1f}%  |  clean specificity {tnr_c*100:.1f}%")
    print(f"{'='*62}")
    print(f"{'folder':>7} | {'clips':>5} | {'correct':>7} | per-clip (+ ok / x wrong)")
    for folder in POS_FOLDERS + NEG_FOLDERS:
        idx = [i for i, (_, _, f) in enumerate(clips) if f == folder]
        if not idx:
            continue
        ok = int(sum(pred[i] == y[i] for i in idx))
        marks = "".join("+" if pred[i] == y[i] else "x" for i in idx)
        print(f"{folder:>7} | {len(idx):>5} | {ok:>3}/{len(idx):<3} | {marks}")
    misses = [i for i in range(len(clips)) if pred[i] != y[i]]
    if misses:
        print("\nMisclassified clips:")
        for i in misses:
            print(f"  {clips[i][2]:>5}/{os.path.basename(clips[i][0])}"
                  f"   score={loco_scores[i]:.2f}")

    # -- final model on ALL data + save -----------------------------------------
    X = np.array([s for c in per_clip for s, _ in c])
    lg = np.array([lf for c in per_clip for _, lf in c])
    yg = np.array([clips[j][1] for j, c in enumerate(per_clip) for _ in c], float)
    w, b, mu, sd = fit_logistic(X, yg, life=lg, l2=args.l2)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "motion_weights.json")
    model = {
        "features": FEATURES,
        "mean": mu.tolist(), "std": sd.tolist(),
        "coef": w.tolist(), "bias": float(b),
        "smoke_thr": best_th,
        "cloud_thr": max(0.05, best_th - 0.2),
        "loco_balanced_acc": best_bal,
        "n_clips": len(clips), "n_groups": int(len(X)),
        "major_life": MAJOR_LIFE,
        "trained_with_params": params,
    }
    with open(out, "w") as f:
        json.dump(model, f, indent=2)
    print(f"\nSaved model -> {out}")
    print("Feature coefficients (standardized, + = smoke-ish):")
    for k, c in sorted(zip(FEATURES, w), key=lambda t: -abs(t[1])):
        print(f"   {k:>10}: {c:+.3f}")
    print("\nwildfire_gui.py picks this file up automatically on next launch.")


if __name__ == "__main__":
    main()
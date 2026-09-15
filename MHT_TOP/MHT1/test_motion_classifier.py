"""
test_motion_classifier.py
===============================================================================
Evaluate a TRAINED motion model (motion_weights.json) on held-out videos.

Two folder layouts are supported, detected automatically:

  LABELED  test_videos/{fire,smoke,both,clean}/*.mp4
           -> full metrics: recall, specificity, balanced accuracy,
              per-folder breakdown, misclassified list.

  FLAT     test_videos/*.mp4  (no label subfolders)
           -> per-clip score + verdict table only.

CONSISTENCY GUARANTEE: the pipeline runs with the EXACT params stored in the
model json (`trained_with_params`) and the decision uses the trained
threshold + major-group rule -- identical to how the model was evaluated in
training. Override params only if you know why (--params).

Usage:
    python test_motion_classifier.py --folder ..\\test_videos
    python test_motion_classifier.py --folder ..\\test_videos --model motion_weights.json
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
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


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def run_clip(path: str, params: dict, model: dict, max_frames: int):
    """Run the full pipeline on one clip. Returns
    (clip_score, n_groups, n_major, per_group [(score, life)])."""
    feats = model["features"]
    mu = np.asarray(model["mean"]); sd = np.asarray(model["std"]) + 1e-9
    coef = np.asarray(model["coef"]); bias = float(model["bias"])
    major_life = int(model.get("major_life", 25))

    pipe = WildfirePipeline(params, max_proc_w=MAX_PROC_W)
    trk = MHTTracker()
    clf = GroupMotionClassifier()
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
        for d in dets:
            _, d.iso = pipe.texture_stats(gray_u8, (d.x, d.y, d.w, d.h), params)
        trk.update(dets)
        clf.update(trk.active_tracks())

        for g in clf.groups:
            if g.misses == 0 and g.signals:
                vec = np.array([float(g.signals.get(k, 0.5)) for k in feats])
                acc.setdefault(g.id, []).append(vec)
    cap.release()

    per_group = []
    for vecs in acc.values():
        v = np.mean(vecs, axis=0)
        x = (v - mu) / sd
        per_group.append((float(sigmoid(x @ coef + bias)), len(vecs), v))

    majors = [s for s, lf, _ in per_group if lf >= major_life]
    pool = majors if majors else [s for s, _, _ in per_group]
    clip_score = max(pool) if pool else 0.0
    return clip_score, len(per_group), len(majors), per_group


def inspect_clip(path, params, model, max_frames, thr):
    """Per-group breakdown: score + ranked feature contributions."""
    feats = model["features"]
    mu = np.asarray(model["mean"]); sd = np.asarray(model["std"]) + 1e-9
    coef = np.asarray(model["coef"])
    major_life = int(model.get("major_life", 25))

    score, ng, nm, per_group = run_clip(path, params, model, max_frames)
    print(f"\n{os.path.basename(path)}")
    print(f"clip score = {score:.2f}  ({'SMOKE' if score >= thr else 'clean'} "
          f"@ thr {thr:.2f})   groups={ng} major={nm}\n")

    per_group.sort(key=lambda t: -t[0])
    for gi, (p, life, v) in enumerate(per_group):
        major = "MAJOR" if life >= major_life else "minor"
        print(f"group #{gi+1}  score={p:.2f}  life={life}f  [{major}]")
        z = (v - mu) / sd
        contrib = coef * z
        order = np.argsort(-np.abs(contrib))
        parts = []
        for j in order[:6]:
            parts.append(f"{feats[j]}={v[j]:.2f} ({contrib[j]:+.2f})")
        print("    drivers: " + ",  ".join(parts))
    print("\n(+) pushes toward SMOKE, (-) toward clean. A clean clip that "
          "alarms will show its lie here -- usually elong/vy/gabor_iso.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="test videos folder")
    ap.add_argument("--model", default=None,
                    help="model json (default: motion_weights.json next to script)")
    ap.add_argument("--max-frames", type=int, default=450)
    ap.add_argument("--params", default=None,
                    help="OVERRIDE pipeline params json (default: the params "
                         "stored in the model -- recommended)")
    ap.add_argument("--thr", type=float, default=None,
                    help="override the decision threshold (default: trained)")
    ap.add_argument("--inspect", default=None, metavar="VIDEO",
                    help="dissect ONE clip: per-group scores and which "
                         "features drove them (coef x standardized value)")
    args = ap.parse_args()

    mpath = args.model or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "motion_weights.json")
    with open(mpath) as f:
        model = json.load(f)
    params = dict(model.get("trained_with_params", {}))
    if not params:
        sys.exit("model json has no trained_with_params; pass --params")
    if args.params:
        with open(args.params) as f:
            params.update(json.load(f))
        print("[WARN] overriding trained params -- results may not reflect "
              "the trained operating point")
    thr = args.thr if args.thr is not None else float(model.get("smoke_thr", 0.5))

    if args.inspect:
        inspect_clip(args.inspect, params, model, args.max_frames, thr)
        return

    print(f"Model : {mpath}")
    print(f"        {model.get('n_clips','?')} training clips | "
          f"LOCO {model.get('loco_balanced_acc',0)*100:.1f}% | "
          f"threshold {thr:.2f} | major_life {model.get('major_life',25)}\n")

    # ── gather clips (labeled layout or flat) ────────────────────────────────
    labeled = any(os.path.isdir(os.path.join(args.folder, d))
                  for d in POS_FOLDERS + NEG_FOLDERS)
    clips: list[tuple[str, int | None, str]] = []
    if labeled:
        for folder, lab in [(f, 1) for f in POS_FOLDERS] + \
                           [(f, 0) for f in NEG_FOLDERS]:
            d = os.path.join(args.folder, folder)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(VIDEO_EXTS):
                    clips.append((os.path.join(d, f), lab, folder))
    else:
        for f in sorted(os.listdir(args.folder)):
            if f.lower().endswith(VIDEO_EXTS):
                clips.append((os.path.join(args.folder, f), None, "-"))
    if not clips:
        sys.exit(f"No videos found under {args.folder}")
    print(f"Found {len(clips)} test clips ({'labeled' if labeled else 'flat'})\n")

    # ── run ─────────────────────────────────────────────────────────────────
    header = f"{'':>3} {'folder':>6} | {'clip':<40} | {'grp':>3} {'maj':>3} | " \
             f"{'score':>5} | verdict"
    print(header); print("-" * len(header))
    scores = []
    t_all = time.time()
    for i, (path, lab, folder) in enumerate(clips):
        t0 = time.time()
        score, ng, nm, _ = run_clip(path, params, model, args.max_frames)
        scores.append(score)
        verdict = "SMOKE" if score >= thr else "clean"
        mark = ""
        if lab is not None:
            ok = (score >= thr) == bool(lab)
            mark = "  ok" if ok else "  << WRONG"
        print(f"{i+1:>3} {folder:>6} | {os.path.basename(path)[:40]:<40} | "
              f"{ng:>3} {nm:>3} | {score:>5.2f} | {verdict}{mark}"
              f"   ({time.time()-t0:.0f}s)")
    print(f"\nDone in {time.time()-t_all:.0f}s")

    # ── metrics (labeled mode) ───────────────────────────────────────────────
    if labeled:
        y = np.array([l for _, l, _ in clips], float)
        s = np.array(scores)
        pred = (s >= thr).astype(float)
        tpr = (pred[y == 1] == 1).mean() if (y == 1).any() else float("nan")
        tnr = (pred[y == 0] == 0).mean() if (y == 0).any() else float("nan")
        print(f"\n{'='*56}")
        print(f"TEST  smoke recall      : {tpr*100:.1f}%  "
              f"({int(pred[y==1].sum())}/{int((y==1).sum())})")
        print(f"TEST  clean specificity : {tnr*100:.1f}%  "
              f"({int((pred[y==0]==0).sum())}/{int((y==0).sum())})")
        print(f"TEST  balanced accuracy : {(tpr+tnr)/2*100:.1f}%")
        print(f"{'='*56}")
        print("\n  thr | smoke recall | clean specificity   (on TEST scores)")
        for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            p = (s >= th).astype(float)
            r = (p[y == 1] == 1).mean() if (y == 1).any() else 0
            sp = (p[y == 0] == 0).mean() if (y == 0).any() else 0
            mark = " <-- operating point" if abs(th - thr) < 0.05 else ""
            print(f"  {th:.1f} |    {r*100:5.1f}%    |     {sp*100:5.1f}%{mark}")


if __name__ == "__main__":
    main()
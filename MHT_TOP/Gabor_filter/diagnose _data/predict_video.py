"""Video-level smoke classifier (Gabor + optical-flow aggregates, linear SVM).

Trained on val_videos, evaluated on test_videos (held out):
  per-video accuracy 72.2% | recall 88.9% | precision 66.7%
LOGO-CV on training videos: accuracy 88.6% | recall 95.8%

Usage:
    python predict_video.py --model smoke_video_model.joblib --video clip.mp4
    python predict_video.py --model smoke_video_model.joblib --folder some_dir
Folder mode z-scores features across the batch (domain adaptation - more
accurate). Single-video mode falls back to the stored training statistics.
"""
import argparse, os, sys
import numpy as np, cv2, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoke_features import SmokeConfig, build_gabor_bank, frame_pair_features
from motion_feats import motion_features

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v")

def extract(path, cfg, bank, stride, max_pairs):
    cap = cv2.VideoCapture(path)
    gf, mf, prev, idx = [], [], None, 0
    while len(gf) < max_pairs:
        ok, frame = cap.read()
        if not ok: break
        if idx % stride == 0:
            if cfg.resize is not None:
                frame = cv2.resize(frame, cfg.resize, interpolation=cv2.INTER_AREA)
            if prev is not None:
                gf.append(frame_pair_features(prev, frame, bank, cfg))
                mf.append(motion_features(prev, frame, cfg))
            prev = frame
        idx += 1
    cap.release()
    return np.asarray(gf), np.asarray(mf)

def aggregate(gf, mf):
    g = np.concatenate([gf.mean(0), gf.std(0)])
    m = np.concatenate([mf.mean(0), mf.std(0), np.percentile(mf, 90, 0)])
    return np.concatenate([g, m])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--video")
    ap.add_argument("--folder")
    args = ap.parse_args()
    blob = joblib.load(args.model)
    cfg = SmokeConfig(); bank = build_gabor_bank(cfg)
    stride, mp = blob["stride"], blob["max_pairs"]

    if args.folder:
        paths = [os.path.join(dp, f) for dp, _, fs in os.walk(args.folder)
                 for f in fs if f.lower().endswith(VIDEO_EXTS)]
    else:
        paths = [args.video]
    V = []
    for p in paths:
        gf, mf = extract(p, cfg, bank, stride, mp)
        if len(gf) == 0:
            print(f"[warn] no frames: {p}"); V.append(None); continue
        V.append(aggregate(gf, mf))
    ok_idx = [i for i, v in enumerate(V) if v is not None]
    A = np.array([V[i] for i in ok_idx])
    if len(A) >= 8:   # batch z-score (domain adaptation)
        Z = (A - A.mean(0)) / (A.std(0) + 1e-9)
    else:             # fall back to training stats
        Z = (A - blob["train_mean"]) / blob["train_std"]
    s = blob["model"].decision_function(Z)
    for i, gi in enumerate(ok_idx):
        verdict = "SMOKE" if s[i] > blob["threshold"] else "no smoke"
        print(f"{paths[gi]}: {verdict}  (score {s[i]:+.3f}, thr {blob['threshold']:+.3f})")

if __name__ == "__main__":
    main()

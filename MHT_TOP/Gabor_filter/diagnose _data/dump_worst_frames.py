r"""
dump_worst_frames.py
--------------------
For the clean videos the model gets wrong, find the frames it's MOST confident
are smoke (highest decision_function score among the true-clean frames) and
dump:
  - the raw downscaled frame
  - a stage-panel (mask / diff / 5 Gabor / wavelet) showing what's tripping it

Mirrors smoke_detection.extract_video_features exactly: same stride, same
resize, same adjacent-pair differencing, same scoring as `eval`.

Usage (PowerShell, from the Gabor_filter folder):
  python dump_worst_frames.py --model .\smoke_svm_at.joblib ^
      --videos ..\test_videos\clean\10397425-hd_3840_2160_30fps.mp4 ^
               ..\test_videos\clean\14158552_3840_2160_30fps.mp4 ^
               ..\test_videos\clean\164360-830461265_medium.mp4 ^
      --label 0 --topk 5 --out worst_frames

  # or just point at a folder of clips:
  python dump_worst_frames.py --model .\smoke_svm_at.joblib --dir ..\test_videos\clean --label 0
"""
from __future__ import annotations
import argparse, os, glob, sys
import numpy as np
import cv2

from smoke_features import (
    SmokeConfig, build_gabor_bank, frame_pair_features, feature_names,
)
from smoke_detection import load_model


def _norm(x):
    x = x.astype(np.float64)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x, dtype=np.uint8)
    return ((x - lo) / (hi - lo) * 255).astype(np.uint8)


def make_panel(frame_bgr, dbg, score, thr, title):
    """Compose raw | mask*cur | diff | 5 gabor | wavelet-ish into one image."""
    h, w = frame_bgr.shape[:2]
    tiles = []

    def label(img_gray_or_bgr, txt):
        img = img_gray_or_bgr
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img.copy()
        cv2.rectangle(img, (0, 0), (w, 18), (0, 0, 0), -1)
        cv2.putText(img, txt, (3, 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    tiles.append(label(frame_bgr, "raw"))
    tiles.append(label(dbg["masked_cur"], "masked_cur"))
    tiles.append(label(_norm(dbg["diff"]), "temporal diff"))
    for i, r in enumerate(dbg["responses"]):
        tiles.append(label(_norm(r), f"gabor o{i}"))

    # pad to a multiple of 4 per row
    while len(tiles) % 4 != 0:
        tiles.append(np.zeros((h, w, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    panel = np.vstack(rows)

    # header bar
    verdict = "PRED=SMOKE (FALSE POS)" if score > thr else "pred=clean"
    bar = np.zeros((34, panel.shape[1], 3), np.uint8)
    cv2.putText(bar, f"{title}  score={score:+.3f}  thr={thr:+.2f}  -> {verdict}",
                (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def score_video(path, pipe, cfg, bank, stride, thr):
    """Return list of (orig_frame_idx, score, frame_bgr, debug) for every pair."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[warn] cannot open {path}", file=sys.stderr)
        return []
    out = []
    prev = None
    idx = 0
    pair_src_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1
            continue
        idx += 1
        if cfg.resize is not None:
            frame = cv2.resize(frame, cfg.resize, interpolation=cv2.INTER_AREA)
        if prev is not None:
            vec, dbg = frame_pair_features(prev, frame, bank, cfg, return_debug=True)
            score = float(pipe.decision_function(vec.reshape(1, -1))[0])
            out.append((idx, score, frame.copy(), dbg))
        prev = frame
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--videos", nargs="*", default=[])
    ap.add_argument("--dir", default=None, help="folder of clips (all scored)")
    ap.add_argument("--label", type=int, default=0,
                    help="true label of these clips (0=clean). topk = worst errors for this label")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--stride", type=int, default=None,
                    help="default: model's training stride")
    ap.add_argument("--threshold", type=float, default=None,
                    help="default: model's stored threshold")
    ap.add_argument("--out", default="worst_frames")
    args = ap.parse_args()

    pipe, cfg, meta = load_model(args.model)
    stride = args.stride if args.stride is not None else int(meta.get("stride", 1))
    thr = args.threshold if args.threshold is not None else float(meta.get("threshold", 0.0))
    bank = build_gabor_bank(cfg)

    vids = list(args.videos)
    if args.dir:
        for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
            vids += glob.glob(os.path.join(args.dir, ext))
    if not vids:
        print("no videos given (use --videos or --dir)", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    print(f"[cfg] stride={stride}  threshold={thr:+.2f}  resize={cfg.resize}  "
          f"gabor={cfg.gabor}  topk={args.topk}  label={args.label}")

    for path in vids:
        scored = score_video(path, pipe, cfg, bank, stride, thr)
        if not scored:
            continue
        # "worst" = most wrong. label 0 -> highest score; label 1 -> lowest score.
        wrong = (lambda s: s > thr) if args.label == 0 else (lambda s: s <= thr)
        scored.sort(key=lambda t: t[1], reverse=(args.label == 0))
        n_err = sum(1 for _, s, _, _ in scored if wrong(s))
        base = os.path.splitext(os.path.basename(path))[0]
        print(f"\n{base}: {len(scored)} pairs, {n_err} misclassified "
              f"({n_err/len(scored):.0%})")

        for rank, (fidx, score, frame, dbg) in enumerate(scored[:args.topk]):
            tag = "ERR" if wrong(score) else "ok"
            panel = make_panel(frame, dbg, score, thr, f"{base} f#{fidx}")
            fn = os.path.join(args.out, f"{base}__rank{rank:02d}_f{fidx:05d}_{tag}.png")
            cv2.imwrite(fn, panel)
            print(f"  rank {rank}: frame {fidx:5d}  score={score:+.3f}  [{tag}] -> {fn}")

    print(f"\ndone. panels in ./{args.out}/")


if __name__ == "__main__":
    main()
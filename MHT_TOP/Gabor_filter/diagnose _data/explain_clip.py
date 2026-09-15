r"""
explain_clip.py
===============================================================================
Watch WHERE the appearance model sees "smoke" on a clip, frame by frame, and
understand why clean clips get flagged (cloud/fog) and smoke clips get missed.

For each sampled frame pair it renders a 2x2 panel:
    [ raw + verdict banner ]   [ HSV smoke-colour mask overlaid on frame ]
    [ temporal diff        ]   [ Gabor response (orientation 0) ]
and writes them to an annotated .mp4 you can scrub. The banner shows the SVM
decision score and whether it fired (PRED=SMOKE) vs the clip's true label, so
false positives (cloud called smoke) and false negatives (smoke missed) are
obvious as you scrub.

The green mask overlay = exactly the pixels the model treats as smoke-coloured.
On a cloud/fog clip you'll see the mask land on the cloud; on a real smoke clip
you'll see whether it lands on the plume or misses it.

Usage:
    python explain_clip.py --video ..\test_videos\clean\93857-642181987_medium.mp4 --model .\smoke_svm_at.joblib --out explain_93857.mp4
    python explain_clip.py --video ..\test_videos\smoke\99182-653447840.mp4 --model .\smoke_svm_at.joblib --label 1 --out explain_99182.mp4

--label is the clip's true class (0=clean, 1=smoke); it only controls the
"CORRECT/WRONG" tag in the banner. --every N samples fewer frames.
"""
from __future__ import annotations
import argparse
import cv2
import numpy as np

from smoke_features import SmokeConfig, build_gabor_bank, frame_pair_features
from smoke_detection import load_model

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def norm_u8(x):
    x = x.astype(np.float64)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros(x.shape, np.uint8)
    return ((x - lo) / (hi - lo) * 255).astype(np.uint8)


def label_tile(img, txt, color=(255, 255, 255)):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(img, txt, (4, 15), _FONT, 0.45, color, 1, cv2.LINE_AA)
    return img


def build_panel(frame_bgr, dbg, score, thr, true_label):
    h, w = frame_bgr.shape[:2]
    fired = score > thr

    # green overlay = smoke-coloured mask pixels (from masked_cur non-black)
    masked = dbg["masked_cur"]
    maskbin = (cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    overlay = frame_bgr.copy()
    overlay[maskbin > 0] = (0, 200, 0)
    blended = cv2.addWeighted(frame_bgr, 0.55, overlay, 0.45, 0)

    diff_u8 = norm_u8(dbg["diff"])
    gabor0 = norm_u8(dbg["responses"][0])

    tiles = [
        label_tile(frame_bgr, "raw"),
        label_tile(blended, "smoke-colour mask (green)", (0, 255, 0)),
        label_tile(diff_u8, "temporal diff"),
        label_tile(gabor0, "gabor o0"),
    ]
    top = np.hstack([tiles[0], tiles[1]])
    bot = np.hstack([tiles[2], tiles[3]])
    grid = np.vstack([top, bot])

    # verdict banner
    if true_label is None:
        tag = ""
    else:
        correct = (int(fired) == int(true_label))
        tag = "  [CORRECT]" if correct else "  [WRONG]"
    verdict = "PRED = SMOKE" if fired else "pred = clean"
    vcolor = (60, 170, 240) if fired else (200, 200, 200)
    bar = np.zeros((30, grid.shape[1], 3), np.uint8)
    cv2.putText(bar, f"score={score:+.3f}  thr={thr:+.2f}  ->  {verdict}{tag}",
                (6, 21), _FONT, 0.55, vcolor, 1, cv2.LINE_AA)
    return np.vstack([bar, grid])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", type=int, default=None, help="true class: 0 clean, 1 smoke")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--out", default="explain.mp4")
    args = ap.parse_args()

    pipe, cfg, meta = load_model(args.model)
    stride = int(meta.get("stride", 12))
    thr = float(meta.get("threshold", 0.0))
    bank = build_gabor_bank(cfg)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"cannot open {args.video}"); return

    prev = None
    idx = 0
    writer = None
    n_frames = fired_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if (idx - 1) % (stride * args.every) != 0:
            continue
        if cfg.resize is not None:
            frame = cv2.resize(frame, cfg.resize, interpolation=cv2.INTER_AREA)
        if prev is not None:
            vec, dbg = frame_pair_features(prev, frame, bank, cfg, return_debug=True)
            score = float(pipe.decision_function(vec.reshape(1, -1))[0])
            panel = build_panel(frame, dbg, score, thr, args.label)
            if writer is None:
                hh, ww = panel.shape[:2]
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                         6, (ww, hh))
            writer.write(panel)
            n_frames += 1
            fired_frames += int(score > thr)
        prev = frame
    cap.release()
    if writer:
        writer.release()

    if n_frames:
        print(f"wrote {args.out}: {n_frames} panels, "
              f"{fired_frames} flagged SMOKE ({fired_frames/n_frames:.0%})")
        if args.label is not None:
            kind = "false positives" if args.label == 0 else "true positives"
            print(f"  clip true label = {args.label} -> "
                  f"{fired_frames}/{n_frames} fired ({kind if args.label==0 else 'recall'})")
    else:
        print("no frame pairs produced (clip too short for this stride?)")


if __name__ == "__main__":
    main()
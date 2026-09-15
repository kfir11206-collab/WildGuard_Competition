"""
smoke_viz.py
------------
Visualise every stage of the Appana et al. (2017) smoke pipeline on a frame
pair: original / HSV mask / masked / temporal diff / the 5 Gabor-orientation
responses / wavelet-energy map, with the SVM verdict banner on top.

Headless (no Qt) — composes one labelled montage image per frame, so it writes
a video or a single PNG and runs anywhere (incl. the Orin). The composed image
comes from make_debug_panel(), which drops straight into a Qt panel too:

    panel, vec, label = make_debug_panel(prev, cur, bank, cfg, model=pipe)

CLI:
    python smoke_viz.py montage --video clip.mp4 --model smoke_svm.joblib --out viz.mp4
    python smoke_viz.py frame   --video clip.mp4 --model smoke_svm.joblib --at 4.0 --out panel.png
"""

from __future__ import annotations

import argparse
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pywt

from smoke_features import (
    SmokeConfig, build_gabor_bank, frame_pair_features, hsv_smoke_mask,
)

# palette lifted from wildfire_gui.py so the two tools look like siblings
P = {
    "bg":          (0x17, 0x11, 0x0d),   # #0d1117 (BGR)
    "panel_bg":    (0x22, 0x1b, 0x16),   # #161b22
    "border":      (0x3d, 0x36, 0x30),   # #30363d
    "fire":        (0x3e, 0x88, 0xf0),   # #f0883e
    "smoke":       (0x9e, 0x94, 0x8b),   # #8b949e
    "go":          (0x50, 0xb9, 0x3f),   # #3fb950
    "red":         (0x3a, 0x49, 0xf8),   # smoke-positive alert
    "text":        (0xf3, 0xed, 0xe6),   # #e6edf3
    "muted":       (0x8b, 0x94, 0x8b),
}

TILE_W, TILE_H = 240, 180
BAR_H = 22          # per-tile title bar
GAP = 6
BANNER_H = 58
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------- #
def _norm8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-9:
        return np.zeros(x.shape, np.uint8)
    return ((x - mn) / (mx - mn) * 255.0).astype(np.uint8)


def _tile(img_bgr: np.ndarray, title: str, accent) -> np.ndarray:
    """One labelled panel: resized image with a coloured title strip."""
    body = cv2.resize(img_bgr, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    tile = np.full((TILE_H + BAR_H, TILE_W, 3), P["panel_bg"], np.uint8)
    tile[BAR_H:, :] = body
    cv2.rectangle(tile, (0, 0), (TILE_W - 1, BAR_H - 1), P["panel_bg"], -1)
    cv2.line(tile, (0, BAR_H - 1), (TILE_W, BAR_H - 1), P["border"], 1)
    cv2.rectangle(tile, (0, 0), (3, BAR_H - 1), accent, -1)          # accent tab
    cv2.putText(tile, title, (9, 15), _FONT, 0.42, P["text"], 1, cv2.LINE_AA)
    cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H + BAR_H - 1), P["border"], 1)
    return tile


def _gray2bgr(g: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _cmap(g: np.ndarray, cmap=cv2.COLORMAP_INFERNO) -> np.ndarray:
    return cv2.applyColorMap(_norm8(g), cmap)


def _wavelet_energy_map(diff: np.ndarray, wavelet: str) -> np.ndarray:
    cA, (cH, cV, cD) = pywt.dwt2(diff.astype(np.float64), wavelet)
    e = cH ** 2 + cV ** 2 + cD ** 2
    return cv2.resize(e, (diff.shape[1], diff.shape[0]), interpolation=cv2.INTER_NEAREST)


def _grid(tiles: List[np.ndarray], cols: int) -> np.ndarray:
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        while len(row) < cols:                                   # pad short row
            row.append(np.full_like(tiles[0], P["bg"]))
        strip = [row[0]]
        for t in row[1:]:
            strip.append(np.full((t.shape[0], GAP, 3), P["bg"], np.uint8))
            strip.append(t)
        rows.append(np.hstack(strip))
    out = [rows[0]]
    for rr in rows[1:]:
        out.append(np.full((GAP, rows[0].shape[1], 3), P["bg"], np.uint8))
        out.append(rr)
    return np.vstack(out)


def _banner(width: int, label: Optional[int], vec: np.ndarray) -> np.ndarray:
    b = np.full((BANNER_H, width, 3), P["panel_bg"], np.uint8)
    if label == 1:
        txt, accent = "SMOKE", P["red"]
    elif label == 0:
        txt, accent = "CLEAR", P["go"]
    else:
        txt, accent = "—", P["muted"]
    cv2.rectangle(b, (0, 0), (10, BANNER_H), accent, -1)
    cv2.putText(b, txt, (24, 38), _FONT, 1.0, accent, 2, cv2.LINE_AA)
    # quick numeric read-out from the 21-d vector
    gabor_mean = float(np.mean(vec[2:20:4]))   # the 5 gabor 'mean' stats
    we = float(vec[-1])
    info = f"wavelet E={we:8.1f}    gabor mean={gabor_mean:6.2f}"
    cv2.putText(b, info, (180, 24), _FONT, 0.5, P["text"], 1, cv2.LINE_AA)
    cv2.putText(b, "Appana 2017  |  HSV -> diff -> Gabor x5 -> wavelet -> RBF-SVM",
                (180, 46), _FONT, 0.42, P["muted"], 1, cv2.LINE_AA)
    cv2.line(b, (0, BANNER_H - 1), (width, BANNER_H - 1), P["border"], 1)
    return b


# --------------------------------------------------------------------------- #
def make_debug_panel(
    prev_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    bank,
    cfg: SmokeConfig,
    model=None,
) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
    """Compose the multi-panel montage for one frame pair."""
    vec, dbg = frame_pair_features(prev_bgr, cur_bgr, bank, cfg, return_debug=True)
    mask = hsv_smoke_mask(cur_bgr, cfg)

    label = None
    if model is not None:
        label = int(model.predict(vec.reshape(1, -1))[0])

    denom = max(cfg.n_orient - 1, 1)
    tiles = [
        _tile(cur_bgr, "01  original", P["smoke"]),
        _tile(_gray2bgr(mask), "02  HSV smoke mask", P["smoke"]),
        _tile(dbg["masked_cur"], "03  masked", P["smoke"]),
        _tile(_cmap(dbg["diff"], cv2.COLORMAP_INFERNO), "04  temporal diff", P["fire"]),
        _tile(_cmap(_wavelet_energy_map(dbg["diff"], cfg.wavelet), cv2.COLORMAP_TURBO),
              "05  wavelet energy", P["go"]),
    ]
    for i, resp in enumerate(dbg["responses"]):
        ang = ["0", "pi/4", "pi/2", "3pi/4", "pi"][i] if i < 5 else f"{i}"
        tiles.append(_tile(_cmap(resp, cv2.COLORMAP_MAGMA),
                           f"Gabor  theta={ang}", P["fire"]))

    grid = _grid(tiles, cols=5)
    panel = np.vstack([_banner(grid.shape[1], label, vec), grid])
    return panel, vec, label


# --------------------------------------------------------------------------- #
def _load(model_path):
    import joblib
    blob = joblib.load(model_path)
    return blob["pipe"], SmokeConfig.from_dict(blob["cfg"]), blob.get("meta", {})


def _read_resized(frame, cfg):
    return frame if cfg.resize is None else cv2.resize(
        frame, cfg.resize, interpolation=cv2.INTER_AREA)


def cmd_montage(args):
    if not args.out and not getattr(args, "show", False):
        args.out = "smoke_viz.mp4"          # no flags -> behave as before
    pipe, cfg, meta = _load(args.model)
    bank = build_gabor_bank(cfg)
    stride = args.stride if args.stride is not None else int(meta.get("stride", 1))
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = (cap.get(cv2.CAP_PROP_FPS) or 25.0) / stride

    writer = None
    show = getattr(args, "show", False)
    prev = None
    idx = wrote = 0
    quit_early = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            cur = _read_resized(frame, cfg)
            if prev is not None:
                panel, _, _ = make_debug_panel(prev, cur, bank, cfg, model=pipe)
                if args.out:
                    if writer is None:
                        h, w = panel.shape[:2]
                        writer = cv2.VideoWriter(
                            args.out, cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (w, h))
                    writer.write(panel)
                wrote += 1
                if show:
                    try:
                        cv2.imshow("smoke pipeline  -  q/esc to quit", panel)
                        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                            quit_early = True
                            break
                    except cv2.error:
                        print("[!] --show needs the GUI build of OpenCV "
                              "(pip uninstall opencv-python-headless ; pip install opencv-python)")
                        show = False
                if args.max_frames and wrote >= args.max_frames:
                    break
            prev = cur
        idx += 1
    cap.release()
    if show:
        cv2.destroyAllWindows()
    if writer:
        writer.release()
        print(f"[saved] {args.out}  ({wrote} panels @ {fps:.1f} fps)")
    elif quit_early:
        print(f"[stopped early]  ({wrote} panels shown)")


def cmd_frame(args):
    pipe, cfg, meta = _load(args.model)
    bank = build_gabor_bank(cfg)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    target = int(args.at * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(target - 1, 0))
    ok, f0 = cap.read()
    ok2, f1 = cap.read()
    if not (ok and ok2):
        raise SystemExit("could not read two frames at that timestamp")
    panel, vec, label = make_debug_panel(
        _read_resized(f0, cfg), _read_resized(f1, cfg), bank, cfg, model=pipe)
    cv2.imwrite(args.out, panel)
    print(f"[saved] {args.out}  verdict={'SMOKE' if label==1 else 'clear'}  "
          f"wavelet_E={vec[-1]:.1f}")


def main():
    ap = argparse.ArgumentParser(description="Visualise the smoke pipeline stages")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("montage", help="write a stage-panel video")
    pm.add_argument("--video", required=True)
    pm.add_argument("--model", required=True)
    pm.add_argument("--out", default=None, help="write the panel video (optional)")
    pm.add_argument("--show", action="store_true",
                    help="live preview window (needs GUI build of OpenCV)")
    pm.add_argument("--stride", type=int, default=None)
    pm.add_argument("--max-frames", type=int, default=None)
    pm.set_defaults(func=cmd_montage)

    pf = sub.add_parser("frame", help="write a single stage-panel PNG")
    pf.add_argument("--video", required=True)
    pf.add_argument("--model", required=True)
    pf.add_argument("--at", type=float, default=2.0, help="timestamp (seconds)")
    pf.add_argument("--out", default="smoke_panel.png")
    pf.set_defaults(func=cmd_frame)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
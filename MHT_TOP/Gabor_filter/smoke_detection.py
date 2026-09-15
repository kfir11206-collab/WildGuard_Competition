"""
smoke_detection.py
------------------
Dataset loading, SVM training/evaluation and single-video inference around the
21-D features in smoke_features.py (Appana et al., 2017).

Dataset layout (one sub-folder per class, videos inside):

    <root>/
      smoke/   *.mp4 ...   -> smoke present  (label 1)
      both/    *.mp4 ...   -> smoke present  (label 1)
      clean/   *.mp4 ...   -> no smoke       (label 0)
      fire/    *.mp4 ...   -> see LABEL note below (label 0 by default)

CLI:
    python smoke_detection.py train  --data <train_root> --out model.joblib [--cv 5]
    python smoke_detection.py eval   --data <val_root>   --model model.joblib
    python smoke_detection.py predict --video clip.mp4   --model model.joblib [--out annotated.mp4]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from smoke_features import (
    SmokeConfig, FEATURE_DIM, build_gabor_bank, frame_pair_features, make_iir,
)


# --------------------------------------------------------------------------- #
# Feature-stage dispatch
# --------------------------------------------------------------------------- #
# cfg.feature_set selects which per-frame stage runs:
#   "v1" -> smoke_features.frame_pair_features      (paper, 21-D)
#   "v2" -> smoke_features_v2.frame_pair_features_v2 (revised, 28-D)
# The choice is stored in the model config, so eval/predict reproduce whatever
# the model was trained with. Models saved before this option default to "v1".
def feature_dim(cfg: SmokeConfig) -> int:
    if getattr(cfg, "feature_set", "v1") == "v2":
        from smoke_features_v2 import FEATURE_DIM_V2
        return FEATURE_DIM_V2
    return FEATURE_DIM


def make_bank(cfg: SmokeConfig):
    if getattr(cfg, "feature_set", "v1") == "v2":
        from smoke_features_v2 import build_bank_v2
        return build_bank_v2(cfg)
    return build_gabor_bank(cfg)


def pair_features(prev_bgr, cur_bgr, bank, cfg, iir=None):
    if getattr(cfg, "feature_set", "v1") == "v2":
        from smoke_features_v2 import frame_pair_features_v2
        return frame_pair_features_v2(prev_bgr, cur_bgr, bank, cfg, iir=iir)
    return frame_pair_features(prev_bgr, cur_bgr, bank, cfg, iir=iir)

try:
    from tqdm import tqdm
except Exception:  # tqdm optional
    def tqdm(it, **kw):  # type: ignore
        return it


# --------------------------------------------------------------------------- #
# Label mapping  --  EDIT HERE if your taxonomy differs
# --------------------------------------------------------------------------- #
SMOKE_LABELS = {"smoke", "both"}     # frames contain smoke      -> 1
NONSMOKE_LABELS = {"clean", "fire"}  # frames contain no smoke   -> 0
# NOTE: "fire" defaults to NON-smoke (flame-only). The paper builds a *smoke*
# detector, so flame without visible smoke is a negative. If your `fire` clips
# actually contain smoke, move "fire" from NONSMOKE_LABELS into SMOKE_LABELS.

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v")


def label_of(folder_name: str) -> Optional[int]:
    name = folder_name.lower()
    if name in SMOKE_LABELS:
        return 1
    if name in NONSMOKE_LABELS:
        return 0
    return None


# --------------------------------------------------------------------------- #
# Dataset discovery + feature extraction
# --------------------------------------------------------------------------- #
def list_videos(root: str) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    for entry in sorted(os.listdir(root)):
        sub = os.path.join(root, entry)
        if not os.path.isdir(sub):
            continue
        lbl = label_of(entry)
        if lbl is None:
            print(f"[skip] '{entry}' not in label map", file=sys.stderr)
            continue
        for ext in VIDEO_EXTS:
            for p in glob.glob(os.path.join(sub, f"*{ext}")):
                items.append((p, lbl))
    return sorted(items)


def extract_video_features(
    path: str,
    bank,
    cfg: SmokeConfig,
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> np.ndarray:
    """(n_pairs, D) features from one video, differencing adjacent kept frames.

    Skipped frames are grab()'d rather than read(), so they are never decoded --
    ~5x less decode work at stride 5, bit-identical output, and it also reads
    past damaged regions where read() aborts early.
    """
    dim = feature_dim(cfg)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[warn] cannot open {path}", file=sys.stderr)
        return np.empty((0, dim))

    feats: List[np.ndarray] = []
    prev = None
    idx = 0
    iir = make_iir(cfg)          # None unless cfg.diff_mode uses IIR
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
            feats.append(pair_features(prev, frame, bank, cfg, iir=iir))
            if max_frames is not None and len(feats) >= max_frames:
                break
        prev = frame
    cap.release()
    return np.asarray(feats, dtype=np.float64) if feats else np.empty((0, dim))


def build_dataset(
    root: str,
    cfg: SmokeConfig,
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X (N,21), y (N,), groups (N,) where group = source-video index."""
    bank = make_bank(cfg)
    videos = list_videos(root)
    if not videos:
        raise SystemExit(f"No labelled videos found under {root}")

    Xs, ys, gs = [], [], []
    for gid, (path, lbl) in enumerate(tqdm(videos, desc="videos")):
        fv = extract_video_features(path, bank, cfg, stride, max_frames)
        if len(fv) == 0:
            continue
        Xs.append(fv)
        ys.append(np.full(len(fv), lbl, dtype=np.int64))
        gs.append(np.full(len(fv), gid, dtype=np.int64))
        print(f"  {os.path.relpath(path, root)}: {len(fv)} samples (label {lbl})")

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    g = np.concatenate(gs)
    print(f"[data] {X.shape[0]} samples / {len(videos)} videos "
          f"| smoke={int((y == 1).sum())} non-smoke={int((y == 0).sum())}")
    return X, y, g


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def make_pipeline(sigma: float = 0.6, C: float = 1.0,
                  gamma: Optional[float] = None, scale: bool = True) -> Pipeline:
    # paper uses an RBF kernel with sigma = 0.6  ->  gamma = 1/(2*sigma^2)
    g = (1.0 / (2.0 * sigma ** 2)) if gamma is None else gamma
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("svc", SVC(kernel="rbf", C=C, gamma=g, class_weight="balanced")))
    return Pipeline(steps)


def save_model(path: str, pipe: Pipeline, cfg: SmokeConfig, meta: dict) -> None:
    joblib.dump({"pipe": pipe, "cfg": cfg.to_dict(), "meta": meta}, path)
    print(f"[saved] {path}")


def load_model(path: str) -> Tuple[Pipeline, SmokeConfig, dict]:
    blob = joblib.load(path)
    return blob["pipe"], SmokeConfig.from_dict(blob["cfg"]), blob.get("meta", {})


def _report(y_true, y_pred, title: str) -> None:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n== {title} ==")
    print(f"accuracy={acc:.4f}  precision={p:.4f}  recall={r:.4f}  f1={f1:.4f}")
    print("confusion [rows=true 0/1, cols=pred 0/1]:")
    print(cm)


# --------------------------------------------------------------------------- #
# Generalisation helpers
# --------------------------------------------------------------------------- #
def _group_stratified_split(X, y, groups, test_frac, seed=0):
    """Hold out whole VIDEOS for test, stratified by class. No frame leakage."""
    rng = np.random.default_rng(seed)
    gid_label = {g: int(y[groups == g][0]) for g in np.unique(groups)}
    test_gids = set()
    for cls in (0, 1):
        gids = np.array([g for g, l in gid_label.items() if l == cls])
        rng.shuffle(gids)
        n_test = max(1, int(round(len(gids) * test_frac))) if len(gids) > 1 else 0
        test_gids.update(gids[:n_test].tolist())
    te = np.isin(groups, list(test_gids))
    tr = ~te
    return (X[tr], y[tr], groups[tr]), (X[te], y[te], groups[te])


def _balance_classes(X, y, groups, seed=0):
    """Down-sample the majority class to match the minority (train set only)."""
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n = min(len(idx0), len(idx1))
    if n == 0:
        return X, y, groups
    keep = np.concatenate([rng.choice(idx0, n, replace=False),
                           rng.choice(idx1, n, replace=False)])
    keep.sort()
    return X[keep], y[keep], groups[keep]


def _n_groups(groups):
    return len(np.unique(groups))


def _grouped_cv_report(X, y, groups, args, n_splits):
    """StratifiedGroupKFold report (both classes guaranteed in every fold)."""
    n = min(n_splits, _n_groups(groups))
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=args.seed)
    accs, bals, ps, rs, fs = [], [], [], [], []
    for k, (tr, te) in enumerate(sgkf.split(X, y, groups), 1):
        pipe = make_pipeline(args.sigma, args.C, args.gamma, not args.no_scale)
        pipe.fit(X[tr], y[tr])
        yp = pipe.predict(X[te])
        a = accuracy_score(y[te], yp)
        b = balanced_accuracy_score(y[te], yp)
        p, r, f1, _ = precision_recall_fscore_support(
            y[te], yp, average="binary", pos_label=1, zero_division=0)
        accs.append(a); bals.append(b); ps.append(p); rs.append(r); fs.append(f1)
        print(f"  fold {k}: acc={a:.3f} bal_acc={b:.3f} prec={p:.3f} rec={r:.3f} f1={f1:.3f}")
    print(f"[{n}-fold stratified grouped CV] "
          f"acc={np.mean(accs):.3f}±{np.std(accs):.3f}  "
          f"bal_acc={np.mean(bals):.3f}  prec={np.mean(ps):.3f}  "
          f"rec={np.mean(rs):.3f}  f1={np.mean(fs):.3f}")


def _grid_search(X, y, groups, args):
    """Grid-search C/gamma over grouped folds; pick the most generalisable."""
    n = min(args.cv or 5, _n_groups(groups))
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=args.seed)
    grid = {"svc__C": [0.1, 1, 10, 100],
            "svc__gamma": ["scale", 0.01, 0.1, 1.0]}
    base = make_pipeline(args.sigma, args.C, args.gamma, not args.no_scale)
    gs = GridSearchCV(base, grid, scoring="f1", cv=sgkf, n_jobs=-1, refit=True)
    gs.fit(X, y, groups=groups)
    print(f"[grid] best params: {gs.best_params_}  (grouped CV f1={gs.best_score_:.3f})")
    return gs.best_estimator_, gs.best_params_


def _tune_threshold_oof(X, y, groups, pipe, args):
    """Pick the decision threshold on out-of-fold scores from the TRAIN videos.

    Each sample is scored by a model that never saw its video, so the chosen
    threshold is honest and does not touch the held-out test set. Maximises the
    requested objective (default: balanced accuracy).
    """
    n = min(args.cv or 5, _n_groups(groups))
    if n < 2:
        return 0.0, None
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=args.seed)
    oof = np.full(len(y), np.nan)
    for tr, te in sgkf.split(X, y, groups):
        p = clone(pipe)
        p.fit(X[tr], y[tr])
        oof[te] = p.decision_function(X[te])
    m = ~np.isnan(oof)
    yv, sv = y[m], oof[m]

    best_thr, best_score = 0.0, -1.0
    for thr in np.linspace(float(sv.min()), float(sv.max()), 80):
        yp = (sv > thr).astype(int)
        if args.threshold_metric == "f1":
            _, _, score, _ = precision_recall_fscore_support(
                yv, yp, average="binary", pos_label=1, zero_division=0)
        elif args.threshold_metric == "recall_at_prec":
            p_, r_, _, _ = precision_recall_fscore_support(
                yv, yp, average="binary", pos_label=1, zero_division=0)
            score = r_ if p_ >= args.min_precision else -1.0  # max recall s.t. precision floor
        else:  # balanced accuracy
            score = balanced_accuracy_score(yv, yp)
        if score > best_score:
            best_score, best_thr = score, float(thr)
    return best_thr, best_score


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_train(args) -> None:
    cfg = _cfg_from_args(args)
    X, y, groups = build_dataset(args.data, cfg, args.stride, args.max_frames)

    # 1) hold out whole videos for an honest test number (before anything else)
    test = None
    if args.test_split > 0:
        (X, y, groups), (Xte, yte, gte) = _group_stratified_split(
            X, y, groups, args.test_split, args.seed)
        test = (Xte, yte, gte)
        print(f"[split] train {len(y)} frames / {_n_groups(groups)} videos   "
              f"|   held-out test {len(yte)} frames / {_n_groups(gte)} videos")

    # 2) balance classes on the training portion only
    if args.balance:
        X, y, groups = _balance_classes(X, y, groups, args.seed)
        print(f"[balance] train -> {len(y)} frames "
              f"(smoke={int((y==1).sum())} non-smoke={int((y==0).sum())})")

    # 3) grouped CV report (honest estimate from training videos)
    if args.cv and _n_groups(groups) >= 2:
        _grouped_cv_report(X, y, groups, args, args.cv)

    # 4) fit final model: grid-searched C/gamma, or fixed params
    if args.grid:
        pipe, best = _grid_search(X, y, groups, args)
    else:
        pipe = make_pipeline(args.sigma, args.C, args.gamma, not args.no_scale)
        pipe.fit(X, y)
        best = {"svc__C": args.C, "svc__gamma": args.gamma}

    # 5) tune the decision threshold on out-of-fold TRAIN scores (honest)
    threshold = 0.0
    if args.auto_threshold:
        threshold, sc = _tune_threshold_oof(X, y, groups, pipe, args)
        tag = f" ({args.threshold_metric}={sc:.3f})" if sc is not None else ""
        print(f"[auto-threshold] chosen={threshold:+.3f}  "
              f"(picked on grouped out-of-fold train scores{tag})")

    # 6) the number that actually matters: held-out, unseen videos
    if test is not None:
        Xte, yte, gte = test
        yp = (pipe.decision_function(Xte) > threshold).astype(int)
        _report(yte, yp, f"HELD-OUT TEST @ threshold {threshold:+.2f}  (videos unseen in training)")

    meta = {"sigma": args.sigma, "params": best, "stride": args.stride,
            "balanced": bool(args.balance), "max_frames": args.max_frames,
            "threshold": threshold, "n_train": int(X.shape[0]), "gabor": cfg.gabor,
            "feature_set": cfg.feature_set, "n_features": int(X.shape[1])}
    save_model(args.out, pipe, cfg, meta)


def _threshold_sweep(y, scores) -> None:
    print("\nthreshold sweep (raise it to trade recall for precision):")
    print("   thr     prec    rec     f1    bal_acc    FP    FN")
    for thr in np.linspace(-1.5, 1.5, 13):
        yp = (scores > thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y, yp, average="binary", pos_label=1, zero_division=0)
        bal = balanced_accuracy_score(y, yp)
        cm = confusion_matrix(y, yp, labels=[0, 1])
        print(f"  {thr:+.2f}   {p:.3f}   {r:.3f}   {f1:.3f}   {bal:.3f}   "
              f"{cm[0,1]:4d}  {cm[1,0]:4d}")


def cmd_eval(args) -> None:
    pipe, cfg, meta = load_model(args.model)
    X, y, groups = build_dataset(args.data, cfg, args.stride, args.max_frames)
    thr = args.threshold if args.threshold is not None else float(meta.get("threshold", 0.0))
    scores = pipe.decision_function(X)
    yp = (scores > thr).astype(int)
    _report(y, yp, f"overall (per-frame, threshold={thr:+.2f})")

    if args.sweep:
        _threshold_sweep(y, scores)

    # per-clip 0..1 grade (0 = clean, 1 = smoke) + verdict, MHT-style table
    videos = list_videos(args.data)
    print("\nper-clip grade (0..1)  ==  fraction of frames scored SMOKE")
    print(f"{'folder':>7} | {'clip':40s} | grade | verdict")
    print("-" * 72)
    gids = np.unique(groups)
    n_wrong = 0
    for gid in gids:
        m = groups == gid
        grade = float(yp[m].mean())           # 0..1 smoke-frame fraction
        true = int(y[m][0])
        pred = int(grade >= 0.5)
        n_wrong += int(pred != true)
        rel = os.path.relpath(videos[gid][0], args.data) if gid < len(videos) else str(gid)
        folder = (os.path.dirname(rel) or "?")
        clip = os.path.basename(rel)
        verdict = "SMOKE" if pred else "clean"
        tag = "ok " if pred == true else "<< WRONG"
        print(f"{folder:>7} | {clip:40s} | {grade:5.2f} | {verdict} {tag}")
    print(f"[clip-level] {len(gids) - n_wrong}/{len(gids)} correct")

    # per-video accuracy (majority vote -> video-level label too)
    print("\nper-video frame accuracy:")
    for gid in np.unique(groups):
        m = groups == gid
        acc = accuracy_score(y[m], yp[m])
        name = os.path.relpath(videos[gid][0], args.data) if gid < len(videos) else str(gid)
        print(f"  {name:40s} {acc:.4f}  ({int(m.sum())} frames)")


def cmd_predict(args) -> None:
    pipe, cfg, meta = load_model(args.model)
    bank = make_bank(cfg)
    # default to the stride the model was TRAINED with, so the temporal-diff
    # gap at inference matches training (and it runs ~stride x faster).
    stride = args.stride if args.stride is not None else int(meta.get("stride", 1))
    thr = args.threshold if args.threshold is not None else float(meta.get("threshold", 0.0))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[predict] stride={stride}  threshold={thr:+.2f}  resize={cfg.resize}  "
          f"gabor={cfg.gabor}  features={getattr(cfg,'feature_set','v1')}")

    writer = None
    show = args.show
    iir = make_iir(cfg)
    prev = None
    label = None            # last decision, carried forward between kept frames
    n_smoke = n_total = 0
    idx = 0
    quit_early = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # only run the expensive Gabor/wavelet extraction on every Nth frame
        if idx % stride == 0:
            proc = frame if cfg.resize is None else cv2.resize(
                frame, cfg.resize, interpolation=cv2.INTER_AREA)
            if prev is not None:
                fv = pair_features(prev, proc, bank, cfg,
                                   iir=iir).reshape(1, -1)
                label = int(pipe.decision_function(fv)[0] > thr)
                n_total += 1
                n_smoke += label
            prev = proc
        idx += 1

        if args.out or show:
            txt = "SMOKE" if label == 1 else ("clear" if label == 0 else "...")
            color = (0, 0, 255) if label == 1 else (0, 180, 0)
            cv2.putText(frame, txt, (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, color, 2, cv2.LINE_AA)
            if args.out:                                  # file keeps just the verdict
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
                writer.write(frame)
            if show:                                      # live window adds running %
                rr = (n_smoke / n_total) if n_total else 0.0
                cv2.putText(frame, f"smoke {rr:.0%}  ({n_smoke}/{n_total})",
                            (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (235, 235, 235), 2, cv2.LINE_AA)
                try:
                    cv2.imshow("smoke detection  -  q/esc to quit", frame)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        quit_early = True
                        break
                except cv2.error:
                    print("[!] --show needs the GUI build of OpenCV "
                          "(pip uninstall opencv-python-headless ; pip install opencv-python)")
                    show = False

    cap.release()
    if show:
        cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
        print(f"[saved] {args.out}")
    ratio = (n_smoke / n_total) if n_total else 0.0
    verdict = "SMOKE DETECTED" if ratio >= args.thresh else "no smoke"
    tag = "  [stopped early]" if quit_early else ""
    print(f"smoke frames: {n_smoke}/{n_total} ({ratio:.1%})  ->  {verdict} "
          f"(threshold {args.thresh:.0%}){tag}")


# --------------------------------------------------------------------------- #
# arg parsing
# --------------------------------------------------------------------------- #
def _add_common(p):
    p.add_argument("--stride", type=int, default=1,
                   help="keep every Nth frame (temporal gap n); paper uses 1")
    p.add_argument("--iir", choices=["frame", "iir", "iir-auto"], default=None,
                   help="temporal diff source: frame (paper default), iir "
                        "(fixed-alpha background), iir-auto (alpha adapts to "
                        "scene speed). Stored in the model; eval/predict "
                        "reuse the training setting automatically.")
    p.add_argument("--iir-alpha", type=float, default=None,
                   help="IIR base alpha (default 0.05)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap feature frames per video")
    p.add_argument("--resize", type=str, default="320x240",
                   help="WxH to resize frames to, or 'none' to keep native")
    p.add_argument("--gabor", choices=["haghighat", "paper"], default="haghighat")
    p.add_argument("--features", choices=["v1", "v2"], default="v2",
                   help="feature stage: v1 = paper 21-D, v2 = revised 28-D "
                        "(default; see README Results). Stored in the model, so "
                        "eval/predict reuse the training setting automatically.")
    p.add_argument("--wavelet", type=str, default="haar")


def _cfg_from_args(args) -> SmokeConfig:
    resize = None
    if args.resize and args.resize.lower() != "none":
        w, h = args.resize.lower().split("x")
        resize = (int(w), int(h))
    cfg = SmokeConfig(gabor=args.gabor, wavelet=args.wavelet, resize=resize)
    if getattr(args, "features", None):
        cfg.feature_set = args.features
    if getattr(args, "iir", None):
        cfg.diff_mode = args.iir
    if getattr(args, "iir_alpha", None) is not None:
        cfg.iir_alpha = args.iir_alpha
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Video smoke detection (Appana et al. 2017)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="extract features and train the SVM")
    pt.add_argument("--data", required=True)
    pt.add_argument("--out", default="smoke_svm.joblib")
    pt.add_argument("--cv", type=int, default=0, help="grouped CV folds (0=off)")
    pt.add_argument("--test-split", type=float, default=0.0,
                    help="fraction of VIDEOS held out for an honest test (e.g. 0.2)")
    pt.add_argument("--balance", action="store_true",
                    help="down-sample majority class to equal counts (train only)")
    pt.add_argument("--auto-threshold", action="store_true",
                    help="pick the decision threshold on out-of-fold train scores, store it")
    pt.add_argument("--threshold-metric", choices=["balanced", "f1", "recall_at_prec"],
                    default="balanced", help="objective the auto-threshold maximises")
    pt.add_argument("--min-precision", type=float, default=0.9,
                    help="precision floor for --threshold-metric recall_at_prec")
    pt.add_argument("--grid", action="store_true",
                    help="grid-search C/gamma over grouped folds (less overfitting)")
    pt.add_argument("--seed", type=int, default=0, help="RNG seed for splits/balance")
    pt.add_argument("--sigma", type=float, default=0.6, help="RBF sigma (paper 0.6)")
    pt.add_argument("--C", type=float, default=1.0)
    pt.add_argument("--gamma", type=float, default=None, help="override RBF gamma")
    pt.add_argument("--no-scale", action="store_true", help="disable feature scaling")
    _add_common(pt)
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("eval", help="evaluate a saved model on a labelled set")
    pe.add_argument("--data", required=True)
    pe.add_argument("--model", required=True)
    pe.add_argument("--threshold", type=float, default=None,
                    help="decision boundary; higher = fewer smoke calls (default 0)")
    pe.add_argument("--sweep", action="store_true",
                    help="print precision/recall/FP/FN across a range of thresholds")
    _add_common(pe)
    pe.set_defaults(func=cmd_eval)

    pp = sub.add_parser("predict", help="run on a single video")
    pp.add_argument("--video", required=True)
    pp.add_argument("--model", required=True)
    pp.add_argument("--out", default=None, help="write annotated mp4")
    pp.add_argument("--show", action="store_true",
                    help="live preview window (needs GUI build of OpenCV)")
    pp.add_argument("--stride", type=int, default=None,
                    help="extract every Nth frame (default: model's training stride)")
    pp.add_argument("--threshold", type=float, default=None,
                    help="decision boundary; higher = fewer smoke calls (default 0)")
    pp.add_argument("--thresh", type=float, default=0.15,
                    help="smoke-frame fraction for a positive verdict")
    pp.set_defaults(func=cmd_predict)

    args = ap.parse_args()
    # eval/predict pull stride/resize from the model's stored cfg; only train
    # needs them up front, but _add_common gives eval them too for overrides.
    args.func(args)


if __name__ == "__main__":
    main()
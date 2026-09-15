"""
train_motion_classifier_kfold.py
===============================================================================
K-fold cross-validated variant of train_motion_classifier.py.

Differences from the original (which is left untouched and still authoritative):

  1. UNBIASED ESTIMATE. The original picks the decision threshold by scanning
     the LOCO scores for the value that maximises the objective, then reports
     accuracy at that same threshold -- a hyperparameter fitted on the
     evaluation data. Here the threshold is chosen INSIDE the training folds
     only (nested), so the reported number is a genuine held-out estimate.

  2. FINDS THE BEST K. Repeated stratified K-fold for several candidate K,
     reporting mean +/- std across repeats. K is selected on estimator
     STABILITY, not on accuracy -- accuracy rises monotonically with K
     (more training data per fold), so "best accuracy" would always pick LOCO.

  3. SEARCHES l2 x major_life by that CV, then refits on all clips.

Splits are BY CLIP and stratified by label: every group from a clip stays on
one side of the split, so a clip's groups can never leak across folds.
major_life is a pooling threshold only (never used during extraction), so the
whole search runs on the cached feature vectors -- no re-extraction.

Output: K.json (same schema as motion_weights.json + a "kfold" block).

Usage:
    python train_motion_classifier_kfold.py --folder /MHT_TOP/val_videos --iir-auto
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import train_motion_classifier as base
from train_motion_classifier import (
    VIDEO_EXTS, POS_FOLDERS, NEG_FOLDERS, FEATURES, DEFAULT_PARAMS,
    extract_clip, fit_logistic, predict_proba,
)

THRESHOLDS = np.linspace(0.05, 0.95, 181)


def clip_score(samples, w, b, mu, sd, major_life):
    if not samples:
        return 0.0
    major = [s for s, lf in samples if lf >= major_life]
    pool = major if major else [s for s, _ in samples]
    return float(predict_proba(np.array(pool), w, b, mu, sd).max())


def fit_fold(per_clip, labels, idx, l2):
    X, y, life = [], [], []
    for j in idx:
        for s, lf in per_clip[j]:
            X.append(s); y.append(labels[j]); life.append(lf)
    if not X or len(set(y)) < 2:
        return None
    return fit_logistic(np.array(X), np.array(y, float),
                        life=np.array(life), l2=l2)


def rates(scores, y, th):
    pred = (scores >= th).astype(float)
    tpr = (pred[y == 1] == 1).mean() if (y == 1).any() else 0.0
    tnr = (pred[y == 0] == 0).mean() if (y == 0).any() else 0.0
    return tpr, tnr


def pick_threshold(scores, y, min_spec):
    tprs = np.array([rates(scores, y, th)[0] for th in THRESHOLDS])
    tnrs = np.array([rates(scores, y, th)[1] for th in THRESHOLDS])
    guard = tnrs >= min_spec - 1e-9
    if not guard.any():
        guard = np.ones_like(tnrs, bool)
    gmax = tprs[guard].max()
    top = guard & (tprs == gmax)
    cand = top & (tnrs == tnrs[top].max())
    return float(np.median(THRESHOLDS[cand]))


def stratified_folds(labels, k, rng):
    folds = [[] for _ in range(k)]
    for lab in (0, 1):
        idx = np.where(labels == lab)[0]
        rng.shuffle(idx)
        for pos, j in enumerate(idx):
            folds[pos % k].append(int(j))
    return [np.array(sorted(f)) for f in folds]


def inner_threshold(per_clip, labels, tr, l2, major_life, min_spec,
                    inner_k, rng):
    """Threshold from an INNER K-fold inside the training set. Scoring clips
    the inner model trained on would give optimistically separated scores and
    a threshold that does not transfer -- these scores are out-of-fold."""
    lab_tr = labels[tr]
    folds = stratified_folds(lab_tr, min(inner_k, len(tr)), rng)
    scores = np.full(len(tr), np.nan)
    for f in folds:
        if len(f) == 0:
            continue
        mask = np.ones(len(tr), bool)
        mask[f] = False
        fit = fit_fold(per_clip, labels, tr[mask], l2)
        if fit is None:
            continue
        w, b, mu, sd = fit
        for loc in f:
            scores[loc] = clip_score(per_clip[tr[loc]], w, b, mu, sd, major_life)
    ok = ~np.isnan(scores)
    if ok.sum() < 2 or len(set(lab_tr[ok].tolist())) < 2:
        return 0.5
    return pick_threshold(scores[ok], lab_tr[ok].astype(float), min_spec)


def oof_scores(per_clip, labels, k, l2, major_life, min_spec, rng, inner_k=3):
    """One repeat of K-fold. Returns pooled out-of-fold binary predictions,
    each produced by a model AND a threshold that never saw that clip."""
    n = len(labels)
    folds = stratified_folds(labels, k, rng)
    pred = np.full(n, np.nan)
    raw = np.full(n, np.nan)
    for f in folds:
        if len(f) == 0:
            continue
        held = set(f.tolist())
        tr = np.array([j for j in range(n) if j not in held])
        fit = fit_fold(per_clip, labels, tr, l2)
        if fit is None:
            continue
        w, b, mu, sd = fit
        th = inner_threshold(per_clip, labels, tr, l2, major_life,
                             min_spec, inner_k, rng)
        for j in f:
            s = clip_score(per_clip[j], w, b, mu, sd, major_life)
            raw[j] = s
            pred[j] = float(s >= th)
    return pred, raw


def repeated_cv(per_clip, labels, k, l2, major_life, min_spec, repeats,
                seed, inner_k=3):
    bals, recalls, specs = [], [], []
    for r in range(repeats):
        rng = np.random.default_rng(seed + r)
        pred, _ = oof_scores(per_clip, labels, k, l2, major_life, min_spec,
                             rng, inner_k)
        ok = ~np.isnan(pred)
        y = labels[ok].astype(float); p = pred[ok]
        tpr = (p[y == 1] == 1).mean() if (y == 1).any() else 0.0
        tnr = (p[y == 0] == 0).mean() if (y == 0).any() else 0.0
        recalls.append(tpr); specs.append(tnr); bals.append(0.5 * (tpr + tnr))
    return (float(np.mean(bals)), float(np.std(bals)),
            float(np.mean(recalls)), float(np.mean(specs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--max-frames", type=int, default=450)
    ap.add_argument("--params", default=None)
    ap.add_argument("--iir-auto", action="store_true")
    ap.add_argument("--auto-gain", type=float, default=1.0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--min-spec", type=float, default=0.75)
    ap.add_argument("--out", default=None)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--inner-k", type=int, default=3,
                    help="inner folds used to choose each fold's threshold")
    ap.add_argument("--k-candidates", default="3,5,10,LOCO")
    ap.add_argument("--l2-grid", default="1e-3,3e-3,1e-2,3e-2,1e-1,3e-1")
    ap.add_argument("--major-life-grid", default="25,30,40,50,60")
    args = ap.parse_args()

    params = dict(DEFAULT_PARAMS)
    if args.params:
        with open(args.params) as f:
            params.update(json.load(f))
    if args.iir_auto:
        params["iir_auto"] = 1
        params["iir_auto_gain"] = args.auto_gain

    clips: list[tuple[str, int, str]] = []
    for folder, lab in [(f, 1) for f in POS_FOLDERS] + \
                       [(f, 0) for f in NEG_FOLDERS]:
        d = os.path.join(args.folder, folder)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(VIDEO_EXTS):
                clips.append((os.path.join(d, f), lab, folder))
    if not clips:
        sys.exit(f"No videos found under {args.folder}")
    labels = np.array([l for _, l, _ in clips])
    print(f"Found {len(clips)} clips ({labels.sum()} positive / "
          f"{len(labels) - labels.sum()} clean)\n")

    cache_dir = None if args.no_cache else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_cache")
    per_clip = []
    t_all = time.time()
    for i, (path, lab, folder) in enumerate(clips):
        t0 = time.time()
        s = extract_clip(path, params, args.max_frames, cache_dir=cache_dir)
        per_clip.append(s)
        print(f"[{i+1:>3}/{len(clips)}] {folder:>5} | "
              f"{os.path.basename(path)[:38]:<38} groups={len(s):<3} "
              f"({time.time()-t0:.1f}s)")
    print(f"\nExtraction done in {time.time()-t_all:.0f}s\n")

    n = len(clips)
    n_pos, n_neg = int(labels.sum()), int(n - labels.sum())
    k_cands = []
    for tok in args.k_candidates.split(","):
        tok = tok.strip()
        k = n if tok.upper() == "LOCO" else int(tok)
        if tok.upper() != "LOCO" and k > min(n_pos, n_neg):
            print(f"[skip] K={tok}: needs <= {min(n_pos, n_neg)} "
                  f"(fewest clips in a class) to stratify")
            continue
        k_cands.append((tok, k))

    print("=" * 68)
    print("STAGE 1 -- choosing K   (l2=1e-2, major_life=40, the current defaults)")
    print("=" * 68)
    print(f"{'K':>6} | {'bal acc':>16} | {'recall':>7} | {'spec':>7} | folds")
    print("-" * 68)
    k_rows = []
    for tok, k in k_cands:
        reps = 1 if k == n else args.repeats
        t0 = time.time()
        m, s, rc, sp = repeated_cv(per_clip, labels, k, 1e-2, 40,
                                   args.min_spec, reps, args.seed,
                                   args.inner_k)
        k_rows.append((tok, k, m, s, rc, sp, reps))
        print(f"{tok:>6} | {m*100:6.1f}% +/- {s*100:4.1f} | {rc*100:6.1f}% | "
              f"{sp*100:6.1f}% | {k} ({time.time()-t0:.0f}s)")

    tunable = [r for r in k_rows if r[6] > 1]
    if tunable:
        best_k_row = min(tunable, key=lambda r: (r[3], -r[2]))
    else:
        best_k_row = k_rows[-1]
    best_k = best_k_row[1]
    print(f"\nChosen K = {best_k_row[0]} -- lowest spread across "
          f"{best_k_row[6]} repeats ({best_k_row[3]*100:.1f} pts), "
          f"i.e. the most reproducible estimate.")
    print("NOTE: K is chosen on stability, not accuracy. Accuracy grows with K "
          "because\n      each fold trains on more clips, so 'highest accuracy' "
          "would always say LOCO.")

    l2_grid = [float(x) for x in args.l2_grid.split(",")]
    ml_grid = [int(x) for x in args.major_life_grid.split(",")]
    print("\n" + "=" * 68)
    print(f"STAGE 2 -- searching l2 x major_life at K={best_k_row[0]} "
          f"({len(l2_grid)}x{len(ml_grid)} = {len(l2_grid)*len(ml_grid)} combos)")
    print("=" * 68)
    print(f"{'l2':>8} | {'major_life':>10} | {'bal acc':>16} | {'recall':>7} | {'spec':>7}")
    print("-" * 68)
    results = []
    for l2 in l2_grid:
        for ml in ml_grid:
            m, s, rc, sp = repeated_cv(per_clip, labels, best_k, l2, ml,
                                       args.min_spec, args.repeats,
                                       args.seed, args.inner_k)
            results.append((m, s, rc, sp, l2, ml))
            print(f"{l2:>8.4g} | {ml:>10} | {m*100:6.1f}% +/- {s*100:4.1f} | "
                  f"{rc*100:6.1f}% | {sp*100:6.1f}%")
    best = max(results, key=lambda r: (round(r[0], 4), -r[1], r[4]))
    bal, bal_sd, rec, spec, best_l2, best_ml = best
    print(f"\nBest: l2={best_l2:g}  major_life={best_ml}  ->  "
          f"{bal*100:.1f}% +/- {bal_sd*100:.1f} balanced")

    print("\n" + "=" * 68)
    print("STAGE 3 -- final model on ALL clips")
    print("=" * 68)
    rng = np.random.default_rng(args.seed)
    _, raw = oof_scores(per_clip, labels, best_k, best_l2, best_ml,
                        args.min_spec, rng, args.inner_k)
    valid = ~np.isnan(raw)
    final_th = pick_threshold(raw[valid], labels[valid].astype(float),
                              args.min_spec)
    X, yg, lg = [], [], []
    for j, samples in enumerate(per_clip):
        for s, lf in samples:
            X.append(s); yg.append(labels[j]); lg.append(lf)
    w, b, mu, sd = fit_logistic(np.array(X), np.array(yg, float),
                                life=np.array(lg), l2=best_l2)

    print(f"deployed threshold : {final_th:.3f} (from out-of-fold scores)")
    print(f"CV balanced acc    : {bal*100:.1f}% +/- {bal_sd*100:.1f}  "
          f"(recall {rec*100:.1f}%  spec {spec*100:.1f}%)")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "K.json")
    model = {
        "features": FEATURES,
        "mean": mu.tolist(), "std": sd.tolist(),
        "coef": w.tolist(), "bias": float(b),
        "smoke_thr": final_th,
        "cloud_thr": max(0.05, final_th - 0.2),
        "loco_balanced_acc": bal,
        "n_clips": len(clips), "n_groups": int(len(X)),
        "major_life": best_ml,
        "trained_with_params": params,
        "kfold": {
            "k": best_k, "k_label": best_k_row[0], "repeats": args.repeats,
            "seed": args.seed, "l2": best_l2, "major_life": best_ml,
            "min_spec": args.min_spec,
            "cv_balanced_acc_mean": bal, "cv_balanced_acc_std": bal_sd,
            "cv_recall_mean": rec, "cv_specificity_mean": spec,
            "nested_threshold": True, "inner_k": args.inner_k,
            "k_search": [{"k": t, "bal": m, "std": s} for t, _, m, s, _, _, _ in k_rows],
        },
    }
    with open(out, "w") as f:
        json.dump(model, f, indent=2)
    print(f"\nSaved -> {out}")
    print("Feature coefficients (standardized, + = smoke-ish):")
    for k_, c in sorted(zip(FEATURES, w), key=lambda t: -abs(t[1])):
        print(f"   {k_:>10}: {c:+.3f}")


if __name__ == "__main__":
    main()

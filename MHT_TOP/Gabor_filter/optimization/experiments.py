"""Model-side experiments over the cached 21-D features.

No feature code is touched -- this is only hyperparameters, class balancing,
decision threshold and evaluation protocol (the "free" fixes).

Run in stages so each finishes inside the sandbox time limit:
    python experiments.py grid1 | grid2 | test | families | repeated | pervideo
"""
import json, os, sys
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support, confusion_matrix)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
OUT = HERE
STATE = os.path.join(tempfile.gettempdir(), "exp_state.json")
PAPER_GAMMA = 1.0 / (2 * 0.6 ** 2)          # 1.3889, what the baseline used

C_ = np.load(os.path.join(OUT, "feats_cache.npz"), allow_pickle=True)
X, y, groups = C_["X"].astype(np.float64), C_["y"], C_["groups"]
META = json.loads(str(C_["meta"]))

IN_VAL = np.array([("val_videos" in m["roots"]) for m in META])
IN_TEST = np.array([("test_videos" in m["roots"]) for m in META])
TRAIN_G = np.where(IN_VAL & ~IN_TEST)[0]     # leaked clip dropped from TRAIN
TEST_G = np.where(IN_TEST)[0]


def svm(C=1.0, gamma=PAPER_GAMMA, balanced=True):
    return Pipeline([("scaler", StandardScaler()),
                     ("svc", SVC(kernel="rbf", C=C, gamma=gamma,
                                 class_weight="balanced" if balanced else None))])


FAMS = {
    "SVM rbf (paper gamma)": svm(1.0, PAPER_GAMMA),
    "LogReg": Pipeline([("s", StandardScaler()),
                        ("m", LogisticRegression(max_iter=3000,
                                                 class_weight="balanced"))]),
    "RandomForest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                           class_weight="balanced",
                                           random_state=0, n_jobs=-1),
    "HistGradBoost": HistGradientBoostingClassifier(max_iter=250,
                                                    learning_rate=0.06,
                                                    random_state=0),
}


def state(**kw):
    s = json.load(open(STATE)) if os.path.exists(STATE) else {}
    if kw:
        s.update(kw)
        json.dump(s, open(STATE, "w"))
    return s


def sub(gids):
    m = np.isin(groups, gids)
    return X[m], y[m], groups[m]


def video_vote(y_true, y_pred, g):
    ok = [int((y_pred[g == q].mean() > 0.5) == bool(y_true[g == q][0]))
          for q in np.unique(g)]
    return float(np.mean(ok))


def report(yt, yp, g, tag):
    a, b = accuracy_score(yt, yp), balanced_accuracy_score(yt, yp)
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="binary",
                                                  pos_label=1, zero_division=0)
    print(f"  {tag:34s} frame_acc={a:.3f} bal={b:.3f} prec={p:.3f} rec={r:.3f} "
          f"f1={f1:.3f} | VIDEO_acc={video_vote(yt, yp, g):.3f}")


def oof(est, X, y, g, seed=0, n=5):
    n = min(n, len(np.unique(g)))
    sg = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    o = np.full(len(y), np.nan)
    for tr, te in sg.split(X, y, g):
        e = clone(est).fit(X[tr], y[tr])
        o[te] = (e.decision_function(X[te]) if hasattr(e, "decision_function")
                 else e.predict_proba(X[te])[:, 1] - 0.5)
    return o


def best_threshold(y, s):
    bt, bv = 0.0, -1.0
    for t in np.linspace(s.min(), s.max(), 120):
        v = balanced_accuracy_score(y, (s > t).astype(int))
        if v > bv:
            bv, bt = v, float(t)
    return bt, bv


GAMMAS_A = ["scale", 0.005, 0.01, 0.02, 0.05]
GAMMAS_B = [0.1, 0.3, 1.0, PAPER_GAMMA]
CS = [0.1, 1, 10, 100]


def do_grid(gammas, key):
    Xtr, ytr, gtr = sub(TRAIN_G)
    rows = state().get("grid", [])
    for gm in gammas:
        for c in CS:
            s = oof(svm(c, gm), Xtr, ytr, gtr, seed=0)
            b0 = balanced_accuracy_score(ytr, (s > 0).astype(int))
            t, bt = best_threshold(ytr, s)
            rows.append([bt, b0, t, str(gm), c])
            print(f"  gamma={str(gm):<8} C={c:<5} oof_bal@0={b0:.3f}  "
                  f"oof_bal@tuned={bt:.3f} (thr={t:+.2f})")
    state(grid=rows)


def cmd_grid1():
    Xtr, ytr, gtr = sub(TRAIN_G)
    Xte, yte, gte = sub(TEST_G)
    print(f"[cache] {X.shape} over {len(META)} clips  "
          f"smoke={int((y==1).sum())} non-smoke={int((y==0).sum())}")
    print(f"[split] train {len(ytr)} frames / {len(TRAIN_G)} clips   "
          f"test {len(yte)} frames / {len(TEST_G)} clips\n")
    print("=== 1. baseline reproduction (sigma=0.6 -> gamma=1.389, thr=0) ===")
    base = svm(1.0, PAPER_GAMMA).fit(Xtr, ytr)
    yp = (base.decision_function(Xte) > 0).astype(int)
    report(yte, yp, gte, "baseline")
    print(confusion_matrix(yte, yp, labels=[0, 1]))
    print("\n=== 2. grouped-CV grid on TRAIN only (part 1) ===")
    do_grid(GAMMAS_A, "a")


def cmd_grid2():
    print("=== 2. grouped-CV grid on TRAIN only (part 2) ===")
    do_grid(GAMMAS_B, "b")
    rows = sorted(state()["grid"], key=lambda r: -r[0])
    print("\n  top 5 by out-of-fold balanced accuracy:")
    for bt, b0, t, gm, c in rows[:5]:
        print(f"    gamma={gm:<8} C={c:<5} oof_bal={bt:.3f} thr={t:+.2f}")
    state(best=rows[0])


def _best():
    bt, b0, t, gm, c = state()["best"]
    return (gm if gm == "scale" else float(gm)), float(c), float(t)


def cmd_test():
    gm, c, thr = _best()
    Xtr, ytr, gtr = sub(TRAIN_G)
    Xte, yte, gte = sub(TEST_G)
    print(f"=== 3. held-out test_videos, TRAIN-selected gamma={gm} C={c} ===")
    fin = svm(c, gm).fit(Xtr, ytr)
    sc = fin.decision_function(Xte)
    report(yte, (sc > 0).astype(int), gte, "tuned gamma/C, thr=0")
    report(yte, (sc > thr).astype(int), gte, f"tuned + oof thr={thr:+.2f}")
    print(confusion_matrix(yte, (sc > thr).astype(int), labels=[0, 1]))
    n_sv = fin.named_steps["svc"].n_support_.sum()
    print(f"  support vectors: {n_sv}/{len(ytr)} ({n_sv/len(ytr):.0%})")


def cmd_families():
    gm, c, thr = _best()
    Xtr, ytr, gtr = sub(TRAIN_G)
    Xte, yte, gte = sub(TEST_G)
    print("=== 4. model family check on held-out test ===")
    fams = dict(FAMS); fams["SVM rbf (tuned)"] = svm(c, gm)
    for name, est in fams.items():
        e = clone(est).fit(Xtr, ytr)
        pr = ((e.decision_function(Xte) > 0).astype(int)
              if hasattr(e, "decision_function") else e.predict(Xte))
        report(yte, pr, gte, name)


def cmd_repeated():
    gm, c, thr = _best()
    fams = dict(FAMS); fams["SVM rbf (tuned)"] = svm(c, gm)
    print("=== 5. all-84-clip repeated grouped CV (low-variance estimate) ===")
    for name, est in fams.items():
        A, B, V = [], [], []
        for seed in range(5):
            s = oof(clone(est), X, y, groups, seed=seed)
            yp = (s > 0).astype(int)
            A.append(accuracy_score(y, yp)); B.append(balanced_accuracy_score(y, yp))
            V.append(video_vote(y, yp, groups))
        print(f"  {name:24s} frame={np.mean(A):.3f}±{np.std(A):.3f}  "
              f"bal={np.mean(B):.3f}  VIDEO={np.mean(V):.3f}±{np.std(V):.3f}")


def cmd_pervideo():
    gm, c, thr = _best()
    Xtr, ytr, gtr = sub(TRAIN_G)
    Xte, yte, gte = sub(TEST_G)
    sc = svm(c, gm).fit(Xtr, ytr).decision_function(Xte)
    yp = (sc > thr).astype(int)
    print("=== 6. per-video frame accuracy on test_videos (tuned model) ===")
    for gid in np.unique(gte):
        m = gte == gid
        md = META[gid]
        a = accuracy_score(yte[m], yp[m])
        flag = "  <-- fails" if a < 0.5 else ""
        print(f"  {md['cls']+'/'+md['name']:46s} {a:.3f}{flag}")


if __name__ == "__main__":
    globals()[f"cmd_{sys.argv[1]}"]()

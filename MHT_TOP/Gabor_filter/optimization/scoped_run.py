"""Final evaluation scoped strictly to val_videos (train) and test_videos (test).

Model selection happens with video-grouped CV *inside val_videos only*.
test_videos is touched exactly once, at the end.

    python scoped_run.py select    # grid over val_videos -> /tmp/scoped.json
    python scoped_run.py final     # train on val, evaluate on test
    python scoped_run.py pooled    # grouped CV over val+test (=84 clips)
"""
import json, os, sys
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support, confusion_matrix)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
# SMOKE_CACHE=feats_cache_v2.npz picks the revised 28-D features
CACHE = os.environ.get("SMOKE_CACHE", "feats_cache.npz")
import tempfile
STATE = os.path.join(tempfile.gettempdir(), "scoped_" + CACHE.replace(".npz", "") + ".json")
PAPER_GAMMA = 1.0 / (2 * 0.6 ** 2)

D = np.load(os.path.join(HERE, CACHE), allow_pickle=True)
X, y, groups = D["X"].astype(np.float64), D["y"], D["groups"]
META = json.loads(str(D["meta"]))

VAL = np.array([i for i, m in enumerate(META) if "val_videos" in m["roots"]])
TEST = np.array([i for i, m in enumerate(META) if "test_videos" in m["roots"]])
TRAIN = np.array([i for i in VAL if i not in set(TEST)])   # drop the leaked clip
assert len(set(TRAIN) & set(TEST)) == 0


def svm(C=1.0, gamma=PAPER_GAMMA):
    return Pipeline([("scaler", StandardScaler()),
                     ("svc", SVC(kernel="rbf", C=C, gamma=gamma,
                                 class_weight="balanced"))])


def sub(gids):
    m = np.isin(groups, gids)
    return X[m], y[m], groups[m]


def video_vote(yt, yp, g):
    return float(np.mean([int((yp[g == q].mean() > .5) == bool(yt[g == q][0]))
                          for q in np.unique(g)]))


def report(yt, yp, g, tag):
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="binary",
                                                  pos_label=1, zero_division=0)
    print(f"  {tag:36s} frame={accuracy_score(yt, yp):.3f} "
          f"bal={balanced_accuracy_score(yt, yp):.3f} prec={p:.3f} rec={r:.3f} "
          f"f1={f1:.3f} | VIDEO={video_vote(yt, yp, g):.3f}")


def oof(est, X, y, g, seed=0, n=5):
    sg = StratifiedGroupKFold(n_splits=min(n, len(np.unique(g))),
                              shuffle=True, random_state=seed)
    o = np.full(len(y), np.nan)
    for tr, te in sg.split(X, y, g):
        e = clone(est).fit(X[tr], y[tr])
        o[te] = (e.decision_function(X[te]) if hasattr(e, "decision_function")
                 else e.predict_proba(X[te])[:, 1] - .5)
    return o


def best_threshold(y, s):
    bt, bv = 0., -1.
    for t in np.linspace(s.min(), s.max(), 120):
        v = balanced_accuracy_score(y, (s > t).astype(int))
        if v > bv:
            bv, bt = v, float(t)
    return bt, bv


def cmd_select():
    """Grid over val_videos only, averaged over 5 CV seeds (seed-robust)."""
    Xtr, ytr, gtr = sub(TRAIN)
    print(f"[scope] model selection on val_videos: {len(ytr)} frames / "
          f"{len(TRAIN)} clips (leaked clip excluded)")
    rows = []
    for gm in [0.005, 0.01, 0.02, 0.05, PAPER_GAMMA]:
        for c in [0.1, 1, 10]:
            v = [best_threshold(ytr, oof(svm(c, gm), Xtr, ytr, gtr, seed=s))
                 for s in range(3)]
            bal = float(np.mean([b for _, b in v]))
            thr = float(np.mean([t for t, _ in v]))
            sd = float(np.std([b for _, b in v]))
            rows.append([bal, sd, thr, gm, c])
            print(f"  gamma={gm:<8.4g} C={c:<5} oof_bal={bal:.3f}±{sd:.3f} "
                  f"thr={thr:+.2f}")
    rows.sort(key=lambda r: -r[0])
    json.dump(rows[0], open(STATE, "w"))
    print(f"\n  -> selected gamma={rows[0][3]} C={rows[0][4]} thr={rows[0][2]:+.3f}")


def cmd_final():
    bal, sd, thr, gm, c = json.load(open(STATE))
    Xtr, ytr, _ = sub(TRAIN)
    Xte, yte, gte = sub(TEST)
    print(f"[scope] train=val_videos {len(TRAIN)} clips -> "
          f"test=test_videos {len(TEST)} clips\n")
    print("=== train on val_videos, evaluate on test_videos ===")
    b = svm(1.0, PAPER_GAMMA).fit(Xtr, ytr)
    report(yte, (b.decision_function(Xte) > 0).astype(int), gte,
           "paper gamma=1.389, thr=0")
    f = svm(c, gm).fit(Xtr, ytr)
    s = f.decision_function(Xte)
    report(yte, (s > 0).astype(int), gte, f"tuned gamma={gm} C={c}, thr=0")
    report(yte, (s > thr).astype(int), gte, f"tuned + val-chosen thr={thr:+.2f}")
    print("\nconfusion (tuned+thr) [rows=true 0/1, cols=pred 0/1]:")
    print(confusion_matrix(yte, (s > thr).astype(int), labels=[0, 1]))
    print("\nper-video frame accuracy on test_videos:")
    yp = (s > thr).astype(int)
    for gid in np.unique(gte):
        m = gte == gid
        a = accuracy_score(yte[m], yp[m])
        print(f"  {META[gid]['cls']+'/'+META[gid]['name']:46s} {a:.3f}"
              f"{'   <-- fails' if a < .5 else ''}")


def cmd_pooled():
    bal, sd, thr, gm, c = json.load(open(STATE))
    pool = np.union1d(VAL, TEST)
    Xp, yp_, gp = sub(pool)
    print(f"[scope] pooled video-grouped CV over val+test = {len(pool)} clips, "
          f"{len(yp_)} frames")
    for name, est in [("SVM paper gamma", svm(1.0, PAPER_GAMMA)),
                      ("SVM tuned", svm(c, gm))]:
        A, V = [], []
        for seed in range(5):
            s = oof(clone(est), Xp, yp_, gp, seed=seed)
            q = (s > 0).astype(int)
            A.append(accuracy_score(yp_, q)); V.append(video_vote(yp_, q, gp))
        print(f"  {name:18s} frame={np.mean(A):.3f}±{np.std(A):.3f}  "
              f"VIDEO={np.mean(V):.3f}±{np.std(V):.3f}")


if __name__ == "__main__":
    globals()[f"cmd_{sys.argv[1]}"]()

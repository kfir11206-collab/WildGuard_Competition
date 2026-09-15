"""Fit the final SVM from a cached feature set and save it as a normal
`smoke_detection` model file.

Equivalent to running `smoke_detection.py train` with the same settings -- the
CLI extractor reproduces these cached features to ~3e-5 (float32 storage
rounding) -- but instant, because extraction is already done.

    python build_model.py --cache feats_cache_v2s55.npz --out ../smoke_v2.joblib
"""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GABOR = os.path.dirname(HERE)
sys.path.insert(0, GABOR)

from smoke_features import SmokeConfig
from smoke_detection import make_pipeline, save_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="feats_cache_v2s55.npz")
    ap.add_argument("--out", default=os.path.join(GABOR, "smoke_v2.joblib"))
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=-0.433)
    ap.add_argument("--s-high", type=float, default=0.55)
    ap.add_argument("--v-low", type=float, default=0.35)
    ap.add_argument("--v-high", type=float, default=1.0)
    a = ap.parse_args()

    D = np.load(os.path.join(HERE, a.cache), allow_pickle=True)
    X, y, groups = D["X"].astype(np.float64), D["y"], D["groups"]
    META = json.loads(str(D["meta"]))

    # train on val_videos only; never on anything that is in test_videos
    val = [i for i, m in enumerate(META) if "val_videos" in m["roots"]]
    test = {i for i, m in enumerate(META) if "test_videos" in m["roots"]}
    train_g = [i for i in val if i not in test]
    assert not (set(train_g) & test), "train/test clip overlap"
    m = np.isin(groups, train_g)
    Xtr, ytr = X[m], y[m]
    print(f"[data] train {Xtr.shape[0]} frames / {len(train_g)} clips "
          f"(smoke {int((ytr==1).sum())} / clean {int((ytr==0).sum())})")

    cfg = SmokeConfig(diff_mode="iir-auto", feature_set="v2",
                      s_high=a.s_high, v_low=a.v_low, v_high=a.v_high)
    pipe = make_pipeline(C=a.C, gamma=a.gamma).fit(Xtr, ytr)

    meta = {"params": {"svc__C": a.C, "svc__gamma": a.gamma},
            "stride": 5, "max_frames": 50, "balanced": False,
            "threshold": a.threshold, "n_train": int(Xtr.shape[0]),
            "gabor": cfg.gabor, "feature_set": "v2",
            "n_features": int(Xtr.shape[1]),
            "source": f"fitted from {a.cache} (optimization/build_model.py)"}
    save_model(a.out, pipe, cfg, meta)
    print(f"[model] gamma={a.gamma} C={a.C} thr={a.threshold:+.3f} "
          f"support_vectors={pipe.named_steps['svc'].n_support_.sum()}")


if __name__ == "__main__":
    main()

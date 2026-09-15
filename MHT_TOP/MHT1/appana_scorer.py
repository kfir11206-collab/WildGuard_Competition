"""
appana_scorer.py
===============================================================================
Bridge between the Gabor_filter project (Appana et al. 2017 per-frame smoke
SVM) and the MHT1 group-motion classifier.

Wraps a trained smoke_detection.py model (.joblib) as a per-frame smoke
probability that the MHT1 pipeline attaches to every group as the "appana"
feature. The two detectors look at DIFFERENT evidence -- the SVM sees
spatial-temporal Gabor/wavelet texture of the whole frame, the motion
classifier sees group kinematics -- so fusing them lets the logistic
regression learn how much to trust each.

Usage:
    scorer = AppanaScorer()            # loads ../Gabor_filter model
    scorer.reset()                     # once per video
    p = scorer.step(frame_bgr)         # every frame -> [0,1], 0.5 = neutral

Fail-soft: if the model or its dependencies (joblib/sklearn/pywt/scipy) are
missing, step() returns 0.5 forever and the rest of the pipeline behaves
exactly as before (the trained logistic sees a constant neutral feature).
===============================================================================
"""

from __future__ import annotations

import math
import os
import sys

_GF_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Gabor_filter"))

# newest per-frame 21-D model first, then fallbacks
DEFAULT_MODEL_CANDIDATES = (
    os.path.join(_GF_DIR, "old_train", "smoke_svm_1.joblib"),
    os.path.join(_GF_DIR, "old_train", "smoke_svm.joblib"),
    os.path.join(_GF_DIR, "old_train", "smoke_iir.joblib"),
)


def default_model_path() -> str | None:
    for p in DEFAULT_MODEL_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


class AppanaScorer:
    """Per-video, stateful smoke-probability scorer around a Gabor_filter SVM."""

    def __init__(self, model_path: str | None = None, ema: float = 0.3):
        self.ok = False
        self.ema = float(ema)
        self.p = 0.5
        self.model_path = model_path or default_model_path()
        self._iir = None
        try:
            if not self.model_path or not os.path.isfile(self.model_path):
                raise FileNotFoundError(f"no SVM model found (looked in {_GF_DIR})")
            if _GF_DIR not in sys.path:
                sys.path.insert(0, _GF_DIR)
            import joblib                                    # noqa: deferred
            from smoke_features import (                     # noqa: deferred
                SmokeConfig, build_gabor_bank, frame_pair_features, make_iir)
            blob = joblib.load(self.model_path)
            self.pipe = blob["pipe"]
            self.cfg = SmokeConfig.from_dict(blob["cfg"])
            meta = blob.get("meta", {}) or {}
            self.stride = max(1, int(meta.get("stride", 1)))
            self.thr = float(meta.get("threshold", 0.0))
            self.bank = build_gabor_bank(self.cfg)
            self._frame_pair_features = frame_pair_features
            self._make_iir = make_iir
            self.ok = True
            print(f"[appana] SVM feature ON: {os.path.basename(self.model_path)} "
                  f"(stride {self.stride}, diff {self.cfg.diff_mode})")
        except Exception as e:
            print(f"[appana] SVM feature OFF ({e}) -- 'appana' stays neutral 0.5")
        self.reset()

    def reset(self):
        """Call between videos (clears frame + IIR state)."""
        self.p = 0.5
        self._prev = None
        self._n = 0
        self._iir = self._make_iir(self.cfg) if self.ok else None

    def step(self, frame_bgr) -> float:
        """Feed EVERY frame (any size; it is resized internally). The heavy
        Gabor/wavelet extraction runs every `stride` frames -- the model's
        own training stride, so the temporal gap matches training. Returns
        an EMA-smoothed smoke probability in [0,1]."""
        if not self.ok:
            return 0.5
        import cv2
        if self._n % self.stride == 0:
            proc = frame_bgr if self.cfg.resize is None else cv2.resize(
                frame_bgr, self.cfg.resize, interpolation=cv2.INTER_AREA)
            if self._prev is not None:
                fv = self._frame_pair_features(
                    self._prev, proc, self.bank, self.cfg,
                    iir=self._iir).reshape(1, -1)
                z = float(self.pipe.decision_function(fv)[0]) - self.thr
                p_now = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
                self.p = (1.0 - self.ema) * self.p + self.ema * p_now
            self._prev = proc
        self._n += 1
        return self.p

    def cache_tag(self) -> str:
        """Identity string for feature-cache keys: model file + mtime."""
        if not self.ok:
            return "appana:off"
        return (f"appana:{os.path.basename(self.model_path)}"
                f":{os.path.getmtime(self.model_path):.0f}")

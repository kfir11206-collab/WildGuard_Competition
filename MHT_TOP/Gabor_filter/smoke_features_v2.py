"""smoke_features_v2.py
--------------------
Revised feature stage. `smoke_features.py` is left untouched (paper-faithful);
this module reuses its mask, IIR background and Gabor bank, and only replaces
the *statistics* step.

Why: in the original pipeline `image_stats()` reduces each of the 5 Gabor
magnitude maps to global entropy/uniformity/mean/std over the WHOLE frame.
Orientation selectivity is a spatial property, so averaging |G_theta| across the
whole image destroys it -- measured on 84 clips the 5 orientations correlate at
r = 1.0000 and PCA finds 3 components explaining 99% of the variance. The 21-D
vector is effectively 3-D. The stats also include the zeroed-out non-mask
region, so with 5% smoke coverage 95% of the histogram sits in one zero bin and
entropy/uniformity mostly measure mask AREA.

Fixes here, all of which need the same extraction pass:
  A  mask coverage / shape                                        (3)
  B  orientation ENERGY PROFILE, L1-normalised -> scale-free      (7)
  C  pixelwise orientation anisotropy over mask pixels only       (3)
  D  spatial heterogeneity from a 4x4 block grid                  (4)
  E  masked texture stats, self-normalised fixed-range histogram  (4)
  F  cross-SCALE energy ratios (smoke suppresses high frequency)  (3)
  G  occlusion cue: wavelet energy of frame vs IIR background     (4)
                                                                 ----
                                                                  28

(G) is the Toreyin/Cetin cue the original pipeline never computes: smoke is a
semi-transparent low-pass filter, so it LOWERS the wavelet energy of the current
frame relative to the background image. That is what separates smoke from cloud,
which the HSV mask cannot do.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import pywt

from smoke_features import (
    SmokeConfig, GaborBank, hsv_smoke_mask, apply_mask, temporal_diff,
    make_iir, build_gabor_kernels,
)

EPS = 1e-9
N_BLOCKS = 4
SCALES = (1, 2, 3)


def fast_hsv_mask(bgr: np.ndarray, cfg: SmokeConfig) -> np.ndarray:
    """Bit-identical to smoke_features.hsv_smoke_mask, but much faster.

    The original filters small components with

        for i in range(1, num): keep[lab == i] = 255

    i.e. one full-image comparison per connected component. With a widened mask
    that is hundreds of passes over the frame per call. Replacing it with a
    single lookup on the label image gives the same result in one pass.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = np.array([cfg.h_low * 179, cfg.s_low * 255, cfg.v_low * 255])
    hi = np.array([cfg.h_high * 179, cfg.s_high * 255, cfg.v_high * 255])
    mask = cv2.inRange(hsv, lo, hi)

    if cfg.min_area > 0:
        num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        good = stats[:, cv2.CC_STAT_AREA] >= cfg.min_area
        good[0] = False                                   # background
        mask = np.where(good[lab], np.uint8(255), np.uint8(0))

    if cfg.close_radius > 0:
        k = 2 * cfg.close_radius + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se)

    if cfg.fill_holes:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
        mask = filled
    return mask

FEATURE_DIM_V2 = 28


def feature_names_v2() -> List[str]:
    n = ["mask_cov", "mask_ncomp", "mask_fill"]
    n += [f"orient_frac{o}" for o in range(5)]
    n += ["orient_logscale", "orient_anisotropy"]
    n += ["px_aniso_mean", "px_aniso_std", "px_domorient_entropy"]
    n += ["blk_cv", "blk_max_ratio", "blk_active_frac", "blk_entropy"]
    n += ["mstat_entropy", "mstat_uniformity", "mstat_mean", "mstat_std"]
    n += ["scale_ratio_12", "scale_ratio_23", "scale_slope"]
    n += ["wav_energy_diff", "wav_ratio_cur_bg", "wav_hf_ratio", "edge_ratio"]
    return n


ASPECT = 3.0        # envelope elongation (1.0 = the original circular envelope)


def oriented_kernels(cfg: SmokeConfig, scale: int, aspect: float = ASPECT):
    """Haghighat Gabor kernels with TWO bugs of the original fixed.

    1. The reference sets gamma = eta = sqrt(2), so alpha == beta and the
       Gaussian envelope is exactly CIRCULAR (measured aspect ratio 1.000).
       A circular envelope with a complex carrier gives a magnitude response
       that barely depends on theta -- which is why the 5 "orientations"
       correlate at r = 1.0000. `cv2.getGaborKernel(..., gamma=1.0)` used by
       gabor="paper" is circular too, so both modes have the same defect.
       Elongating to aspect 3:1 raises the orientation spread on a directional
       test image from 0.0024 to 0.1384.

    2. theta = j/(n_orient-1) * pi walks 0, 45, 90, 135, 180 degrees -- but 180
       degrees IS 0 degrees, so only 4 of the 5 orientations are distinct.
       Using j/n_orient * pi spans [0, pi) with n distinct orientations.
    """
    import math
    m = n = cfg.gabor_ksize
    g, e = 1.0, aspect
    fu = cfg.gabor_fmax / (math.sqrt(2.0) ** (scale - 1))
    alpha, beta = fu / g, fu / e
    c = (m + 1) / 2.0
    X = np.arange(1, m + 1)[:, None] * np.ones((1, n))
    Y = np.ones((m, 1)) * np.arange(1, n + 1)[None, :]
    out = []
    for j in range(cfg.n_orient):
        th = (j / cfg.n_orient) * math.pi          # spans [0, pi), no duplicate
        xp = (X - c) * math.cos(th) + (Y - c) * math.sin(th)
        yp = -(X - c) * math.sin(th) + (Y - c) * math.cos(th)
        env = (fu ** 2 / (math.pi * g * e)) * np.exp(
            -(alpha ** 2 * xp ** 2 + beta ** 2 * yp ** 2))
        out.append((env * np.exp(1j * 2 * math.pi * fu * xp)).astype(np.complex128))
    return out


class MultiScaleBank:
    """Gabor banks at several scales, using the orientation-selective kernels."""

    def __init__(self, cfg: SmokeConfig, scales=SCALES, aspect: float = ASPECT):
        self.banks = []
        for s in scales:
            c = SmokeConfig(**{**cfg.to_dict(), "gabor_scale": s})
            b = GaborBank(c)
            b.kernels = oriented_kernels(c, s, aspect)   # swap in fixed kernels
            b._plans = {}
            self.banks.append(b)

    def responses(self, diff):
        return [b.responses(diff) for b in self.banks]     # [scale][orient]


# --------------------------------------------------------------------------- #
# helpers -- every statistic below is computed over MASK PIXELS ONLY
# --------------------------------------------------------------------------- #
def _masked(a: np.ndarray, m: np.ndarray) -> np.ndarray:
    v = a[m]
    return v if v.size else np.zeros(1, dtype=a.dtype)


def _norm_hist_stats(v: np.ndarray, bins: int = 64, hi: float = 8.0):
    """entropy/uniformity/mean/std of v after dividing by its own mean.

    Self-normalising + FIXED histogram range makes these comparable between
    clips of different contrast, unlike the original data-dependent range.
    """
    mu = float(v.mean())
    sd = float(v.std())
    if mu <= EPS:
        return 0.0, 1.0, 0.0, 0.0
    z = np.clip(v / mu, 0.0, hi)
    h, _ = np.histogram(z, bins=bins, range=(0.0, hi))
    p = h.astype(np.float64)
    p /= max(p.sum(), 1.0)
    nz = p > 0
    return (float(-(p[nz] * np.log(p[nz])).sum()), float((p * p).sum()),
            mu, sd)


def _block_stats(resp: np.ndarray, m: np.ndarray, k: int = N_BLOCKS):
    """Energy per cell of a k x k grid, using mask pixels only -> heterogeneity.

    Smoke is spatially localised and coherent; sensor noise and global camera
    motion are spread evenly. Global averaging cannot tell these apart.
    """
    H, W = resp.shape
    e = np.zeros(k * k)
    for i in range(k):
        for j in range(k):
            sl = (slice(i * H // k, (i + 1) * H // k),
                  slice(j * W // k, (j + 1) * W // k))
            mm = m[sl]
            e[i * k + j] = resp[sl][mm].mean() if mm.any() else 0.0
    tot = e.sum()
    if tot <= EPS:
        return 0.0, 0.0, 0.0, 0.0
    mu = e.mean()
    cv = float(e.std() / (mu + EPS))
    mx = float(e.max() / (mu + EPS))
    act = float((e > 0.5 * e.max()).mean())
    p = e / tot
    nz = p > 0
    ent = float(-(p[nz] * np.log(p[nz])).sum())
    return cv, mx, act, ent


def _wavelet_detail(x: np.ndarray, wavelet="haar"):
    _cA, (cH, cV, cD) = pywt.dwt2(x.astype(np.float64), wavelet)
    return cH ** 2 + cV ** 2 + cD ** 2


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def frame_pair_features_v2(prev_bgr, cur_bgr, bank: MultiScaleBank,
                           cfg: SmokeConfig, iir=None) -> np.ndarray:
    mask_u8 = fast_hsv_mask(cur_bgr, cfg)
    m = mask_u8 > 0
    cov = float(m.mean())

    gp = cv2.cvtColor(apply_mask(prev_bgr, fast_hsv_mask(prev_bgr, cfg)),
                      cv2.COLOR_BGR2GRAY)
    gc = cv2.cvtColor(apply_mask(cur_bgr, mask_u8), cv2.COLOR_BGR2GRAY)
    full_cur = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)

    if iir is not None:
        diff = iir.step(gp, gc, full_cur)
        bg = iir._bg  # float32 background estimate maintained by the IIR
    else:
        diff = temporal_diff(gp, gc)
        bg = None

    f: List[float] = []

    # --- A. mask coverage / shape -----------------------------------------
    ncomp, _lab, stats, _c = cv2.connectedComponentsWithStats(mask_u8, 8)
    areas = stats[1:, cv2.CC_STAT_AREA] if ncomp > 1 else np.array([0])
    fill = float(areas.max() / (m.sum() + EPS)) if m.any() else 0.0
    f += [cov, float(min(ncomp - 1, 50)) / 50.0, fill]

    if not m.any():
        return np.array(f + [0.0] * (FEATURE_DIM_V2 - len(f)), dtype=np.float64)

    # --- Gabor responses, all scales ---------------------------------------
    per_scale = bank.responses(diff)          # [scale][orient] -> HxW
    R = np.asarray(per_scale[0])              # scale 1, shape (5,H,W)

    # --- B. orientation energy profile (L1-normalised => scale-free) -------
    E = np.array([_masked(R[o], m).mean() for o in range(R.shape[0])])
    tot = E.sum()
    frac = E / (tot + EPS)
    f += list(frac)
    f += [float(np.log1p(tot)),
          float((E.max() - E.min()) / (E.max() + E.min() + EPS))]

    # --- C. pixelwise orientation anisotropy -------------------------------
    Rm = R[:, m]                              # (5, Nmask)
    mx = Rm.max(0); mn = Rm.mean(0)
    aniso = (mx - mn) / (mx + EPS)
    dom = Rm.argmax(0)
    ph = np.bincount(dom, minlength=R.shape[0]).astype(np.float64)
    ph /= max(ph.sum(), 1.0)
    nz = ph > 0
    f += [float(aniso.mean()), float(aniso.std()),
          float(-(ph[nz] * np.log(ph[nz])).sum())]

    # --- D. spatial heterogeneity ------------------------------------------
    Rmean = R.mean(0)
    f += list(_block_stats(Rmean, m))

    # --- E. masked texture stats, fixed-range histogram --------------------
    f += list(_norm_hist_stats(_masked(Rmean, m)))

    # --- F. cross-scale ratios (smoke suppresses high frequency) -----------
    Es = [float(np.asarray(per_scale[s]).mean(0)[m].mean()) for s in range(len(per_scale))]
    f += [float(np.log((Es[0] + EPS) / (Es[1] + EPS))),
          float(np.log((Es[1] + EPS) / (Es[2] + EPS)))]
    ls = np.log(np.array(Es) + EPS)
    f.append(float(np.polyfit(np.arange(len(ls)), ls, 1)[0]))

    # --- G. occlusion cue: current frame vs background ---------------------
    wd = _wavelet_detail(diff, cfg.wavelet)
    f.append(float(wd.mean()))
    if bg is not None:
        bgu = cv2.convertScaleAbs(bg)
        mh = cv2.resize(mask_u8, (wd.shape[1], wd.shape[0]),
                        interpolation=cv2.INTER_NEAREST) > 0
        wc = _wavelet_detail(cv2.bitwise_and(full_cur, mask_u8), cfg.wavelet)
        wb = _wavelet_detail(cv2.bitwise_and(bgu, mask_u8), cfg.wavelet)
        if mh.any():
            # <1 means the current frame lost detail vs the background = smoke
            f.append(float(np.log((wc[mh].mean() + EPS) / (wb[mh].mean() + EPS))))
            hf = wc[mh] > np.percentile(wb[mh], 75)
            f.append(float(hf.mean()))
        else:
            f += [0.0, 0.0]
        ec = cv2.Laplacian(full_cur, cv2.CV_32F)
        eb = cv2.Laplacian(bgu, cv2.CV_32F)
        f.append(float(np.log((np.abs(ec)[m].mean() + EPS) /
                              (np.abs(eb)[m].mean() + EPS))))
    else:
        f += [0.0, 0.0, 0.0]

    v = np.asarray(f, dtype=np.float64)
    assert v.shape[0] == FEATURE_DIM_V2, (v.shape, FEATURE_DIM_V2)
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def build_bank_v2(cfg: SmokeConfig) -> MultiScaleBank:
    return MultiScaleBank(cfg)

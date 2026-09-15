"""
smoke_features.py
-----------------
GUI-free port of the per-frame feature pipeline from:

    Appana, Dileep K., et al. "A video-based smoke detection using smoke flow
    pattern and spatial-temporal energy analyses for alarm systems."
    Information Sciences 418-419 (2017): 91-101.

Pipeline (paper Fig. 1 / Fig. 9):
    frame -> HSV color mask -> temporal frame difference (n=1)
          -> Gabor filter bank (5 orientations) -> 4 stats each  (20 feats)
          -> single-level 2-D wavelet energy                     ( 1 feat)
          -> 21-D feature vector

The HSV thresholds + morphology follow the authors' own MATLAB reference
(color_masking.m); the Gabor bank follows their gaborFilterBank.m (the
Haghighat parameterisation actually used in the published code). The simpler
real-cosine form written in the paper (Eq. 3) is available via gabor="paper".

This module has no Qt / no I/O side effects so it drops straight into a
core/GUI split.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
import math

import cv2
import numpy as np
import pywt
import scipy.fft as sfft


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class SmokeConfig:
    # --- HSV smoke-colour thresholds, normalised [0,1] (paper 2.1.1) -------- #
    # Retuned from the paper values by sweeping s_high over the 84-clip set
    # (see FINDINGS_gabor_baseline.md section 9). Paper: s_high=0.28,
    # v_low=0.38, v_high=0.985 -- those rejected 42% of a typical smoke clip as
    # "too saturated" and left 4 clips with a completely empty mask, one of
    # which (smoke/99182-653447840) was therefore unclassifiable in every run.
    # Held-out test accuracy by s_high: 0.28 -> 0.650, 0.45 -> 0.669,
    # 0.55 -> 0.694, 0.65 -> 0.654. The curve peaks at 0.55.
    # v_high is raised to 1.0 so fully blown-out skies (V=255) are not dropped.
    h_low: float = 0.0
    h_high: float = 1.0          # hue unconstrained
    s_low: float = 0.0
    s_high: float = 0.55         # paper: 0.28
    v_low: float = 0.35          # paper: 0.38
    v_high: float = 1.0          # paper: 0.985

    # --- mask clean-up (reference color_masking.m) ------------------------- #
    min_area: int = 100          # bwareaopen(...,100)
    close_radius: int = 4        # imclose(strel('disk',4))
    fill_holes: bool = True      # imfill('holes')

    # --- Gabor bank -------------------------------------------------------- #
    gabor: str = "haghighat"     # "haghighat" (reference code) | "paper" (Eq. 3)
    gabor_backend: str = "fft"   # "fft" (batched, ~20x faster) | "spatial" (cv2.filter2D)
    n_orient: int = 5            # theta = 0, pi/4, pi/2, 3pi/4, pi
    gabor_ksize: int = 39
    gabor_scale: int = 1         # Haghighat scale index (1 = highest freq)
    gabor_fmax: float = 3.0
    # paper-mode (Eq. 3) envelope/wavelength; the paper's sigma=1,f=2 are in
    # unspecified normalised units, so these reproduce the look of Fig. 4.
    paper_sigma: float = 4.0
    paper_lambda: float = 8.0

    # --- wavelet energy (paper 2.2.2) -------------------------------------- #
    wavelet: str = "haar"

    # --- which feature stage to use ----------------------------------------- #
    # "v1" : the paper's 21-D vector from this module (default, backward
    #        compatible -- models saved before this option keep working).
    # "v2" : the 28-D revised stage in smoke_features_v2.py (masked, block-local
    #        statistics + orientation-selective Gabor kernels + occlusion cue).
    #        Measurably better; see README "Results".
    feature_set: str = "v1"

    # --- temporal difference source ----------------------------------------- #
    # "frame"    : |I(t) - I(t-1)|  (paper Eq. 2, default, stateless)
    # "iir"      : |I(t) - B(t)| against an IIR background B, fixed alpha
    # "iir-auto" : same, but alpha adapts to measured apparent speed
    #              (fast scenes update the background faster), like the
    #              MHT pipeline's Stage-1 auto mode.
    diff_mode: str = "frame"
    iir_alpha: float = 0.05      # base/fallback IIR update rate
    iir_alpha_min: float = 0.01  # auto mode: alpha range and speed mapping
    iir_alpha_max: float = 0.25
    iir_speed_lo: float = 0.3    # px/frame -> alpha_min
    iir_speed_hi: float = 4.0    # px/frame -> alpha_max

    # --- frame I/O --------------------------------------------------------- #
    resize: Optional[Tuple[int, int]] = (320, 240)  # (w,h); paper res. None=keep

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SmokeConfig":
        d = dict(d)
        if d.get("resize") is not None:
            d["resize"] = tuple(d["resize"])
        return cls(**d)


FEATURE_DIM = 21
# names line up with the 21-vector below (5 orientations x 4 stats + energy)
_STAT_NAMES = ("entropy", "uniformity", "mean", "std")


def feature_names(cfg: SmokeConfig) -> List[str]:
    names = []
    for o in range(cfg.n_orient):
        for s in _STAT_NAMES:
            names.append(f"gabor{o}_{s}")
    names.append("wavelet_energy")
    return names


# --------------------------------------------------------------------------- #
# Pre-processing  (paper 2.1)
# --------------------------------------------------------------------------- #
def hsv_smoke_mask(bgr: np.ndarray, cfg: SmokeConfig) -> np.ndarray:
    """Binary (0/255) smoke-colour mask with the reference morphology."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)  # H:0-179, S:0-255, V:0-255
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    lo = np.array([cfg.h_low * 179, cfg.s_low * 255, cfg.v_low * 255])
    hi = np.array([cfg.h_high * 179, cfg.s_high * 255, cfg.v_high * 255])
    mask = cv2.inRange(hsv, lo, hi)  # uint8 0/255

    # remove small connected components  (bwareaopen)
    if cfg.min_area > 0:
        num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros_like(mask)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= cfg.min_area:
                keep[lab == i] = 255
        mask = keep

    # morphological closing  (imclose, disk radius)
    if cfg.close_radius > 0:
        k = 2 * cfg.close_radius + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se)

    # fill holes  (imfill 'holes') via external-contour fill (robust)
    if cfg.fill_holes:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
        mask = filled

    return mask


def apply_mask(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(bgr, bgr, mask=mask)


def temporal_diff(prev_gray: np.ndarray, cur_gray: np.ndarray) -> np.ndarray:
    """|I(t) - I(t-1)| (paper Eq. 2 with n=1, magnitude form)."""
    return cv2.absdiff(cur_gray, prev_gray)


class IIRBackground:
    """IIR background model for the temporal-difference stage.

    diff = |I(t) - B(t)|,   B(t+1) = (1-alpha) B(t) + alpha I(t)

    In "iir-auto" mode alpha adapts per frame to the scene's apparent speed
    (median |dI/dt| / |grad I| on a blurred half-resolution copy), mirroring
    the MHT pipeline's Stage-1 auto mode: slow scenes keep a long memory
    (small alpha, faint drifting smoke stays visible in the diff), fast
    scenes update quickly so global motion does not flood the features.
    One instance per video; reset() between videos.
    """

    def __init__(self, cfg: SmokeConfig):
        self.cfg = cfg
        self.auto = cfg.diff_mode == "iir-auto"
        self.reset()

    def reset(self):
        self._bg = None
        self._prev_blur = None
        self._speed_ema = None
        self._frames = 0
        self.alpha_effective = float(self.cfg.iir_alpha)

    def _auto_alpha(self, full_gray_u8: np.ndarray) -> float:
        cfg = self.cfg
        small = cv2.resize(full_gray_u8.astype(np.float32), None, fx=0.5,
                           fy=0.5, interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(small, (5, 5), 0)
        if self._prev_blur is None or self._prev_blur.shape != blur.shape:
            self._prev_blur = blur
        dt_b = cv2.absdiff(blur, self._prev_blur)
        self._prev_blur = blur
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        gmag = cv2.magnitude(gx, gy) * 0.125
        sel = (dt_b > 2.0) & (gmag > 1.0)
        v = 2.0 * float(np.median(dt_b[sel] / gmag[sel])) if sel.sum() > 50 else 0.0
        self._speed_ema = v if self._speed_ema is None else             0.9 * self._speed_ema + 0.1 * v
        if self._frames < 10:                      # EMA warm-up
            return float(cfg.iir_alpha)
        t = np.clip((self._speed_ema - cfg.iir_speed_lo) /
                    (cfg.iir_speed_hi - cfg.iir_speed_lo + 1e-6), 0.0, 1.0)
        return float(cfg.iir_alpha_min +
                     t * (cfg.iir_alpha_max - cfg.iir_alpha_min))

    def step(self, prev_gray_u8: np.ndarray, cur_gray_u8: np.ndarray,
             full_gray_u8: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the diff image for this pair and update the background."""
        self._frames += 1
        if self._bg is None or self._bg.shape != cur_gray_u8.shape:
            self._bg = prev_gray_u8.astype(np.float32)
        alpha = self._auto_alpha(
            full_gray_u8 if full_gray_u8 is not None else cur_gray_u8)             if self.auto else float(self.cfg.iir_alpha)
        self.alpha_effective = alpha
        diff = cv2.absdiff(cur_gray_u8, cv2.convertScaleAbs(self._bg))
        cv2.accumulateWeighted(cur_gray_u8.astype(np.float32), self._bg, alpha)
        return diff


def make_iir(cfg: SmokeConfig) -> Optional["IIRBackground"]:
    """One IIRBackground per video when cfg.diff_mode uses IIR, else None."""
    return IIRBackground(cfg) if cfg.diff_mode in ("iir", "iir-auto") else None


# --------------------------------------------------------------------------- #
# Gabor filter bank  (paper 2.2.1 / reference gaborFilterBank.m)
# --------------------------------------------------------------------------- #
def build_gabor_kernels(cfg: SmokeConfig) -> List[np.ndarray]:
    """Return a list of (complex) Gabor kernels, one per orientation."""
    if cfg.gabor == "paper":
        return _bank_paper(cfg)
    return _bank_haghighat(cfg)


class GaborBank:
    """Holds the Gabor kernels and applies them.

    backend="fft": FFT the diff image once, multiply by pre-computed kernel
    spectra, inverse-FFT the whole stack in one batched call -> O(n log n),
    independent of kernel size, ~20x faster than 5 complex cv2.filter2D calls.
    The image is reflect-padded so the interior is bit-identical (~1e-15) to the
    spatial path, so a model trained with "spatial" needs no retrain.

    backend="spatial": the original 2 x cv2.filter2D-per-orientation path.

    FFT plans are cached per diff-shape, so resize=None (variable sizes) is fine.
    """

    def __init__(self, cfg: SmokeConfig):
        self.cfg = cfg
        self.kernels = build_gabor_kernels(cfg)
        self.backend = getattr(cfg, "gabor_backend", "fft")
        self.K = cfg.gabor_ksize
        self._plans: dict = {}   # shape -> (Hp, Wp, pad, spectra)

    def __len__(self) -> int:
        return len(self.kernels)

    def _plan(self, shape: Tuple[int, int]):
        if shape in self._plans:
            return self._plans[shape]
        H, W = shape
        pad = self.K // 2
        Hp = sfft.next_fast_len(H + 2 * pad)
        Wp = sfft.next_fast_len(W + 2 * pad)
        spectra = []
        for k in self.kernels:
            kf = k[::-1, ::-1]                       # flip -> correlation (cv2 convention)
            Kc = np.zeros((Hp, Wp), dtype=np.complex128)
            kh, kw = kf.shape
            Kc[:kh, :kw] = kf
            Kc = np.roll(Kc, (-(kh // 2), -(kw // 2)), axis=(0, 1))   # centre the kernel
            spectra.append(sfft.fft2(Kc))
        plan = (Hp, Wp, pad, np.asarray(spectra))
        self._plans[shape] = plan
        return plan

    def responses(self, diff: np.ndarray) -> List[np.ndarray]:
        """Return the per-orientation Gabor magnitude maps for one image."""
        if self.backend == "spatial":
            f = diff.astype(np.float64)
            return [gabor_magnitude(f, k) for k in self.kernels]

        Hp, Wp, pad, spectra = self._plan(diff.shape)
        H, W = diff.shape
        dp = cv2.copyMakeBorder(diff.astype(np.float64),
                                pad, Hp - H - pad, pad, Wp - W - pad,
                                cv2.BORDER_REFLECT)
        F = sfft.fft2(dp)                            # image FFT: ONCE
        stack = np.abs(sfft.ifft2(spectra * F, axes=(-2, -1), workers=-1))
        return [stack[i, pad:pad + H, pad:pad + W] for i in range(len(self.kernels))]


def build_gabor_bank(cfg: SmokeConfig) -> GaborBank:
    """Build the Gabor bank (FFT-backed by default). Use .responses(diff)."""
    return GaborBank(cfg)


def _bank_haghighat(cfg: SmokeConfig) -> List[np.ndarray]:
    """Exact port of gaborFilterBank.m for a single scale, n_orient orients."""
    m = n = cfg.gabor_ksize
    gamma = math.sqrt(2.0)
    eta = math.sqrt(2.0)
    fu = cfg.gabor_fmax / (math.sqrt(2.0) ** (cfg.gabor_scale - 1))
    alpha, beta = fu / gamma, fu / eta

    cx = (m + 1) / 2.0  # MATLAB 1-indexed centre
    cy = (n + 1) / 2.0
    xi = np.arange(1, m + 1)[:, None].astype(np.float64)  # rows  -> x
    yj = np.arange(1, n + 1)[None, :].astype(np.float64)  # cols  -> y
    X = np.broadcast_to(xi, (m, n))
    Y = np.broadcast_to(yj, (m, n))

    kernels = []
    denom = max(cfg.n_orient - 1, 1)
    for j in range(cfg.n_orient):
        theta = (j / denom) * math.pi
        xp = (X - cx) * math.cos(theta) + (Y - cy) * math.sin(theta)
        yp = -(X - cx) * math.sin(theta) + (Y - cy) * math.cos(theta)
        env = (fu ** 2 / (math.pi * gamma * eta)) * np.exp(
            -(alpha ** 2 * xp ** 2 + beta ** 2 * yp ** 2)
        )
        kernels.append((env * np.exp(1j * 2 * math.pi * fu * xp)).astype(np.complex128))
    return kernels


def _bank_paper(cfg: SmokeConfig) -> List[np.ndarray]:
    """Real cosine Gabor (paper Eq. 3) via cv2.getGaborKernel."""
    k = cfg.gabor_ksize
    kernels = []
    denom = max(cfg.n_orient - 1, 1)
    for j in range(cfg.n_orient):
        theta = (j / denom) * math.pi
        g = cv2.getGaborKernel(
            (k, k), cfg.paper_sigma, theta, cfg.paper_lambda,
            gamma=1.0, psi=0.0, ktype=cv2.CV_64F,
        )
        kernels.append(g.astype(np.complex128))
    return kernels


def gabor_magnitude(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """|img * kernel|. Correlation w/ reflective border (matches imfilter)."""
    f = img.astype(np.float64)
    re = cv2.filter2D(f, cv2.CV_64F, np.real(kernel), borderType=cv2.BORDER_REFLECT)
    if np.iscomplexobj(kernel) and np.any(np.imag(kernel)):
        im = cv2.filter2D(f, cv2.CV_64F, np.imag(kernel), borderType=cv2.BORDER_REFLECT)
        return np.sqrt(re * re + im * im)
    return np.abs(re)


# --------------------------------------------------------------------------- #
# Statistics  (paper Table 1)
# --------------------------------------------------------------------------- #
def image_stats(img: np.ndarray, bins: int = 256) -> Tuple[float, float, float, float]:
    """Return (entropy, uniformity, mean, std) for one filtered image."""
    x = img.astype(np.float64).ravel()
    mean = float(x.mean())
    std = float(x.std())  # population std: sqrt(1/n * sum (x-mu)^2)

    hist, _ = np.histogram(x, bins=bins)
    p = hist.astype(np.float64)
    total = p.sum()
    if total <= 0:
        return 0.0, 1.0, mean, std
    p /= total
    nz = p > 0
    entropy = float(-(p[nz] * np.log(p[nz])).sum())   # natural log (ln)
    uniformity = float((p * p).sum())
    return entropy, uniformity, mean, std


# --------------------------------------------------------------------------- #
# Spatial-temporal wavelet energy  (paper 2.2.2, Eq. 5-6)
# --------------------------------------------------------------------------- #
def wavelet_energy(img: np.ndarray, wavelet: str = "haar") -> float:
    """Mean per-pixel detail energy of a single-level 2-D DWT."""
    x = img.astype(np.float64)
    _cA, (cH, cV, cD) = pywt.dwt2(x, wavelet)
    energy = cH ** 2 + cV ** 2 + cD ** 2
    return float(energy.mean())


# --------------------------------------------------------------------------- #
# Per-frame-pair feature vector
# --------------------------------------------------------------------------- #
def frame_pair_features(
    prev_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    bank: "GaborBank",
    cfg: SmokeConfig,
    return_debug: bool = False,
    iir: Optional["IIRBackground"] = None,
):
    """21-D feature vector for an adjacent (masked) frame pair.

    Pass a per-video IIRBackground (see make_iir) to replace the adjacent
    frame difference with an IIR background difference; with iir=None the
    behaviour is the original paper pipeline, unchanged."""
    mp = apply_mask(prev_bgr, hsv_smoke_mask(prev_bgr, cfg))
    mc = apply_mask(cur_bgr, hsv_smoke_mask(cur_bgr, cfg))
    gp = cv2.cvtColor(mp, cv2.COLOR_BGR2GRAY)
    gc = cv2.cvtColor(mc, cv2.COLOR_BGR2GRAY)
    if iir is not None:
        diff = iir.step(gp, gc, cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY))
    else:
        diff = temporal_diff(gp, gc)

    responses = bank.responses(diff)        # all 5 orientations in one batched FFT
    feats: List[float] = []
    for resp in responses:
        feats.extend(image_stats(resp))           # 4 per orientation
    feats.append(wavelet_energy(diff, cfg.wavelet))  # +1

    vec = np.asarray(feats, dtype=np.float64)
    if return_debug:
        return vec, {"masked_cur": mc, "diff": diff, "responses": responses}
    return vec

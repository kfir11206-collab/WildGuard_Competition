"""
wildfire_core.py
===============================================================================
Pure, Qt-free image-processing core for the Wildfire Detection pipeline.

Single source of truth for the three stages:
    1. iir              - IIR background subtraction (motion mask)
    2. hsv              - HSV smoke colour candidates
    3. gabor            - Gabor smoke-texture verification

Both wildfire_gui.py (interactive) and optimize_pipeline.py (Optuna tuning)
import `WildfirePipeline` from here, so the detection logic lives in exactly
one place. There are NO PyQt / GUI imports in this file by design.

Visualization without duplication
---------------------------------
Each stage method takes an optional `panel` argument:
  * panel is None  -> headless: only the binary mask is computed/returned
                      (this is what the optimizer uses; the draw_* calls are
                      cheap no-ops because the helper returns immediately).
  * panel given    -> the SAME method also draws the GUI overlay onto `panel`,
                      pixel-identical to the original wildfire_gui.py.

This means the contour loops, thresholds and mask logic exist once. The GUI
passes a panel; the optimizer doesn't. Both see identical detection results.
===============================================================================
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    from gabor_cupy import CUPY_AVAILABLE, HybridGaborBank
except Exception:
    CUPY_AVAILABLE = False
    HybridGaborBank = None


# Keep identical to the GUI so tuned params transfer 1:1.
MAX_PROC_W = 640

# Font constant reused by the overlay helpers.
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# Pixel budget for one Gabor-bank evaluation. A contour bbox can span the whole
# frame, which made the bank convolve 640x360 six times (90% of frame cost).
GABOR_MAX_ROI_PX = 64_000
GABOR_TILE = 112


def _tile_origins(dim, tile, count):
    if count <= 1:
        return [max(0, (dim - tile) // 2)]
    return [int(round(v)) for v in np.linspace(0, dim - tile, count)]


def _gabor_tiles(roi, max_px=GABOR_MAX_ROI_PX, tile=GABOR_TILE):
    """Small ROIs pass through untouched. An oversized one is sampled with
    evenly spaced equal-size tiles, so the mean response stays representative
    of the whole region at a bounded cost."""
    if roi.size <= max_px:
        return (roi,)
    h, w = roi.shape
    th, tw = min(tile, h), min(tile, w)
    n = max(1, max_px // (th * tw))
    ny = max(1, min(int(round((n * h / w) ** 0.5)), h // th))
    nx = max(1, min(n // ny, w // tw))
    return tuple(roi[y:y + th, x:x + tw]
                 for y in _tile_origins(h, th, ny)
                 for x in _tile_origins(w, tw, nx))



def bank_response_cpu(roi, bank):
    """Mean |response| per orientation, CPU/OpenCV, through the tile budget."""
    tiles = _gabor_tiles(roi)
    return np.array(
        [float(np.mean([np.mean(np.abs(cv2.filter2D(t, cv2.CV_32F, k)))
                        for t in tiles]))
         for k in bank], np.float32)


class WildfirePipeline:
    """Headless pipeline. Construct with a params dict, call process_frame()."""

    def __init__(self, params: dict, max_proc_w: int = MAX_PROC_W):
        self.params = params
        self.max_proc_w = max_proc_w

        # ---- internal state (mirrors the old ProcessingThread) ------------
        self._bg_model = None
        self._gabor_bank: list = []
        self._gabor_cache: tuple = ()
        self._gabor_gpu = None

        # ---- adaptive-IIR state --------------------------------------------
        self._prev_gray = None          # previous frame (float32) for |dI/dt|
        self._prev_blur = None          # blurred previous frame (speed estimate)
        self._speed_ema = None          # EMA of apparent speed  [px/frame]
        self._noise_ema = None          # EMA of noise sigma     [gray levels]
        self._auto_frames = 0           # frames seen since reset (warm-up)
        # Effective (alpha, threshold) actually used on the last frame —
        # equals the sliders in manual mode, the auto values in auto mode.
        # The GUI reads this to display the live auto choice.
        self.iir_effective: tuple = (0.0, 0)

    def reset(self):
        """Clear background state between videos."""
        self._bg_model = None
        self._prev_gray = None
        self._prev_blur = None
        self._speed_ema = None
        self._noise_ema = None
        self._auto_frames = 0

    # ── Gabor bank ──────────────────────────────────────────────────────────
    def _refresh_gabor(self, ksize, sigma, lam, gamma):
        key = (ksize, sigma, lam, gamma)
        if key == self._gabor_cache:
            return
        self._gabor_bank = []
        for deg in [0, 30, 60, 90, 120, 150]:
            k = cv2.getGaborKernel(
                (ksize, ksize), sigma, np.deg2rad(deg),
                lam, gamma, psi=0.0, ktype=cv2.CV_32F
            )
            k -= k.mean()                     # exact zero DC -> true bandpass
            k /= (np.abs(k).sum() + 1e-6)     # L1 normalize -> responses comparable
            self._gabor_bank.append(k)
        self._gabor_gpu = (HybridGaborBank(self._gabor_bank)
                           if CUPY_AVAILABLE and HybridGaborBank else None)
        self._gabor_cache = key

    # ── Stage 1: IIR ────────────────────────────────────────────────────────
    #
    # Two modes, selected by p["iir_auto"] (0 = manual sliders, 1 = auto):
    #
    #   MANUAL  uses p["iir_alpha"] / p["iir_threshold"] exactly as before.
    #
    #   AUTO    picks both per frame from the video itself:
    #     * threshold — robust noise floor. Static pixels dominate any frame,
    #       so the median + MAD of the inter-frame diff |I_t - I_{t-1}|
    #       estimates the sensor-noise sigma without being polluted by the
    #       moving object. threshold = 3*sigma + margin.
    #     * alpha — apparent motion speed. From brightness constancy
    #       (I_x*u + I_t = 0), the mean speed over the moving pixels is
    #           v  ~  mean(|dI/dt|) / mean(|grad I|)      [px/frame]
    #       which is contrast-invariant and ~100x cheaper than optical flow.
    #       Slow drift (smoke creeping) -> low alpha so the object is NOT
    #       absorbed into the background; fast motion -> high alpha so the
    #       background catches up and ghost trails don't linger.
    #       v is EMA-smoothed and mapped linearly onto
    #       [iir_alpha_min, iir_alpha_max] over [iir_speed_lo, iir_speed_hi].
    #
    # All auto knobs have defaults via .get(), so existing param dicts and
    # the Optuna optimizer keep working unchanged (auto off by default).
    def iir(self, gray_f32, p):
        first = self._bg_model is None
        if first:
            self._bg_model = gray_f32.copy()
            self._prev_gray = gray_f32.copy()

        if p.get("iir_auto", 0):
            alpha, thresh = self._auto_iir_params(gray_f32, p)
        else:
            alpha, thresh = float(p["iir_alpha"]), int(p["iir_threshold"])
            self._prev_gray = gray_f32.copy()   # keep estimator state fresh

        cv2.accumulateWeighted(gray_f32, self._bg_model, alpha)
        diff = cv2.absdiff(cv2.convertScaleAbs(gray_f32),
                           cv2.convertScaleAbs(self._bg_model))
        _, mask = cv2.threshold(diff, int(thresh), 255, cv2.THRESH_BINARY)

        self.iir_effective = (alpha, int(thresh))
        return mask

    def _auto_iir_params(self, gray_f32, p) -> tuple[float, int]:
        """Estimate (alpha, threshold) for the current frame. O(1 pass)."""
        # -- tunable bounds (all optional in the params dict) ----------------
        a_min    = float(p.get("iir_alpha_min",  0.01))
        a_max    = float(p.get("iir_alpha_max",  0.25))
        v_lo     = float(p.get("iir_speed_lo",   0.3))   # px/frame -> a_min
        v_hi     = float(p.get("iir_speed_hi",   4.0))   # px/frame -> a_max
        thr_k    = float(p.get("iir_thresh_k",   3.0))   # sigma multiplier
        thr_min  = int(p.get("iir_thresh_min",   6))
        thr_max  = int(p.get("iir_thresh_max",   40))
        gain_c   = float(p.get("iir_auto_gain",  1.0))    # user knob: c * auto
        ema_a    = 0.1                                    # smoothing factor

        dt_raw = cv2.absdiff(gray_f32, self._prev_gray)   # |dI/dt|, float32
        self._prev_gray = gray_f32.copy()
        self._auto_frames += 1

        # -- adaptive threshold: robust sigma of the (mostly static) diff ----
        # Raw (unblurred) diff, because the actual mask threshold is applied
        # to the raw |frame - bg_model|. 2x2 subsampling: same noise stats,
        # 4x cheaper median.
        dts   = dt_raw[::2, ::2]
        med   = float(np.median(dts))
        mad   = float(np.median(np.abs(dts - med)))
        sigma = 1.4826 * mad + 1e-3
        self._noise_ema = sigma if self._noise_ema is None else \
            (1 - ema_a) * self._noise_ema + ema_a * sigma
        # c > 1 -> stricter (collects less), c < 1 -> looser. Applied to the
        # auto threshold BEFORE clamping so the knob stays effective.
        thresh = int(np.clip(gain_c * (thr_k * self._noise_ema + 3.0),
                             thr_min, thr_max))

        # -- apparent speed ----------------------------------------------------
        # Estimated on a BLURRED copy so per-pixel sensor noise cannot fake
        # motion; only real structure (edges) survives the blur. Gate on both
        # temporal change AND significant spatial gradient, then take the
        # MEDIAN per-pixel velocity (robust to outliers).
        # Runs at HALF resolution (4x cheaper); a half-res pixel of motion is
        # 2 full-res px, hence the *2 on v at the end.
        small = cv2.resize(gray_f32, None, fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(small, (5, 5), 0)
        if self._prev_blur is None or self._prev_blur.shape != blur.shape:
            self._prev_blur = blur
        dt_b = cv2.absdiff(blur, self._prev_blur)
        self._prev_blur = blur

        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        gmag = cv2.magnitude(gx, gy) * 0.125   # Sobel3 gain ~8 -> px units

        sel = (dt_b > max(thresh * 0.5, 2.0)) & (gmag > 1.0)
        if sel.sum() > 50:               # need enough pixels to be meaningful
            v = 2.0 * float(np.median(dt_b[sel] / gmag[sel]))
        else:
            v = 0.0                       # static scene -> slowest adaptation
        self._speed_ema = v if self._speed_ema is None else \
            (1 - ema_a) * self._speed_ema + ema_a * v

        # -- map speed -> alpha ------------------------------------------------
        t = np.clip((self._speed_ema - v_lo) / (v_hi - v_lo + 1e-6), 0.0, 1.0)
        alpha = a_min + t * (a_max - a_min)

        # warm-up: EMA state is unreliable for the first few frames
        if self._auto_frames < 10:
            alpha = float(p.get("iir_alpha", 0.05))
            thresh = int(p.get("iir_threshold", 12))
        return alpha, thresh

    # ── Stage 2: HSV ──────────────────────────────────────────────────────
    def hsv(self, frame, motion, p, panel=None):
        """Returns smoke_mask. If `panel` is given, draws the HSV overlay +
        legend onto it."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]

        sm = cv2.inRange(hsv,
            np.array([int(p["smoke_h_min"]), int(p["smoke_s_min"]), int(p["smoke_v_min"])], np.uint8),
            np.array([int(p["smoke_h_max"]), int(p["smoke_s_max"]), int(p["smoke_v_max"])], np.uint8)
        )

        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        sm = cv2.bitwise_and(sm, motion)
        sm = cv2.morphologyEx(cv2.morphologyEx(sm, cv2.MORPH_OPEN, ko), cv2.MORPH_CLOSE, kc)

        if panel is not None:
            panel[sm > 0] = (190, 190, 190)
            cv2.rectangle(panel, (4, h-28), (122, h-4), (18, 18, 18), -1)
            cv2.circle(panel, (14, h-15), 5, (190, 190, 190), -1)
            cv2.putText(panel, "Smoke", (24, h-10), _FONT, 0.44, (200, 200, 200), 1)

        return sm

    # ── Stage 3: Gabor ────────────────────────────────────────────────────
    def _bank_response(self, roi):
        """Mean |response| per orientation over a pixel-budgeted view of roi."""
        if self._gabor_gpu is not None:
            return self._gabor_gpu(roi)
        return bank_response_cpu(roi, self._gabor_bank)

    def gabor(self, gray_u8, smoke_mask, p, panel=None):
        h, w = gray_u8.shape
        verified = np.zeros((h, w), np.uint8)

        if not cv2.countNonZero(smoke_mask):
            if panel is not None:
                cv2.putText(panel, "SMOKE (Gabor verified)", (6, 22),
                            _FONT, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
            return verified

        ksize = max(5, int(p["gabor_ksize"]) | 1)
        self._refresh_gabor(ksize, p["gabor_sigma"], p["gabor_lambda"], p["gabor_gamma"])
        cv_thr = p["gabor_std_thresh"]   # NOTE: now a coeff-of-variation threshold

        cnts, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) < 300:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 4 or bh < 4:
                continue
            roi = gray_u8[y:y+bh, x:x+bw].astype(np.float32)

            resp = self._bank_response(roi)
            mean_r = float(resp.mean())
            cv_iso = float(resp.std()) / (mean_r + 1e-6)   # contrast-invariant isotropy

            if panel is not None:
                cv2.rectangle(panel, (x, y), (x+bw, y+bh), (80, 180, 220), 1)

            # smoke = isotropic texture; flat patch (mean_r ~ 0) is rejected
            if mean_r > 1e-3 and cv_iso <= cv_thr:
                roi_m = smoke_mask[y:y+bh, x:x+bw]
                verified[y:y+bh, x:x+bw] = cv2.bitwise_or(
                    verified[y:y+bh, x:x+bw], roi_m)

        if panel is not None:
            panel[verified > 0] = (190, 190, 190)
            cv2.putText(panel, "SMOKE (Gabor verified)", (6, 22),
                        _FONT, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
        return verified

    # ── Texture stats for one bbox (used by the motion classifier) ─────────
    def texture_stats(self, gray_u8, bbox, p) -> tuple[float, float]:
        """(mean_response, isotropy_cv) of the Gabor bank over a bbox.
        Smoke texture is isotropic -> LOW cv; directional structure (waves,
        foliage, cloud striations) -> HIGH cv. Returns iso = -1.0 when the
        patch is too small/flat to measure. Same bank as the gabor() stage,
        so this costs one extra set of filter2D calls per detection."""
        x, y, w, h = bbox
        if w < 4 or h < 4:
            return 0.0, -1.0
        ksize = max(5, int(p["gabor_ksize"]) | 1)
        self._refresh_gabor(ksize, p["gabor_sigma"], p["gabor_lambda"],
                            p["gabor_gamma"])
        roi = gray_u8[y:y+h, x:x+w].astype(np.float32)
        resp = self._bank_response(roi)
        mean_r = float(resp.mean())
        if mean_r < 1e-3:
            return mean_r, -1.0
        return mean_r, float(resp.std() / (mean_r + 1e-6))
    # ── One frame: headless binary verdict for smoke ───────────────────────
    def process_frame(self, frame, min_area: int = 40) -> bool:
        """Run all stages on one BGR frame -> smoke_present.
        No panels are drawn (optimizer path)."""
        h0, w0 = frame.shape[:2]
        if w0 > self.max_proc_w:
            frame = cv2.resize(
                frame, (self.max_proc_w, int(h0 * self.max_proc_w / w0)),
                interpolation=cv2.INTER_AREA)

        p = dict(self.params)
        gray_f32 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_u8 = cv2.convertScaleAbs(gray_f32)

        motion = self.iir(gray_f32, p)
        smoke_m = self.hsv(frame, motion, p)
        smoke_v = self.gabor(gray_u8, smoke_m, p)

        return cv2.countNonZero(smoke_v) > min_area
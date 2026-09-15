"""
gabor_cupy.py
===============================================================================
CuPy implementation of the Gabor filter bank used by wildfire_core.

Unified-memory notes (Jetson Orin)
----------------------------------
CPU and GPU share physical DRAM, so the cost to avoid is not the wire transfer
but the pageable staging copy cudaMemcpy makes behind a plain cp.asarray().
This module avoids it three ways:

  * inputs are staged through a pinned (page-locked) host buffer allocated once
    per ROI shape, so host->device is a mapped-memory copy with no staging,
  * the six responses are reduced to scalars on the device, so no response
    image is ever read back -- only 6 floats,
  * the kernel stack, device buffers and the pinned buffer are all built once
    and reused across calls.

Falls back cleanly: CUPY_AVAILABLE is False when CuPy is missing or no GPU is
visible, and callers keep using the OpenCV path.
===============================================================================
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.ndimage import correlate as cp_correlate
    CUPY_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    cp_correlate = None
    CUPY_AVAILABLE = False


class GaborBankCuPy:
    """Device-resident Gabor bank for a fixed ROI shape.

    Build once per shape, then call with a float32 ROI of that shape. Returns
    the same 6-vector of mean |response| that wildfire_core._bank_response
    produces on the CPU.
    """

    def __init__(self, bank, shape):
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy unavailable")
        self.h, self.w = shape
        self.npix = float(self.h * self.w)
        self.d_kernels = [cp.asarray(k, dtype=cp.float32) for k in bank]
        self.pinned = cp.cuda.alloc_pinned_memory(self.h * self.w * 4)
        self.host = np.frombuffer(self.pinned, np.float32,
                                  self.h * self.w).reshape(self.h, self.w)
        self.d_src = cp.empty((self.h, self.w), cp.float32)
        self.d_out = cp.empty(len(bank), cp.float32)

    def __call__(self, roi_f32):
        np.copyto(self.host, roi_f32)
        self.d_src.set(self.host)
        for i, k in enumerate(self.d_kernels):
            resp = cp_correlate(self.d_src, k, mode="mirror")
            self.d_out[i] = cp.abs(resp).mean()
        return cp.asnumpy(self.d_out)


class TiledGaborCuPy:
    """Applies the ROI pixel budget from wildfire_core, then runs the bank on
    each tile and averages -- the GPU counterpart of _bank_response."""

    def __init__(self, bank):
        self.bank = bank
        self._cache = {}

    def __call__(self, roi_f32):
        from wildfire_core import _gabor_tiles
        tiles = _gabor_tiles(roi_f32)
        acc = None
        for t in tiles:
            g = self._cache.get(t.shape)
            if g is None:
                g = self._cache[t.shape] = GaborBankCuPy(self.bank, t.shape)
            r = g(np.ascontiguousarray(t))
            acc = r if acc is None else acc + r
        return (acc / len(tiles)).astype(np.float32)


class HybridGaborBank:
    """Size-dispatched bank: GPU for ROIs large enough to be tiled, OpenCV for
    the rest.

    Measured on an Orin Nano, a CuPy call costs ~2.5 ms of launch and
    synchronisation overhead regardless of size, so it only pays off once the
    ROI is big enough to be worth it. Small blobs -- the common case -- finish
    on the CPU in ~1 ms and are dispatched there.
    """

    def __init__(self, bank, min_gpu_px=None):
        from wildfire_core import GABOR_MAX_ROI_PX
        self.bank = bank
        self.min_gpu_px = GABOR_MAX_ROI_PX if min_gpu_px is None else min_gpu_px
        self.gpu = TiledGaborCuPy(bank) if CUPY_AVAILABLE else None

    def __call__(self, roi_f32):
        if self.gpu is not None and roi_f32.size > self.min_gpu_px:
            return self.gpu(roi_f32)
        from wildfire_core import bank_response_cpu
        return bank_response_cpu(roi_f32, self.bank)

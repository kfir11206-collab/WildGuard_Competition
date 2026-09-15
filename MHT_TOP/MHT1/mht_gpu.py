"""
mht_gpu.py
===============================================================================
Vectorised and CuPy backends for the MHT association cost matrix.

The stock `MHTTracker._cost_matrix` is a Python double loop over
tracks x detections, each iteration calling `Track.gating_cost` which itself
loops over the hypothesis family. That is the O(tracks x dets x hyps) term and
it is interpreted Python, so it dominates everything at scale.

The maths is a per-pair 2x2 Mahalanobis:

    dx = det.cx - h.x[0],  dy = det.cy - h.x[1]
    (a, b, c, d) = (P00 + R00, P01, P10, P11 + R11)
    m = (d*dx^2 - (b+c)*dx*dy + a*dy^2) / (a*d - b*c)
    cost[track, det] = min over the track's hypotheses of m

which is a batched outer product -- exactly the shape that vectorises. Both
backends below produce a matrix numerically identical to the loop.

Memory note: the padded (T, D) work arrays are allocated once per hypothesis
slot, not per track, so peak extra memory is ~2 * T * D * 4 bytes rather than
T * max_hyps * D.

IMPORTANT: NumpyCostMixin is the control. Without it you cannot tell whether a
speedup came from the GPU or merely from not being a Python loop.
===============================================================================
"""

from __future__ import annotations

import numpy as np

from mht_tracker import MHTTracker, _DET_FLOOR

try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    CUPY_AVAILABLE = False


def _pack_hypotheses(tracks, max_slots):
    """Flatten track hypothesis families into slot-major arrays.

    Returns (slots, n_used) where slots[k] = (hx, hy, ha, hb, hc, hd, valid)
    for the k-th hypothesis of every track; `valid` masks tracks with fewer
    than k+1 hypotheses.
    """
    n_t = len(tracks)
    out = []
    for k in range(max_slots):
        hx = np.zeros(n_t, np.float32)
        hy = np.zeros(n_t, np.float32)
        ha = np.ones(n_t, np.float32)
        hb = np.zeros(n_t, np.float32)
        hc = np.zeros(n_t, np.float32)
        hd = np.ones(n_t, np.float32)
        valid = np.zeros(n_t, bool)
        any_valid = False
        for ti, t in enumerate(tracks):
            if k < len(t.hyps):
                kf = t.hyps[k].kf
                x = kf.x
                P, R = kf.P, kf.R
                hx[ti] = x[0]
                hy[ti] = x[1]
                ha[ti] = P[0, 0] + R[0, 0]
                hb[ti] = P[0, 1]
                hc[ti] = P[1, 0]
                hd[ti] = P[1, 1] + R[1, 1]
                valid[ti] = True
                any_valid = True
        if not any_valid:
            break
        out.append((hx, hy, ha, hb, hc, hd, valid))
    return out


def _mahal_block(xp, dx, dy, ha, hb, hc, hd):
    """(T, D) Mahalanobis^2 for one hypothesis slot. Mirrors _mahal2 exactly,
    including the determinant floor."""
    det = ha * hd - hb * hc
    det = xp.where(xp.abs(det) < _DET_FLOOR,
                   xp.where(det >= 0, _DET_FLOOR, -_DET_FLOOR), det)
    a = ha[:, None]
    bc = (hb + hc)[:, None]
    d = hd[:, None]
    return (d * dx * dx - bc * dx * dy + a * dy * dy) / det[:, None]


class NumpyCostMixin:
    """Vectorised CPU backend. The control for the GPU measurement."""

    def _cost_matrix(self, detections):
        n_t, n_d = len(self.tracks), len(detections)
        if not n_t or not n_d:
            return np.full((n_t, n_d), np.inf, np.float32)

        detx = np.fromiter((d.cx for d in detections), np.float32, n_d)
        dety = np.fromiter((d.cy for d in detections), np.float32, n_d)
        cost = np.full((n_t, n_d), np.inf, np.float32)

        for hx, hy, ha, hb, hc, hd, valid in _pack_hypotheses(
                self.tracks, self.max_hyps):
            dx = detx[None, :] - hx[:, None]
            dy = dety[None, :] - hy[:, None]
            m = _mahal_block(np, dx, dy, ha, hb, hc, hd)
            m = np.where(valid[:, None], m, np.inf)
            np.minimum(cost, m, out=cost)

        cost[cost > self.gate] = np.inf
        return cost


class CuPyCostMixin:
    """GPU backend. Falls back to the vectorised CPU path when CuPy is absent."""

    def _cost_matrix(self, detections):
        n_t, n_d = len(self.tracks), len(detections)
        if not CUPY_AVAILABLE or not n_t or not n_d:
            return NumpyCostMixin._cost_matrix(self, detections)

        detx = cp.asarray(np.fromiter((d.cx for d in detections), np.float32, n_d))
        dety = cp.asarray(np.fromiter((d.cy for d in detections), np.float32, n_d))
        cost = cp.full((n_t, n_d), cp.inf, cp.float32)

        for hx, hy, ha, hb, hc, hd, valid in _pack_hypotheses(
                self.tracks, self.max_hyps):
            g = [cp.asarray(a) for a in (hx, hy, ha, hb, hc, hd)]
            gvalid = cp.asarray(valid)
            dx = detx[None, :] - g[0][:, None]
            dy = dety[None, :] - g[1][:, None]
            m = _mahal_block(cp, dx, dy, g[2], g[3], g[4], g[5])
            m = cp.where(gvalid[:, None], m, cp.inf)
            cp.minimum(cost, m, out=cost)

        cost = cp.where(cost > self.gate, cp.inf, cost)
        return cp.asnumpy(cost)


class NumpyMHTTracker(NumpyCostMixin, MHTTracker):
    pass


class CuPyMHTTracker(CuPyCostMixin, MHTTracker):
    pass


BACKENDS = {
    "loop": MHTTracker,
    "numpy": NumpyMHTTracker,
    "cupy": CuPyMHTTracker,
}


try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_CUDA = torch.cuda.is_available()
except Exception:
    torch = None
    TORCH_AVAILABLE = False
    TORCH_CUDA = False


class TorchCostMixin:
    """PyTorch backend. `torch_device` selects cuda or cpu; cpu is a useful
    control because it isolates torch's kernels from the GPU itself."""

    torch_device = "cuda"

    def _cost_matrix(self, detections):
        n_t, n_d = len(self.tracks), len(detections)
        if not TORCH_AVAILABLE or not n_t or not n_d:
            return NumpyCostMixin._cost_matrix(self, detections)
        dev = self.torch_device
        if dev == "cuda" and not TORCH_CUDA:
            dev = "cpu"

        detx = torch.from_numpy(
            np.fromiter((d.cx for d in detections), np.float32, n_d)).to(dev)
        dety = torch.from_numpy(
            np.fromiter((d.cy for d in detections), np.float32, n_d)).to(dev)
        cost = torch.full((n_t, n_d), float("inf"), dtype=torch.float32, device=dev)

        for hx, hy, ha, hb, hc, hd, valid in _pack_hypotheses(
                self.tracks, self.max_hyps):
            t = [torch.from_numpy(a).to(dev) for a in (hx, hy, ha, hb, hc, hd)]
            tvalid = torch.from_numpy(valid).to(dev)
            dx = detx[None, :] - t[0][:, None]
            dy = dety[None, :] - t[1][:, None]
            det = t[2] * t[5] - t[3] * t[4]
            det = torch.where(det.abs() < _DET_FLOOR,
                              torch.where(det >= 0,
                                          torch.full_like(det, _DET_FLOOR),
                                          torch.full_like(det, -_DET_FLOOR)), det)
            m = (t[5][:, None] * dx * dx
                 - (t[3] + t[4])[:, None] * dx * dy
                 + t[2][:, None] * dy * dy) / det[:, None]
            m = torch.where(tvalid[:, None], m, torch.full_like(m, float("inf")))
            torch.minimum(cost, m, out=cost)

        cost = torch.where(cost > self.gate, torch.full_like(cost, float("inf")), cost)
        return cost.cpu().numpy()


class TorchMHTTracker(TorchCostMixin, MHTTracker):
    torch_device = "cuda"


class TorchCPUMHTTracker(TorchCostMixin, MHTTracker):
    torch_device = "cpu"


BACKENDS["torch"] = TorchMHTTracker
BACKENDS["torchcpu"] = TorchCPUMHTTracker

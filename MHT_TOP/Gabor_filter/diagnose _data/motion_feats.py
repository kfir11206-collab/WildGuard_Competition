"""Per-pair motion/appearance descriptor (14-D) to complement the 21-D Gabor vector."""
import numpy as np, cv2, sys
sys.path.insert(0, "/sessions/jolly-magical-lovelace/mnt/Gabor_filter")
from smoke_features import hsv_smoke_mask

MDIM = 14

def motion_features(prev_bgr, cur_bgr, cfg):
    pg = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(pg, cg, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    fx, fy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)
    mask = hsv_smoke_mask(cur_bgr, cfg) > 0
    h, w = cg.shape
    area = mask.mean()
    f = [area]
    # global flow stats
    f += [mag.mean(), mag.std()]
    mv = mag > 0.3
    f.append((fy[mv] < 0).mean() if mv.any() else 0.5)  # global upward fraction
    if mask.sum() > 20:
        mm, mfx, mfy = mag[mask], fx[mask], fy[mask]
        f += [mm.mean(), mm.std()]
        moving = mm > 0.3
        f.append((mfy[moving] < 0).mean() if moving.any() else 0.5)  # mask upward frac
        ang = np.arctan2(mfy[moving], mfx[moving]) if moving.any() else np.array([0.0])
        hist, _ = np.histogram(ang, bins=8, range=(-np.pi, np.pi))
        p = hist / max(hist.sum(), 1)
        p = p[p > 0]
        f.append(float(-(p * np.log2(p)).sum()))  # direction entropy
        ys, xs = np.nonzero(mask)
        cy, cx = ys.mean(), xs.mean()
        # expansion: flow projected on radial direction from centroid
        ry, rx = (ys - cy), (xs - cx)
        rn = np.sqrt(ry * ry + rx * rx) + 1e-6
        exp = ((fx[mask] * rx + fy[mask] * ry) / rn).mean()
        f += [float(exp), cy / h, cx / w,
              float(mask[:h // 2].mean()), float(mask[h // 2:].mean())]
    else:
        f += [0, 0, 0.5, 3.0, 0, 0.5, 0.5, 0, 0]
    return np.array(f, dtype=np.float64)

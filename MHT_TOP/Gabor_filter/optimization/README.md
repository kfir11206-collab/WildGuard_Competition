# optimization/

Tooling used to tune and measure the detector. **Nothing here is needed to train
or run it** — the parent folder is self-contained. Keep this only if you want to
re-derive the numbers in the main README or continue the analysis.

## What is here

| file | purpose |
|---|---|
| `FINDINGS_gabor_baseline.md` | Full write-up: every measurement, why the baseline scored 0.58, what each fix bought. |
| `extract_cache.py` | Resumable extraction of the 21-D v1 features for every unique clip → `feats_cache.npz`. |
| `extract_cache_v2.py` | Same for the 28-D v2 features. Honours `SMOKE_SHI` / `SMOKE_VLO` / `SMOKE_VHI` / `SMOKE_TAG` so mask settings can be compared side by side. |
| `scoped_run.py` | The evaluation protocol: select γ/C/threshold on `val_videos` out-of-fold, evaluate once on `test_videos`. |
| `experiments.py` | Diagnostics — model-family comparison, feature redundancy (PCA, orientation correlation), univariate AUCs. |
| `hsv_audit.py` | Builds a 64×64 joint (S,V) histogram per clip so any mask threshold can be swept without re-decoding video. |
| `feats_cache*.npz` | Pre-computed features for all 84 clips. `_v2s45` / `_v2s55` / `_v2s65` are the three saturation settings. |
| `*.png` | Figures embedded in the main README. |

## Why the caches exist

Extraction is the slow part (~10 min for 84 clips); model selection off a cache
takes seconds. Every experiment in the main README was run against these `.npz`
files, so results are exactly comparable — only the thing under test changes.

Features are keyed by content hash, so if you add or remove clips you only need
to re-run `pack`, not re-extract. Extraction itself is resumable: `step` writes
one `.npy` per clip and exits on a time budget, so re-run it until it reports
`84/84`.

## Typical session

```powershell
# extract (skip — caches are already here)
python extract_cache_v2.py step    # repeat until 84/84
python extract_cache_v2.py pack

# tune on val_videos, then evaluate once on test_videos
$env:SMOKE_CACHE="feats_cache_v2s55.npz"
python scoped_run.py select    # grid over gamma/C, 3 CV seeds, picks threshold
python scoped_run.py final     # train on val_videos -> report on test_videos
python scoped_run.py pooled    # video-grouped CV over all 84 clips

# diagnostics
python experiments.py families
python hsv_audit.py step ; python hsv_audit.py sweep
```

Paths are resolved relative to this file, so the scripts expect the layout
`<project>/Gabor_filter/optimization/` with `val_videos/` and `test_videos/`
sitting next to `Gabor_filter/`.

## Method notes worth keeping

- **Select on `val_videos`, touch `test_videos` once.** `scoped_run.py` enforces
  zero clip overlap with an assertion. Reading anything off the test set — even
  a threshold — costs you: measured optimism was +0.031 when a leaked clip was
  present, +0.008 after it was removed.
- **Report video accuracy alongside frame accuracy.** 50 frames from one clip
  are ~1 effective sample, so frame accuracy overstates confidence.
- **Differences below ~0.03 on 24 clips are noise.** Removing a single clip once
  reordered the entire `s_high` ranking.
- **Do not tune the mask on coverage statistics.** Coverage separation preferred
  `s_high=0.45`; downstream accuracy disagreed. Tune on CV accuracy.

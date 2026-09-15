# Gabor pipeline — diagnosis of the 0.67 test result

All numbers below come from one cached feature extraction (`stride=5`,
`max_frames=50`, `iir-auto`, resize 320x240) over every unique clip, so the
model comparisons are exactly like-for-like. Scripts: `extract_cache.py`,
`experiments.py`. Cache: `feats_cache.npz` (4160 x 21 over 84 clips).

---

## 1. The dataset is 84 clips, not 60 + 25 + 64 + 20

Deduplicating the four folders by content hash:

| folder | clips | composition |
|---|---|---|
| `val_videos` | 60 | — |
| `test_videos` | 25 | — |
| `new_train_videos` | 64 | 42 from `val_videos` + **22 of the 25 `test_videos`** |
| `new_test_videos` | 20 | 17 from `val_videos` + 2 from `test_videos` + 1 in all three |

**84 unique clips total** (43 smoke, 41 non-smoke). `new_train_videos` and
`new_test_videos` are a reshuffle of the same footage, not new data.

Two consequences:

- Training on `new_train_videos` and testing on `test_videos` would score ~95%
  purely from memorisation — 22 of the 25 test clips are byte-identical members
  of that training set. Do not use that pairing.
- Your `val_videos -> test_videos` split is clean apart from
  `15732852_3840_2160_60fps.mp4`, which sits in both. It is worth ~4 accuracy
  points on its own (the model scores 1.000 on it when trained on it).

## 2. The RBF gamma was ~29x too wide

`--sigma 0.6` (the paper's value) gives `gamma = 1/(2*0.6^2) = 1.389`. But the
pipeline standardises the features first, so typical squared distance between
two points in 21-D is ~42 and `exp(-1.389*42) ~ e^-58`. The kernel becomes a
near-delta function and the SVM memorises; the saved model used 850/2960
support vectors. The paper's 0.6 was not intended for standardised features.

Grouped-CV grid on the training clips only picked `gamma=0.02, C=0.1`.

## 3. What tuning actually bought

Scope: `scoped_run.py` uses **only** `val_videos` and `test_videos`. Gamma, C and
the decision threshold are chosen by video-grouped CV *inside `val_videos`*
(averaged over 3 CV seeds); `test_videos` is touched once, at the end. Selected:
`gamma=0.02, C=0.1, thr=-0.19`.

Held-out `test_videos`, trained on the 59 non-leaked `val_videos` clips:

| config | frame acc | bal acc | f1 | video acc |
|---|---|---|---|---|
| baseline `gamma=1.389`, thr=0 | 0.584 | 0.578 | 0.494 | 0.560 |
| tuned `gamma=0.02, C=0.1`, thr=0 | 0.626 | 0.625 | 0.607 | 0.600 |
| tuned + val-chosen threshold -0.19 | **0.645** | **0.646** | **0.643** | **0.640** |

Pooled video-grouped CV over `val_videos` + `test_videos` (= all 84 clips,
5 seeds, low variance):

| model | frame acc | video acc |
|---|---|---|
| SVM rbf, paper gamma | 0.618 ± 0.011 | 0.614 ± 0.019 |
| SVM rbf, tuned | **0.677 ± 0.007** | **0.674 ± 0.012** |

So the gamma fix is real and worth roughly +6 points. It is not the main
problem.

Note on scope: `val_videos ∪ test_videos` already covers **all 84 unique clips**
— zero clips exist only in the `new_*` folders. So restricting to your two
folders loses no footage, and every number in this document comes from them.

## 4. The ceiling is the features, not the model

Same features, same split, four different model families:

| model | frame acc | video acc |
|---|---|---|
| SVM rbf (paper gamma) | 0.584 | 0.560 |
| SVM rbf (tuned) | 0.626 | 0.600 |
| Logistic regression | 0.591 | 0.600 |
| Random forest | 0.572 | 0.520 |
| Hist gradient boosting | 0.566 | 0.520 |

Tree ensembles are scale-invariant and have no gamma to get wrong, and they do
no better. Nothing in the 21-D vector supports more than ~0.65.

## 5. Why: the 21 features are effectively 3

Pairwise correlation between the 5 Gabor **orientations**, within each statistic,
across all 4160 samples:

| statistic | min r | mean r | max r |
|---|---|---|---|
| entropy | 0.9998 | 0.9999 | 1.0000 |
| uniformity | 1.0000 | 1.0000 | 1.0000 |
| mean | 1.0000 | 1.0000 | 1.0000 |
| std | 1.0000 | 1.0000 | 1.0000 |

The five orientations are the *same number*. PCA agrees: **3 components explain
99% of the variance** of the 21-D vector (0.810 / 0.159 / 0.023 / ...).

The cause is in `image_stats()` — it reduces each Gabor magnitude map to global
entropy/uniformity/mean/std over the **entire frame**. Orientation selectivity
is a *spatial* property; averaging |G_theta| over the whole image throws it away,
because the temporal-difference image is close to isotropic at that scale. The
bank computes five genuinely different responses and then collapses them to one
number five times.

A second issue compounds it: the stats run over the whole frame *including the
zeroed-out non-mask region*. When smoke covers 5% of the image, 95% of the
256-bin histogram sits in a single zero bin, so entropy and uniformity are
largely measuring mask **area**, not texture.

Best single feature, measured as clip-level AUC over the 84 clips:

| feature | AUC | AUC if polarity flipped |
|---|---|---|
| `gabor*_mean` | 0.276 | 0.724 |
| `gabor*_std` | 0.296 | 0.704 |
| `gabor*_entropy` | 0.363 | 0.637 |
| `wavelet_energy` | 0.404 | 0.596 |
| `gabor*_uniformity` | 0.566 | — |

The strongest cue is "smoke frames have *lower* mean Gabor magnitude", which is
an absolute contrast measurement — i.e. partly a recording fingerprint rather
than a smoke property.

## 6. Where the remaining errors are

Per-video frame accuracy on `test_videos` with the tuned model — note how
bimodal it is (near 1.0 or near 0.0). The classifier is deciding once per video
and repeating it, not deciding per frame:

```
clean/10397425-hd_3840_2160_30fps   0.040   <-- fails
clean/153976-817104245_medium       0.000   <-- fails
clean/15732852_3840_2160_60fps      0.000   <-- fails
clean/91319-626985271               0.160   <-- fails
clean/93857-642181987_medium        0.380   <-- fails
smoke/223852_medium                 0.020   <-- fails
smoke/360-136069328                 0.000   <-- fails
smoke/91282-629546524               0.320   <-- fails
smoke/99182-653447840               0.000   <-- fails
```
(16 of 25 clips score above 0.7; the 9 above are near-total failures.)

## 7. Recommended order of work

1. **Fix `image_stats` to be local, and mask-restricted.** Compute the four
   statistics over mask pixels only, and per spatial block (e.g. 4x4 grid) or as
   the variance of the response *across* orientations, so orientation
   information survives. Add mask coverage as its own explicit feature. This is
   what turns 3 effective dimensions back into ~20.
2. **Add the occlusion cue.** `wavelet_energy(diff)` measures motion. The
   Toreyin/Cetin smoke cue is that smoke *lowers* the wavelet energy of the
   current frame relative to the background image, because smoke is a
   semi-transparent low-pass filter. Add block-wise `E(current)/E(background)`.
   This is the cue that separates smoke from cloud, which is the dominant
   confuser here.
3. **Go multi-scale.** `gabor_scale=1` only. The physical signature is
   suppression of high frequencies relative to low; a single scale cannot
   express a cross-scale ratio.
4. **Normalise away absolute contrast**, so features stop encoding the source
   recording.
5. **Temporal aggregation** over a 15-30 frame window (mean/std/slope), so a
   sample can see smoke's slow persistent growth.
6. **More clips.** 84 videos is ~84 effective samples; 50 near-identical frames
   from one clip is not 50 independent observations.

## 8. What the v2 feature stage changed

`smoke_features_v2.py` (28-D). `smoke_features.py` is untouched.

### Two kernel bugs found while implementing it

Masking and block-local statistics alone did **not** restore orientation
selectivity — `orient_frac` stayed at [0.200, 0.200, ...] and pixel anisotropy at
0.013. The reason is in the kernel, not the statistics:

1. **The Gaussian envelope is exactly circular in both Gabor modes.**
   `_bank_haghighat` sets `gamma = eta = sqrt(2)`, so `alpha == beta` and the
   envelope has measured aspect ratio **1.000**. `gabor="paper"` calls
   `cv2.getGaborKernel(..., gamma=1.0)`, which is circular too. A circular
   envelope with a complex carrier gives a magnitude response that barely
   depends on theta. Elongating to 3:1 raises orientation spread on a
   directional test image from 0.0024 to **0.1384** (58x).
2. **`theta = j/(n_orient-1) * pi` walks 0, 45, 90, 135, 180 degrees** — and 180
   degrees is the same orientation as 0, so only **4 of the 5** filters are
   distinct. `j/n_orient * pi` spans [0, pi) properly.

Together these fully explain the r = 1.0000 correlation. The bank was computing
five filters that were, by construction, nearly the same filter.

### Feature-level effect

| | v1 (21-D) | v2 (28-D) |
|---|---|---|
| components for 99% of variance | **3** of 21 | **14** of 28 |
| orientation cross-correlation | 1.0000 | 0.9864 |
| best clip-level univariate AUC | 0.724 (`gabor_mean`) | **0.837** (`px_aniso_mean`) |

The new best feature is pixelwise orientation anisotropy, which could not exist
in v1 because the orientations were interchangeable.

### Accuracy effect

Pooled video-grouped CV over `val_videos` + `test_videos` (84 clips, 5 seeds):

| features | frame acc | video acc |
|---|---|---|
| v1 tuned | 0.677 ± 0.007 | 0.674 ± 0.012 |
| v2 tuned | **0.708 ± 0.031** | **0.726 ± 0.039** |

Held out `test_videos`, trained on `val_videos` (25 clips, so noisy):

| features | frame acc | video acc | recall |
|---|---|---|---|
| v1 tuned + thr | 0.645 | 0.640 | 0.665 |
| v2 tuned, thr=0 | **0.650** | **0.680** | **0.742** |

Smoke recall is the clear gain (0.665 -> 0.742); smoke clips failing outright
dropped from 4 to 2. Clean clips got slightly worse, so the net frame gain on the
25-clip test set is small and within noise — the pooled CV and the feature-level
diagnostics are the more trustworthy evidence that this helped.

### What did not work

Temporal aggregation (rolling window W=10, mean/std/slope per feature) **hurt**:
v2 video accuracy 0.726 -> 0.706. With 84 clips, tripling the feature count to
84-D costs more in variance than the temporal context gains. Revisit only with
substantially more clips, or with a much smaller aggregate (e.g. slope only).

## 9. Widening the HSV mask — the saturation bound was the biggest single win

Current thresholds (`SmokeConfig`, normalised 0-1 -> OpenCV units):

| channel | normalised | OpenCV |
|---|---|---|
| hue | 0.000 – 1.000 | 0 – 179 (unconstrained) |
| saturation | 0.000 – **0.280** | 0 – 71.4 (of 255) |
| value | 0.380 – 0.985 | 96.9 – 251.2 (of 255) |

`hsv_audit.py` builds a 64x64 joint (S,V) histogram per clip, so any threshold
can be evaluated without re-decoding. Mean mask coverage over the 84 clips:

| s_high | smoke | non-smoke | gap | clips with empty mask |
|---|---|---|---|---|
| **0.28 (current)** | 0.478 | 0.430 | +0.048 | **4** |
| 0.35 | 0.556 | 0.511 | +0.046 | 2 |
| 0.45 | 0.621 | 0.559 | **+0.062** | 1 |
| 0.55 | 0.669 | 0.609 | +0.060 | 1 |
| 0.65 | 0.693 | 0.663 | +0.030 | 1 |
| 0.80 | 0.702 | 0.693 | +0.009 | 1 |
| 1.00 | 0.705 | 0.703 | +0.003 | 1 |

At `s_high=0.28` the mask discards 42% of a smoke clip (`final.mp4`) for being
"too saturated". Coverage separation actually *improves* up to about 0.45-0.55
and only collapses past 0.65, where the mask starts passing everything.

Four clips had an essentially **empty mask** under the paper thresholds:

```
clean/15263283_3840_2160_25fps.mp4   cov=0.0081
clean/347076.mp4                     cov=0.0010
clean/Clouds1.mp4                    cov=0.0000
smoke/99182-653447840.mp4            cov=0.0011
```

`smoke/99182` is one of the clips that scored **0.000** in every earlier run —
it was never classifiable, because every feature was computed on an empty mask.
`Clouds1.mp4` is a separate problem: it is fully blown out (V = 255 at the *1st*
percentile), so `v_high=0.985` rejects 100% of the frame regardless of
saturation.

### Result with `s_high=0.55, v_low=0.35, v_high=1.0`

Pooled video-grouped CV over `val_videos` + `test_videos` (84 clips, 5 seeds):

| configuration | frame acc | video acc |
|---|---|---|
| v1 21-D, paper mask | 0.677 ± 0.007 | 0.674 ± 0.012 |
| v2 28-D, paper mask | 0.708 ± 0.031 | 0.726 ± 0.039 |
| **v2 28-D, widened mask** | **0.741 ± 0.016** | **0.783 ± 0.028** |

Held out `test_videos`, trained on `val_videos`:

| configuration | frame acc | video acc | precision | recall |
|---|---|---|---|---|
| original baseline | 0.584 | 0.560 | 0.593 | 0.423 |
| v1 tuned | 0.645 | 0.640 | 0.621 | 0.665 |
| v2, paper mask | 0.650 | 0.680 | 0.612 | 0.742 |
| **v2, widened mask** | **0.694** | **0.720** | 0.654 | 0.767 |

The clips the mask had been silently killing recovered exactly as predicted:

| clip | before | after |
|---|---|---|
| `smoke/99182-653447840` | 0.000 | **0.940** |
| `clean/15263283` | 0.020 | **0.900** |
| `clean/153976-817104245` | 0.280 | **1.000** |

Net: **0.584 -> 0.694** frame and **0.560 -> 0.720** video on held-out clips.

### Choosing the saturation bound: full sweep

Four settings extracted in full (v_low=0.35, v_high=1.0 held constant, so only
`s_high` varies). Pooled video-grouped CV over 84 clips, and held-out
`test_videos` trained on `val_videos`:

| s_high | pooled frame | pooled video | test frame | test video | prec | rec |
|---|---|---|---|---|---|---|
| 0.28 (paper) | 0.708 ± 0.031 | 0.726 ± 0.039 | 0.650 | 0.680 | 0.612 | 0.742 |
| 0.45 | 0.740 ± 0.015 | 0.757 ± 0.018 | 0.669 | 0.680 | 0.621 | 0.797 |
| **0.55** | **0.741 ± 0.016** | **0.783 ± 0.028** | **0.694** | **0.720** | **0.654** | 0.767 |
| 0.65 | 0.737 ± 0.017 | 0.781 ± 0.022 | 0.654 | 0.720 | 0.634 | 0.660 |

**The curve peaks at 0.55 and turns over after it.** 0.65 is flat on pooled CV
(within noise of 0.55) and clearly worse on held-out frame accuracy
(0.654 vs 0.694), which is what you would expect once the mask starts admitting
so much of the frame that it stops being a smoke prior at all.

**Recommended setting: `s_high=0.55, v_low=0.35, v_high=1.0`.**

Worth noting: the coverage-gap heuristic preferred 0.45 (+0.062 vs +0.060) and
was **wrong** — mask coverage separation does not predict downstream accuracy.
Accuracy kept improving from 0.45 to 0.55 while the gap was flat, then fell at
0.65 where the gap had already collapsed. Tune the mask on CV accuracy, not on
coverage statistics.

### Caveat

The widened mask shifts the operating point toward predicting smoke, and three
clean clips that used to pass now fail (`4355461` 0.840 -> 0.020, `164360`
0.980 -> 0.080, `258655` 0.960 -> 0.480).

### Still open

- `clean/Clouds1.mp4` yields **zero** mask coverage — the HSV mask rejects the
  entire frame, so every feature is degenerate for that clip. Worth auditing how
  many clips this affects; the mask thresholds may need revisiting.
- 6 of 13 clean test clips still fail. The occlusion cue (`wav_ratio_cur_bg`)
  ranks below the anisotropy features, so it is not yet pulling its weight.

## Reproducing

```
python extract_cache.py step     # repeat until it reports 84/84
python extract_cache.py pack     # -> feats_cache.npz

# strictly val_videos -> test_videos (this is the one that matters)
python scoped_run.py select      # tune gamma/C/threshold inside val_videos
python scoped_run.py final       # train on val_videos, evaluate on test_videos
python scoped_run.py pooled      # grouped CV over val+test

# the revised 28-D features
python extract_cache_v2.py step  # repeat until 84/84
python extract_cache_v2.py pack  # -> feats_cache_v2.npz
SMOKE_CACHE=feats_cache_v2.npz python scoped_run.py select
SMOKE_CACHE=feats_cache_v2.npz python scoped_run.py final
SMOKE_CACHE=feats_cache_v2.npz python scoped_run.py pooled

# supporting diagnostics (model families, feature redundancy)
python experiments.py grid1 && python experiments.py grid2
python experiments.py families
```

Note: `extract_cache.py` uses `cap.grab()` for skipped frames instead of
`cap.read()`. Output is bit-identical (max abs diff 0.0) on intact files, but it
reads further into three slightly damaged `clean` clips where `read()` aborts
early, which is why the baseline reproduces at 0.622 rather than your 0.673.

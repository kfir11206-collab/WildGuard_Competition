# Wildfire Smoke Detection (MHT)

Detects **smoke** in video by running a full pipeline (IIR background subtraction → HSV colour → Gabor texture → Multiple-Hypothesis Tracking → group motion clustering) and classifying tracked groups as SMOKE vs. CLOUD/clean with a small logistic-regression model.

## Files

| File | What it does |
|------|--------------|
| `wildefire_gui.py` | Interactive desktop GUI — watch a **single video** run through every stage live |
| `train_motion_classifier.py` | **Train** the classifier on labeled videos → writes `motion_weights.json` |
| `test_motion_classifier.py` | **Test** a trained model on held-out videos (metrics or per-clip verdicts) |
| `wildfire_core.py` | Shared image-processing pipeline (used by GUI, train, and test) |
| `mht_tracker.py` | Multiple-Hypothesis tracker |
| `motion_classifier.py` | Group motion clustering + classifier |
| `motion_weights.json` | The trained model (auto-loaded by the GUI) |
| `feature_cache/` | Cached per-clip features, speeds up re-training |

## Setup

```bash
pip install PyQt5 opencv-python numpy PyWavelets
```

Python 3.10+ recommended.

---

## 1. Show a single video (GUI)

Watch one clip go through all four stages (Original · IIR Motion · HSV Colour · Gabor·Smoke). The GUI auto-loads `motion_weights.json` on startup.

```bash
# Play a specific video file
python wildefire_gui.py --video path/to/clip.mp4

# Use the webcam instead (device 0)
python wildefire_gui.py
```

---

## 2. Train

Point it at a parent folder with labeled subfolders. `fire/`, `smoke/`, `both/` are **positive** (produce smoke); `clean/` is **negative** (clouds / fog / nothing).

```
val_videos/
    fire/    *.mp4
    smoke/   *.mp4
    both/    *.mp4
    clean/   *.mp4
```

```bash
# Recommended — adaptive background + a specificity floor so the saved
# threshold lands at a balanced operating point (~0.5, no --thr needed at test)
python train_motion_classifier.py --folder path/to/val_videos --iir-auto --min-spec 0.75

# Basic training
python train_motion_classifier.py --folder path/to/val_videos
```

It runs leave-one-clip-out evaluation, prints a recall/specificity table, picks a decision threshold, and saves `motion_weights.json` next to the script. The GUI picks up the new model on next launch.

Useful flags:

- `--iir-auto` — adaptive IIR background mode. Recommended: it rescues clips the fixed mode finds nothing in (several smoke clips went from 0 → 10+ tracked groups with it on).
- `--min-spec S` — recall-first, but keep clean specificity ≥ S when picking the threshold (default 0.5). Use `0.75` so the auto-chosen threshold doesn't collapse to a low value like 0.28.
- `--max-frames N` — frames processed per clip (default 450, ≈ 18s @ 25fps)
- `--min-recall 0.9` — instead: best specificity subject to smoke recall ≥ 0.9
- `--fn-cost K` — cost-weighted threshold (a missed smoke clip is K× worse than a false alarm)
- `--major-life N` — min group lifetime (frames) to decide a clip (default 40). Higher = fewer short-blob false alarms.
- `--l2 X` — L2 regularization for the logistic fit (default 0.01).
- `--no-cache` — ignore `feature_cache/` and recompute features
- `--out FILE` — write the model somewhere other than `motion_weights.json`

**Tuning note (from a `l2 × major-life` grid search):** `--iir-auto --min-spec 0.75` with the defaults `major-life 40`, `l2 0.01` is the robust sweet spot. `major-life 25` shows a higher *peak* cross-validation score, but only at a fragile high threshold (~0.7) — and it still misclassifies the 164360 test clip (which scores 0.90 at life 25), so that peak doesn't generalize. `l2 0.03` is a reasonable alternative with a slightly flatter CV curve. `MAJOR_LIFE` also has a constant at the top of the script; the `--major-life` flag overrides it per-run. Retrain after any change.

---

## 3. Test

Evaluate the trained model on videos it hasn't seen. Two folder layouts are auto-detected:

**Labeled** (`test_videos/{fire,smoke,both,clean}/*.mp4`) → full metrics: recall, specificity, balanced accuracy, per-folder breakdown, and a list of misclassified clips.

**Flat** (`test_videos/*.mp4`, no subfolders) → a per-clip score + verdict table.

```bash
# Run the test set
python test_motion_classifier.py --folder path/to/test_videos

# Use a specific model file
python test_motion_classifier.py --folder test_videos --model motion_weights.json

# Dissect ONE clip — per-group scores and which features drove the decision
python test_motion_classifier.py --folder test_videos --inspect path/to/clip.mp4
```

The test runs with the exact pipeline params stored inside the model (`trained_with_params`), so results match how the model was evaluated during training. Override the threshold with `--thr` or the params with `--params` only if you know why.

---

## Typical workflow

```bash
# 1. Train on labeled clips (adaptive + specificity floor)
python train_motion_classifier.py --folder val_videos --iir-auto --min-spec 0.75

# 2. Check it on held-out clips
python test_motion_classifier.py --folder test_videos

# 3. Eyeball a single clip live
python wildefire_gui.py --video test_videos/smoke/clip.mp4
```

---

## Current model & results

Trained with `--iir-auto --min-spec 0.75`, `MAJOR_LIFE = 40`:

- **Held-out test set: 88.9% smoke recall, 88.9% clean specificity** (balanced 88.9%).
- **LOCO (cross-validation): 85.8% balanced**, threshold auto-set to ~0.49 — so no `--thr` override is needed at test time.

Known remaining failure modes (not fixed by the decision rule — different root causes):

- **Smoke fused with cloud:** when a plume touches the clouds behind it, the tracker groups them together and the blob reads as cloud, so the smoke is missed. Needs finer segmentation.
- **Cloud approaching the camera:** a cloud moving toward the lens grows and drifts upward in the 2D frame, mimicking a rising plume. Single-camera video can't separate "approaching" from "rising."
- **Windblown foliage / thin cloud edges:** their texture can fool the Gabor stage into reading smoke.

To debug any single clip, use `--inspect` (Test section) to see per-group scores and which features drove them.

"""Set the stored decision threshold in a saved smoke model without retraining.

Lower threshold  -> more smoke calls -> higher recall (fewer missed fires).
Usage:
    python set_threshold.py smoke.joblib -0.3
    python set_threshold.py smoke.joblib          # just print current value
"""
import sys, joblib

if len(sys.argv) < 2:
    sys.exit("usage: python set_threshold.py <model.joblib> [new_threshold]")

path = sys.argv[1]
blob = joblib.load(path)
meta = blob.setdefault("meta", {})
cur = float(meta.get("threshold", 0.0))
print(f"current threshold: {cur:+.3f}")

if len(sys.argv) >= 3:
    new = float(sys.argv[2])
    meta["threshold"] = new
    joblib.dump(blob, path)
    print(f"updated threshold: {new:+.3f}  (lower = higher recall)  -> saved {path}")

#!/bin/bash
ROOT=/home/student/Documents/projects/wildfire_detection
cd "$ROOT" || exit 1
python3 resident_vlm/wake_test.py --preflight "$@" || { echo; echo "pre-flight FAILED - nothing was started."; exit 1; }
OUT=$ROOT/resident_vlm/results/waketest_$(date +%Y%m%d_%H%M%S)
setsid nohup python3 -u resident_vlm/wake_test.py --out "$OUT" "$@" > "$OUT.log" 2>&1 < /dev/null &
echo
echo "pre-flight passed. detached as pid $!. safe to close this terminal AND VS Code."
echo "progress:  tail -f $OUT.log"

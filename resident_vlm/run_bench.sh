#!/bin/bash
# Detached benchmark run. Launch from a PLAIN terminal (not VS Code), then
# close VS Code so the desktop stops competing for RAM. Keep the desktop
# session logged in - the baseline arm renders to DISPLAY=:1, and container
# X access comes from xhost (see the xhost check below), not from an xauth file.
ROOT=/home/student/Documents/projects/wildfire_detection
cd "$ROOT" || exit 1

REPS=${REPS:-3}
fail=0
ok()   { printf "  %-24s %s\n" "$1" "$2"; }
bad()  { printf "  %-24s %s\n" "$1" "$2"; fail=1; }

echo "=== pre-flight ==="

if [ -S /tmp/.X11-unix/X1 ]; then ok "X display :1" "ok"
else bad "X display :1" "MISSING - keep the desktop session logged in"; fi

if xhost 2>/dev/null | grep -q "access control disabled"; then
  ok "xhost" "access control disabled"
else
  xhost +local:root >/dev/null 2>&1 && ok "xhost" "granted +local:root" \
                                    || bad "xhost" "could not grant - run: xhost +local:root"
fi

if [ -f resident_vlm/results/fused_state.pt ]; then
  ok "fused checkpoint" "$(du -h resident_vlm/results/fused_state.pt | cut -f1)"
else ok "fused checkpoint" "absent - daemon will rebuild it (+15s)"; fi

if [ -c /dev/video0 ]; then ok "camera" "ok"; else bad "camera" "MISSING /dev/video0"; fi

running=$(docker ps -q | wc -l)
if [ "$running" -eq 0 ]; then ok "containers" "0 (clean)"
else bad "containers" "$running running - stop them first"; fi

avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
if [ "$avail" -ge 6000 ]; then ok "RAM available" "${avail}MB"
else bad "RAM available" "${avail}MB - close VS Code/browser (need >=6000)"; fi

mode=$(cat /var/lib/nvpmodel/status 2>/dev/null)
cap=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null)
if [ "$mode" = "pmode:0001" ] && [ "$cap" = "1344000" ]; then
  ok "power mode" "25W applied (verified live)"
else bad "power mode" "$mode cap=$cap - expected pmode:0001 / 1344000"; fi

if [ "$fail" -ne 0 ]; then
  echo; echo "pre-flight FAILED - nothing was started."; exit 1
fi

LOG=$ROOT/resident_vlm/results/bench_$(date +%Y%m%d_%H%M%S).log
echo
echo "pre-flight passed. starting $REPS reps x 2 arms (~10 min/rep)."
setsid nohup python3 -u "$ROOT/resident_vlm/bench.py" --reps "$REPS" > "$LOG" 2>&1 < /dev/null &
echo "detached as pid $!. safe to close this terminal AND VS Code."
echo
echo "progress:  tail -f $LOG"

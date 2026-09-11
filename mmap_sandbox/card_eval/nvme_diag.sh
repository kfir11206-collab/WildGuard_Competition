#!/bin/bash
ROOT=/home/student/Documents/projects/wildfire_detection
MODEL=$(xargs < /sys/block/nvme0n1/device/model)
case "$MODEL" in
  *SDSQXFN*) LABEL=sd_express ;;
  *CT1000P5PSSD8*) LABEL=ssd ;;
  *) LABEL=unknown ;;
esac
ADDR=$(cat /sys/block/nvme0n1/device/address)
OUT=$ROOT/mmap_sandbox/results/card_eval/nvme_diag_${LABEL}_$(date +%Y%m%d_%H%M%S).txt
mkdir -p "$(dirname "$OUT")"
{
  echo "=== drive ==="
  echo "$MODEL  fw=$(xargs < /sys/block/nvme0n1/device/firmware_rev)  pcie=$ADDR"
  echo "=== kernel log (nvme / pcie / aer) ==="
  sudo dmesg -T | grep -iE 'nvme|pcie|aer'
  echo "=== card health ==="
  sudo nvme smart-log /dev/nvme0
  echo "=== card error log (non-empty entries only) ==="
  sudo nvme error-log /dev/nvme0 | grep -B2 -A4 -E 'error_count\s*: [1-9]' || echo "(no errors logged)"
  echo "=== autonomous power states ==="
  sudo nvme get-feature /dev/nvme0 -f 0x0c -H | head -24
  echo "=== power states / thermal limits / host memory buffer ==="
  sudo nvme id-ctrl /dev/nvme0 | grep -E '^(ps |wctemp|cctemp|hmpre|hmmin|mntmt|mxtmt|apsta)'
  echo "=== pcie link ==="
  sudo lspci -vvv -s "$ADDR" | grep -E 'LnkCap|LnkCtl|LnkSta|DevSta|UESta|CESta'
} > "$OUT" 2>&1
echo "saved $OUT"
grep -c 'completion polled' "$OUT" | xargs echo "'completion polled' lines in the kernel log:"

#!/bin/bash
# Burst vs continuous load time: 3 arms x 3 reps, interleaved.
#   resident             demand paging, 128KB faults - what ships today
#   resident_burst       64MB reads overlapped with the load
#   resident_continuous  whole file streamed first, then load
# --pace-mbps 100000 is a no-op pace: the sleep in _paced_preread never fires,
# so the continuous arm reads at whatever the drive gives it.
# To skip the reference arm and save ~15 min, drop "resident" from --arms.
# See RUNBOOK_LOADTIME.txt.
exec bash "$(dirname "$0")/run_bench.sh" \
  --reps 3 --arms resident resident_burst resident_continuous \
  --pace-mbps 100000 --prefetch-mb 64

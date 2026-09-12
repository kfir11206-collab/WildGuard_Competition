#!/bin/bash
exec bash "$(dirname "$0")/run_bench.sh" \
  --reps 3 --arms resident resident_burst resident_continuous \
  --pace-mbps 100000 --prefetch-mb 64

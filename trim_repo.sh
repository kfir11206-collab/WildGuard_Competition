#!/usr/bin/env bash
#
# trim_repo.sh — reduce the WildGuard tree to what the SDA report depends on.
#
# Run from inside the repository you want to trim:
#     bash trim_repo.sh            # show what would be removed, change nothing
#     bash trim_repo.sh --apply    # actually remove it
#
# Refuses to delete anything unless every path the report cites is present,
# so a wrong working directory or a partial clone fails loudly instead of
# quietly removing the wrong tree.

set -euo pipefail

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

# ---------------------------------------------------------------- must exist
# Every figure and table in the report traces to one of these.
KEEP=(
  "mmap_sandbox/results/card_eval/sd_express_20260911_211324"   # Table A.1, card
  "mmap_sandbox/results/card_eval/ssd_20260912_004218"          # Table A.1, SSD
  "mmap_sandbox/results/perf_burst_vs_continuous"               # 3.3 and Figure 2
  "resident_vlm/results/bench_20260911_220613"                  # Table 1, card
  "resident_vlm/results/bench_20260912_013746"                  # Table 1, SSD
  "resident_vlm/results/waketest_20260913_150739"               # Table 2, card
  "resident_vlm/results/waketest_20260913_154137"               # Table 2, SSD
  "mmap_sandbox/card_eval/HANDOFF.md"
  "REPORT_SOURCES.md"
  "README.md"
  "fire_watcher"
  "resident_vlm/bench.py"
)

missing=0
for p in "${KEEP[@]}"; do
  if [[ ! -e "$p" ]]; then
    echo "MISSING required path: $p" >&2
    missing=1
  fi
done
if (( missing )); then
  echo >&2
  echo "Refusing to trim. Are you in the repository root?" >&2
  exit 1
fi

# stall probes and drive diagnostics are globs, so check they exist at all
shopt -s nullglob
probes=( mmap_sandbox/results/card_eval/stall_probe_* )
diags=(  mmap_sandbox/results/card_eval/nvme_diag_* )
if (( ${#probes[@]} == 0 || ${#diags[@]} == 0 )); then
  echo "Refusing to trim: stall probes or nvme diagnostics are missing." >&2
  exit 1
fi

# ---------------------------------------------------------------- remove
REMOVE=(
  # Superseded: July cold-start design, replaced by the resident architecture
  "mmap_sandbox/results/burst_vs_continuous_run1"
  "mmap_sandbox/results/burst_vs_continuous_lowpace.log"
  "mmap_sandbox/results/wake_latency"

  # Superseded September benches (each carries its own SUPERSEDED.txt)
  "resident_vlm/results/bench_20260912_225203"
  "resident_vlm/results/bench_20260913_012802"
  "resident_vlm/results/bench_20260913_030701"
  "resident_vlm/results/bench_20260913_125256"

  # Earlier development runs, not cited
  "resident_vlm/results/bench_20260911_115331"
  "resident_vlm/results/bench_20260911_124406"
  "resident_vlm/results/bench_20260911_153617"
  "resident_vlm/results/bench_20260828_140619"
  "resident_vlm/results/bench_20260828_153513"
  "resident_vlm/results/dryrun_baseline"
  "resident_vlm/results/smoke_continuous"

  # mmap characterisation: supports the showcase, not the report
  "mmap_sandbox/results/beta_madvise"
  "mmap_sandbox/results/beta_scale"
  "mmap_sandbox/results/beta_wake"
  "mmap_sandbox/results/full_model_hysteresis"
  "mmap_sandbox/results/mht_integration_run1"
  "mmap_sandbox/results/seq_read_rerun"
  "mmap_sandbox/results/rand_read_sweep"
  "mmap_sandbox/results/write_curve"
  "mmap_sandbox/results/sustained_rate"
  "mmap_sandbox/results/20260705_103333_sweep"
  "mmap_sandbox/results/20260705_103652_sweep"

  # Scratch
  "patches"
  "tasks.txt"
  "tree_project"          # stale `tree` dump from another machine; layout no longer matches

  # Internal notes: working briefs, not documentation. Keep these in a private clone.
  # Before deleting CLAUDE.md, move its section 5 (directory map) into README.md.
  "REPORT_HANDOFF.md"
  "CLAUDE.md"
)

echo "=== paths to remove ==="
total=0
for p in "${REMOVE[@]}"; do
  if [[ -e "$p" ]]; then
    sz=$(du -sh "$p" 2>/dev/null | cut -f1)
    printf '  %-62s %s\n' "$p" "$sz"
    (( total++ )) || true
  else
    printf '  %-62s (absent)\n' "$p"
  fi
done

echo
echo "before: $(du -sh . | cut -f1)"

if (( APPLY )); then
  for p in "${REMOVE[@]}"; do
    rm -rf -- "$p"
  done
  echo "after:  $(du -sh . | cut -f1)"
  echo
  echo "Removed $total paths. Note that this does not shrink the git history —"
  echo "see the history section of TRIM_MANIFEST.md."
else
  echo
  echo "Dry run: nothing was changed. Re-run with --apply to remove."
fi

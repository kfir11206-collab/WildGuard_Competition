# Repository trim — what was removed and why

Applied by `trim_repo.sh`. Tree size before: 133 MB. After: 51 MB.

The rule used: keep everything the SDA report cites, everything needed to run the
system, and the documentation that explains both. Remove development history that
no figure or table depends on.

---

## Kept, and what depends on it

| Path | Report dependency |
|---|---|
| `mmap_sandbox/results/card_eval/sd_express_20260911_211324` | Table A.1, microSD Express column |
| `mmap_sandbox/results/card_eval/ssd_20260912_004218` | Table A.1, M.2 SSD column |
| `mmap_sandbox/results/card_eval/stall_probe_*` | the two excluded tests in §3.2 / Appendix A |
| `mmap_sandbox/results/card_eval/nvme_diag_*` | link width and speed, power states, Appendix E thermal counters |
| `mmap_sandbox/results/perf_burst_vs_continuous` | §3.3 and Figure 2 |
| `resident_vlm/results/bench_20260911_220613` | Table 1, microSD Express column |
| `resident_vlm/results/bench_20260912_013746` | Table 1, M.2 SSD column |
| `resident_vlm/results/waketest_20260913_150739` | Table 2, microSD Express column |
| `resident_vlm/results/waketest_20260913_154137` | Table 2, M.2 SSD column |

Also kept: all source (`fire_watcher/`, `resident_vlm/*.py`, `classifier/`, `src/`,
`nano_llm/`, `mmap_sandbox/card_eval/`, `mmap_sandbox/showcase/`), the compose files
and `Dockerfile.mht`, `README.md`, `REPORT_SOURCES.md`, both `PROVENANCE.md` files,
and `mmap_sandbox/card_eval/HANDOFF.md`.

---

## Removed

### Superseded design — the cold-start wake, replaced by the resident architecture
- `mmap_sandbox/results/burst_vs_continuous_run1` (29 MB) — July, whole-wake burst vs
  paced reads on the old design. Its ~150 s wake numbers describe an architecture that
  no longer exists and must not be quoted.
- `mmap_sandbox/results/burst_vs_continuous_lowpace.log`
- `mmap_sandbox/results/wake_latency` (3.5 MB) — phase breakdown of the same old wake.

### Superseded benches — each carried its own `SUPERSEDED.txt`
- `resident_vlm/results/bench_20260912_225203`
- `resident_vlm/results/bench_20260913_012802`
- `resident_vlm/results/bench_20260913_030701`
- `resident_vlm/results/bench_20260913_125256`

### Earlier development runs, not cited
- `resident_vlm/results/bench_20260911_115331`, `_124406`, `_153617`
- `resident_vlm/results/bench_20260828_140619`
- `resident_vlm/results/bench_20260828_153513` — the first clean resident result
  (156.6 s → 12.70 s). Correct at the time, but a different procedure from the
  September sittings, so it cannot be compared against them and is not quoted.
- `resident_vlm/results/dryrun_baseline`
- `resident_vlm/results/smoke_continuous` — the paced pre-read arm, dropped because the
  pre-read was evicted before it could be used.

### mmap characterisation — supports the showcase document, not the report
- `beta_madvise`, `beta_scale`, `beta_wake`, `full_model_hysteresis`,
  `mht_integration_run1`, `seq_read_rerun`, `rand_read_sweep`, `write_curve`,
  `sustained_rate`, `20260705_103333_sweep`, `20260705_103652_sweep`

These characterise raw read and write behaviour under the earlier cold-start design.
Their file sets vary — `sustained_rate` and `seq_read_rerun` read the full 2,813.7 MB
weight set (DistilBERT, VILA vision tower, VILA LLM) and reported 703 MB/s, while
`rand_read_sweep` and `beta_madvise` use a single 268 MB file — so their throughput
figures are not comparable either to each other or to the wake measurements in Table 1,
and none is quoted in the report.

Two are worth noting before deletion, in case they are wanted later:

- `beta_wake` holds `resident_cold`, `resident_prewarm`, `spawn_cold` and
  `spawn_prewarm` traces — the original resident-versus-spawn experiment that the
  current architecture descends from.
- `full_model_hysteresis` holds the lifecycle logs behind the 0.60 / 0.85 / 0.40
  thresholds shown in the Appendix B state machine. The thresholds themselves are
  defaults in `fire_watcher.py`, so nothing in the report depends on this directory,
  but it is the evidence for how they were chosen.

### Scratch
- `patches/`, `tasks.txt`
- `tree_project` — an 859-line `tree` dump captured on a different machine
  (`tesla@ubuntu`). The layout it shows no longer matches this repository, so it
  misleads rather than documents.

**Kept:** `important commands` — despite the name this is the operational runbook for
the system: start, monitor, tear down and recover, plus the gotchas that cost real time
(AF_UNIX socket path limit, `docker compose stop` taking 10–13 s with exit 137 being
normal, the OpenCV 4.8 / 5.0 split between system Python and the venv, and why
`fire_watcher` has no `mem_limit` because cgroup v2 charges mmap page cache to the
faulting container). Consider renaming it `RUNNING.md`.

### Internal notes
- `REPORT_HANDOFF.md` and `CLAUDE.md` — working notes rather than documentation. Kept in
  the authors' private repository. The directory map from `CLAUDE.md` has been moved into
  `README.md`.

`REPORT_SOURCES.md` and `card_eval/HANDOFF.md` are written as documentation and remain
published.

---

## Git history

Deleting files in a commit does **not** shrink the repository. The 133 MB remains in
history and anyone cloning still downloads it. Two options:

**Fresh repository, no history** — simplest, and appropriate for a submission:

```bash
rm -rf .git
git init
git add -A
git commit -m "WildGuard: SDA Student Competition 2026 submission"
git remote add origin <url>
git push -u origin main --force
```

**Rewrite history, keeping commits** — requires `git filter-repo`:

```bash
pip install git-filter-repo
git filter-repo --invert-paths \
  --path mmap_sandbox/results/burst_vs_continuous_run1 \
  --path resident_vlm/results/bench_20260828_153513
  # ...one --path per removed directory
```

Rewriting history changes every commit hash, so anyone with an existing clone must
re-clone. Do this only on the duplicate repository, never on a shared one.

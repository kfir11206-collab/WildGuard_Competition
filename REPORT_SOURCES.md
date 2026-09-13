# Sources for the project report

**Writing the report? Read `REPORT_HANDOFF.md` first** — the summary, the guardrails and what not to use.

A map of which file holds which piece of the work, ordered by how much each one can be
argued with: the comparisons first, then the single-drive characterisation that supports
them, with the mmap work last.

Every number in these files is measured on this project's Jetson Orin Nano Super
(JetPack R39.2, 25 W mode). The drive under test is always `/dev/nvme0n1`, which is
either the SD Express card or the SSD — check `device.json` in any run folder.

---

## Ordering principle

A number on its own persuades nobody; a number next to its alternative does. So the
comparisons come first, and the single-drive characterisation is evidence they draw on.

| # | Comparison | What is held constant |
|---|---|---|
| 1 | SD Express vs SSD | same board, same 19 tests, same 25 W settings |
| 2 | Resident wake vs cold start | same drive, same model, same trigger |
| 3 | Gated wake vs always-on | same 11.5 min window, two fire episodes |
| 4 | Burst vs steady reads | same bytes, same elapsed time |
| 5 | The card against itself | sequential vs random, cold vs warm, SLC vs TLC |

---

## 1. SD Express vs SSD — the only drive-against-drive comparison

| File | What it gives the report |
|---|---|
| `mmap_sandbox/card_eval/HANDOFF.md` | **The best single source.** Fairness rules, the side-by-side table, the three claims worth making, the energy trade and its mechanism, every caveat, and which runs do not count. ~30 KB, written to be read start to finish. |
| `mmap_sandbox/results/card_eval/sd_express_20260911_211324/analysis.json` | SD Express, 19 tests: sequential read 871 MB/s, sustained 870.6 MB/s, 80,114 random read IOPS, idle 2.539 W on the non-compute rails. |
| `mmap_sandbox/results/card_eval/ssd_20260912_004218/analysis.json` | The SSD through the identical suite: 2,585.7 MB/s, 95,254 IOPS, idle 3.034 W. |
| `resident_vlm/results/bench_20260911_220613/analysis.json` | Whole system on the card: cold start 117.3 s / 797.6 J, resident wake 11.18 s / 84.4 J (median of the two clean reps; `resident_0` hit the board's stall and is excluded). |
| `resident_vlm/results/bench_20260912_013746/analysis.json` | Same on the SSD: 109.41 s / 737.5 J and 10.24 s / 77.6 J. |
| `.../stall_probe_sd_express_20260911_141641/summary.json` and `.../stall_probe_ssd_20260912_021844/summary.json` | The 30-second stall, reproduced on **both** drives — the evidence it belongs to the board, not the card. |
| `mmap_sandbox/results/card_eval/nvme_diag_*.txt` | Drive health, power states, thermal counters, PCIe link state, before and after each sitting. |
| Published page | https://claude.ai/code/artifact/746549ba-ae80-48d2-a115-0dcac1a32518 (version 4, 2026-09-13: void rule, weights throughput, stall wording) |

The card's case in one line: it gives up 0.94 s and 6.8 J per wake, and saves 0.50 W
continuously, so the SSD only repays its idle draw above ~6,300 wakes a day.

---

## 2. Resident wake vs cold start — the design comparison

Measured on both drives, which makes it a comparison twice over.

| File | What it gives the report |
|---|---|
| `resident_vlm/results/bench_20260828_153513/analysis.json` | The first clean result: 12.70 s against 156.6 s, 78 J against 850 J — 12.3× faster, 10.9× less energy. |
| `resident_vlm/results/bench_20260911_220613/analysis.json` | September rerun with the drive-read phase separated out (36.7 J of the 84.4 J wake, clean reps). |
| `resident_vlm/RUNBOOK.txt` | What the benchmark does and every condition that must hold for its numbers to count. |
| `resident_vlm/bench.py`, `sampler.py`, `analyze.py`, `resident_daemon.py` | The harness, if the report describes the method. |
| `resident_vlm/results/smoke_continuous/` | Why the paced pre-read variant was dropped: it read the file twice. |
| Published pages | https://claude.ai/code/artifact/69559262-b224-4af8-8f08-c576d5a62574 · https://claude.ai/code/artifact/c72988e6-e5b9-4c89-ba68-262d76b44cc7 |

---

## 3. Gated wake vs always-on — the system-level energy comparison

This is where the 29% energy figure comes from, and it is a genuine A/B: the same
11.5-minute window with two fire episodes, run both ways.

| File | What it gives the report |
|---|---|
| `mmap_sandbox/results/perf_burst_vs_continuous/summary_stats.json` | The six paired metrics: energy 2.11 vs 2.98 Wh (−29%), average power 11.1 vs 15.7 W, T_j 58.0 vs 63.9 °C, RAM 5.2 vs 6.9 GB, CPU 38.7 vs 66.1%, GPU 29.4 vs 55.0%. |
| `.../tegrastats_burst.jsonl`, `.../tegrastats_continuous.jsonl`, `marks_*.json` | The raw power traces behind the timeline: gated returns to a ~5.5 W idle floor between fires while always-on holds 15–18 W. |
| `mmap_sandbox/showcase/figures/F5_system_metrics.png`, `F6_power_timeline.png` | Both figures, already drawn. |

---

## 4. Burst vs steady reads

Earlier experiments at different scopes found no advantage for bursts; the September 13 wake test
found burst reading faster than continuous on both drives:

| File | What it gives the report |
|---|---|
| `mmap_sandbox/results/burst_vs_continuous_run1/summary.json` + `latency.png`, `timeseries.png`, `summary_panel.png` | **July, whole wake.** 150.6 s burst vs 152.7 s paced at 703 MB/s vs 162.1 s at 350 MB/s — pacing the read barely moved anything, because the wake was CUDA-bound. |
| `mmap_sandbox/results/card_eval/*/analysis.json` → `burst_vs_continuous` | **September, drive level with fio.** Same bytes in the same time: energy per GB ratio 0.999 on the card, 1.000 on the SSD. The supplier's "good at bursts" claim is not visible at this level. |
| `mmap_sandbox/burst_vs_continuous/*.py` | How the July run worked. |
| `resident_vlm/results/waketest_20260913_150739/` (SD Express), `waketest_20260913_154137/` (SSD), `resident_vlm/RUNBOOK_WAKETEST.txt` | **September 13, the wake test — the one to use.** Model wake only, 5 burst + 5 continuous wakes per drive, first 3 valid count, all 20 valid. Burst 8.58 s / 65.2 J (SD Express) vs 7.18 s / 56.4 J (SSD); continuous 10.75 s / 76.2 J vs 8.36 s / 63.3 J. Published as https://claude.ai/code/artifact/78d26989-52a0-4f33-8110-94eb420ef68d. Check with `resident_vlm/wake_checks.py`. |
| `resident_vlm/READ_STRATEGIES.md` | **SUPERSEDED 2026-09-13, not used.** **Start here for the wake read methods.** Step by step what demand paging, 64 MB bursts and continuous reading each do, why they finished in that order, and why the SSD is only ~1.4 s faster. |
| `resident_vlm/results/bench_20260912_225203/` (SD Express), `bench_20260913_012802/` + top-up `bench_20260913_030701/` (SSD) + HANDOFF.md → "Wake read-strategy test" | **SUPERSEDED 2026-09-13, not used** (a shorter burst-vs-continuous test replaces it). **September, the real wake, uncapped.** SD Express: demand 11.73 s, burst 12.01 s, continuous 15.97 s (n=2, top-up pending). SSD: 10.25 / 10.69 / 13.63 s. Same order on both drives. Every method sends the drive ~128 KB requests a few at a time; continuous cannot fit the file in RAM and reads it twice. Stall-hit reps excluded with `resident_vlm/wake_checks.py`. |

---

## 5. The SD Express against itself — characterisation

Not a device comparison, but the section that explains *why* the card behaves as it does,
and it contains three internal comparisons worth quoting.

| File | What it gives the report |
|---|---|
| `mmap_sandbox/showcase/SD_EXPRESS_SHOWCASE.md` | Sequential read 720 ± 13 MB/s over five cold runs; **SLC burst 546 MB/s vs folded TLC ~200 MB/s**; **sequential vs random 15× at 4 KB**; **cold 763 MB/s vs warm page cache 2,647 MB/s**. |
| `mmap_sandbox/showcase/figures/F1…F4*.png` | The figures for those sections. |
| `mmap_sandbox/showcase/bench_*.py`, `make_figures.py` | How each figure was produced. |

**Caution:** the showcase's §7 wake-latency numbers (~118 s wake, 150.6 s burst) describe the
**old cold-start design**. Section 2 above replaced it with an 11.26 s wake. Use §7 only as
the "before" state, or the report will contradict itself.

---

## 6. mmap characterisation — lowest priority

| File | What it gives the report |
|---|---|
| `mmap_sandbox/results/beta_madvise/` + `beta_comparison.png` | The one that matters: cold sequential ~700–760 MB/s whatever the advice, `MADV_RANDOM` collapsing to 56 MB/s with 65 k major faults, warm cache 2.6 GB/s. |
| `mmap_sandbox/results/seq_read_rerun/`, `rand_read_sweep/`, `write_curve/`, `sustained_rate/` | Raw JSON behind showcase sections 2–4. |
| `mmap_sandbox/results/20260705_*_sweep/` + `sweep.png` | Chunk-size sweep, 1–64 MB: all within noise. |
| `mmap_sandbox/mmap_raw_io/`, `model_beta_bench/` | The benchmark scripts. |

---

## Context files worth including

| File | Why |
|---|---|
| `README.md` | What EmberEye is: camera → VLM → classifier, the two containers, the hardware. |
| `mmap_sandbox/results/PROVENANCE.md` and `resident_vlm/results/PROVENANCE.md` | Which drive produced which results, and the cutoff date that separates them. |
| `CLAUDE.md` | The directory map, if the report needs to explain the repository layout. |

---

## Reproducing any number

    python3 mmap_sandbox/card_eval/summarize.py mmap_sandbox/results/card_eval/<run>
    python3 resident_vlm/analyze.py resident_vlm/results/<run>

    python3 resident_vlm/wake_checks.py resident_vlm/results/<run> [<top-up run>]

The SSD benchmark's raw samples, once missing, were recovered on 2026-09-13; every
committed run can now be recomputed from its samples.

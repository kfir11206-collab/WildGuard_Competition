# Sources for the project report

A map of which file holds which piece of the work, in priority order for the report:
SD Express performance and strengths first, then SD vs SSD, then the resident VLM,
then burst vs continuous, with mmap last.

Every number in these files is measured on this project's Jetson Orin Nano Super
(JetPack R39.2, 25 W mode). The drive under test is always `/dev/nvme0n1`, which is
either the SD Express card or the SSD — check `device.json` in any run folder.

---

## 1. SD Express performance and strengths — highest priority

| File | What it gives the report |
|---|---|
| `mmap_sandbox/showcase/SD_EXPRESS_SHOWCASE.md` | **Start here.** The card's case in seven sections: sequential read 720 ± 13 MB/s over 5 cold runs, the SLC write cliff (546 → ~200 MB/s), sequential vs random as the dominant lever (15× at 4 K), page-cache pre-warm reaching 2,647 MB/s, and a 29% system energy win from sleeping the model on the card. |
| `mmap_sandbox/showcase/figures/F1…F7*.png` | The seven figures that document references, ready to drop into a report. |
| `mmap_sandbox/results/card_eval/sd_express_20260911_211324/analysis.json` | The current standard-tool numbers (fio/iostat/gdsio), 19 tests, September sitting. |
| `mmap_sandbox/card_eval/HANDOFF.md` § "The energy trade" | Why the card wins on energy despite losing on bandwidth, and the honest mechanism behind it. |

**Caution for the writer:** the showcase's §7 wake-latency numbers (~118 s wake,
150.6 s burst) describe the **old cold-start design**. The resident VLM work replaced
that with an 11.26 s wake. Use §7 only as the "before" state.

---

## 2. SD Express vs SSD — the September comparison

| File | What it gives the report |
|---|---|
| `mmap_sandbox/card_eval/HANDOFF.md` | **The single best source.** Fairness rules, the side-by-side table, the three claims worth making, every caveat, and which runs do not count and why. ~30 KB, written to be read start to finish. |
| `mmap_sandbox/results/card_eval/sd_express_20260911_211324/` | Official SD Express sitting — `analysis.json` is the summary, the rest is raw. |
| `mmap_sandbox/results/card_eval/ssd_20260912_004218/` | Official SSD sitting, same 19 tests. |
| `resident_vlm/results/bench_20260911_220613/analysis.json` | SD Express, full-system: cold start 117.3 s / 797.6 J vs resident wake 11.26 s / 84.9 J. |
| `resident_vlm/results/bench_20260912_013746/analysis.json` | Same on the SSD: 109.41 s / 737.5 J vs 10.24 s / 77.6 J. |
| `mmap_sandbox/results/card_eval/stall_probe_sd_express_20260911_141641/summary.json` | The 30-second stall on the card, with its verdict line. |
| `mmap_sandbox/results/card_eval/stall_probe_ssd_20260912_021844/summary.json` | The same stall on the SSD — the evidence it is the board's fault, not the card's. |
| `mmap_sandbox/results/card_eval/nvme_diag_*_2026091*.txt` | Drive health, power states, thermal counters and PCIe link state, before and after each sitting. |
| Published page | https://claude.ai/code/artifact/746549ba-ae80-48d2-a115-0dcac1a32518 — the finished comparison, already written up. |

---

## 3. Resident VLM — the wake-up redesign

| File | What it gives the report |
|---|---|
| `resident_vlm/RUNBOOK.txt` | What the benchmark does, how it is run, and every condition that has to hold for the numbers to be valid. |
| `resident_vlm/results/bench_20260828_153513/analysis.json` | The first clean result: 12.70 s vs 156.6 s, 78 J vs 850 J. |
| `resident_vlm/results/bench_20260911_220613/analysis.json` | The September rerun with read-phase energy separated out. |
| `resident_vlm/bench.py`, `sampler.py`, `analyze.py`, `resident_daemon.py` | The harness itself, if the report needs to describe the method. |
| `resident_vlm/results/smoke_continuous/` | Evidence for why the paced-pre-read variant was dropped (it read the file twice). |
| Published pages | https://claude.ai/code/artifact/69559262-b224-4af8-8f08-c576d5a62574 (The Eight-Second Wake) · https://claude.ai/code/artifact/c72988e6-e5b9-4c89-ba68-262d76b44cc7 (The Card Did Not Stall) |

---

## 4. Burst vs continuous

Two separate experiments, both worth reporting, with different conclusions:

| File | What it gives the report |
|---|---|
| `mmap_sandbox/results/burst_vs_continuous_run1/summary.json` + `latency.png`, `timeseries.png`, `summary_panel.png` | **July, on the cold-start wake.** Pacing the read barely moved wake latency: 150.6 s burst vs 152.7 s at 703 MB/s vs 162.1 s at 350 MB/s — the wake was CUDA-bound, not storage-bound. |
| `mmap_sandbox/burst_vs_continuous/*.py` | How that was run (`run_burst_vs_continuous.py`, `continuous_watcher.py`, `measure_sustained_rate.py`). |
| `mmap_sandbox/results/card_eval/*/analysis.json` → `burst_vs_continuous` | **September, at drive level with fio.** Same bytes in the same time, burst vs paced: energy per GB ratio 0.999 on the card and 1.000 on the SSD. The supplier's "good at bursts" claim is not visible at this level. |
| `mmap_sandbox/results/perf_burst_vs_continuous/` | The earlier perf-counter version of the same question. |

---

## 5. mmap characterisation — lowest priority

| File | What it gives the report |
|---|---|
| `mmap_sandbox/results/beta_madvise/` + `beta_comparison.png` | madvise/readahead sweep on real model files: cold sequential ~700–760 MB/s regardless of advice, `MADV_RANDOM` collapsing to 56 MB/s with 65 k major faults, warm cache 2.6 GB/s. |
| `mmap_sandbox/results/seq_read_rerun/`, `rand_read_sweep/`, `write_curve/`, `sustained_rate/` | The raw JSON behind showcase sections 2–4. |
| `mmap_sandbox/results/20260705_*_sweep/` + `sweep.png` | mmap chunk-size sweep (1–64 MB): all within noise. |
| `mmap_sandbox/mmap_raw_io/`, `model_beta_bench/` | The benchmark scripts themselves. |

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

One gap: the SSD benchmark's raw samples were never committed, so its energy figures
can be read from its stored `analysis.json` but not recomputed. Recovery steps are at
the end of `mmap_sandbox/card_eval/HANDOFF.md`.

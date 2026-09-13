# START HERE — writing the report

Written 2026-09-13 at the end of the testing phase, for the Claude that helps Kfir write the
competition report on Kfir's personal PC. That Claude has this repository as a zip and, if Kfir
uploads them, the two published pages as HTML files. **It cannot run anything on the Jetson,
and it cannot open claude.ai artifact links** — so everything needed is summarised here, with
the file that proves each number.

Read this file first. Then `REPORT_SOURCES.md` (the map of every result file). Open
`mmap_sandbox/card_eval/HANDOFF.md` only when a detail is missing — it is the long lab
notebook, ~800 lines, and parts of it describe a test that was dropped (see "Do not use").

---

## 1. Who and why

- **Kfir** wrote the VLM + classifier side, the integration, the resident wake and all the
  storage tests. Kfir is a beginner: explain step by step, define terms, and explain before
  changing anything.
- **Ori** owns everything under `MHT_TOP` (the motion/heat trigger, "MHT"). Never credit MHT
  work to Kfir.
- **The competition is run by the makers of the SD Express card.** The report must show the
  card is important to this project and performs very well for it. Kfir prefers optimistic
  sentences — every one must still be true to the measurements (section 6).

The project (EmberEye / WildGuard, `README.md`): a wildfire sensor on a Jetson Orin Nano Super.
A cheap motion/heat detector (MHT) watches the camera; when it fires, a vision-language model
(VLM) wakes, looks at the frame and writes a smoke verdict, and a classifier acts on it.

## 2. The focus Kfir set for the report (2026-09-13) — to develop together

Not "a benchmark anyone could run", but **what was unique about using the SD Express in this
project**. Directions Kfir named:

1. **Almost as good as the SSD despite being far less powerful** — one PCIe lane against four,
   a third of the sequential speed, yet within ~1 s on the real wake (sections 4–5).
2. **The low idle power suits the resident design specifically.** The sensor sleeps almost all
   the time with the model resident and its weights on the card; 0.50 W less at idle outweighs a
   few joules per wake.
3. More to find in the thinking session.

**First step Kfir wants: check the specifications of both drives** — what is printed on the
label and what each maker claims — and set them against the measurements. Useful facts already
measured (from `mmap_sandbox/results/card_eval/nvme_diag_*_20260913_*.txt`):

| | SD Express | SSD |
|---|---|---|
| Model (as reported by the drive) | SanDisk SDSQXFN, 456 GB | Crucial P5 Plus (CT1000P5PSSD8), 1 TB |
| PCIe link the drive supports | Gen3 (8 GT/s) ×1 | **Gen4 (16 GT/s)** ×4 |
| Link it runs at on this board | Gen3 ×1 — **its full rating** | Gen3 ×4 — **downgraded**, the board's M.2 slot is Gen3 |
| Max power, drive's own power-state table | 1.80 W | 8.25 W |
| Deepest sleep state, same table | 0.015 W | 0.005 W |

So the card runs at everything it was designed for, while the SSD's advertised Gen4 speeds are
out of reach on this board. Still to collect: the label and datasheet claims (sequential
read/write MB/s, IOPS, power, temperature range, endurance, price) for both.

## 3. Three tests, kept separate in the report

Numbers from different tests are never mixed; each answers its own question.

| | A. Drive evaluation | B. Full-system benchmark | C. Wake test |
|---|---|---|---|
| Question | How fast is each drive with standard tools? | What does the drive change in the whole sensor's wake? | Which way of reading the weights wakes the model fastest, on each drive? |
| When | SD Express 11 Sep, SSD 12 Sep | same sittings as A | both drives 13 Sep |
| What runs | fio, iostat, gdsio — 19 tests | trigger stops MHT, starts classifier, wakes the model; also a cold-start arm | wakes the model only; model loaded once, 10 wakes |
| Reads the weights by | — | demand paging (what the sensor ships) | burst and continuous, alternating |
| Runbook | `mmap_sandbox/card_eval/card_eval.py`, steps in `mmap_sandbox/card_eval/HANDOFF.md` | `resident_vlm/RUNBOOK.txt` | `resident_vlm/RUNBOOK_WAKETEST.txt` |
| Results | `mmap_sandbox/results/card_eval/sd_express_20260911_211324/`, `ssd_20260912_004218/` | `resident_vlm/results/bench_20260911_220613/` (SD), `bench_20260912_013746/` (SSD) | `resident_vlm/results/waketest_20260913_150739/` (SD), `waketest_20260913_154137/` (SSD) |
| Published page | One Lane Against Four | One Lane Against Four | Waking on One Lane |

Why B and C times differ (11.18 s vs 8.58 s on the card): B's trigger also stops the MHT and
starts the classifier (~2.6 s), and B reads the weights by demand paging.

## 4. The numbers

All at 25 W, Jetson Orin Nano Super, JetPack R39.2. "SD" = SD Express.

### A. Drive evaluation (fio, O_DIRECT, 30 s per test)

| | SD Express | SSD |
|---|---|---|
| Sequential read 1 MB, 8 in flight | 871.0 MB/s (88% of a Gen3 lane) | 2,585.7 MB/s (66% of Gen3 ×4) |
| Sustained read, 120 s | 870.6 MB/s, peak 84.8 °C, no throttling in the timed window | 2,563.3 MB/s, 68.8 °C |
| 4 KB random read, 32 in flight | 80,114 IOPS | 95,254 IOPS |
| 4 KB random write, one at a time | **18,247 IOPS** | 17,861 IOPS |
| Sequential write 1 MB, 8 in flight | 569.3 MB/s | 2,570.3 MB/s |
| gdsio into GPU memory (compat mode) | 878.3 MB/s | 2,621.5 MB/s |
| Idle, non-compute rails | **2.539 W** | 3.034 W |

The 0.495 W idle difference is an attribution (the drive sits in a rail group with other
parts); One Lane Against Four explains why it is read as the drive's PCIe link.

### B. Full-system benchmark

| | SD Express | SSD |
|---|---|---|
| Resident wake, trigger → verdict | **11.18 s** (median of 2) | 10.24 s (median of 3) |
| Energy per resident wake | **84.4 J** | 77.6 J |
| Cold start (old design) | 117.3 s, 797.6 J | 109.4 s, 737.6 J |
| Weights throughput (demand paging) | 292 MB/s | 340 MB/s |

One SD wake (`resident_0`, 15.12 s) was hit by the board's stall (a request held 3.8 s) and is
excluded under the void rule — Kfir's decision 2026-09-13, applied to the published page.
The SSD is 0.94 s / 6.8 J faster per wake (8%). Energy break-even: the SSD's idle costs
0.495 W × 86,400 s = 42,768 J/day, so it only uses less energy above ~6,300 wakes a day.
The resident design itself: ~11 s against ~117 s cold start, ~10× faster and ~9× less energy.

### C. Wake test (13 Sep) — first three valid wakes per method, medians

| | SD Express | SSD |
|---|---|---|
| Burst: trigger → verdict | **8.58 s**, 65.2 J | 7.18 s, 56.4 J |
| Continuous: trigger → verdict | 10.75 s, 76.2 J | 8.36 s, 63.3 J |
| Burst saves over continuous | **2.17 s, 11.0 J** | 1.18 s, 6.9 J |
| Weights throughput, burst | **285 MB/s** = 78% of the SSD's | 366 MB/s |
| Weights throughput, continuous | 210 MB/s | 292 MB/s |
| Burst adds to throughput | **+36%** | +25% |
| Weights phase / camera + first verdict (burst) | 5.92 s / 2.64 s | 4.61 s / 2.56 s |

All 20 measured wakes were valid (no request held > 1 s, all read ≥ 1.6 GB), no drive errors.
Energy trade with the idle figure from A: a burst wake costs the card 8.8 J more, repaid by
18 s of idle; the SSD wins on energy only above ~4,900 wakes a day.
Free memory before each sitting: SD 5,475 MB, SSD 5,730 MB.

**Throughput is the headline speed measure (Kfir's decision 2026-09-13):** the 1.69 GB weights file
(1,689,241,949 bytes) divided by the time from the wake command until the model holds it on the GPU —
`tput` in `resident_vlm/wake_checks.py`. The file counts once, so bytes read twice cost time without
adding throughput. It is the wake time expressed as MB/s, so it agrees with the times by construction.

**Burst** = a helper thread reads the 1.7 GB weights file in 64 MB pieces while the model loads
from it. **Continuous** = the whole file is streamed first, then the model loads. In both, the
operating system sends the drive requests of ≤ 128 KB, 1–3 at a time.

### The stall (a board fault, not the card)

Single 1 MB reads one at a time sometimes freeze up to 30 s on **both** drives, with the kernel
line `timeout, completion polled`; a tiny read once a second clears it. Blamed on the board's
`nvme.use_threaded_interrupts=1`. Evidence: `mmap_sandbox/results/card_eval/stall_probe_sd_express_20260911_141641/` and
`mmap_sandbox/results/card_eval/stall_probe_ssd_20260912_021844/`.

## 5. Published pages

| Page | Link | What it holds |
|---|---|---|
| **One Lane Against Four** | https://claude.ai/code/artifact/746549ba-ae80-48d2-a115-0dcac1a32518 | Tests A and B: throughput, idle power, energy break-even, thermals, the stall, full results, limits. Version 4 (2026-09-13) applies the void rule (11.18 s / 84.4 J), weights throughput (292 / 340 MB/s) and the corrected stall wording. |
| **Waking on One Lane** | https://claude.ai/code/artifact/78d26989-52a0-4f33-8110-94eb420ef68d | Test C (version 2: throughput as the headline): drive-activity traces from trigger to verdict, where the seconds go, burst vs continuous, idle trade, all 20 wakes, how C differs from B. |
| The Card Did Not Stall | https://claude.ai/code/artifact/c72988e6-e5b9-4c89-ba68-262d76b44cc7 | How the 117 s cold start became an ~11 s resident wake. |

## 6. Guardrails — what the data supports

**Supported, say it confidently:**
- The card nearly fills its lane (88%); on the real wake it is within 0.94 s (B) / 1.40 s (C)
  of an SSD with four lanes and ~3× the sequential speed.
- The wake is software-bound: 2.97× the bandwidth buys 1.16× the weights throughput in B.
- With bursts the card delivers 78% of the SSD's weights throughput (285 vs 366 MB/s) from 34% of its
  sequential speed (871 vs 2,586 MB/s).
- The card draws 0.50 W less at idle (attribution, see A); for a sleeping sensor that outweighs
  the few joules per wake below thousands of wakes a day.
- Burst beats continuous on both drives; on the card it saves 2.17 s and 11.0 J and adds 36%
  throughput, on the SSD 1.18 s, 6.9 J and 25% (C's counted medians).
- On single small writes the card is marginally ahead (18,247 vs 17,861 IOPS).
- The stall is the board's, reproduced on both drives.

**Not supported — do not write:**
- That the card is faster than the SSD, or uses less energy **per wake**.
- That bursts use a hardware strength of the card: the drive only ever receives ≤ 128 KB
  requests, and fio's burst-vs-steady test found no difference (ratio 0.999).
- That burst reading helps "in general" more on the card. Known but not featured (Kfir's
  choice): the continuous − burst gap shrank rep by rep as free RAM grew, and in the last pair
  the card's gap vanished. Quote the counted medians as measured, not as a law.
- That the SSD numbers are the SSD's rated performance — it runs at Gen3 on this board.
- Any figure from the dropped test below.

## 7. Do not use

- **The wake read-strategy "load-time" test** (demand / burst / continuous with `bench.py`):
  `resident_vlm/READ_STRATEGIES.md`, `resident_vlm/RUNBOOK_LOADTIME.txt`, run folders
  `bench_20260912_225203`, `bench_20260913_125256`, `bench_20260913_012802`,
  `bench_20260913_030701` (each has `SUPERSEDED.txt`), and the "Wake read-strategy test" section
  of `mmap_sandbox/card_eval/HANDOFF.md`. Kfir dropped it on 2026-09-13.
- `mmap_sandbox/showcase/SD_EXPRESS_SHOWCASE.md` §7 wake numbers (~118 s): the old cold-start
  design, usable only as the "before" state.
- Any "read speed" MB/s as the headline: `analyze.py`'s `sd_read_mb_s` (362 / 427 MB/s, a window
  average), and the drive-busy speed (`wake_checks.py` `MB/s` column, `analyze.py`
  `sd_read_mb_s_while_reading`: 437 / 540, 346 / 482, 599 / 723 MB/s). The busy speed is a
  diagnostic only — it flatters continuous reading, which streams uninterrupted and then reads
  much of the file again. Quote weights throughput instead (section 4).
- Throughput figures from different tests side by side (B's demand-paged 292 MB/s next to C's
  burst 285 MB/s): the conditions differ, so that comparison was never tested.

## 8. Words to define in the report

- **SD Express** — an SD card that talks PCIe/NVMe instead of the old SD bus.
- **PCIe lane** — one serial link, ~985 MB/s at Gen3; the card uses 1, the SSD 4.
- **fio** — the standard Linux storage benchmark. **QD / in flight** — requests queued at once.
- **Resident wake** — the model stays initialised while asleep; only its weights leave memory and
  are read back from the drive on wake. **Cold start** — load everything from scratch.
- **Demand paging** — the model reads each part of the weights file the moment it needs it.
- **Void wake** — excluded by the rule fixed before the run (a request held > 1 s, or < 1.6 GB read).
- **Weights throughput** — MB of model weights delivered into the model per second of loading.
- **J (joule)** — energy; W × s. **Break-even** — the number of wakes a day at which the SSD's
  cheaper wakes would pay for its higher idle power.

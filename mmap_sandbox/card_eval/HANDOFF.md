# SD Express vs SSD — handoff

Written 2026-09-11 on the SD Express card, for the Claude session that runs on the SSD.
**The SSD is a clone made 2026-09-10, so its Claude memory knows nothing from 2026-09-11.
This file is the source of truth. Read all of it before running anything, and save the
key points to memory.**

**Status 2026-09-13, 03:00: the card comparison is done and published; the wake read-strategy test has
run on both drives and is being topped up.** Kfir asked for a wake read-strategy test — demand paging vs
64 MB bursts vs continuous, uncapped, on the real resident wake. SD Express half 2026-09-12 22:52–23:45,
SSD half 2026-09-13 01:28–02:23. The board's rescued stalls voided 3 of the 18 reps, leaving three arms
at 2 valid reps, so a pre-registered **top-up** (2 more reps per arm, same code) runs on the SSD
2026-09-13 and **on the SD Express next time it is in the slot**. Go straight to the last section, "Wake
read-strategy test", and its "Top-up" part. The earlier SSD results are in "SSD sitting, 2026-09-12".

## What this is for

The project is entered in an SD-card competition. The competition managers asked for an
**SD Express card performance evaluation with benchmark results from standard tools — fio,
iostat or gdsio** (no block sizes or parameters specified). The card is the subject; the
Crucial P5 Plus SSD is the reference point, not the thing being promoted. Present the card
honestly: claims that do not survive scrutiny hurt more than a weaker number.

Kfir asked for the comparison to be **as fair as possible**. The rules that follow from that
are in "Fairness rules" below — do not relax them on the SSD side.

The two drives take turns in the same M.2 slot. Only one can be installed at a time.

| | SD Express (Gen3 x1, slot 0007:01:00.0) | SSD (Gen3 x4, slot 0004:01:00.0) |
|---|---|---|
| model string | `ST51271_WDC SanDisk SD Express SDSQXFN` | `CT1000P5PSSD8` |
| `device.json` label | `sd_express` | `ssd` |
| PCIe link | Gen3 **x1** (~985 MB/s ceiling) | expected Gen3 **x4** (slot supports it) |
| controller | DRAM-less, uses 128 MiB host memory buffer | DRAM cache |
| power states | PS0 1.80 W max; PS4 0.015 W after 2 s idle (APST) | read from `nvme_diag.sh` output |

## Fairness rules

1. **One identical sitting per drive**: RUNBOOK STEP 0 → STEP 3 (diagnostics, card_eval, both
   benchmark arms interleaved, diagnostics again), same commands, same flags, same order,
   VS Code closed, the desktop's background helpers killed, 25 W power mode (pre-flight checks it).
   The stall probe is run by Claude from VS Code on both drives, never right before STEP 0.
2. **Each card test starts from the drive's own resting temperature**: after a 20 s rest,
   card_eval waits until the drive is within 5 °C of the coolest reading it had during the
   idle window (at most 120 s). A drive that idles warmer is not penalized; heat carried over
   from the previous test is removed. Wait time and start temperature are recorded per test.
3. **Burst vs continuous is compared within each drive** (ratio), at the same average rate and
   the same request size. The fair pair is the stall-free 128 KB one-at-a-time pair; the 1 MB
   pair is kept because it exposes the stall (finding 1).
4. **Idle** is measured twice, with and without the sampler's 10×/s drive-temperature reads
   (they are NVMe admin commands and may keep the drive awake). For a drive-level comparison
   use "rest of board" = `VDD_IN − VDD_CPU_GPU_CV − VDD_SOC`, not whole-board power —
   whole-board idle drifts ~1 W between days on host rails.
5. **Do not expand the SSD partition** — the clone keeps the filesystem layout identical.
6. Prefer comparisons made **within one sitting** (burst/continuous, resident/baseline, % of the
   drive's own link ceiling) — they cancel day-to-day drift. The two drives necessarily run on
   different days; say so in the write-up.
7. **25 W must really be applied.** After a reboot the board comes up at MAXN_SUPER clocks while
   `/var/lib/nvpmodel/status` still says `pmode:0001`: `nvpmodel.service` fails at every boot
   (exit 234). Best explanation: the GPU's `tpc_pg_mask` reads 0 but every mode in the conf wants
   `TPC_PG_MASK 240`, so nvpmodel always wants a reboot, prompts with no one to answer, and quits
   before setting any clocks. 25 W and MAXN_SUPER differ ONLY in CPU max (1344000), GPU max
   (918000000) and EMC max (already 3199 MHz), so `set_25w_clocks.sh` sets exactly those — no
   nvpmodel, no reboot, lasts until the next boot. Checked 2026-09-11: `sudo nvpmodel -m 1` asks
   to reboot, and answering no prints `optMask is 1, no request for power mode` and changes nothing;
   the script then set CPU 1344000 / GPU 918000000 / EMC 3199000000. Every other 25 W setting
   (CPU min 729600, GPU min, GPU power control, EMC) already matched the live state, so the result
   equals 25 W except the two GPU gating masks, which read 0 whichever method is used. `card_eval.py`, `stall_probe.py` and
   `run_bench.sh` refuse to start unless CPU 1344000 and GPU 918000000. Every run's `host.json`
   records the CPU cap (and `gpu_max_hz` from the 2026-09-11 evening on) — check before using a run.
8. **Start from a rested drive** — nothing heavy for 30 min before STEP 0. A stopped STEP 1 is
   restarted from STEP 0 after 30 min, never straight away. The SD throttles above ~85 °C, so a
   warm start changes the sustained results (finding 8).
9. **Nobody else on the board during a sitting** — no MHT GUI, containers or other sessions.

## Tools (all in git)

| file | what it does | who runs it |
|---|---|---|
| `mmap_sandbox/card_eval/card_eval.py` | fio suite + gdsio, `iostat -x` running throughout, board power and drive temperature sampled per test, cool-start before each test. ~25–35 min (cool-down pauses vary). 8 GB O_DIRECT test file in `~/card_eval_work`, deleted at the end | Kfir, VS Code closed |
| `mmap_sandbox/card_eval/stall_probe.py` | reproduces the 1 MB QD1 stall (finding 1), prints a verdict. ~4 min, 2 GB file | Claude, VS Code open is fine |
| `mmap_sandbox/card_eval/nvme_diag.sh` | kernel log, SMART, error log, power states, PCIe link. Needs sudo, saves into `mmap_sandbox/results/card_eval/` | Kfir (Claude has no sudo) |
| `mmap_sandbox/card_eval/set_25w_clocks.sh` | sets the 25 W clock limits after a boot (fairness rule 7). Needs sudo | Kfir, after every boot |
| `mmap_sandbox/card_eval/summarize.py <run dir>` | per-test table, idle with/without temperature reads, burst-vs-continuous ratios, sustained timeline, stall seconds; writes `analysis.json` | Claude |
| `resident_vlm/run_bench.sh --reps 3 --arms baseline resident` | full-system benchmark: cold container start vs resident wake, time + energy to first verdict. ~45 min | Kfir, VS Code closed |
| `resident_vlm/analyze.py <run dir>` | per-window energy / throughput / thermal table | Claude |

`resident_vlm/RUNBOOK.txt` is the exact terminal procedure Kfir follows.

card_eval steps, in order: 8.6 GB sequential write laying out the test file; idle 60 s with
and 60 s without drive-temperature reads; sequential read 1M at QD8 and QD1; random read 4K at
QD32 and QD1; `burst_read` (128 MB bursts of 1M at QD8, 1 s gaps) and `continuous_read`
(paced to exactly burst_read's average); `burst_read_128k_qd1` and `continuous_read_128k_qd1`
(same 128 MB bursts and pacing, one 128K request at a time — stall-free on the SD, verified:
both 105.3 MB/s, slowest read 2.0 / 4.6 ms); `sustained_read` (120 s flat out); gdsio in three
transfer modes; then the four write tests (sequential 1M QD8/QD1, random 4K QD32/QD1). Writes
go last so their background housekeeping cannot disturb the reads.

## SD results

**Official SD sitting**: see "SD runs added after this file was written" at the bottom.
Compare the SSD against that sitting — it follows the fairness rules. Runs from 2026-09-11 that
do NOT count (kept in git as raw data):

| run | why it does not count |
|---|---|
| card_eval `145100`, `150838`, `151049`, `194739` | stopped part-way (fio's once-a-second progress line looked like a fault; now `--eta=never`) |
| card_eval `151124` | complete at 25 W, but started minutes after three stopped runs (drive pre-loaded) — secondary data |
| `bench_20260911_153617` | a second user started the MHT GUI mid-run (`baseline_2` suspect); run killed during `resident_2` |
| card_eval `200018` | complete, but at **MAXN_SUPER clocks** (cap 1728000, after the 16:27 reboot) and started 90 s after a stopped run — shows the throttling in finding 8, not comparable otherwise |

**SD first pass (committed; repeatability data only)** — run before the fairness rules existed:

- card_eval `mmap_sandbox/results/card_eval/sd_express_20260911_122548/`: seq read 1M QD8
  877.6 MB/s (89% of the x1 link); sustained 120 s flat at 880 MB/s while the card went
  64 → 80 °C; rand read 4K QD32 / QD1 87,972 / 6,403 IOPS; seq write 1M QD8 / QD1
  578.5 / 490.4 MB/s; rand write 4K QD32 / QD1 78,003 / 17,682 IOPS; gdsio ~877 MB/s in all
  three modes (**compat mode — never call it GPUDirect**). Its `seq_read_1m_qd1` (8.5 MB/s)
  and 1M burst/continuous were hit by the stall; no 128K pair, no cool-start.
- resident benchmark `resident_vlm/results/bench_20260911_124406/`: resident arm valid —
  trigger → verdict median 11.85 s, wake 89.0 J, drive-read phase 38.8 J at ~344 MB/s. Its
  baseline arm failed (xauth, finding 2).
- older clean SD run `bench_20260828_153513`: baseline 156.6 s / 850 J vs resident 12.70 s / 78 J.

**Stall probe** — `mmap_sandbox/results/card_eval/stall_probe_sd_express_20260911_141641/`:
verdict **STALLS, RESCUED**. 1M QD1 without help: 612.9 MB/s (slowest read 3.5 s) and
12.2 MB/s (slowest 15.4 s). With a 4K read on the same queue once a second: 687.0 and
424.2 MB/s (slowest 0.65 s and 0.98 s). 128K QD1 481.7, 256K QD1 591.1, 128K QD8 888.6 MB/s,
none slower than 10 ms. The spread between the two unassisted runs is expected — a 30 s window
may or may not catch a long stall — so compare verdicts and slowest reads, not MB/s.

**Diagnostics** — `mmap_sandbox/results/card_eval/nvme_diag_sd_express_20260911_140645.txt`
(after the first pass and the scratch stall experiments): five `completion polled` lines,
SMART clean (0 media errors, 0 error-log entries), PCIe link clean.

## Findings and caveats — read before comparing

1. **1 MB QD1 read stall = missed completion interrupt, proven.** Single 1 MB reads
   (8 × 128 KB NVMe commands completing together, then silence) sometimes wait up to 30 s,
   which is `nvme_core.io_timeout`. The kernel log shows `nvme nvme0: I/O tag N QID Q timeout,
   completion polled` — the card had already finished; the host only noticed when the timeout
   polled. A 4K read on the same queue once a second releases every stall within ~1 s. Not the
   cause: the sampler's temperature reads (A/B tested), PCIe ASPM (off), thermal (below
   WCTEMP), link errors (none), media errors (none). 128K/256K at QD1 and anything at QD8+ is
   fine. Suspect: kernel cmdline `nvme.use_threaded_interrupts=1` with edge-triggered MSI.
   **Not yet proven which side drops the interrupt** — the card not raising it vs the host
   missing it. The SSD in the same slot and kernel decides: run `stall_probe.py` and compare
   verdicts. If the SSD stalls too, it is the board and must not be counted against the card;
   if it does not, report it honestly as a card-plus-this-platform issue. The resident wake is
   unaffected. Do not change kernel parameters without asking Kfir (needs a reboot).
2. **Baseline arm needs `/tmp/.docker.xauth` to be a FILE.** `/tmp` is wiped every boot; if
   any container starts first, Docker creates a folder there and `wildfire_detection_fused`
   cannot start. `run_bench.sh` pre-flight creates the empty file and refuses if it is a
   folder (fix: `sudo rmdir /tmp/.docker.xauth`). `bench.py` prints compose errors and fails
   that arm immediately instead of waiting out the 420 s timeout. X access itself comes from
   `xhost +local:root`; the file only has to exist.
3. **The wake is software-bound, not card-bound.** The card reads 878 MB/s sequentially but
   the resident wake pulls only ~344–367 MB/s (mmap demand paging, 128 KB requests). Expect
   the SSD's four lanes to help the wake much less than 4×. This is the strongest honest point
   for the card: on one lane it may deliver a wake close to a four-lane SSD.
4. **Idle power is probably not where the card wins** — both drives drop to milliwatt states
   after 2 s idle. **Active power** is the likelier win: SD PS0 ceiling is 1.80 W — compare with
   the SSD's `ps 0` line from `nvme_diag.sh`, and compare energy per GB and wake energy.
5. **Thermals.** SD lifetime SMART: 334 min above warning temperature, 58 min above critical,
   thermal management T2 41,782 s. It runs hot, yet held full speed for 120 s at 80 °C. The
   before/after `nvme_diag.sh` pair gives each drive's throttle-counter delta for its sitting.
6. **`resident_continuous` benchmark arm dropped.** Its pre-read gets evicted because the GPU
   copy of the weights shares RAM with the page cache, so it read the file twice. Code kept
   (off by default); burst vs continuous is measured by fio instead.
7. **Everything that matters for I/O is on the drive in the slot**, including Docker's image
   store (`/var/lib/docker`, 92 GB image) and the model weights (bind-mounted `hf-cache`).
   The SSD is a clone, so it serves the identical workload.
8. **The SD slows itself down above its 84.85 °C warning temperature.** In `200018` (full-power
   clocks, warm start, resting at 57.9 °C) the 120 s sustained read fell from ~879 to ~640 MB/s
   in every 10 s window where the card passed ~85 °C. In `151124` (25 W, resting 54.9 °C) it
   peaked at 82.8 °C and held ~878 MB/s. Report it, with the conditions it needs.
9. **Our temperature polling does not change idle power**: rest-of-board 2.409 W with vs
   2.403 W without it (`200018`). The open question in finding 4 is answered for the SD.

## SSD procedure

Kfir: power off, swap, boot, `git pull`, open VS Code, tell Claude to read this file. Then:

1. **Claude** — confirm the drive, link and partition (do not expand):
   ```bash
   cat /sys/block/nvme0n1/device/model
   cat /sys/bus/pci/devices/$(cat /sys/block/nvme0n1/device/address)/current_link_width
   lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT /dev/nvme0n1; df -h /
   ```
   Model must be `CT1000P5PSSD8`. Kfir can run `sudo parted /dev/nvme0n1 print free` to see
   the unallocated tail.
   Then Kfir runs `bash mmap_sandbox/card_eval/set_25w_clocks.sh` — a fresh boot comes up at full
   power (fairness rule 7). It must print cpu 1344000 and gpu 918000000.
2. **Kfir, VS Code closed** — RUNBOOK "BEFORE RUNNING", then STEP 0 → STEP 3, exactly as on the SD.
3. **Claude**, after STEP 3 — `python3 mmap_sandbox/card_eval/stall_probe.py` (~4 min); compare
   the verdict with the SD's. Never run it right before STEP 0: it loads the drive.
4. **Claude** — check `device.json` label is `ssd` in every new run dir, then:
   ```bash
   python3 mmap_sandbox/card_eval/summarize.py mmap_sandbox/results/card_eval/ssd_<timestamp>
   python3 resident_vlm/analyze.py resident_vlm/results/bench_<timestamp>
   ```
   and the same for the official SD sitting's dirs.
5. **Claude** — build the comparison and publish it as a **new** artifact dedicated to the
   card comparison (Kfir chose this; do not merge it into the older resident-VLM artifact —
   cross-link it). Standard-tool results first, the resident benchmark as the "what this means
   for a real AI system" section.
6. Commit and push the SSD results; update this file.

## How to compare

- Per fio test: MB/s, IOPS, p99, board J/GB, drive temperature at start and peak, cool-down
  wait — SD vs SSD side by side, plus each result as a percentage of that drive's link ceiling.
- Burst vs continuous: within each drive, `summarize.py` prints the J/GB ratio; the 128K pair
  is the fair one, the 1M pair shows the stall. Energy per GB is the headline; latency at QD8
  is inflated by queueing.
- Resident benchmark: trigger → verdict, wake J, drive-read phase J and MB/s; baseline likewise;
  resident/baseline ratios within each sitting.
- Stall probe verdicts side by side, with the `completion polled` counts from `nvme_diag.sh`.
- Idle: "rest of board" with and without temperature reads, and the two power-state tables.
- Thermals: temperature peaks and the before/after SMART thermal-counter deltas.

## SD runs added after this file was written

**Official SD sitting, 2026-09-11 evening** (25 W via `set_25w_clocks.sh`, drive rested 45 min):

- STEP 0 diagnostics: `mmap_sandbox/results/card_eval/nvme_diag_sd_express_20260911_211305.txt`
- STEP 1 card_eval: `mmap_sandbox/results/card_eval/sd_express_20260911_211324/` — 19/19 steps,
  no failures, host.json CPU 1344000 / GPU 918000000, drive resting at 51.85 °C.
  `sustained_read`: flat ~872 MB/s for the timed 120 s with the card rising to 84.8 °C and no
  throttling — then a 30 s lost-completion stall while fio waited for its last in-flight reads,
  so fio's own figure (697 MB/s, step 150.6 s) includes that wait. Use the timed-window rate
  that `summarize.py` prints, for both drives. `seq_read_1m_qd1` stalled as expected (273 MB/s,
  slowest 15.4 s); the 128K burst/continuous pair was clean (J/GB ratio 0.999).
  Kfir checked in with Claude between STEP 1 and STEP 2 (VS Code open for a few minutes, drive
  idle) — fine to do the same on the SSD.
- STEP 2 benchmark: `resident_vlm/results/bench_20260911_220613/` — 6/6 reps, no errors,
  host.json CPU 1344000 / GPU 918000000. Run with `MINRAM=5300` because only ~5400 MB was available
  after the 16:27 reboot (VS Code closed, helpers killed, nothing extra running) — use the same floor
  on the SSD. trigger → verdict: baseline 129.8 / 117.3 / 114.9 s (median **117.3 s**, wake
  **797.6 J**); resident 15.1 / 11.1 / 11.3 s (median **11.26 s**, wake **84.9 J**, drive-read phase
  37.4 J at ~362–373 MB/s). Resident is 10.4× faster and 9.4× less energy. `resident_0` is slow
  because its drive read paused 4.1 s (read phase 9.8 s at 250 MB/s) — the missed-completion stall,
  rescued before the 30 s timeout; the medians are unaffected. Free RAM before the trigger was
  5280–5498 MB and the baseline swapped 1.0–1.3 GB. The baseline is ~40 s faster than on Aug 28
  (156.6 s) for reasons outside the drive (it was already ~116 s in the afternoon run) — compare SD
  and SSD only between runs made with this same procedure, never against Aug 28.
- STEP 3 diagnostics: `mmap_sandbox/results/card_eval/nvme_diag_sd_express_20260911_225131.txt`.
  Change since STEP 0: no media errors, no new error-log entries, **0 minutes above the 84.85 °C
  warning temperature**, thermal-management level 1 entered 5 times for 69 s in total, level 2
  never. One `timeout, completion polled` line during the sitting (21:28:27 — the end of
  card_eval's sustained read); the benchmark's 4.1 s pause cleared before a timeout. The drive read
  ~318 GB and wrote ~102 GB (host data units) during the sitting.

**The SD side is complete.**

## SSD sitting, 2026-09-12

Same procedure, same commands, 25 W via `set_25w_clocks.sh`, board booted 23:10 and nothing
heavy run before STEP 0. The clone had **no fio and no sysstat** (they were installed on the SD
on 2026-09-11, after the 09-10 clone): installed 23:58 from Ubuntu noble, giving fio 3.36 and
sysstat 12.6.1, the same versions the SD sitting recorded, then a full 30 min rest before STEP 0.
An aborted STEP 1 at 23:49 stopped at card_eval's pre-flight check ("fio not installed") before
writing anything, and an aborted STEP 2 at 01:32 was killed 12 s in and deleted.

- STEP 0 diagnostics: `nvme_diag_ssd_20260912_004211.txt`. Raw data that does not count:
  `nvme_diag_ssd_20260912_004149.txt` (identical, taken 22 s earlier) and
  `nvme_diag_ssd_20260911_234932.txt` (before the fio install).
- STEP 1 card_eval: `mmap_sandbox/results/card_eval/ssd_20260912_004218/` — 19/19 steps, no
  failures, host.json CPU 1344000 / GPU 918000000, drive resting at 42.85 °C.
  Seq read 1M QD8 **2585.7 MB/s** (66% of the Gen3 x4 ceiling, against the card's 89% of x1);
  `sustained_read` flat at **2563.3 MB/s** over the timed 120 s window, 48 → 69 °C, no throttling;
  rand read 4K QD32 / QD1 **95,254 / 11,151 IOPS**; seq write 1M QD8 / QD1 2570.3 / 1431.6 MB/s;
  rand write 4K QD32 / QD1 123,792 / 17,861 IOPS; gdsio 2621 / 2644 / 3077 MB/s (compat mode).
  Idle rest-of-board 3.034 W with temperature polling, 3.031 W without.
  `seq_read_1m_qd1` was hit hard by the stall: 6.2 MB/s, slowest read 30,095 ms (exactly
  `nvme_core.io_timeout`), 26 s stuck; `continuous_read` lost 38 s, `burst_read` 3 s,
  `gdsio_x2_cpu_gpu` 15 s. The 128K QD1 pair was clean on both drives (116.3 MB/s both arms,
  J/GB ratio 1.0), so it stays the fair burst-vs-continuous pair.
- STEP 2 benchmark: `resident_vlm/results/bench_20260912_013746/` — 6/6 reps, no errors.
  trigger → verdict: baseline 116.4 / 107.1 / 109.4 s (median **109.41 s**, wake **737.5 J**);
  resident 10.24 / 9.62 / 10.27 s (median **10.24 s**, wake **77.6 J**, read phase 31.5 J at
  405–453 MB/s). Resident is 10.7× faster and 9.5× less energy. Free RAM at the trigger was
  5601–5705 MB against the SD's 5280–5501, and the baseline swapped 0.95–1.1 GB — the extra RAM
  slightly favours the SSD's baseline arm, so say so in the write-up. Run with the default
  `MINRAM=5500`, which is stricter than the SD's 5300; `MINRAM` is only a pre-flight gate and
  never reaches `bench.py`, so it changes no measurement.
- STEP 3 diagnostics: `nvme_diag_ssd_20260912_021539.txt`. No new errors during the sitting
  (error_count 19 before and after, all pre-existing "Invalid Field in Command"), 0 media errors,
  and **no throttling at all** — thermal T1 transition count 10 → 10 and 0 s above the warning
  temperature, where the SD entered level 1 five times for 69 s. Two `timeout, completion polled`
  lines (00:49:26 and 01:00:50, both during card_eval) against the SD's one. The drive read
  ~734 GB and wrote ~194 GB.
- Stall probe: `mmap_sandbox/results/card_eval/stall_probe_ssd_20260912_021844/` — verdict
  **STALLS, RESCUED**, the same as the SD. 1M QD1 unassisted 8.8 and 9.7 MB/s (slowest 26.5 s and
  17.4 s); with a 4K read on the same queue once a second, 162.4 and 157.2 MB/s (slowest ~1.0 s).
  128K QD1 879.0, 256K QD1 1141.5, 128K QD8 1933.6 MB/s, none slower than 8 ms.

### What this changes

- **Finding 1 is answered: the stall is the board's, not the card's.** A different drive, on the
  same slot and kernel, produces the same verdict and the same 30 s `io_timeout` signature. Report
  it as a platform issue and do not count it against the card. The suspect is unchanged
  (`nvme.use_threaded_interrupts=1` with edge-triggered MSI); it is still not proven which side
  drops the interrupt, and the kernel cmdline was not changed on either drive.
- **Finding 3 holds and is now measured on both drives.** The SSD reads 2586 MB/s sequentially but
  its resident wake pulls only 405–453 MB/s, against the card's 878 and 362–373. Roughly 3× the
  raw sequential bandwidth buys about 1.2× on the wake and 9% on trigger → verdict (10.24 s vs
  11.26 s) — the wake is software-bound, which is the card's strongest honest point.
- **Idle comparison needs a caveat.** The SD sitting's idle had 1.156 W on the CPU/GPU rail against
  the SSD's 0.498 W, so the host was not equally quiet. "Rest of board" subtracts that rail
  (SD 2.539 W, SSD 3.034 W) and stays the metric to use, but say in the write-up that the two
  sittings differed in host activity.
- **Thermals reverse the expected story**: the card ran to 84.8 °C and throttled briefly during its
  sitting, the SSD peaked at 69 °C and never throttled.

## Side-by-side results

Both sittings ran the same procedure at 25 W with a cool-start before every test. The SD figures
are its official evening sitting, the SSD figures its 2026-09-12 sitting.

| | SD Express (Gen3 x1, slot 0007:01:00.0) | SSD (Gen3 x4, slot 0004:01:00.0) |
|---|---|---|
| seq read 1M QD8 | 871.0 MB/s — 88% of its x1 link | 2585.7 MB/s — 66% of its x4 link |
| sustained read, timed 120 s | 870.6 MB/s, peak 84.8 °C, no throttle | 2563.3 MB/s, peak 69 °C, no throttle |
| rand read 4K QD32 / QD1 | 80,114 / 6,482 IOPS | 95,254 / 11,151 IOPS |
| seq write 1M QD8 / QD1 | 569.3 / 523.9 MB/s | 2570.3 / 1431.6 MB/s |
| rand write 4K QD32 / QD1 | 82,850 / 18,247 IOPS | 123,792 / 17,861 IOPS |
| gdsio (compat mode) | 878.3 / 878.5 / 878.1 MB/s | 2621 / 2644 / 3077 MB/s — the x2 step lost 15 s to a stall, treat it as suspect |
| 128K QD1 burst vs continuous (the fair pair) | J/GB ratio 0.999 | J/GB ratio 1.0 |
| 1M QD1 burst vs continuous | ratio 0.994, no stuck seconds | ratio 0.644 — 38 stuck seconds, an artifact, do not quote |
| 1M QD1 stall probe | STALLS, RESCUED | STALLS, RESCUED |
| idle, rest of board | 2.539 W | 3.034 W |
| baseline trigger → verdict | 117.3 s, 797.6 J | 109.41 s, 737.5 J |
| resident trigger → verdict | 11.26 s, 84.9 J | 10.24 s, 77.6 J |
| resident drive-read phase | median 362.1 MB/s (250–373), 37.4 J | median 426.5 MB/s (405–453), 31.5 J |
| resident vs baseline | 10.4× faster, 9.4× less energy | 10.7× faster, 9.5× less energy |

**The energy trade — the SD Express's actual case.** Per wake the SSD saves 7.29 J and 1.02 s. At idle
the board draws 0.495 W less on the non-compute rails with the SD Express installed, so the SSD's
per-wake saving only repays that above ~5,900 wakes/day — one every 15 s. At 24 wakes/day the SD
Express is ~42.6 kJ/day ahead. State the mechanism, or the claim is not honest: both drives sleep at
hundredths of a watt (SD Express PS4 0.015 W, SSD PS4 0.005 W, APST enabled on both, ITPT 2 s and
100 ms→PS3/2 s→PS4), so **the 0.5 W is not the flash**. It is most likely the interface — this board
runs `LnkCtl: ASPM Disabled` on both drives, so an idle link stays powered, and the SSD holds up four
lanes against the SD Express's one. Note the whole-board idle favours the SSD by 0.151 W, because the
two sittings' CPU/GPU rails differed (1.156 vs 0.498 W); that confound runs AGAINST the SD Express,
which still measured lower on the rail group that contains the drive. Other places it is competitive:
4K random writes one at a time 18,247 vs 17,861 IOPS (102%), 4K random reads at QD32 84% of the SSD on
a quarter of the lanes, 88.4% of its own link against 65.6%, spec maximum draw 1.80 W against 8.25 W.

The three claims worth building the write-up on:

1. **The stall is the platform's, not the card's** — same verdict and same 30 s `io_timeout`
   signature from a completely different drive on a DIFFERENT PCIe controller (card 0007:01:00.0 x1,
   SSD 0004:01:00.0 x4 — they are not the same slot). Two controllers showing the same fault makes the
   platform explanation stronger, not weaker.
2. **The wake is software-bound.** About 3× the sequential bandwidth buys ~1.2× on the drive-read
   phase and 9% on trigger → verdict. On one lane the card nearly matches a four-lane SSD on the
   workload that matters.
3. **State the handicaps**: the SSD baseline had 5601–5705 MB free against the SD's 5280–5501 and
   the baseline swaps; host activity differed at idle (CPU/GPU rail 1.156 W vs 0.498 W); the SSD
   has ~510 GB of unpartitioned NAND for SLC caching and wear levelling that the full card lacks;
   the two sittings ran on different days; the SSD's cool-start gate timed out on 3 of 16 tests and
   waited ~100 s on most of the rest, where the SD waited 0-11 s except its write tests, so the SSD's
   tests began further above its own resting temperature; and the SSD's 1M `continuous_read` and
   `gdsio_x2_cpu_gpu` steps were disrupted by stalls (38 s and 15 s stuck) and must not be quoted.

## State of play — read this first

The standard-tool comparison and resident benchmark are **finished on both drives** — do not re-run
them — and
the comparison write-up is **published**: "One Lane Against Four",
https://claude.ai/code/artifact/746549ba-ae80-48d2-a115-0dcac1a32518 (2026-09-12) — standard-tool
results first, the resident benchmark as the "what this means for a real AI system" section,
cross-linked to the older resident-VLM artifact rather than merged into it. Every figure on that
page comes from the official runs named above; if a number here is corrected, update the page too.
Two open items, both needing the SSD installed, so do them in one session: the SSD half of the wake
read-strategy test (last section of this file), and the SSD raw-sample recovery described below.

Note for whichever drive is installed: each drive carries its own Claude memory, and they diverge
after 2026-09-10. **This file is the shared record — `git pull` first, then read it.** Every number
above is reproducible from the committed run directories with `summarize.py` and `analyze.py`,
**with one caveat**: the SSD benchmark (`resident_vlm/results/bench_20260912_013746/`) was committed
without its `*.jsonl` samples, so its per-window figures can be read from its committed `analysis.json`
but cannot be recomputed from raw data the way the SD's can. Its trigger → verdict times and drive-read
rates are recomputable from `summary.json` (the daemon's own byte counters give 387.7 MB/s for the SD
and 466.2 MB/s for the SSD, measured slightly differently from the table's 362.1 / 426.5).
(Running `analyze.py` on a directory with no samples used to overwrite `analysis.json` with an empty
list — it now refuses and says so.)

Kfir decided 2026-09-12 to recover them — with the SSD installed:

    cd ~/Documents/projects/wildfire_detection && git pull
    git add -f resident_vlm/results/bench_20260912_013746/*.jsonl
    python3 resident_vlm/analyze.py resident_vlm/results/bench_20260912_013746
    git add resident_vlm/results/bench_20260912_013746/analysis.json
    git commit -m 'Add the SSD benchmark raw samples and analysis' && git push

`.gitignore` now keeps `resident_vlm/results/**/*.jsonl` tracked, so this cannot happen again.

## Wake read-strategy test — burst vs continuous (2026-09-12 evening)

Written on the SD Express right after its half ran, for the Claude session on the SSD.
**This is the only measurement left to run.** Nothing earlier in this file needs re-running.

### What Kfir asked for, and why the earlier "burst vs continuous" did not answer it

Kfir's question: **when the MHT hands off, how long until the model gives its first verdict, and
how many MB/s come off the drive? Is burst reading faster than continuous, does the answer change
between the SD Express and the SSD, and which configuration is fastest?** On the real system (the
resident wake), not a lab test, and with **no rate cap on any arm** ("what is the point if we are
restricting it?").

Neither earlier test answered that:
- card_eval's `burst_read` / `continuous_read` (and the 128K pair) ran continuous at `--rate=` the
  burst arm's own average, so both took the same time by construction and only energy could differ
  (J/GB ratio 0.999 / 1.0). A synthetic file, not the wake.
- the `resident_continuous` wake arm was capped at 185 MB/s (bench.py's `--pace-mbps` default), ran
  once as a smoke test, and was dropped.

Kfir also said: **do not re-measure what already exists** — no card_eval, no idle power, no dedicated
energy tests, no baseline container arm (wake energy still falls out of the same 10 Hz samples at no
extra cost). He wants the design discussed with him and a runbook before anything runs.

### The three arms, and what "burst" means here

| arm | bench.py name | how the 1.689 GB `fused_state.pt` comes off the drive |
|---|---|---|
| A — demand | `resident` | `torch.load(mmap=True)` page faults, 128 KB readahead. What ships. The reference, and the sitting's self-check against the official 11.26 s (SD) / 10.24 s (SSD) |
| B — burst 64 MB | `resident_burst` | daemon `--prefetch-mb 64`: a thread reads the file in 64 MB `read()` chunks while the load runs |
| C — continuous | `resident_continuous` | daemon `--pace-mbps 100000` (so high the pacing sleep never fires = uncapped) with `--chunk-mb 64`: stream the whole file, then load |

"Burst" took several rounds to settle with Kfir — do not reopen it:
- It does **not** mean read-then-rest. Idle inserted into a latency path can only slow the handoff; a
  wake never takes the drive near throttling (≤ 62 °C against 84.85 °C) and fio already showed resting
  saves no energy. Kfir agreed: no read-rest arm.
- 64 MB is the SD Association guidance Kfir cited: hand the drive large contiguous requests.
  `max_hw_sectors_kb` is 128 on the SD, so a 64 MB request is ~512 queued 128 KB commands, never one
  transfer.
- B and C differ only in overlap: B reads while the load runs; C finishes reading, then loads.

### Code (commits `f287e80`, `5e14783`, and the commits carrying this section)

- `resident_daemon.py`: `_prefetch()` + `--prefetch-mb` (default 0 = off); `_paced_preread`'s chunk
  is now `--chunk-mb` (default 64); the wake reply adds `prefetch_seconds` / `prefetch_bytes`.
- `bench.py`: per-arm daemon flags, `--prefetch-mb` (default 64). The cooldown now also waits for the
  **drive** to be within 5 °C of its temperature when bench.py started, inside the same
  `--cooldown-max` 240 s budget as the Tj ≤ 52 °C wait, and records `card_start_c`. Reason: in the
  official SD sitting the drive started the three resident reps at 57.85 / 60.85 / 61.85 °C, and
  because arms interleave, that drift always lands on whichever arm runs later.
- `analyze.py`: `card_start_c` / `card_peak_c` per window (`cd_st`, `cd_pk` columns).
- `sampler.py`: `read_card_temp()`; `device.json` records `read_ahead_kb`, `max_sectors_kb`,
  `max_hw_sectors_kb`, `nr_requests`.
- `resident_vlm/run_loadtime.sh`: `--reps 3 --arms resident resident_burst resident_continuous
  --pace-mbps 100000 --prefetch-mb 64`. **Always launch through it** — bench.py's `--pace-mbps`
  default of 185 silently caps arm C.
- `resident_vlm/RUNBOOK_LOADTIME.txt`: Kfir's procedure, same shape as `RUNBOOK.txt`.

Every default keeps the old behaviour: re-running analyze.py on `bench_20260911_220613` reproduced
11.26 s / 84.9 J exactly and only added the two temperature fields.

### SD Express half — `resident_vlm/results/bench_20260912_225203/`

25 W verified (CPU 1344000, GPU 918000000), 9/9 reps, no errors, 22:52–23:45. Drive resting at
51.85 °C. Diagnostics `nvme_diag_sd_express_20260912_225135.txt` (before) and `..._234556.txt`
(after). `device.json`: `max_hw_sectors_kb` 128, `read_ahead_kb` 128.

| arm | trigger → verdict per rep | **median** | daemon `wake_seconds` median | `bytes_read` | read-phase MB/s | wake J median |
|---|---|---|---|---|---|---|
| A — demand | 11.73 / 11.89 / 11.53 s | **11.73 s** | 8.90 s | 2.28–2.32 GB | 341–364 | 88.7 |
| B — burst 64 MB | 12.01 / 13.93 / 11.85 s | **12.01 s** | 8.99 s | 2.26–2.36 GB | 304–338 | 90.8 |
| C — continuous | 16.72 / 16.28 / 15.67 s | **16.28 s** | 13.25 s | 4.24–4.30 GB | 381–403 | 116.7 |

**On the SD Express, demand paging (what ships) is fastest; 64 MB bursts tie with it; continuous is
~4.5 s slower.** Camera open (~0.12 s) and first inference (~2.5 s) were identical in every arm, so
the whole difference is in the read. `resident_burst_1`'s 13.93 s carries 2.3 s extra **before**
`wake_sent` (MHT stop / classifier start), not in the read — always report `wake_seconds` beside
trigger → verdict. `bytes_read` exceeds the 1.689 GB file in every arm (other I/O during the wake; the
source was not established); what matters is that C reads ~2 GB more than A and B.

Why:
- **Buffered reads keep the queue shallow, whatever the chunk size.** During arm C's uncapped stream:
  requests in flight median 2–3, max 7; 10 Hz MB/s median ~490–500, peak ~600; over the whole file
  412–422 MB/s (`preread_seconds` 4.0–4.1 s). fio's O_DIRECT 1M QD8 keeps ~64 in flight and reaches
  871 MB/s. The page cache issues 128 KB readahead a few at a time no matter how much Python asks for —
  consistent with the July chunk-size sweep (1–64 MB, all within noise, `mmap_sandbox/results/20260705_*_sweep/`).
  The prediction that 64 MB would approach 871 MB/s and save ~2.7 s per wake was wrong.
- **Burst:** the prefetch thread (293–328 MB/s) and the page faults shared one budget (in-flight median
  3 against demand's 5), read the same bytes, and gained nothing.
- **Continuous:** the stream finishes in ~4 s, but the GPU copy of the weights lives in the same
  unified RAM, the kernel evicts the cached file, and the load reads it again. C = demand + ~4 s of
  discarded reading. It is the same double read as the 185 MB/s smoke, so the cap never caused it.
- Free RAM at the trigger was 1.66–1.92 GB in every rep, and every arm swapped out 238–505 MB during
  the wake — a board property, not specific to C.

Acceptance checks — **CORRECTED 2026-09-13: `resident_continuous_0` is VOID** (a read request waited
~1.6 s inside the wake). The check (b) below passed it because it only looks for requests in flight
while no bytes complete, and other reads kept flowing past the stuck one. The replacement is committed
as `resident_vlm/wake_checks.py` — see "Top-up". With that rep removed, continuous is 16.28 / 15.67 s,
**median 15.97 s, n=2**; demand and burst keep n=3 and their medians. The ranking is unchanged.
- **(a) nothing warm:** every rep read ≥ 2.26 GB off the drive.
- **(b) no stall inside a wake (the original, insufficient check):** the longest stretch with `disk_io_in_flight > 0` and no change in
  `disk_read_bytes` between consecutive 10 Hz samples, `wake_sent` → `first_verdict`, was 0.23 s. The
  one `timeout, completion polled` line (23:35:42, stuck since ~23:35:12) maps, by the samples' `wall`
  times, into the gap between `resident_2` (sampling ended 23:32:52) and `resident_burst_2` (started
  23:36:20) — burst_2's unmeasured daemon load and warm-up. **Do not use analyze.py's `stallms` column
  for this**: it sums read-ms over every request finishing in a 100 ms window, so a healthy read at
  queue depth 5 shows 500+ (all 9 reps did). The runbook's rule (b) was first written that way and is
  now corrected.
- **(c) temperature:** `cd_st` 51.85–53.85 °C at the start of every wake, peaks 57.85–61.85 °C.

These checks and the phase breakdown were ad-hoc Python over `summary.json` (`wake_reply`),
`<tag>.jsonl` and `<tag>.marks.json`. Since 2026-09-13 `resident_vlm/wake_checks.py <run dir>...` computes
the phase breakdown, the corrected stall test and (a)/(c) the same way for every run — use it.
Fields: `wake_reply.preread_seconds/bytes`, `prefetch_seconds/bytes`, `sd_read_seconds`,
`camera_open_seconds`, `first_verdict_seconds`, `wake_seconds`, `bytes_read`; per record
`cooldown_seconds`, `card_start_c`, `trigger_to_verdict_seconds`. In-flight depth and MB/s come from
the jsonl between `wake_sent` and `preread_end` (arm C) or `sd_read_end` (A, B).

Runs that do not count: `resident_vlm/results/bench_20260911_115331/` — aborted during `baseline_0`
on 2026-09-11 (configured with the 185 MB/s continuous arm; only `baseline_0.jsonl`, no summary).
Committed as raw data only.

### SSD half — to-do, in order

1. **Kfir:** power off, swap, boot, `git pull`, open VS Code, tell Claude to read this file.
   **Claude: save the key points to memory** — the SSD's memory predates all of this.
2. **Claude:** confirm the model is `CT1000P5PSSD8`, `pgrep -af bench.py` is empty and no containers
   run. Check the live clocks — after every boot they are 1728000 / 1020000000 (full power) even though
   `pmode:0001` says 25 W:
   `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq /sys/class/devfreq/17000000.gpu/max_freq`.
   Walk Kfir through `resident_vlm/RUNBOOK_LOADTIME.txt`.
3. **Kfir, VS Code closed:** `set_25w_clocks.sh` (must print 1344000 / 918000000), kill the four
   desktop helpers, `free -m` (write it down), STEP 0 `nvme_diag.sh`, STEP 1
   `bash resident_vlm/run_loadtime.sh` (~55 min on the SD), STEP 2 `nvme_diag.sh`.
   Same arms, reps and flags as the SD — do not edit bench.py defaults or the launcher between drives.
4. **Claude:** `python3 resident_vlm/analyze.py resident_vlm/results/bench_<timestamp>`; check
   `device.json` label is `ssd` and `host.json` shows CPU 1344000 / GPU 918000000; build the phase
   breakdown and run checks (a)–(c) exactly as above, mapping every `completion polled` line from the
   before/after diagnostics onto the rep timelines. A stall inside a wake voids that rep — tell Kfir;
   any re-run only after a 30 min rest.
5. **Record the SSD's `max_hw_sectors_kb` and `read_ahead_kb`** from `device.json`. If they exceed
   128, the SSD gets larger commands for free and the comparison must say so.
6. **Compare**, three arms × two drives. Kfir's headline: burst vs continuous on each drive and whether
   the ranking changes between drives, with demand as the reference, plus which configuration is
   fastest overall. Per arm: all three reps and the median of trigger → verdict and `wake_seconds`,
   read-phase MB/s, in-flight depth during C's stream, `bytes_read` (does C double-read on the SSD
   too?), free RAM at the trigger (SD: 1.66–1.92 GB), swap-out during the wake. Only compare
   within-sitting figures against each other; the two sittings ran on different days.
7. **Also with the SSD installed:** the raw-sample recovery in "State of play" —
   `bench_20260912_013746`'s `*.jsonl` (still 0 files in git as of 2026-09-12 23:50). Do it after STEP 2,
   never before STEP 0.
8. **Commit and push** the SSD run folder (check `git status` shows its `*.jsonl` staged — last time
   they were missed), both diagnostics files, and an update to this section.
9. **Then the graph Kfir asked for:** per arm per drive, MB/s over time from 5 s before the trigger to
   5 s after the first verdict (the samples already cover 20 s before and 60 s after). Ask Kfir whether it
   goes on a new artifact or onto "One Lane Against Four"; publish it as an Artifact, not a local HTML file.

What to expect: the SSD's earlier demand wake was 10.24 s with a ~426 MB/s read phase. If the
shallow-queue explanation is right, the SSD's advantage stays small in all three arms and C
double-reads again (same board, same RAM). Either outcome is worth reporting.

### SSD half — `resident_vlm/results/bench_20260913_012802/` (2026-09-13)

25 W verified (CPU 1344000, GPU 918000000), 9/9 reps, no errors, 01:28–02:23, same wrapper and flags as
the SD. Drive resting at 41.85 °C. Diagnostics `nvme_diag_ssd_20260913_012747.txt` (before) and
`nvme_diag_ssd_20260913_022319.txt` (after): **0 `completion polled` lines and no new kernel lines at all**,
0 media errors, error log 20 → 20, thermal T1 10 → 10, 0 s above warning temperature.
`device.json`: **`max_hw_sectors_kb` 128 and `read_ahead_kb` 128, identical to the SD Express** — the SSD
gets no larger commands, so item 5's confound does not exist.

Boot note: on this boot `nvpmodel.service` did **not** fail (`inactive`, `Result=success`) and the clocks
came up at 25 W before `set_25w_clocks.sh` ran, so `host.json` says `nvpmodel_service: inactive` where the
SD says `failed`. Every clock knob matched the SD sitting (CPU online 0-5, CPU max 1344000 / min 729600,
schedutil, GPU max 918000000, EMC 3199000000) — the route differed, not the state. Fairness rule 7's "fails
at every boot" is not always true.

A false start at 00:55 (launched before STEP 0, stopped ~2 min in during the daemon load, no rep
measured) was deleted; the drive then rested 30 min before STEP 0.

| arm | trigger → verdict per rep | valid median | `wake_seconds` median | read phase | `bytes_read` | wake J median |
|---|---|---|---|---|---|---|
| A — demand | **VOID 14.90** / 10.25 / 10.16 s | **10.20 s** (n=2) | 7.36 s | 4.69–4.89 s | 2.33–2.35 GB | 77.6 |
| B — burst 64 MB | 10.61 / 10.69 / 10.84 s | **10.69 s** (n=3) | 8.03 s | 5.29–5.54 s | 2.36–2.41 GB | 82.5 |
| C — continuous | 13.63 / **VOID 16.05** / 13.62 s | **13.62 s** (n=2) | 10.87 s | pre-read 3.52–3.54 s + load 4.64–4.74 s | 4.25–4.27 GB | 99.3 |

**Same ranking as the SD Express: demand ≤ burst < continuous, and every valid range is disjoint.**
- **Voids:** `resident_0` (a request waited ~4.4 s; its read phase 9.74 s at 255 MB/s) and
  `resident_continuous_1` (~2.3 s). Both were released before the 30 s timeout, hence no kernel line.
  During `resident_0`'s stuck request the other reads kept completing at 0.14–0.30 ms each but only ~13 KB
  apiece — which briefly looked like a host-side access-pattern effect. It was not: average ms/read per
  sample hides one stuck request. Same signature as the SD 09-11 official `resident_0` (~3.8 s), whose
  original stall attribution is confirmed.
- **Burst is slower than demand on the SSD, but tied on the SD — and the reason is the join.** `wake()`
  waits for the prefetch thread to finish the whole file (`pf.join`, `resident_daemon.py`). On the SSD
  burst's read phase equals the prefetch thread's time exactly (5.29 / 5.52 / 5.54 s both): the load had
  finished and the wake waited for a thread running at only 305–320 MB/s. On the SD the load outlasted the
  thread (6.23–6.83 vs 5.14–5.76 s), so the join cost nothing. The SD half ran the same code, so it must
  NOT change before the top-ups — but the write-up must say burst's SSD figure is "load + wait for the
  prefetch", not "64 MB reads are slower". Nothing suggests burst could beat demand without the join: on
  the SD the loader with a prefetch thread was no faster than alone. Burst also swapped out more during
  the wake (SSD 449–507 MB vs demand 180–337; SD 414–494 vs 250–385).
- **Continuous double-reads on the SSD too** (4.25–4.27 GB vs ~2.34), so it is the shared RAM, not the
  drive. Uncapped 64 MB buffered streaming reached only **~478 MB/s — 18% of the SSD's 2586 MB/s fio
  figure** (SD: ~420 MB/s, 48% of its 871), with 1–2 requests in flight: the shallow-queue explanation
  holds on both drives. All three arms send the drive 128 KB commands a few at a time; they differ in
  wasted work, not in what the drive sees.
- **The SD-vs-SSD wake gap is real, not drift.** The SSD's demand arm reproduced its 09-12 sitting
  (9.62 / 10.24 / 10.27 s) to ~0.05 s median, where the SD moved 0.47 s between its two sittings. All 5
  valid SSD demand reps (9.62–10.27 s) beat all 5 valid SD ones (11.10–11.89 s). The ~1.3–1.5 s gap sits
  in the read phase (median 6.22 vs 4.79 s). Kfir had questioned whether 9% was noise; with a second SSD
  sitting it is not. Small for four lanes, which is the point the published page already makes.
- **Check (c):** bench.py's cooldown gate passed every rep at 46.85–47.85 °C (5–6 °C over resting), but
  the drive then warmed to 53.85–54.85 °C by the trigger, so the literal "within 5 °C of resting" is not
  met. Start temperature was within 1 °C across all arms, peaks ≤ 56.85 °C, no throttling: the rule's
  purpose (no arm starting warmer than another) holds. Report both.
- Free RAM at the trigger 1719–1871 MB (SD 1659–1924), swap-out during the wake 180–507 MB (SD 250–505) —
  comparable.

### Top-up — rules fixed and committed 2026-09-13 BEFORE it ran

Three arms now have 2 valid reps (SSD demand, SSD continuous, SD Express continuous); Kfir's standard is 3.
The stall hit 2/9 SSD reps and 1/9 SD reps, so a fresh 3-rep sitting is clean on all 9 reps only ~10–20% of
the time — a full re-run would most likely come up short again. Kfir agreed (2026-09-13) to a top-up: 2 more
reps of every arm, same code and order, via `resident_vlm/run_loadtime_topup.sh` (`--reps 2`, otherwise
identical to `run_loadtime.sh`). Procedure: `resident_vlm/RUNBOOK_LOADTIME.txt`, section "TOP-UP".

1. An arm below 3 valid reps takes its **first** valid top-up reps, in run order, until it has 3 — never
   the better of two, never replacing a rep from the original sitting.
2. An arm that already has 3 valid reps does not use its top-up reps; they are the **drift check**. If
   their median is within 0.5 s of that arm's median in the original sitting, the top-up joins the
   sitting; if not, the top-up does not count and that drive gets a full re-run.
   SSD: burst (10.69 s). SD Express: demand (11.73 s) and burst (12.01 s).
3. Every top-up rep goes through `wake_checks.py` exactly like the sitting; void reps are listed with the
   reason, never dropped silently.
4. If an arm is still below 3 after the top-up, report it at the n it has and say so — no third batch.

**SSD top-up:** 2026-09-13, STEP 0 no earlier than 02:53. Results go here when done.

**SD Express top-up — to-do for the SD session:** `git pull`, read this section, save it to memory, then
the RUNBOOK "TOP-UP" procedure with the identical command. Afterwards
`python3 resident_vlm/wake_checks.py resident_vlm/results/bench_20260912_225203 resident_vlm/results/bench_<top-up>`,
apply rules 1–4, map `completion polled` lines, commit the folder (check its `*.jsonl` are staged) and both
diagnostics files, and update this section.

### How Kfir works — the SSD memory does not know this

Explain each command before running it, and end with a short summary of what the code does and how.
Discuss the design and give him a runbook before any test; explain and get a go-ahead before patching.
No explanatory comments in code or config edits — explain in chat. Always write "SD Express", never
"card" alone. Kfir is a beginner working on VLM/classifier/integration; Ori owns all of MHT_TOP. Time
before the competition is short, but he wants data he can defend: three reps per configuration, no
single-measurement conclusions.

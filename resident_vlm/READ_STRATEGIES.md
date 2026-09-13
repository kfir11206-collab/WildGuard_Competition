# How the weights come off the drive — demand, burst, continuous

> **SUPERSEDED 2026-09-13 — not used.** Kfir decided not to use the load-time test this document
> describes (demand / burst / continuous, run with `bench.py`). Its four runs are kept as raw data only.
> See the STATUS block at the top of `mmap_sandbox/card_eval/HANDOFF.md`.

This explains the three ways the resident wake reads the model weights, step by step, and why they
finished in the order they did. Every number comes from the committed runs listed at the end; anything
that is an explanation rather than a measurement is marked **(reasoned)**.

Measured on both drives in the same M.2 slot: the SanDisk SD Express (PCIe Gen3 x1) on 2026-09-12 and
the Crucial P5 Plus SSD (Gen3 x4) on 2026-09-13.

---

## 1. What is being loaded, and where it has to go

**The file.** `resident_vlm/results/fused_state.pt` is 1.689 GB. Inside it:

- a 70 KB **index** (`data.pkl`) that says which weight is where;
- **547 pieces of weight data** stored one after another, uncompressed. Two are 164 MB each, three are
  10 MB, and the median piece is only 0.33 MB.

**Where it goes.** The weights must end up in GPU memory. On the Jetson there is no separate GPU memory
card: "GPU memory" is taken from the **same 7.5 GB of RAM** the rest of the system uses.

**How much room there is.** When the MHT hands off, only about **1.7–1.9 GB of RAM is available** on
either drive. The weights alone need 1.69 GB of it. This one fact decides most of what follows.

**Nothing is left over from the last wake.** When the model goes to sleep, the daemon deletes the
weights from GPU memory and tells the kernel to drop the file from its memory as well
(`posix_fadvise DONTNEED`); `bench.py` does the same again just before the trigger. Every counted wake
read at least 2.26 GB off the drive, which proves the weights were really read from the drive each time.

---

## 2. Every wake, every method: the parts that do not change

From the MHT's trigger to the first verdict, in order:

| step | who | time |
|---|---|---|
| stop the MHT container | `bench.py` | ~0.8 s |
| start the classifier container | `bench.py` | ~1.8 s |
| **read the weights and put them in GPU memory** | the daemon | **this is the only step that differs** |
| open the camera | the daemon | 0.11–0.18 s |
| first inference → verdict | the daemon | 2.40–2.71 s |

So about **5.2 s of every wake is identical in all three methods and on both drives.** Everything below
is about the one row in bold.

---

## 3. Method A — demand paging (`resident`, what the system ships)

The code is three lines in `resident_daemon.py` (`_load_from_card`). Follow one weight through them —
say a 9.8 MB piece.

**Step 1 — `torch.load(fused_state.pt, mmap=True)`.**
PyTorch reads only the 70 KB index. For our weight it learns "9.8 MB, starting at this position in the
file". Then it **maps** the file: the program is given memory addresses that cover all 1.689 GB, as if the
file were already in RAM. **No weight data has been read yet.**

**Step 2 — `load_state_dict(..., assign=True)`.**
The model's weight is pointed at those addresses. **Still nothing read.**

**Step 3 — `model.to("cuda")`.**
For each of the 547 pieces, in file order: set aside 9.8 MB of GPU memory, then copy the bytes from the
mapped addresses into it. **This copy is where the reading happens.**

**Step 4 — the copy hits bytes that are not in RAM.**
The copy reaches the first byte of our weight. That part of the file is not in memory, so the processor
raises a **page fault**: the copying thread is **paused**, and the kernel goes to fetch the data.

**Step 5 — the kernel reads ahead, 128 KB at a time.**
The kernel sees the program moving forward through the file, so instead of fetching only the 4 KB it was
asked for, it **reads ahead**: it sends the drive requests of up to **128 KB each** (the drive's
`read_ahead_kb` setting, 128 on both drives) and keeps a few of them waiting in line ahead of the copy.
Measured: **1–5 requests in flight** at any moment.

**Step 6 — the drive answers, the copy continues.**
Each request takes the drive, on average, **0.35–0.37 ms on the SSD** and **0.64–0.70 ms on the SD Express**
(kernel's own count, from sending the request to getting the data back). The bytes land in the kernel's
file cache in RAM, the paused copy wakes up, copies them into GPU memory, runs into the next part that is
not there yet, and pauses again. Our 9.8 MB weight takes roughly 80–110 requests like this.

**Step 7 — used bytes are thrown away, harmlessly.**
Once our weight is in GPU memory, its copy in the file cache is not needed any more. RAM is running out
fast — the GPU copy is using it up — so the kernel frees those cached pages. **(reasoned)** That costs
nothing, because the copy never goes back to them — which fits A reading only 2.33–2.35 GB in total.

**Step 8 — repeat for all 547 pieces.**
Measured on the SSD (`bench_20260913_012802`, `resident_1`), one-second steps from the wake request:

| time | read off the drive so far | file cache in RAM | RAM available |
|---|---|---|---|
| +0 s | 0 MB | 716 MB | 1887 MB |
| +1 s | 509 MB | 941 MB | 1563 MB |
| +2 s | 1081 MB | 1004 MB | 1001 MB |
| +3 s | 1639 MB | 659 MB | 651 MB |
| +4 s | 2059 MB | 372 MB | 427 MB |

Available RAM drains from 1.9 GB to 0.4 GB because the weights are filling GPU memory. The cache never
holds more than about 1 GB, and it does not need to.

**What the drive sees:** a steady stream of ~128 KB requests, a few at a time, at roughly 400–660 MB/s,
paced by how fast the copy moves. **It is not burst reading** — nothing waits and then reads a big
block. It is the drive being asked for exactly the next bytes the model needs, and only once.

**Why it is called "demand":** the drive only reads when the copy demands a byte that is not there.

**Result:** read phase **4.69–4.89 s on the SSD**, **5.80–6.22 s on the SD Express.**

---

## 4. Method B — 64 MB bursts (`resident_burst`)

**What it does.** Steps 1–8 exactly as in A. At the same moment the load starts, a **second thread**
(`_prefetch`) opens the file and reads it from the beginning with `read()` calls of **64 MB** each,
throwing the bytes away. The idea: put the file into the kernel's cache ahead of the copy, and give the
drive large pieces of work (64 MB is the size the SD Association recommends).

**What actually happens:**

**B1 — a 64 MB `read()` does not reach the drive as 64 MB.** The thread's read goes through the same
kernel file cache as the page faults in A, and the kernel turns it into the same ~128 KB requests. The
drive sees **1–3 requests in flight**, the same shallow line as A. The July sweep saw the same thing:
Python read sizes from 1 MB to 64 MB all gave the same speed.

**B2 — two readers, one set of bytes.** The thread (305–320 MB/s on the SSD, 293–328 MB/s on the SD
Express) and the copy both start at the beginning of the file and move through it at similar speeds.
**(reasoned)** Whichever reaches a part first reads it from the drive; the other finds it already
cached. **No drive work is removed** — the total read barely changes (SSD 2.36–2.41 GB against A's 2.33–2.35 GB).

**B3 — it costs RAM.** Each 64 MB `read()` needs a 64 MB buffer in the Python process, on a board with
under 2 GB to spare. Measured: more of the running system is pushed out to swap during the wake
(**(reasoned)** the buffers and the second reader are the likely cause):
**SSD 449–507 MB swapped out, against A's 180–337 MB; SD Express 414–494 MB against 250–385 MB.**

**B4 — the wake waits for the thread to finish the whole file.** After the load, `wake()` calls
`pf.join()`: it does not continue to the camera until the thread has read the last byte, even if the
model is already loaded.
- **On the SSD the copy finished first.** Burst's read phase was 5.29 / 5.52 / 5.54 s — **exactly the
  thread's own time** — against 4.69–4.89 s for A. The wake spent ~0.5 s waiting for a thread that had
  become useless.
- **On the SD Express the copy took longer than the thread** (6.23–6.83 s against 5.14–5.76 s), so the
  wait cost nothing.

That is why burst **ties with demand on the SD Express and loses to it on the SSD**. Nothing suggests
burst could beat demand even without the wait: on the SD Express, where the wait cost nothing, the copy
with a helper thread (6.23–6.83 s) was no faster than the copy alone (5.80–6.22 s).

---

## 5. Method C — continuous (`resident_continuous`)

**What it does.** **Before** Step 1, read the whole file from the first byte to the last in 64 MB
`read()` calls, with no pause (`--pace-mbps 100000` is so high the rate limiter never acts), throwing the
bytes away (`_paced_preread`). **Then** run Steps 1–8 exactly as in A. The idea: fill the kernel's cache
with the whole file first, so the load finds everything already in RAM.

**What actually happens:**

**C1 — the pre-read is fast but shallow.** It takes **3.52–3.58 s on the SSD (473–480 MB/s)** and
**4.00–4.03 s on the SD Express (~420 MB/s)**, with 1–2 requests in flight — the same 128 KB requests as A
and B (see B1). That is 18% of the SSD's fio sequential speed and 48% of the SD Express's.

**C2 — the file does not fit, so most of it is thrown out while it is being read.** The file is 1.69 GB
and only ~1.8 GB of RAM is available, part of which is already cache. Measured on the SSD
(`resident_continuous_0`):

| time | read off the drive so far | file cache in RAM | RAM available |
|---|---|---|---|
| +0 s | 0 MB | 770 MB | 1847 MB |
| +1 s | 565 MB | 969 MB | 1945 MB |
| +2 s | 1068 MB | 1049 MB | 1771 MB |
| +3 s | 1714 MB | 1216 MB | 1742 MB |
| *pre-read ends at +3.52 s, load starts* | | | |
| +4 s | 2215 MB | 1371 MB | 1679 MB |
| +5 s | 2714 MB | 1201 MB | 1223 MB |
| +6 s | 3159 MB | 905 MB | 895 MB |
| +7 s | 3649 MB | 462 MB | 490 MB |
| +8 s | 3934 MB | 596 MB | 617 MB |

By +3 s, 1.7 GB had come off the drive but the cache had grown by only ~450 MB. The SD Express did the
same: 1963 MB read during its pre-read, cache up by only 507 MB. **Most of the file was already gone from
memory before the load began.**
**(reasoned)** When RAM runs short, the kernel frees the cached pages that have gone unused longest.
During a front-to-back pre-read those are the **start of the file** — exactly where the load begins.

**C3 — the load reads it all again.** Steps 1–8 now run into parts of the file that are no longer in
memory and fetch them from the drive a second time. The load alone read 2.55–2.61 GB — a little more
than A reads from scratch. **Total: 4.24–4.30 GB off the drive, against A's 2.33–2.35 GB.** And once the GPU copy
starts filling RAM (from +4 s), available memory drains just as in A.

**So C = A, plus a 3.5–4 s pre-read whose work is mostly thrown away.**

This is a **RAM limit, not a drive limit**: both drives show the same double read. **(reasoned)** It
would only go away if the board had room for the file and the GPU copy at the same time (~3.4 GB free),
which it does not while the system is running.

---

## 6. Results

Trigger → first verdict, counted reps only (why some reps are not counted: section 8).

| method | SD Express, per rep | SD Express median | SSD, per rep | SSD median |
|---|---|---|---|---|
| **A — demand** | 11.73 / 11.89 / 11.53 s | **11.73 s** | 10.25 / 10.16 / 10.29 s | **10.25 s** |
| **B — burst 64 MB** | 12.01 / 13.93 / 11.85 s | **12.01 s** | 10.61 / 10.69 / 10.84 s | **10.69 s** |
| **C — continuous** | 16.28 / 15.67 s — *third rep pending (top-up)* | **15.97 s** (n=2) | 13.63 / 13.62 / 14.19 s | **13.63 s** |

| method | SD Express: read phase / wake energy (median) | SSD: read phase / wake energy (median) |
|---|---|---|
| A — demand | 5.80–6.22 s / 88.7 J | 4.69–4.89 s / 78.0 J |
| B — burst 64 MB | 6.23–6.83 s / 90.8 J | 5.29–5.54 s / 82.5 J |
| C — continuous | pre-read 4.00–4.03 s + load 6.00–6.50 s / 114.3 J | pre-read 3.52–3.58 s + load 4.64–4.74 s / 100.2 J |

`resident_burst_1` on the SD Express (13.93 s) spent 2.3 s longer **before** the wake request (MHT stop /
classifier start), not in the read.

**The answer to the question:**
- **Is burst faster than continuous?** Yes, on both drives, by 3–4 s — because continuous reads the file
  twice (C2–C3).
- **Is burst faster than demand?** No. It ties on the SD Express and loses ~0.45 s on the SSD (B4).
- **Does the order change between drives?** No: demand ≤ burst < continuous on both.
- **Fastest configuration measured:** demand paging on the SSD, 10.25 s; on the SD Express, demand paging,
  11.73 s.

**In one sentence:** demand is fastest because it is the only method that does no extra work — it
reads each byte once, in order, exactly when the model needs it. Burst adds a second reader that removes
no work and makes the wake wait for it; continuous adds a read that the RAM cannot keep.

---

## 7. Why the SSD is faster in every method — and by so little

All three methods send **the same kind of requests** to both drives: ~128 KB each, 1–5 at a time. So the
only thing the drive changes is **how long each request takes**:

| | SD Express (x1) | SSD (x4) |
|---|---|---|
| average time per read request, demand paging | 0.64–0.70 ms | 0.35–0.37 ms |
| read phase, demand paging | 5.80–6.22 s | 4.69–4.89 s |
| fio sequential read (1 MB requests, 8 at a time) | 871 MB/s | 2586 MB/s |

The SSD answers each small request in about half the time, and that is the whole difference in the
wake: ~1.3–1.5 s of read phase.

**(reasoned)** Moving a ~90 KB request over one PCIe lane takes about 0.1 ms, over four lanes about
0.02 ms. So most of the SD Express's 0.7 ms is the drive locating and reading the data, not the lane
count. That is why the SSD's ~3× advantage in fio sequential speed turns into only ~1.3–1.5 s on the
wake: requests this small, this few at a time, never use the extra lanes.

The gap is real, not day-to-day noise: the SSD's demand arm gave 9.62 / 10.24 / 10.27 s on 2026-09-12
and 10.25 / 10.16 / 10.29 s on 2026-09-13; all six are faster than every SD Express demand rep not hit
by a stall (11.10–11.89 s across its two sittings).

---

## 8. Reps that do not count

The board has a fault, proven on both drives, where a finished read request is sometimes not noticed
by the host. It is usually released within seconds by the next request on the same queue, so it leaves
no kernel log line — but a wake that hits it stalls while it waits. `resident_vlm/wake_checks.py` finds
these: for every 100 ms sample it subtracts the normal cost of the reads that completed from the
read-time that was actually spent; more than 1 s left over means one request waited that long. Healthy
reps stay under 0.7 s.

| drive | run | rep | request waited | trigger → verdict |
|---|---|---|---|---|
| SD Express | `bench_20260912_225203` | `resident_continuous_0` | ~1.6 s | 16.72 s |
| SSD | `bench_20260913_012802` | `resident_0` | ~4.4 s | 14.90 s |
| SSD | `bench_20260913_012802` | `resident_continuous_1` | ~2.3 s | 16.05 s |
| SSD | `bench_20260913_030701` (top-up) | `resident_1` | ~1.1 s (read phase 6.72 s vs 4.69–4.89 s) | 12.22 s |

The missing reps were made up by a **top-up** whose rules were committed (`322063e`) before it ran: an
arm below 3 counted reps takes its *first* valid top-up reps; an arm already at 3 uses its top-up reps
only to check the day had not drifted (SSD burst: 10.70 s against the sitting's 10.69 s — joined). Full
rules: `resident_vlm/RUNBOOK_LOADTIME.txt`, section "TOP-UP". The SD Express continuous arm's top-up is
still to run.

---

## 9. What this does not show

In all three methods the drive only ever received small requests, a few at a time. So this measures
**which way of loading the weights is fastest in this software on this board**. It does **not** test
whether the SD Express is faster when it is given large requests — fio shows it is, at the drive level
(871 MB/s with 1 MB requests 8 at a time). Whether the wake can be made to send it such requests is an
open question, not answered here.

---

## Sources and how to reproduce

| run | drive | what |
|---|---|---|
| `resident_vlm/results/bench_20260912_225203/` | SD Express | 3 methods × 3 reps, 2026-09-12 22:52–23:45 |
| `resident_vlm/results/bench_20260913_012802/` | SSD | 3 methods × 3 reps, 2026-09-13 01:28–02:23 |
| `resident_vlm/results/bench_20260913_030701/` | SSD | top-up, 3 methods × 2 reps, 2026-09-13 03:07–03:44 |
| `mmap_sandbox/results/card_eval/nvme_diag_*_2026091{2,3}_*.txt` | both | kernel log and SMART before/after each run |

    python3 resident_vlm/wake_checks.py resident_vlm/results/bench_20260912_225203
    python3 resident_vlm/wake_checks.py resident_vlm/results/bench_20260913_012802 resident_vlm/results/bench_20260913_030701
    python3 resident_vlm/analyze.py resident_vlm/results/<run>

`wake_checks.py` prints each rep's phase breakdown, requests in flight, RAM and swap at the trigger, and
the stall verdict. The memory tables in sections 3 and 5 come from the same `<rep>.jsonl` samples
(`disk_read_bytes`, `cached_mb` = the kernel's `Cached:`, `mem_available_mb`), read at one-second steps
from the `wake_sent` mark. The per-request times in section 7 are `disk_read_ms` divided by
`disk_reads_completed` over each read phase.

Two numbers are not explained: every method reads more than the 1.689 GB file (A: 2.33–2.35 GB). The
drive counters include every read on the drive during the wake, such as the classifier container
starting; where the extra ~0.65 GB comes from was not established.

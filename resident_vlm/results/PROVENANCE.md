# Storage device provenance

Every result in this directory was produced on the **SD Express card**:
WDC SanDisk SD Express `SDSQXFN`, 456 GB, SN530 DRAM-less controller.

That card was the only storage device in the Jetson's M.2 slot until 2026-09-10.
On 2026-09-10 a Crucial P5 Plus 1 TB NVMe SSD (`CT1000P5PSSD8`, SN 220836F7F73B,
FW P7CR403) was installed in the same slot as a clone of the card, to run the
same benchmarks on SSD for comparison.

SD Express presents as NVMe over PCIe, so BOTH drives appear as `/dev/nvme0n1`
and are indistinguishable in the result JSON without an explicit device stamp.
This is the reason this file exists.

Cutoff: no file under this directory has an mtime later than 2026-08-28.
The SSD was first booted 2026-09-10 21:16. Anything dated on or after
2026-09-10 was NOT produced on the SD card.

**Testing on the SD Express card is NOT finished.** The card is retained intact
and will be re-installed. The SSD is a parallel measurement, not a replacement,
and the two drives will alternate in the same slot.

To confirm which drive is installed in any future run:

    cat /sys/block/nvme0n1/device/{model,serial,firmware_rev}

SD Express card -> SDSQXFN.  SSD -> CT1000P5PSSD8 / 220836F7F73B.

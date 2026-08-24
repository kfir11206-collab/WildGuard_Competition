import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"


def img(name):
    b = base64.b64encode((FIG / name).read_bytes()).decode()
    return f"data:image/png;base64,{b}"


def table(headers, rows, hot_col=None, good_col=None):
    h = "".join(f"<th>{c}</th>" for c in headers)
    body = ""
    for r in rows:
        tds = ""
        for i, c in enumerate(r):
            cls = "num"
            if i == 0:
                cls = "lbl"
            if good_col is not None and i == good_col:
                cls = "num good"
            tds += f'<td class="{cls}">{c}</td>'
        body += f"<tr>{tds}</tr>"
    return f'<div class="tbl-wrap"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def stat(value, unit, label):
    return f'''<div class="stat">
      <div class="stat-v">{value}<span class="stat-u">{unit}</span></div>
      <div class="stat-l">{label}</div>
    </div>'''


def figure(name, cap):
    return f'''<figure class="plate">
      <img src="{img(name)}" alt="{cap}">
      <figcaption>{cap}</figcaption>
    </figure>'''


def section(num, eyebrow, title, body_html):
    return f'''<section>
      <div class="sec-head">
        <span class="sec-num">{num}</span>
        <div>
          <div class="eyebrow">{eyebrow}</div>
          <h2>{title}</h2>
        </div>
      </div>
      {body_html}
    </section>'''


stats = "".join([
    stat("720", "MB/s", "cold sequential read · ±13 over 5 runs"),
    stat("15", "×", "sequential vs random penalty at 4 KB"),
    stat("2.6", "GB/s", "warm page-cache read after pre-warm"),
    stat("29", "%", "less energy — gated vs always-on"),
    stat("4", "s", "SD read inside a 118 s cold wake"),
])

s1 = section("01", "The device under test", "A DRAM-less card doing an SSD's job",
    """<p>The card is the root drive: benchmarking <code>/</code> benchmarks the SD. Everything
    below runs on this exact hardware, with the real 2.8&nbsp;GB wildfire model set as the payload.</p>"""
    + table(["Property", "Value"], [
        ["Card", "WDC SanDisk SD Express <code>SDSQXFN</code> · SN530, DRAM-less"],
        ["Interface", "PCIe → presented as NVMe (<code>nvme0n1</code>), M.2 slot"],
        ["Capacity", "456 GB device · 78 GB root partition"],
        ["Host", "NVIDIA Jetson Orin Nano (Super) · JetPack 7"],
        ["Memory", "7.4 GB <strong>unified</strong> — CPU + GPU share one pool"],
        ["Workload", "VILA1.5-3B-AWQ VLM + DistilBERT · 2.8 GB weights"],
    ]))

s2 = section("02", "Read throughput", "Fast and flat — no read-side cliff",
    """<p>A full cold read of the 2.8&nbsp;GB weight set holds ~700&nbsp;MB/s from first byte to
    last, with no burst-then-throttle on the read path, and repeats tightly across five cold runs.</p>"""
    + table(["Metric", "Value"], [
        ["Overall (single trace)", "703 MB/s"],
        ["5× cold re-run", "<strong>720 ± 13 MB/s</strong> · min 703 · max 737"],
        ["Shape", "flat across all 2.8 GB"],
    ])
    + figure("F1_sequential_read.png",
             "Figure 1 — Throughput across the full 2.8 GB read, and run-to-run consistency over 5 cold runs."))

s3 = section("03", "Write endurance", "The SLC burst, then the throttle",
    """<p>Writes tell the DRAM-less story that reads hide. The first few gigabytes land in the fast
    SLC cache; once it saturates near 4&nbsp;GiB the card folds to TLC and throughput drops. This is the
    one place the controller's class shows — and it only bites large sustained writes, which this
    pipeline never does on the hot path.</p>"""
    + table(["Phase", "Throughput"], [
        ["SLC burst (first ~4 GiB)", "peak <strong>546 MB/s</strong>"],
        ["Folded / sustained (TLC)", "~180–230 MB/s"],
        ["Overall (12 GiB)", "282 MB/s"],
    ])
    + figure("F2_write_curve.png",
             "Figure 2 — Write throughput vs cumulative data written: SLC burst region, then the fold to TLC."))

s4 = section("04", "Access pattern", "The one lever that actually moves",
    """<p>The dominant factor on this card is sequential vs random — not block size, not chunk size.
    Small random reads pay a steep penalty that closes as blocks grow, while sequential throughput
    stays flat no matter how the reader chunks it; the kernel does the read-ahead.</p>"""
    + table(["Read type", "4 KB", "64 KB", "1 MB"], [
        ["Sequential (MB/s)", "435", "470", "464"],
        ["Random (MB/s)", "<strong>29.7</strong>", "225", "416"],
        ["Random IOPS", "7 255", "3 427", "397"],
    ])
    + """<p class="fine">mmap chunk sweep 1–64 MB: 613–680 MB/s (within noise). Same-method
    per-block <code>pread</code> comparison; the 703 MB/s headline uses 16 MB streaming reads.</p>"""
    + figure("F3_access_pattern.png",
             "Figure 3 — Random vs sequential across block sizes, and throughput vs mmap chunk size."))

s5 = section("05", "Page cache", "Pre-warming turns SD reads into RAM reads",
    """<p>The page cache is cross-process, so one tiny always-on watcher can pre-warm the weights and
    the <em>later</em> VLM loader reads them from RAM at ~2.6&nbsp;GB/s. The flip side is the cost of
    getting the pattern wrong: forcing <code>MADV_RANDOM</code> collapses throughput to 56&nbsp;MB/s with
    65k major faults.</p>"""
    + table(["Condition", "Throughput"], [
        ["<code>MADV_RANDOM</code> (worst case)", "56 MB/s"],
        ["Cold (page cache empty)", "763 MB/s"],
        ["Warm (pre-warmed cache)", "<strong>2 647 MB/s</strong>"],
    ])
    + figure("F4_cache.png",
             "Figure 4 — Random-advice floor, cold read, and warm pre-warmed read."))

s6 = section("06", "System payoff", "Sleep on the card, spend 29% less energy",
    """<p>Because the card reloads the model quickly and cheaply, the VLM can sleep on it between fires
    instead of staying resident. Over an 11.5-minute window with two fire episodes (~73% active), the
    gated design wins on every axis and drops to the ~5.5&nbsp;W idle floor between fires while the
    always-on baseline holds ~15–18&nbsp;W.</p>"""
    + table(["Metric", "Burst (gated)", "Always-on", "Δ"], [
        ["Energy", "2.11 Wh", "2.98 Wh", "−29%"],
        ["Avg power", "11.1 W", "15.7 W", "−29%"],
        ["Avg T<sub>j</sub>", "58.0 °C", "63.9 °C", "−9%"],
        ["Avg RAM", "5.2 GB", "6.9 GB", "−25%"],
        ["Avg CPU", "38.7%", "66.1%", "−41%"],
        ["Avg GPU", "29.4%", "55.0%", "−47%"],
    ], good_col=3)
    + figure("F5_system_metrics.png",
             "Figure 5 — Gated vs always-on across energy, power, thermals, memory and utilization.")
    + figure("F6_power_timeline.png",
             "Figure 6 — Board power over the two-fire window: the gated run returns to idle between fires."))

s7 = section("07", "Wake latency", "The card is never the bottleneck",
    """<p>The price of sleeping is wake latency — but that cost is framework- and CUDA-bound, not
    storage-bound. Reading all 2.8&nbsp;GB of weights is a ~4-second sliver of a ~118-second wake;
    AWQ dequant plus GPU init alone is 66 seconds. Halving the card's read bandwidth adds only ~7%.</p>"""
    + table(["Wake phase", "Seconds"], [
        ["docker compose up", "3.9"],
        ["python + imports", "13.3"],
        ["torch / CUDA init", "14.0"],
        ["<strong>AWQ LLM load + GPU init</strong>", "<strong>66.2</strong>"],
        ["startup self-benchmark", "4.2"],
        ["vision tower load", "4.0"],
        ["camera + first inference", "12.8"],
        ["<span class='hot'>actual SD read (embedded)</span>", "<span class='hot'>~4.0</span>"],
    ])
    + table(["Read pacing", "Wake latency"], [
        ["Burst", "150.6 ± 3.5 s"],
        ["Continuous @ 703 MB/s", "152.7 ± 4.0 s"],
        ["Continuous @ 350 MB/s", "162.1 ± 5.7 s"],
    ])
    + figure("F7_wake_anatomy.png",
             "Figure 7 — Anatomy of a 118 s wake with the SD read isolated, and latency vs read pacing."))

takeaways = "".join(f"<li><span>{i:02d}</span><p>{t}</p></li>" for i, t in enumerate([
    "<strong>Fast, consistent reads</strong> — ~700 MB/s flat across the full model, 720 ± 13 MB/s over five cold runs.",
    "<strong>Honest write characterization</strong> — SLC burst 546 MB/s folding to ~200 MB/s sustained, exactly the DRAM-less behavior to expect.",
    "<strong>Access pattern is the lever</strong> — 15× sequential-over-random at 4 KB; chunk size is irrelevant.",
    "<strong>Page cache is free leverage</strong> — cross-process pre-warm reaches 2.6 GB/s.",
    "<strong>Enables a 29% energy win</strong> — quick enough to sleep the whole VLM on the card and reload on demand, and it never becomes the wake bottleneck.",
], start=1))

HTML = f'''<title>SD Express Under Fire</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{{
  --ground:#f6f7f9; --surface:#ffffff; --surface-2:#eef1f5;
  --ink:#151a21; --muted:#59636f; --faint:#8a94a3; --line:#e2e7ee;
  --accent:#0e7c66; --accent-soft:#0e7c6614; --ember:#c05617; --ember-soft:#c0561714;
  --plate:#ffffff; --plate-line:#e6eaf0;
}}
@media (prefers-color-scheme:dark){{
  :root:not([data-theme="light"]){{
    --ground:#0c1015; --surface:#141a22; --surface-2:#1b232d;
    --ink:#e7ecf3; --muted:#9aa5b4; --faint:#69737f; --line:#242e3a;
    --accent:#33c39e; --accent-soft:#33c39e1f; --ember:#e8813f; --ember-soft:#e8813f1f;
    --plate:#f7f8fa; --plate-line:#0c1015;
  }}
}}
:root[data-theme="dark"]{{
  --ground:#0c1015; --surface:#141a22; --surface-2:#1b232d;
  --ink:#e7ecf3; --muted:#9aa5b4; --faint:#69737f; --line:#242e3a;
  --accent:#33c39e; --accent-soft:#33c39e1f; --ember:#e8813f; --ember-soft:#e8813f1f;
  --plate:#f7f8fa; --plate-line:#0c1015;
}}
*{{box-sizing:border-box}}
body{{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,sans-serif; font-size:17px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}}
.wrap{{max-width:900px; margin:0 auto; padding:clamp(28px,6vw,80px) clamp(20px,5vw,40px) 96px}}
.eyebrow{{
  font-family:"IBM Plex Mono",monospace; font-size:12px; font-weight:500;
  letter-spacing:.18em; text-transform:uppercase; color:var(--accent);
}}
h1,h2,h3{{font-family:"Archivo",sans-serif; text-wrap:balance; margin:0}}
p{{margin:0 0 1em; max-width:66ch}}
code{{font-family:"IBM Plex Mono",monospace; font-size:.86em; background:var(--surface-2);
  padding:.08em .38em; border-radius:4px}}
strong{{font-weight:600}}
.hot{{color:var(--ember); font-weight:600}}

/* masthead */
.mast{{border-bottom:1px solid var(--line); padding-bottom:36px; margin-bottom:44px}}
.mast .eyebrow{{margin-bottom:18px}}
.mast h1{{font-size:clamp(38px,7vw,68px); font-weight:800; line-height:1.02; letter-spacing:-.02em}}
.mast h1 em{{font-style:normal; color:var(--accent)}}
.dek{{font-size:clamp(18px,2.4vw,22px); color:var(--muted); margin-top:20px; max-width:60ch;
  font-weight:400}}
.meta{{display:flex; flex-wrap:wrap; gap:8px 26px; margin-top:26px;
  font-family:"IBM Plex Mono",monospace; font-size:12.5px; color:var(--faint);
  letter-spacing:.03em}}
.meta b{{color:var(--muted); font-weight:500}}

/* stat strip */
.stats{{display:grid; grid-template-columns:repeat(5,1fr); gap:1px; background:var(--line);
  border:1px solid var(--line); border-radius:14px; overflow:hidden; margin-bottom:64px}}
.stat{{background:var(--surface); padding:22px 18px}}
.stat-v{{font-family:"Archivo",sans-serif; font-weight:800; font-size:clamp(26px,3.6vw,38px);
  letter-spacing:-.02em; line-height:1; font-variant-numeric:tabular-nums}}
.stat-u{{font-family:"IBM Plex Mono",monospace; font-size:.42em; font-weight:500;
  color:var(--accent); margin-left:.28em; letter-spacing:.02em; vertical-align:.14em}}
.stat-l{{font-size:12px; color:var(--muted); margin-top:11px; line-height:1.35}}
@media(max-width:760px){{
  .stats{{grid-template-columns:repeat(2,1fr)}}
  .stat:last-child{{grid-column:1 / -1}}
}}

/* sections */
section{{margin-bottom:60px}}
.sec-head{{display:flex; gap:20px; align-items:baseline; margin-bottom:20px}}
.sec-num{{font-family:"IBM Plex Mono",monospace; font-size:15px; font-weight:600;
  color:var(--accent); background:var(--accent-soft); border-radius:8px;
  padding:6px 11px; flex:none; letter-spacing:.02em}}
.sec-head h2{{font-size:clamp(25px,3.6vw,34px); font-weight:700; letter-spacing:-.015em;
  line-height:1.08; margin-top:4px}}
.sec-head .eyebrow{{margin-bottom:6px}}

/* tables */
.tbl-wrap{{overflow-x:auto; margin:22px 0; border:1px solid var(--line); border-radius:12px}}
table{{width:100%; border-collapse:collapse; font-size:15px}}
thead th{{font-family:"IBM Plex Mono",monospace; font-size:11px; font-weight:600;
  letter-spacing:.1em; text-transform:uppercase; color:var(--faint); text-align:right;
  padding:13px 18px; background:var(--surface-2); border-bottom:1px solid var(--line)}}
thead th:first-child{{text-align:left}}
tbody td{{padding:12px 18px; border-bottom:1px solid var(--line); background:var(--surface)}}
tbody tr:last-child td{{border-bottom:none}}
td.lbl{{color:var(--ink); text-align:left; font-weight:400}}
td.num{{text-align:right; font-family:"IBM Plex Mono",monospace; font-size:14px;
  font-variant-numeric:tabular-nums; color:var(--muted)}}
td.num strong{{color:var(--ink)}}
td.good{{color:var(--accent); font-weight:600}}

/* figure plates */
.plate{{margin:26px 0 4px; background:var(--plate); border:1px solid var(--plate-line);
  border-radius:14px; padding:18px 18px 0; box-shadow:0 1px 3px rgba(10,20,40,.06)}}
.plate img{{display:block; width:100%; height:auto; border-radius:6px}}
.plate figcaption{{font-size:13px; color:#5b6675; padding:14px 4px 16px; line-height:1.5;
  border-top:1px solid #eef0f4; margin-top:14px; font-family:"IBM Plex Sans",sans-serif}}
.fine{{font-size:13.5px; color:var(--faint); max-width:70ch}}

/* takeaways */
.close{{border-top:1px solid var(--line); padding-top:40px; margin-top:24px}}
.close h2{{font-size:clamp(24px,3.4vw,30px); font-weight:700; letter-spacing:-.015em;
  margin-bottom:8px}}
ol.take{{list-style:none; padding:0; margin:26px 0 0; display:flex; flex-direction:column; gap:2px}}
ol.take li{{display:flex; gap:18px; align-items:baseline; padding:16px 0;
  border-bottom:1px solid var(--line)}}
ol.take li:last-child{{border-bottom:none}}
ol.take span{{font-family:"IBM Plex Mono",monospace; font-size:13px; font-weight:600;
  color:var(--accent); flex:none}}
ol.take p{{margin:0}}
ol.take strong{{font-weight:600}}
.foot{{margin-top:48px; font-family:"IBM Plex Mono",monospace; font-size:12px;
  color:var(--faint); letter-spacing:.03em; line-height:1.7}}
</style>

<div class="wrap">
  <header class="mast">
    <div class="eyebrow">Storage characterization · edge AI</div>
    <h1>SD Express, <em>under fire</em></h1>
    <p class="dek">A DRAM-less SD Express card, measured inside a real wildfire-detection VLM pipeline
    on a 7.4&nbsp;GB Jetson — fast enough that storage never bottlenecks, quiet enough to let the model
    sleep on it.</p>
    <div class="meta">
      <span><b>Device</b> SanDisk SDSQXFN · SN530</span>
      <span><b>Host</b> Jetson Orin Nano</span>
      <span><b>Payload</b> 2.8 GB VLM + classifier</span>
    </div>
  </header>

  <div class="stats">{stats}</div>

  {s1}{s2}{s3}{s4}{s5}{s6}{s7}

  <section class="close">
    <div class="eyebrow">For the judges</div>
    <h2>Five things this card gives the pipeline</h2>
    <ol class="take">{takeaways}</ol>
    <p class="foot">All figures generated from raw measurements on the device · benchmarks:
    bench_seq_read · bench_rand_read · bench_write · make_figures</p>
  </section>
</div>'''

out = HERE / "sd_express_showcase.html"
out.write_text(HTML)
print("wrote", out, f"({len(HTML)/1024:.0f} KB)")

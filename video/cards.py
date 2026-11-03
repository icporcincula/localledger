"""Render the title, Stage B results and closing cards as 1920x1080 PNGs (same look as the app).

  python cards.py        # writes cards/title.png, cards/stageb.png, cards/close.png
The Stage B card reads results/recall.jsonl directly, so its numbers cannot drift from the data.
"""
import collections
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
RECALL = HERE.parent / "results" / "recall.jsonl"
OUT = HERE / "cards"

CSS = """
:root { --bg:#f3f1ec; --panel:#fffdf8; --panel2:#f6f3ea; --ink:#1d2328; --mute:#667078; --line:#d9d5cb;
        --accent:#0b6e5c; --warn:#b4540a; }
* { box-sizing:border-box } html,body { margin:0; width:1920px; height:1080px; }
body { background:var(--bg); color:var(--ink); font:30px/1.45 "Segoe UI",system-ui,sans-serif;
       display:flex; flex-direction:column; justify-content:center; padding:0 190px; }
.brand { font-size:34px; font-weight:650; letter-spacing:.2px; margin-bottom:34px; }
.brand span { color:var(--mute); font-weight:400; font-size:26px; margin-left:14px; }
h1 { font-size:84px; line-height:1.08; margin:0 0 26px; letter-spacing:-.5px; }
p.lead { font-size:38px; color:var(--mute); margin:0 0 54px; max-width:1350px; }
.two { display:grid; grid-template-columns:1fr 1fr; gap:28px; }
.box { background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:30px 36px; }
.box h3 { margin:0 0 10px; font-size:22px; letter-spacing:.1em; text-transform:uppercase; color:var(--mute); font-weight:600; }
.box.hit { border-color:var(--accent); } .box.hit h3 { color:var(--accent); }
.box p { margin:0; font-size:32px; line-height:1.4; }
table { border-collapse:separate; border-spacing:0; width:100%; background:var(--panel); border:1px solid var(--line);
        border-radius:18px; overflow:hidden; font-variant-numeric:tabular-nums; }
th, td { padding:22px 26px; text-align:center; font-size:40px; border-bottom:1px solid var(--line); }
th { font-size:22px; letter-spacing:.1em; text-transform:uppercase; color:var(--mute); font-weight:600; background:var(--panel2); }
td:first-child, th:first-child { text-align:left; font-weight:650; }
tr:last-child td { border-bottom:0; }
td.ok { color:var(--accent); font-weight:650; } td.part { color:var(--warn); font-weight:650; }
td.mute { color:var(--mute); font-size:32px; }
.foot { margin-top:30px; font-size:25px; color:var(--mute); max-width:1500px; line-height:1.5; }
.url { font-size:62px; font-weight:650; color:var(--accent); margin:12px 0 40px; letter-spacing:-.3px; }
ul { margin:0; padding:0; list-style:none; font-size:30px; color:var(--mute); line-height:1.7; } ul b { color:var(--ink); font-weight:600; }
"""


def page(body):
    return f"<!doctype html><meta charset=utf-8><style>{CSS}</style><body>{body}</body>"


def title_card():
    return page("""
<div class=brand>LocalLedger<span>Private answers from your confidential documents</span></div>
<h1>Ask a confidential document questions.<br>Nothing leaves your computer.</h1>
<p class=lead>Meta's Muse Glimmer 30B reads the whole document on one 16 GB gaming graphics card.</p>
<div class=two>
 <div class=box><h3>Before this week</h3><p>Private speed tests and a command-line demo of the model on one gaming card.</p></div>
 <div class="box hit"><h3>Built this week</h3><p>The app, the answer checker, and the recall tests.</p></div>
</div>""")


def stageb_card():
    rows = [json.loads(l) for l in open(RECALL, encoding="utf-8") if l.strip()]
    cell = collections.defaultdict(lambda: [0, 0])
    peak = collections.defaultdict(int)
    for r in rows:
        c = cell[(r["kv"], r["length"])]
        c[0] += r["ok"]
        c[1] += 1
        peak[r["kv"]] = max(peak[r["kv"]], r.get("peak_mib") or 0)
    total = {kv: [sum(cell[(kv, L)][0] for L in ("32k", "64k", "124k")), sum(cell[(kv, L)][1] for L in ("32k", "64k", "124k"))]
             for kv in ("f16", "q8_0", "q4_0")}
    names = {"f16": "Full precision (f16)", "q8_0": "8-bit (q8_0)", "q4_0": "4-bit (q4_0)"}
    tr = ""
    for kv in ("f16", "q8_0", "q4_0"):
        tds = ""
        for L in ("32k", "64k", "124k"):
            ok, n = cell[(kv, L)]
            cls = "mute" if n == 0 else ("ok" if ok == n else "part")
            tds += f"<td class={cls}>{f'{ok} / {n}' if n else '-'}</td>"
        ok, n = total[kv]
        cls = "mute" if n == 0 else ("ok" if ok == n else "part")
        tds += f"<td class={cls}>{f'{ok} / {n}' if n else '-'}</td>"
        tds += f"<td class=mute>{peak[kv] / 1024:.1f} GB</td>" if peak[kv] else "<td class=mute>-</td>"
        tr += f"<tr><td>{names[kv]}</td>{tds}</tr>"
    html = page(f"""
<div class=brand>LocalLedger<span>Does a smaller memory cost recall?</span></div>
<h1 style="font-size:64px;margin-bottom:34px">Hidden codes found, by memory type and document length</h1>
<table><tr><th>Cache type</th><th>32K tokens</th><th>64K tokens</th><th>124K tokens</th><th>All</th><th>Peak GPU memory</th></tr>{tr}</table>
<div class=foot>Six random codes hidden in each synthetic document, with look-alike decoys, at 5% to 95% depth. Two documents per cell,
so 12 questions per cell. Exact code must appear in the answer. One machine (RTX 5060 Ti 16 GB). Small samples: read the counts, not the percentages.</div>""")
    return html


def close_card():
    return page("""
<div class=brand>LocalLedger</div>
<h1 style="font-size:72px">Documents that cannot leave the building<br>can finally be asked questions.</h1>
<div class=url>github.com/icporcincula/localledger</div>
<ul>
 <li><b>Open source</b>, MIT license. Runs offline on one 16 GB NVIDIA card.</li>
 <li><b>Model:</b> Meta Muse Glimmer 30B (open weights) with llama.cpp. <b>Built with</b> Claude Code.</li>
 <li><b>Narration:</b> AI voice (Audio8 TTS, Apache 2.0), cloned from the author's own voice.</li>
</ul>""")


def main():
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1920, "height": 1080})
        for name, html in (("title", title_card()), ("stageb", stageb_card()), ("close", close_card())):
            pg.set_content(html)
            pg.wait_for_timeout(300)
            pg.screenshot(path=str(OUT / f"{name}.png"))
            print("wrote", OUT / f"{name}.png")
        b.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

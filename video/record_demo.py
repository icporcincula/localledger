"""Drive the real LocalLedger page and record it.

  python record_demo.py --rehearse        # headless dry run: checks every step, writes rehearsal_marks.json
  python record_demo.py --record          # headless page screencast at 1920x1080 -> take/take.mp4 + take/marks.json

Needs the app already running (python -m localledger.app ...) with the model loaded. --record also waits for the
internet to be switched off first (the page's own "Offline" pill is what ends up on screen). Nothing here is
scripted output: every answer, quote, page number and timing on screen comes from the live model.
"""
import os
import argparse
import json
import math
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

FF = os.environ.get("FFMPEG", "ffmpeg")
HERE = Path(__file__).parent
RFC = str(Path(__file__).resolve().parent.parent / "results" / "rfc7540.txt")
DSF = 1.5  # 1920x1080 screen -> 1280x720 css px, so the page reads at video size

Q1 = "Which locker holds the spare chlorine cartridges?"
Q2 = "Who is authorised to issue boil-water notices, and which extension?"
Q3 = "What is the annual budget of the Pellam depot?"          # not in the document
Q4 = "Which error code is used when the peer seems to generate excessive load, and what is its value?"
Q5 = "What colour is the emergency shutoff wheel at Ravensmoor Tank?"

CURSOR_JS = """(() => {
  const mk = () => {
    if (document.getElementById('__cur')) return;
    const c = document.createElement('div'); c.id = '__cur';
    c.style.cssText = 'position:fixed;left:0;top:0;width:26px;height:26px;z-index:2147483647;pointer-events:none;transform:translate(-100px,-100px)';
    c.innerHTML = '<svg width="26" height="26" viewBox="0 0 24 24"><path d="M3 2 L3 19 L8 14.5 L11.5 22 L14.5 20.6 L11 13.2 L18 13 Z" fill="#111" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    document.documentElement.appendChild(c);
    const r = document.createElement('div'); r.id = '__rip';
    r.style.cssText = 'position:fixed;width:34px;height:34px;margin:-17px 0 0 -17px;border-radius:50%;border:3px solid #0b6b5a;z-index:2147483646;pointer-events:none;opacity:0';
    document.documentElement.appendChild(r);
    addEventListener('mousemove', e => { c.style.transform = `translate(${e.clientX - 3}px,${e.clientY - 2}px)`; }, true);
    addEventListener('mousedown', e => { r.style.left = e.clientX + 'px'; r.style.top = e.clientY + 'px';
      r.animate([{opacity: .9, transform: 'scale(.4)'}, {opacity: 0, transform: 'scale(1.6)'}], {duration: 450}); }, true);
  };
  if (document.readyState === 'loading') addEventListener('DOMContentLoaded', mk); else mk();
})();"""

STATE_JS = """() => ({
  go_disabled: document.getElementById('go').disabled,
  prog_hidden: document.getElementById('prog').hidden,
  progtxt: document.getElementById('progtxt').textContent,
  bar: document.getElementById('progbar').style.width,
})"""

RESULT_JS = """() => {
  const b = document.querySelector('#log .qa'); if (!b) return null;
  const src = b.querySelector('.src');
  return {q: b.querySelector('.q').textContent, a: b.querySelector('.a').textContent,
          src: src && !src.hidden ? src.innerText : '',
          tiles: [...b.querySelectorAll('.tile')].map(t => t.innerText.replace(/\\n/g, ' ')),
          bad: !!(src && src.classList.contains('bad'))};
}"""


class Screencast:
    """Record only the page (no desktop) via CDP screencast: JPEG frames with timestamps, assembled to take.mp4."""

    def __init__(self, ctx, page, outdir):
        self.dir = Path(outdir).resolve() / "frames"
        if self.dir.exists():
            for f in self.dir.glob("*.jpg"):
                f.unlink()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.frames = []
        self.t0 = time.time()
        self.cdp = ctx.new_cdp_session(page)
        self.cdp.on("Page.screencastFrame", self._on)

    def _on(self, p):
        import base64
        i = len(self.frames)
        path = self.dir / f"{i:06d}.jpg"
        path.write_bytes(base64.b64decode(p["data"]))
        self.frames.append((p["metadata"]["timestamp"], path))
        self.cdp.send("Page.screencastFrameAck", {"sessionId": p["sessionId"]})

    def start(self):
        self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": 1920, "maxHeight": 1080,
                                                "everyNthFrame": 1})

    def stop_and_encode(self, out_mp4):
        self.cdp.send("Page.stopScreencast")
        fr = self.frames
        if not fr:
            raise SystemExit("no frames captured")
        base = fr[0][0]
        lines = []
        for k, (ts, path) in enumerate(fr):
            nxt = fr[k + 1][0] if k + 1 < len(fr) else ts + 0.5
            lines.append(f"file '{path.as_posix()}'\nduration {max(nxt - ts, 0.001):.4f}\n")
        lines.append(f"file '{fr[-1][1].as_posix()}'\n")
        lst = self.dir / "list.txt"
        lst.write_text("".join(lines), encoding="utf-8")
        subprocess.run([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-vf", "fps=30,scale=1920:1080:flags=lanczos,format=yuv420p", "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "16", str(out_mp4)], check=True)
        return fr[0][0] - self.t0   # how long after t0 the first frame arrived


def online():
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=1.5).close()
        return True
    except OSError:
        return False


class Rec:
    def __init__(self, page, capture):
        self.page, self.capture = page, capture
        self.t0 = time.perf_counter()
        self.marks, self.results = {}, {}
        self.mx, self.my = 640, 360

    def t(self):
        return round(time.perf_counter() - self.t0, 2)

    def mark(self, name):
        self.marks[name] = self.t()
        print(f"[{self.t():7.1f}s] {name}", flush=True)

    def hold(self, s):
        self.page.wait_for_timeout(int(s * 1000))

    def glide(self, x, y):
        x0, y0 = self.mx, self.my
        n = int(min(90, max(25, math.hypot(x - x0, y - y0) / 8)))
        for i in range(1, n + 1):
            f = i / n
            f = f * f * (3 - 2 * f)  # smoothstep
            self.page.mouse.move(x0 + (x - x0) * f, y0 + (y - y0) * f)
            self.page.wait_for_timeout(12)
        self.mx, self.my = x, y

    def bring(self, sel, block="center", settle=0.9):
        self.page.evaluate(f"document.querySelector({json.dumps(sel)}).scrollIntoView({{behavior:'smooth',block:{json.dumps(block)}}})")
        self.hold(settle)

    def scroll_to(self, y, settle=1.0):
        self.page.evaluate(f"window.scrollTo({{top:{y},behavior:'smooth'}})")
        self.hold(settle)

    def click(self, sel, dx=0.0, dy=0.0):
        box = self.page.locator(sel).first.bounding_box()
        vh = self.page.evaluate("innerHeight")
        if box is None or box["y"] < 40 or box["y"] + box["height"] > vh - 40:
            self.bring(sel)
            box = self.page.locator(sel).first.bounding_box()
        x, y = box["x"] + box["width"] * (0.5 + dx), box["y"] + box["height"] * (0.5 + dy)
        self.glide(x, y)
        self.hold(0.25)
        self.page.mouse.down()
        self.hold(0.08)
        self.page.mouse.up()
        self.hold(0.3)

    def type_into(self, sel, text):
        self.click(sel)
        self.page.keyboard.press("Control+A")
        self.page.keyboard.type(text, delay=42)
        self.hold(0.5)

    def ask(self, name, question, via=None, timeout=900):
        """Type (or click a chip) and wait for the answer to finish. Marks <name>_asked/_reading/_answering/_done."""
        if via == "chip":
            self.click(f"#chips button[title={json.dumps(question)}]")
        else:
            self.type_into("#q", question)
            self.hold(0.3)
            self.click("#go")
        self.mark(f"{name}_asked")
        self.page.wait_for_function("document.getElementById('go').disabled", timeout=15000)
        t_start, seen = time.time(), set()
        while True:
            s = self.page.evaluate(STATE_JS)
            txt = s["progtxt"]
            if txt.startswith("Reading") and "reading" not in seen:
                seen.add("reading"); self.mark(f"{name}_reading")
            if txt.startswith("Reading only") and "new" not in seen:
                seen.add("new"); self.mark(f"{name}_readnew")
            if txt.startswith("restored") and "restore" not in seen:
                seen.add("restore"); self.mark(f"{name}_restored"); self.results[f"{name}_note"] = txt
            if txt == "Answering..." and "answering" not in seen:
                seen.add("answering"); self.mark(f"{name}_answering")
            if not s["go_disabled"]:
                break
            if time.time() - t_start > timeout:
                raise TimeoutError(f"{name}: no answer within {timeout}s")
            self.page.wait_for_timeout(250)
        self.mark(f"{name}_done")
        self.results[name] = self.page.evaluate(RESULT_JS)
        print("   ->", json.dumps(self.results[name], ensure_ascii=False)[:400], flush=True)


def run(rec, hold_scale=1.0):
    h = lambda s: rec.hold(s * hold_scale)
    page = rec.page
    page.goto("http://127.0.0.1:8802")
    page.wait_for_function("!document.getElementById('net').textContent.includes('checking')", timeout=15000)
    rec.hold(1.0)
    rec.mark("hero")
    h(8.0)

    rec.click("#sample")
    rec.mark("sample_clicked")
    page.wait_for_function("document.getElementById('docinfo').textContent.includes('pages loaded')", timeout=120000)
    rec.mark("loaded")
    rec.results["docinfo"] = page.evaluate("document.getElementById('docinfo').textContent")
    h(2.5)
    rec.bring("#q", block="start", settle=1.0)
    rec.scroll_to(page.evaluate("document.getElementById('q').getBoundingClientRect().top + scrollY - 70"))

    rec.ask("q1", Q1)
    h(7.0)
    rec.ask("q2", Q2, via="chip")
    h(5.0)
    rec.ask("q3", Q3)
    h(4.0)

    # a second, real document: the HTTP/2 RFC. Upload, read it once, then jump back to the binder.
    rec.mark("upload_start")
    page.set_input_files("#file", RFC)
    page.wait_for_function("document.querySelectorAll('#docs button').length >= 2", timeout=60000)
    rec.mark("upload_done")
    rec.scroll_to(page.evaluate("document.getElementById('docs').getBoundingClientRect().top + scrollY - 40"))
    h(2.5)
    rec.ask("q4", Q4)
    h(6.0)
    rec.mark("switch_start")
    rec.click("#docs button:nth-child(1)")
    h(1.5)
    rec.ask("q5", Q5)
    h(5.0)

    # for the IT person
    rec.mark("it_start")
    rec.bring("details.tech summary", block="center")
    rec.click("details.tech summary")
    rec.scroll_to(page.evaluate("document.querySelector('details.tech').getBoundingClientRect().top + scrollY - 20"), 1.2)
    rec.mark("it_open")
    h(6.0)
    rec.bring("#vram", block="center")
    box = page.locator("#vram").bounding_box()
    rec.glide(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
    page.mouse.down(); h(0.1); page.mouse.up()
    rec.mark("adv_start")
    for _ in range(4):
        page.keyboard.press("ArrowLeft")
        h(0.45)
    h(5.0)
    rec.mark("adv_done")
    rec.bring("#chart", block="center")
    rec.mark("chart")
    h(5.0)
    rec.scroll_to(0, 1.4)
    rec.mark("end")
    h(3.0)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rehearse", action="store_true")
    g.add_argument("--record", action="store_true")
    ap.add_argument("--out", default=str(HERE / "take"))
    ap.add_argument("--no-offline-wait", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    if a.record:
        if not a.no_offline_wait:
            print(">>> Turn Wi-Fi OFF now (and enable Do Not Disturb). Waiting for the internet to drop ...", flush=True)
            ok = 0
            while ok < 3:
                ok = ok + 1 if not online() else 0
                time.sleep(2)
            print("offline confirmed, starting in 5 s. Keep Wi-Fi off until this finishes.", flush=True)
            import winsound
            winsound.Beep(880, 400)
            time.sleep(5)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 720}, device_scale_factor=DSF, color_scheme="light")
        ctx.add_init_script(CURSOR_JS)
        page = ctx.new_page()
        page.goto("about:blank")
        page.wait_for_timeout(500)
        rec = Rec(page, a.record)
        sc = None
        if a.record:
            sc = Screencast(ctx, page, out)
            sc.start()
            rec.t0 = time.perf_counter()
            sc.t0 = time.time()
            rec.mark("capture_start")
        try:
            run(rec)
        finally:
            name = "marks.json" if a.record else "rehearsal_marks.json"
            json.dump({"marks": rec.marks, "results": rec.results, "online_at_end": online()},
                      open(out / name, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
            if sc:
                lag = sc.stop_and_encode(out / "take.mp4")
                print(f"take.mp4 written, {len(sc.frames)} frames, first frame {lag:.2f}s after start")
            browser.close()
    print("done")
    try:
        import winsound
        for _ in range(3):
            winsound.Beep(1200, 300)
            time.sleep(0.15)
    except Exception:
        pass


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

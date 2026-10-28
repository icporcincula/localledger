"""Offline long-document Q&A: one page, one local model, nothing leaves the machine.

  python -m localledger.app --ngl 48 --kv q8_0          # starts llama-server itself
  python -m localledger.app --attach 8801               # reuse a running llama-server
                                                            # (start it with --slot-save-path to
                                                            # enable instant switching between documents)
"""
import argparse
import json
import re
import shutil
import socket
import threading
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import advisor, corpus
from .llama import PeakVram, Server, count_tokens, slot_restore, slot_save, stream_chat, vram_mib

ROOT = Path(__file__).parent.parent
RESULTS = str(ROOT / "results" / "bench.jsonl")
SLOT_DIR = ROOT / "results" / "slots"
MAX_UPLOAD = 2_000_000  # ~500K tokens of text; the context window is the real limit
ANSWER_ROOM = 1500      # tokens kept free for the question, reasoning and answer

STATE = {"docs": {}, "active": None, "slot_doc": None, "saved": set(), "base": None,
         "cfg": {}, "slots_ok": False, "seq": 0}
LOCK = threading.Lock()  # one model, one slot: serialise questions and document switches
PAGE = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
SYSTEM = ("Answer using only the document. Be brief, but answer every part of the question. "
          "If the document does not say, reply 'not in the document' and nothing else. "
          "Otherwise end with a last line: Source: \"<the one sentence from the document that "
          "supports the answer, copied word for word>\"")
WORDS_PER_PAGE = 500  # same convention as the page counts in the UI (index.html)


def find_evidence(doc_text, answer):
    """Pull the quoted source line out of the answer and check it really is in the document."""
    m = re.search(r"\n\s*Source:\s*(.+?)\s*$", answer, re.S)
    if not m:
        return None
    quote = m.group(1).strip().strip('"“”').strip()
    if not quote:
        return None
    flat = re.sub(r"\s+", " ", doc_text)
    idx = flat.find(re.sub(r"\s+", " ", quote))
    if idx < 0:
        return {"quote": quote, "found": False}
    return {"quote": quote, "found": True, "page": len(flat[:idx].split()) // WORDS_PER_PAGE + 1,
            "pages": len(flat.split()) // WORDS_PER_PAGE + 1}


def internet_reachable():
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=1.5).close()
        return True
    except OSError:
        return False


def model_up():
    try:
        with urllib.request.urlopen(STATE["base"] + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


MODEL_DOWN = ("The model is not responding. It may have run out of memory or been closed. "
              "Restart the app, then load the document again.")


def add_doc(name, text):
    STATE["seq"] += 1
    did = f"d{STATE['seq']}"
    tokens = count_tokens(STATE["base"], text)
    STATE["docs"][did] = {"name": name, "text": text, "tokens": tokens}
    STATE["active"] = did
    return did


def doc_summary():
    ctx = STATE["cfg"].get("ctx", 0)
    return [{"id": k, "name": v["name"], "tokens": v["tokens"], "active": k == STATE["active"],
             "cached": k == STATE["slot_doc"] or k in STATE["saved"],
             "too_long": bool(ctx) and v["tokens"] + ANSWER_ROOM > ctx}
            for k, v in STATE["docs"].items()]


def switch_slot(did):
    """Make the model's cache hold `did`: park the old document to disk, restore the new one."""
    note = None
    cur = STATE["slot_doc"]
    if cur == did or not STATE["slots_ok"]:
        STATE["slot_doc"] = did
        return note
    try:
        if cur and cur in STATE["docs"]:
            slot_save(STATE["base"], f"{cur}.bin")
            STATE["saved"].add(cur)
        if did in STATE["saved"]:
            secs = slot_restore(STATE["base"], f"{did}.bin")
            note = f"restored cache from disk in {secs:.1f} s"
    except Exception as e:
        STATE["slots_ok"] = False  # server was not started with --slot-save-path
        STATE["saved"].clear()
        note = "could not restore the saved memory for this document, so it is being re-read (slower this once)"
    STATE["slot_doc"] = did
    return note


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/":
            return self._send(200, PAGE, "text/html")
        if u.path == "/api/status":
            used, total = vram_mib()
            return self._send(200, json.dumps({
                "docs": doc_summary(), "cfg": STATE["cfg"], "slots": STATE["slots_ok"],
                "online": internet_reachable(), "model_up": model_up(),
                "vram_used": used, "vram_total": total}))
        if u.path == "/api/sample":
            if not model_up():
                return self._send(503, json.dumps({"error": MODEL_DOWN}))
            words = int(q.get("words", ["30000"])[0])
            text, qs = corpus.build(words, seed=words)
            with LOCK:
                add_doc(f"Water-utility binder ({words // 1000}K words)", text)
            return self._send(200, json.dumps({"questions": [x["question"] for x in qs]}))
        if u.path == "/api/select":
            did = q.get("id", [""])[0]
            if did not in STATE["docs"]:
                return self._send(404, json.dumps({"error": "unknown document"}))
            STATE["active"] = did
            return self._send(200, "{}")
        if u.path == "/api/bench":
            return self._bench()
        if u.path == "/api/advise":
            return self._advise(int(q.get("vram_gb", ["16"])[0]), int(q.get("tokens", ["100000"])[0]))
        if u.path == "/api/ask":
            return self._ask(q.get("q", [""])[0], careful=q.get("mode", ["quick"])[0] == "careful")
        self._send(404, "{}")

    def do_POST(self):
        if self.path == "/api/upload":
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_UPLOAD:
                return self._send(413, json.dumps({"error": f"file over {MAX_UPLOAD // 1_000_000} MB"}))
            name = urllib.parse.unquote(self.headers.get("X-Filename", "document.txt"))
            raw = self.rfile.read(n)
            if raw[:5] == b"%PDF-" or raw[:2] == b"PK" or b"\x00" in raw[:4096]:
                return self._send(400, json.dumps({"error": (
                    "this looks like a PDF or Office file. Save or export it as plain text "
                    "(.txt or .md) and load that instead.")}))
            text = raw.decode("utf-8", "replace")
            if not text.strip():
                return self._send(400, json.dumps({"error": "the file is empty"}))
            if not model_up():
                return self._send(503, json.dumps({"error": MODEL_DOWN}))
            with LOCK:
                add_doc(name, text)
            return self._send(200, "{}")
        self._send(404, "{}")

    def _bench(self):
        rows, _ = advisor.load_rows(RESULTS)
        out = []
        for r in rows:
            qs = r["questions"]
            dec = [x["decode_per_s"] for x in qs if x["decode_per_s"]]
            out.append({"ngl": r["ngl"], "kv": r["kv"], "tokens": r["doc_tokens"],
                        "decode": sum(dec) / len(dec) if dec else 0, "peak": r["peak_mib"],
                        "prefill": qs[0]["prompt_per_s"], "cold_s": qs[0]["wall_s"],
                        "follow_s": sum(x["wall_s"] for x in qs[1:]) / max(len(qs) - 1, 1),
                        "correct": sum(x["ok"] for x in qs), "asked": len(qs)})
        self._send(200, json.dumps(out))

    def _advise(self, vram_gb, tokens):
        rows, failures = advisor.load_rows(RESULTS)
        p = advisor.Profile(rows, failures)
        recs = advisor.recommend(p, vram_gb * 1024, tokens, 600, advisor.HEADROOM_MIB)
        self._send(200, json.dumps({"recs": recs[:5], "ctx": p.ctx, "total_mib": vram_gb * 1024}))

    def _ask(self, question, careful=False):
        doc = STATE["docs"].get(STATE["active"])
        if not doc:
            return self._send(400, json.dumps({"error": "load a document first"}))
        ctx = STATE["cfg"].get("ctx", 0)
        if ctx and doc["tokens"] + ANSWER_ROOM > ctx:
            return self._send(400, json.dumps({"error": (
                f"this document is {doc['tokens']:,} tokens; the loaded context holds {ctx:,} "
                f"with room for the answer. Shorten it or restart with a larger --ctx.")}))
        if not model_up():
            return self._send(503, json.dumps({"error": MODEL_DOWN}))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"<document>\n{doc['text']}\n</document>\n\nQuestion: {question}"}]
        try:
            with LOCK:
                note = switch_slot(STATE["active"])
                if note:
                    self._event("note", note)
                finished, answer = False, []
                with PeakVram() as pv:
                    for kind, val in stream_chat(STATE["base"], msgs, max_tokens=1024,
                                                   reasoning="high" if careful else "low"):
                        finished = finished or kind == "timings"
                        if kind == "text":
                            answer.append(val)
                        self._event(kind, val)
                    ev = find_evidence(doc["text"], "".join(answer))
                    if ev:
                        self._event("evidence", ev)
                    self._event("peak", pv.peak)
                if not finished:
                    self._event("error", MODEL_DOWN if not model_up() else "The model stopped before finishing. Ask again.")
        except Exception:
            # a reset can come from the browser closing (nothing to tell it) or from the model dying
            try:
                self._event("error", MODEL_DOWN if not model_up() else
                            "Something went wrong reading this document. Ask again, or try a shorter one.")
            except OSError:
                pass

    def _event(self, kind, val):
        self.wfile.write(f"data: {json.dumps({'k': kind, 'v': val})}\n\n".encode("utf-8"))
        self.wfile.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngl", type=int, default=48)
    ap.add_argument("--kv", default="q8_0")
    ap.add_argument("--ctx", type=int, default=106496)
    ap.add_argument("--attach", type=int, help="port of a running llama-server")
    ap.add_argument("--llama-port", type=int, default=8801)
    ap.add_argument("--port", type=int, default=8802)
    a = ap.parse_args()
    srv = None
    shutil.rmtree(SLOT_DIR, ignore_errors=True)  # parked caches contain document content
    SLOT_DIR.mkdir(parents=True, exist_ok=True)
    if a.attach:
        STATE["base"] = f"http://127.0.0.1:{a.attach}"
        STATE["slots_ok"] = True  # verified on first switch; falls back if the flag is missing
    else:
        print("loading model ...", flush=True)
        srv = Server(a.ngl, a.kv, a.ctx, a.llama_port,
                     extra=["--slot-save-path", str(SLOT_DIR)]).start()
        STATE["base"] = srv.base
        STATE["slots_ok"] = True
    STATE["cfg"] = {"ngl": a.ngl, "kv": a.kv, "ctx": a.ctx}
    print(f"ready: http://127.0.0.1:{a.port}", flush=True)
    try:
        ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
    finally:
        if srv:
            srv.stop()
        shutil.rmtree(SLOT_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()

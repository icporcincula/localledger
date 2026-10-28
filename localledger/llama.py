"""Thin wrapper around llama-server: launch, health, VRAM sampling, streaming chat."""
import json
import os
import subprocess
import threading
import time
import urllib.request

DEFAULT_SERVER = os.environ.get("LLAMA_SERVER", "llama-server")  # a full path, or on PATH
DEFAULT_MODEL = os.environ.get("GLIMMER_GGUF", "")


def vram_mib():
    """(used, total) MiB for GPU 0 via nvidia-smi."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    used, total = (int(x) for x in out.split(","))
    return used, total


class PeakVram:
    """Background poller that remembers the highest VRAM use seen."""

    def __init__(self, every=0.4):
        self.every, self.peak, self._stop = every, 0, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, vram_mib()[0])
            except Exception:
                pass
            self._stop.wait(self.every)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=3)


class Server:
    def __init__(self, ngl, kv="q8_0", ctx=106496, port=8801, model=DEFAULT_MODEL,
                 exe=DEFAULT_SERVER, extra=(), log_path=None):
        self.ngl, self.kv, self.ctx, self.port = ngl, kv, ctx, port
        self.model, self.exe, self.extra, self.log_path = model, exe, list(extra), log_path
        self.proc = None

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout=420):
        if not self.model:
            raise SystemExit("Set GLIMMER_GGUF to the path of the Muse Glimmer 30B GGUF file "
                             "(and LLAMA_SERVER to llama-server if it is not on PATH).")
        cmd = [self.exe, "-m", self.model, "-c", str(self.ctx), "-ngl", str(self.ngl),
               "-fa", "on", "-ctk", self.kv, "-ctv", self.kv, "--jinja", "-np", "1",
               "--host", "127.0.0.1", "--port", str(self.port), "--no-webui", *self.extra]
        log = open(self.log_path, "wb") if self.log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early (code {self.proc.returncode})")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                    if r.status == 200:
                        return self
            except Exception:
                time.sleep(2)
        self.stop()
        raise TimeoutError("llama-server did not become healthy")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                           capture_output=True)
        self.proc = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def stream_chat(base, messages, max_tokens=1024, reasoning="low", cache=True):
    """Yield events: ('progress', fraction), ('reasoning', text), ('text', text), ('timings', dict)."""
    body = {"messages": messages, "max_tokens": max_tokens, "stream": True,
            "cache_prompt": cache, "return_progress": True, "temperature": 0.2,
            "chat_template_kwargs": {"reasoning_strength": reasoning}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            ev = json.loads(line[5:])
            prog = ev.get("prompt_progress")
            if prog and prog.get("total"):
                yield "progress", (prog["processed"] / prog["total"], prog.get("cache", 0))
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("reasoning_content"):
                    yield "reasoning", d["reasoning_content"]
                if d.get("content"):
                    yield "text", d["content"]
            if ev.get("timings"):
                yield "timings", ev["timings"]


def _post(base, path, body, timeout=600):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def count_tokens(base, text):
    """Exact token count from the running model; falls back to ~4.2 chars/token."""
    try:
        return len(_post(base, "/tokenize", {"content": text}, timeout=120)["tokens"])
    except Exception:
        return int(len(text) / 4.2)


def slot_save(base, filename):
    """Write the slot's KV cache to the server's --slot-save-path. Returns seconds taken."""
    t0 = time.time()
    _post(base, "/slots/0?action=save", {"filename": filename})
    return time.time() - t0


def slot_restore(base, filename):
    t0 = time.time()
    _post(base, "/slots/0?action=restore", {"filename": filename})
    return time.time() - t0


def ask(base, doc, question, max_tokens=1024, cache=True):
    """Blocking helper: returns dict(answer, timings, wall_s)."""
    sys_prompt = ("Answer using only the document. Be brief. "
                  "If the document does not say, reply 'not in the document'.")
    messages = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"<document>\n{doc}\n</document>\n\nQuestion: {question}"}]
    t0, text, timings = time.time(), [], {}
    for kind, val in stream_chat(base, messages, max_tokens, cache=cache):
        if kind == "text":
            text.append(val)
        elif kind == "timings":
            timings = val
    return {"answer": "".join(text).strip(), "timings": timings, "wall_s": time.time() - t0}

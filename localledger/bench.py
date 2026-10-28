"""Measure offload configs on long documents. Appends one JSON line per (config, doc length).

Per row: cold first question (full prefill), then follow-ups over the cached prefix,
with accuracy per needle depth. Resumable: finished (ngl, kv, ctx, words) rows are skipped.
"""
import argparse
import json
import os
import sys

from . import corpus
from .llama import PeakVram, Server, ask, vram_mib

TOKENS_PER_WORD = 1.45  # rough; real count is read back from the server


def done_keys(path):
    if not os.path.exists(path):
        return set()
    keys = set()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        keys.add((r["ngl"], r["kv"], r["ctx"], r["words"]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="48:q8_0,44:q8_0,44:f16,40:f16",
                    help="comma list of ngl:kv")
    ap.add_argument("--lengths", default="8000,32000,64000,96000", help="doc token targets")
    ap.add_argument("--ctx", type=int, default=106496)
    ap.add_argument("--out", default="results/bench.jsonl")
    ap.add_argument("--port", type=int, default=8801)
    ap.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    seen = done_keys(a.out)
    lengths = [int(x) for x in a.lengths.split(",")]
    for cfg in a.configs.split(","):
        ngl, kv = cfg.split(":")
        ngl = int(ngl)
        todo = [t for t in lengths if (ngl, kv, a.ctx, int(t / TOKENS_PER_WORD)) not in seen]
        if not todo:
            continue
        idle = vram_mib()[0]
        print(f"== ngl {ngl} kv {kv} (idle {idle} MiB)", flush=True)
        log = f"results/server_{ngl}_{kv}.log"
        try:
            srv = Server(ngl, kv, a.ctx, a.port, log_path=log).start()
        except Exception as e:
            print("  load failed:", e, flush=True)
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ngl": ngl, "kv": kv, "ctx": a.ctx, "words": 0,
                                    "error": str(e), "idle_mib": idle}) + "\n")
            continue
        try:
            for target in todo:
                words = int(target / TOKENS_PER_WORD)
                doc, qs = corpus.build(words, seed=words)
                row = {"ngl": ngl, "kv": kv, "ctx": a.ctx, "words": words, "idle_mib": idle,
                       "questions": []}
                with PeakVram() as pv:
                    for i, q in enumerate(qs):
                        try:
                            r = ask(srv.base, doc, q["question"], a.max_tokens)
                        except Exception as e:
                            row["error"] = f"q{i}: {e}"
                            break
                        t = r["timings"]
                        ok = q["expect"].lower() in r["answer"].lower()
                        row["questions"].append({
                            "depth": q["depth"], "ok": ok, "wall_s": round(r["wall_s"], 1),
                            "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
                            "prompt_per_s": t.get("prompt_per_second"),
                            "decode_per_s": t.get("predicted_per_second"),
                            "decode_n": t.get("predicted_n"), "answer": r["answer"][:200]})
                        print(f"  {words}w q{i} d={q['depth']:.2f} ok={ok} "
                              f"{r['wall_s']:.0f}s dec={t.get('predicted_per_second')}", flush=True)
                row["peak_mib"] = pv.peak
                row["doc_tokens"] = (row["questions"][0]["prompt_n"] or 0) + \
                                    (row["questions"][0]["cache_n"] or 0) if row["questions"] else 0
                with open(a.out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
        finally:
            srv.stop()


if __name__ == "__main__":
    main()

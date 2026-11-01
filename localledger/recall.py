"""Does quantising the KV cache cost recall? Needle lookups at ~32K / ~64K / ~124K tokens, per KV dtype.

Protocol (fixed before the first scored run):
  - arms: -ctk/-ctv f16, q8_0, q4_0, all at -ngl 44 -c 131072 (the one setting every dtype fits at;
    ngl moves layers between GPU and CPU and does not change the numbers)
  - lengths: 22000, 44000, 85000 words of filler (about 32K, 63K, 123K tokens)
  - per (arm, length, seed): one document with 6 planted facts at depths 0.05/0.20/0.40/0.60/0.80/0.95
    and 6 questions asked in order over the same cached prefix. Seeds 1 and 2 use different filler
    and different code values.
  - each fact is a random code, so the answer cannot be guessed; the filler has look-alike facts
    (other pump houses' passcodes, extensions) as distractors
  - score: the exact code appears in the final answer (reasoning text is not scored); temperature 0
  - 3 arms x 3 lengths x 2 seeds x 6 needles = 108 trials, 12 per cell. That is small: read counts,
    not percentages.
Resumable: finished trials in --out are skipped.

  python -m localledger.recall            # run all arms
  python -m localledger.recall --summary  # print the table from results/recall.jsonl
"""
import argparse
import collections
import json
import os
import random
import sys
import time

from . import corpus
from .llama import PeakVram, Server, _post, vram_mib

LENGTHS = {"32k": 22000, "64k": 44000, "124k": 85000}
DEPTHS = [0.05, 0.20, 0.40, 0.60, 0.80, 0.95]
KV_ARMS = ("f16", "q8_0", "q4_0")
NAMES = ["Alder", "Brindle", "Cormac", "Dessa", "Ebbfleet", "Farrow", "Gantry", "Harrowgate",
         "Ingram", "Juniper", "Kestrel", "Lindqvist"]
# (planted sentence, question); {n} is the place name, {c} the code
TEMPLATES = [
    ("The maintenance passcode for {n} pump house is {c}.",
     "What is the maintenance passcode for {n} pump house?"),
    ("The calibration certificate for the {n} pressure transducer carries serial number {c}.",
     "What serial number is on the calibration certificate for the {n} pressure transducer?"),
    ("The valve-key cabinet at {n} Tank opens with the code {c}.",
     "What code opens the valve-key cabinet at {n} Tank?"),
    ("Incident report {c} covers the burst main on {n} Road.",
     "Which incident report covers the burst main on {n} Road?"),
    ("The chemical delivery permit for {n} treatment works is numbered {c}.",
     "What is the number of the chemical delivery permit for {n} treatment works?"),
    ("The override code for the {n} surge tank is {c}.",
     "What is the override code for the {n} surge tank?"),
]
SYSTEM = ("Answer using only the document. Be brief. "
          "If the document does not say, reply 'not in the document'.")


def build(words, seed):
    """Return (text, [(question, code, depth)]). Same (words, seed) gives the same document."""
    rng = random.Random(seed * 1000 + words)
    names = rng.sample(NAMES, len(TEMPLATES))
    needles = []
    for (fact, q), n in zip(TEMPLATES, names):
        code = f"{rng.choice('QXZKW')}{rng.choice('QXZKW')}-{rng.randint(1000, 9999)}"
        needles.append((fact.format(n=n, c=code), q.format(n=n), code))
    paras, total = [], 0
    while total < words:
        p = corpus._para(rng)
        paras.append(p)
        total += len(p.split())
    n = len(paras)
    for depth, (fact, _, _) in sorted(zip(DEPTHS, needles), reverse=True):  # back to front keeps indexes valid
        paras.insert(int(depth * n), fact)
    return "\n\n".join(paras), [(q, code, d) for (_, q, code), d in zip(needles, DEPTHS)]


def ask(base, doc, question, max_tokens=2048):
    body = {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"<document>\n{doc}\n</document>\n\nQuestion: {question}"}],
            "max_tokens": max_tokens, "temperature": 0, "cache_prompt": True,
            "chat_template_kwargs": {"reasoning_strength": "low"}}
    r = _post(base, "/v1/chat/completions", body, timeout=3600)
    ch = r["choices"][0]
    return ch["message"].get("content") or "", ch.get("finish_reason"), r.get("timings", {}), r["usage"]


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def summarize(rows):
    cells = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        c = cells[(r["kv"], r["length"])]
        c[0] += r["ok"]
        c[1] += 1
    lines = [f"{'kv':>5} " + " ".join(f"{k:>8}" for k in LENGTHS)]
    for kv in KV_ARMS:
        lines.append(f"{kv:>5} " + " ".join(
            f"{cells[(kv, k)][0]:>3}/{cells[(kv, k)][1]:<4}" if (kv, k) in cells else f"{'-':>8}" for k in LENGTHS))
    cut = sum(r["finish"] == "length" for r in rows)
    if cut:
        lines.append(f"{cut} answer(s) cut off at the token limit (finish=length)")
    return "\n".join(lines)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kv", nargs="+", default=list(KV_ARMS))
    ap.add_argument("--lengths", nargs="+", default=list(LENGTHS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--ngl", type=int, default=44)
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--port", type=int, default=8803)
    ap.add_argument("--out", default="results/recall.jsonl")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        print(summarize(load(a.out)))
        return
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = {(r["kv"], r["length"], r["seed"], r["depth"]) for r in load(a.out)}
    for kv in a.kv:
        todo = [(L, s) for L in a.lengths for s in a.seeds
                if any((kv, L, s, d) not in done for d in DEPTHS)]
        if not todo:
            continue
        idle = vram_mib()[0]
        if idle > 1024:
            sys.exit(f"GPU has {idle} MiB in use; free it first")
        print(f"== kv {kv} ngl {a.ngl} ctx {a.ctx}", flush=True)
        with Server(a.ngl, kv, a.ctx, a.port, log_path=f"results/server_recall_{a.ngl}_{kv}.log") as srv, \
                PeakVram() as pv:
            for L, s in todo:
                doc, qs = build(LENGTHS[L], s)
                for q, code, depth in qs:
                    if (kv, L, s, depth) in done:
                        continue
                    t0 = time.time()
                    ans, fin, t, usage = ask(srv.base, doc, q)
                    row = dict(kv=kv, length=L, seed=s, depth=depth, ok=code in ans, code=code, answer=ans[:300],
                               finish=fin, wall_s=round(time.time() - t0, 1), prompt_n=t.get("prompt_n"),
                               cache_n=t.get("cache_n"), prompt_tokens=usage.get("prompt_tokens"),
                               completion_tokens=usage.get("completion_tokens"), ngl=a.ngl, ctx=a.ctx,
                               idle_mib=idle, peak_mib=pv.peak)
                    with open(a.out, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"{kv} {L} seed{s} d={depth}: {'PASS' if row['ok'] else 'FAIL'} {row['wall_s']}s "
                          f"tokens={row['prompt_tokens']} finish={fin} ans={ans[:50]!r}", flush=True)
    print(summarize(load(a.out)))


if __name__ == "__main__":
    main()

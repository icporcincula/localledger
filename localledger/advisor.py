"""Offload advisor: fit a profile from bench rows, then recommend -ngl / KV dtype for a document.

Usage:
  python -m localledger.advisor probe                 # measure this machine (a few minutes)
  python -m localledger.advisor --doc-tokens 100000   # recommend from results/bench.jsonl
"""
import argparse
import json
import statistics
import subprocess
import sys

from .llama import vram_mib

HEADROOM_MIB = 250  # smallest margin that ran stable in measurements (322 MiB free)


def load_rows(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    ok = []
    for r in rows:
        qs = r.get("questions") or []
        if r.get("error") or not qs or not r.get("peak_mib"):
            continue
        ok.append(r)
    return ok, [r for r in (json.loads(l) for l in open(path, encoding="utf-8")) if r.get("error")]


def linfit(xs, ys):
    """Least-squares y = a + b*x; falls back to a flat line with one point."""
    if len(set(xs)) < 2:
        return statistics.mean(ys), 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


class Profile:
    def __init__(self, rows, failures):
        self.ctx = max(r["ctx"] for r in rows)
        self.idle = statistics.mean(r["idle_mib"] for r in rows)
        self.failed = {(f["ngl"], f["kv"]) for f in failures}
        self.cfg = {}  # (ngl, kv) -> dict(peak, d0, slope, prefill, acc)
        by = {}
        for r in rows:
            by.setdefault((r["ngl"], r["kv"]), []).append(r)
        for key, rs in by.items():
            dec = [(r["doc_tokens"], statistics.mean(q["decode_per_s"] for q in r["questions"]
                                                     if q["decode_per_s"])) for r in rs]
            d0, slope = linfit([x for x, _ in dec], [y for _, y in dec])
            pre = [r["questions"][0]["prompt_per_s"] for r in rs if r["questions"][0]["prompt_per_s"]]
            qs = [q for r in rs for q in r["questions"]]
            self.cfg[key] = {"peak": max(r["peak_mib"] for r in rs), "d0": d0, "slope": slope,
                             "prefill": statistics.mean(pre) if pre else None,
                             "acc": sum(q["ok"] for q in qs) / len(qs)}
        # per-layer VRAM cost shared across kv dtypes: fit peak on ngl with a per-kv intercept
        pts = [(k[0], v["peak"], k[1]) for k, v in self.cfg.items()]
        kvs = sorted({p[2] for p in pts})
        centred = []
        for kv in kvs:
            sub = [p for p in pts if p[2] == kv]
            m = statistics.mean(p[1] for p in sub)
            mg = statistics.mean(p[0] for p in sub)
            centred += [(p[0] - mg, p[1] - m) for p in sub]
        den = sum(x * x for x, _ in centred)
        self.mib_per_layer = sum(x * y for x, y in centred) / den if den else 0.0
        self.kv_off = {}
        for kv in kvs:
            sub = [p for p in pts if p[2] == kv]
            self.kv_off[kv] = statistics.mean(p[1] - self.mib_per_layer * p[0] for p in sub)

    def predict(self, ngl, kv):
        if (ngl, kv) in self.cfg:
            return self.cfg[(ngl, kv)], True
        if kv not in self.kv_off:
            return None, False
        near = sorted((k for k in self.cfg if k[1] == kv), key=lambda k: abs(k[0] - ngl))
        peak = self.kv_off[kv] + self.mib_per_layer * ngl
        base = self.cfg[near[0]]
        # decode rate interpolated along the measured ngl points of this dtype
        pts = sorted(k for k in self.cfg if k[1] == kv)
        lo = [k for k in pts if k[0] <= ngl]
        hi = [k for k in pts if k[0] >= ngl]
        if lo and hi and lo[-1] != hi[0]:
            f = (ngl - lo[-1][0]) / (hi[0][0] - lo[-1][0])
            mix = lambda a: self.cfg[lo[-1]][a] * (1 - f) + self.cfg[hi[0]][a] * f
            return {"peak": peak, "d0": mix("d0"), "slope": mix("slope"),
                    "prefill": mix("prefill"), "acc": None}, False
        return {"peak": peak, "d0": base["d0"], "slope": base["slope"],
                "prefill": base["prefill"], "acc": None}, False


def recommend(p, total_mib, doc_tokens, answer_tokens, headroom, layers=52):
    out = []
    if doc_tokens + answer_tokens > p.ctx:
        print(f"warning: document needs more than the measured context ({p.ctx} tokens)")
    for kv in sorted(p.kv_off):
        lo_n = min(k[0] for k in p.cfg if k[1] == kv)
        hi_n = max(k[0] for k in p.cfg if k[1] == kv)
        for ngl in range(lo_n, hi_n + 1, 2):  # never extrapolate past measured layers
            c, measured = p.predict(ngl, kv)
            if c is None or (ngl, kv) in p.failed:
                continue
            if c["peak"] > total_mib - headroom:
                continue
            dec = max(c["d0"] + c["slope"] * doc_tokens, 0.5)
            cold = doc_tokens / c["prefill"] if c["prefill"] else float("nan")
            out.append({"ngl": ngl, "kv": kv, "peak": c["peak"], "decode": dec,
                        "cold_s": cold, "follow_s": answer_tokens / dec, "measured": measured})
    out.sort(key=lambda r: -r["decode"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="recommend", choices=["recommend", "probe"])
    ap.add_argument("--results", default="results/bench.jsonl")
    ap.add_argument("--doc-tokens", type=int, default=100000)
    ap.add_argument("--answer-tokens", type=int, default=600)
    ap.add_argument("--vram-mib", type=int, default=None, help="default: query nvidia-smi")
    ap.add_argument("--headroom-mib", type=int, default=HEADROOM_MIB)
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "probe":
        used, total = vram_mib()
        print(f"probing: {total} MiB card, {used} MiB already in use")
        cfgs = ",".join(f"{n}:{kv}" for kv in ("q8_0", "f16") for n in (40, 44, 48))
        sys.exit(subprocess.call([sys.executable, "-m", "localledger.bench", "--configs", cfgs,
                                  "--lengths", "8000,48000", "--out", a.results]))
    rows, failures = load_rows(a.results)
    p = Profile(rows, failures)
    total = a.vram_mib or vram_mib()[1]
    recs = recommend(p, total, a.doc_tokens, a.answer_tokens, a.headroom_mib)
    print(f"{total} MiB card, {a.doc_tokens} token document, headroom {a.headroom_mib} MiB, "
          f"fitted {p.mib_per_layer:.0f} MiB/layer")
    print(f"{'ngl':>4} {'kv':>5} {'peak MiB':>9} {'decode t/s':>11} {'cold prefill':>13} "
          f"{'follow-up':>10}  source")
    for r in recs[:a.top]:
        print(f"{r['ngl']:>4} {r['kv']:>5} {r['peak']:>9.0f} {r['decode']:>11.1f} "
              f"{r['cold_s']:>11.0f} s {r['follow_s']:>8.0f} s  "
              f"{'measured' if r['measured'] else 'interpolated'}")
    if not recs:
        print("nothing fits: lower the headroom or shorten the document")


if __name__ == "__main__":
    main()

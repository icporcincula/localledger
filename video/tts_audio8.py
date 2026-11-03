"""Render narration chunks with Edge0/Audio8-TTS-Realistic-0.6b (local copy in ../tts-audio8/model).

  tts_audio8.py chunks.json outdir [--ref ref.wav --ref-text "exact transcript"] [--device cuda|cpu] [--seed N]

chunks.json: [{"id": "s1_0", "text": "One sentence."}, ...]. One WAV per chunk, 44.1 kHz mono.
Chunks already rendered (same id, same text) are skipped, so a re-run only redoes what changed.
"""
import os
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor

MODEL = os.environ.get("AUDIO8_MODEL", "./audio8-model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks")
    ap.add_argument("outdir")
    ap.add_argument("--ref")
    ap.add_argument("--ref-text")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    a = ap.parse_args()

    chunks = json.load(open(a.chunks, encoding="utf-8"))
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    voice = "clone" if a.ref else "default"
    manifest_path = out / "manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else {}

    def key(c):
        return hashlib.sha1(f"{voice}|{a.ref_text}|{a.seed}|{a.temperature}|{c['text']}".encode()).hexdigest()

    todo = [c for c in chunks if manifest.get(c["id"], {}).get("key") != key(c) or not (out / f"{c['id']}.wav").exists()]
    if not todo:
        print("nothing to render")
        return
    dtype = torch.bfloat16 if a.device == "cuda" else torch.float32
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=dtype).eval().to(a.device)
    print(f"loaded in {time.time() - t0:.1f}s on {a.device}", flush=True)
    for c in todo:
        kw = {}
        if a.ref:
            kw = dict(reference_audio=[a.ref], reference_text=[a.ref_text])
        inputs = processor(text=[c["text"]], return_tensors="pt", **kw)
        inputs = {k: (v.to(a.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        torch.manual_seed(a.seed)
        t1 = time.time()
        with torch.inference_mode():
            res = model.generate(**inputs, max_new_tokens=a.max_new_tokens, temperature=a.temperature, top_p=0.95,
                                 top_k=50, do_sample=True, return_dict_in_generate=True)
            waves, lengths = model.decode_audio(res.codes)
        audio = waves[0, : int(lengths[0])].float().cpu().numpy()
        sf.write(out / f"{c['id']}.wav", audio, model.config.codec_sample_rate)
        secs = len(audio) / model.config.codec_sample_rate
        manifest[c["id"]] = {"key": key(c), "seconds": round(secs, 3), "text": c["text"]}
        json.dump(manifest, open(manifest_path, "w"), indent=1)
        peak = torch.cuda.max_memory_allocated() / 2**20 if a.device == "cuda" else 0
        print(f"{c['id']}: {secs:.1f}s audio in {time.time() - t1:.1f}s (peak {peak:.0f} MiB) {c['text'][:50]!r}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

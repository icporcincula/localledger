"""Round-trip check: transcribe rendered WAVs with Whisper and diff against the script text."""
import difflib, json, re, sys
import whisper
chunks_json, wav_dir = sys.argv[1], sys.argv[2]
model = whisper.load_model("small", device="cpu")
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower().replace("-", " ")).split()
bad = 0
for c in json.load(open(chunks_json, encoding="utf-8")):
    r = model.transcribe(f"{wav_dir}/{c['id']}.wav", language="en", fp16=False)
    a, b = norm(c["text"]), norm(r["text"])
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    bad += ratio < 0.9
    print(f"{c['id']}: match {ratio:.2f}\n   want: {c['text']}\n   heard:{r['text']}")
print("chunks below 0.90:", bad)

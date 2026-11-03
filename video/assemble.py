"""Build the final video from the screen take, the narration WAVs and the cards.

  tts-venv python assemble.py scenes.json --take take --audio audio_clone --out final/localledger_demo.mp4

scenes.json: {"scenes": [{"id", "sentences": [...], "lead": 0.3, "video": [segment, ...]}]}
  segment: {"src": "take", "a": "q1_asked", "b": "q1_reading+2", "speed": 1, "label": "..."}   (a/b: seconds or mark[+-]secs)
           {"src": "take", "a": ..., "b": ..., "dur": 7}              (dur: play the span in this many seconds, i.e. sped up)
           {"src": "card:title", "dur": 8}
Narration WAVs come from tts_audio8.py (ids "<scene id>_<n>"). Each scene lasts as long as the longer of its video and
its narration; a short video is held on its last frame, short narration leaves the screen running with no voice.
"""
import os
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

FF = os.environ.get("FFMPEG", "ffmpeg")
HERE = Path(__file__).parent
SR = 44100
GAP = 0.28   # silence between sentences
TAIL = 0.45  # breathing room after the last sentence of a scene
FONT = "segoeui.ttf"  # copied next to this script; ffmpeg runs with cwd=HERE (a drive colon breaks filter parsing)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=HERE)
    if r.returncode:
        sys.exit(f"ffmpeg failed: {' '.join(map(str, cmd))[:300]}\n{r.stderr[-1500:]}")


def at(ref, marks):
    if isinstance(ref, (int, float)):
        return float(ref)
    m = re.fullmatch(r"([A-Za-z0-9_]+)\s*([+-]\s*[\d.]+)?", ref.strip())
    return marks[m.group(1)] + (float(m.group(2).replace(" ", "")) if m.group(2) else 0.0)


def trim_voice(x, pad=0.06):
    idx = np.where(np.abs(x) > 0.004)[0]
    if not len(idx):
        return x
    p = int(pad * SR)
    return x[max(0, idx[0] - p): idx[-1] + p]


def seg_clip(seg, marks, take, cards, tmp, name):
    out = tmp / f"{name}.mp4"
    common = ["-an", "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p"]
    if seg["src"].startswith("card:"):
        dur = float(seg["dur"])
        png = cards / f"{seg['src'][5:]}.png"
        vf = f"scale=1920:1080,format=yuv420p,fade=t=in:st=0:d=0.3,fade=t=out:st={dur - 0.3:.2f}:d=0.3"
        run([FF, "-y", "-loglevel", "error", "-loop", "1", "-framerate", "30", "-t", f"{dur}", "-i", str(png),
             "-vf", vf, *common, str(out)])
        return out, dur
    a, b = at(seg["a"], marks), at(seg["b"], marks)
    src_len = b - a
    dur = float(seg["dur"]) if "dur" in seg else src_len / float(seg.get("speed", 1))
    speed = src_len / dur
    vf = f"setpts=PTS/{speed:.5f},fps=30,scale=1920:1080:flags=lanczos"
    if seg.get("label"):
        txt = seg["label"].replace(":", r"\:").replace("'", "")
        vf += (f",drawtext=fontfile={FONT}:text='{txt}':fontsize=34:fontcolor=white:box=1:boxcolor=0x1d2328@0.82:"
               f"boxborderw=16:x=48:y=48")
    vf += ",format=yuv420p"
    run([FF, "-y", "-loglevel", "error", "-ss", f"{a:.3f}", "-t", f"{src_len:.3f}", "-i", str(take / "take.mp4"),
         "-vf", vf, "-t", f"{dur:.3f}", *common, str(out)])
    return out, dur


def ass_time(t):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes")
    ap.add_argument("--take", default=str(HERE / "take"))
    ap.add_argument("--audio", required=True)
    ap.add_argument("--cards", default=str(HERE / "cards"))
    ap.add_argument("--out", default=str(HERE / "final" / "localledger_demo.mp4"))
    ap.add_argument("--no-subs-burn", action="store_true")
    a = ap.parse_args()
    take, cards, audio = Path(a.take), Path(a.cards), Path(a.audio)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = HERE / "tmp_build"
    tmp.mkdir(exist_ok=True)
    marks = json.load(open(take / "marks.json", encoding="utf-8"))["marks"] if (take / "marks.json").exists() else {}
    manifest = json.load(open(audio / "manifest.json", encoding="utf-8"))
    scenes = json.load(open(a.scenes, encoding="utf-8"))["scenes"]

    t_cursor, scene_files, cues, narration = 0.0, [], [], []
    for sc in scenes:
        # narration for this scene
        lead = float(sc.get("lead", 0.3))
        parts, offs, pos = [], [], lead
        for i, text in enumerate(sc["sentences"]):
            cid = f"{sc['id']}_{i}"
            if manifest.get(cid, {}).get("text") != text:
                sys.exit(f"audio for {cid} is missing or stale; re-run tts_audio8.py")
            x, sr = sf.read(audio / f"{cid}.wav", dtype="float32")
            assert sr == SR, sr
            x = trim_voice(x if x.ndim == 1 else x[:, 0])
            offs.append((pos, pos + len(x) / SR, text))
            parts.append(x)
            pos += len(x) / SR + GAP
        voice_end = pos - GAP
        # picture
        clips, vdur = [], 0.0
        for k, seg in enumerate(sc["video"]):
            f, d = seg_clip(seg, marks, take, cards, tmp, f"{sc['id']}_v{k}")
            clips.append(f)
            vdur += d
        L = max(vdur, voice_end + TAIL)
        lst = tmp / f"{sc['id']}_list.txt"
        lst.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips), encoding="utf-8")
        vid = tmp / f"{sc['id']}_video.mp4"
        pad = L - vdur
        run([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
             "-vf", f"tpad=stop_mode=clone:stop_duration={pad:.3f}" if pad > 0.01 else "null",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p", "-r", "30", str(vid)])
        wav = np.zeros(int(L * SR) + SR, dtype="float32")
        p = int(lead * SR)
        for x in parts:
            wav[p:p + len(x)] = x
            p += len(x) + int(GAP * SR)
        wav_path = tmp / f"{sc['id']}_voice.wav"
        sf.write(wav_path, wav[: int(L * SR)], SR)
        scene_mp4 = tmp / f"{sc['id']}.mp4"
        run([FF, "-y", "-loglevel", "error", "-i", str(vid), "-i", str(wav_path), "-t", f"{L:.3f}",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-ar", "44100", str(scene_mp4)])
        scene_files.append(scene_mp4)
        for s0, s1, text in offs:
            cues.append((t_cursor + s0, t_cursor + s1 + 0.15, text))
        print(f"{sc['id']}: {L:5.1f}s (video {vdur:.1f}s, voice {voice_end:.1f}s) starts at {t_cursor:.1f}s", flush=True)
        t_cursor += L

    # subtitles: SRT for upload, ASS for burning
    def srt_time(t):
        return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d},{int(round((t % 1) * 1000)):03d}"
    srt = "".join(f"{i}\n{srt_time(s)} --> {srt_time(e)}\n{tx}\n\n" for i, (s, e, tx) in enumerate(cues, 1))
    out.with_suffix(".srt").write_text(srt, encoding="utf-8")
    ass = ("[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nWrapStyle: 0\n\n[V4+ Styles]\n"
           "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,"
           "StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
           "Style: Sub,Segoe UI,40,&H00FFFFFF,&H00FFFFFF,&H70000000,&H70000000,0,0,0,0,100,100,0,0,3,10,0,2,260,260,34,1\n\n"
           "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
    ass += "".join(f"Dialogue: 0,{ass_time(s)},{ass_time(e)},Sub,,0,0,0,,{tx}\n" for s, e, tx in cues)
    (HERE / "subs.ass").write_text(ass, encoding="utf-8")

    lst = tmp / "all.txt"
    lst.write_text("".join(f"file '{f.as_posix()}'\n" for f in scene_files), encoding="utf-8")
    vf = "null" if a.no_subs_burn else "ass=subs.ass:fontsdir=."
    run_cwd = subprocess.run([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                              "-vf", vf, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                              "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                              "-c:a", "aac", "-ar", "48000", "-b:a", "192k", "-movflags", "+faststart", str(out)],
                             cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if run_cwd.returncode:
        sys.exit(run_cwd.stderr[-1500:])
    print(f"wrote {out}  total {t_cursor:.1f}s ({int(t_cursor // 60)}:{int(t_cursor % 60):02d})")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

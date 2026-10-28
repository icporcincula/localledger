# Demo video script (3:00)

Audience: business owners and hackathon judges. Show only work built since Oct 27.
Setup: `chrome --app=http://127.0.0.1:8802 --start-fullscreen`, Wi-Fi off, sample already loaded once
(so the cache is warm for the follow-ups), a second document pre-loaded for the switch demo.
Cold read is ~138 s: do NOT wait for it on camera. Record it separately and cut, or show the progress bar for 5 s and jump.

| Time | On screen | Narration |
|---|---|---|
| 0:00-0:15 | Empty page, offline pill reads "Offline: working without internet" | "Your contracts, patient notes and audit files can't go to a cloud chatbot. So most people just don't get to ask questions of them. This is LocalLedger. The internet is off right now." |
| 0:15-0:35 | Click the sample, about 140 pages loaded | "This is a 140-page operations binder. I load it here. No upload, no account. The whole thing goes into the model's memory at once. There is no search index guessing which snippets matter." |
| 0:35-0:50 | Progress bar "Reading the whole document (first time only)", then cut | "The first read takes a couple of minutes on a single gaming graphics card. It happens once." |
| 0:50-1:20 | Ask "Which locker holds the spare chlorine cartridges?" Answer in about 7 s. Tiles: 6.9 s, almost none re-read, nothing sent | "Now questions take seconds. Locker 9, behind the depot office. Notice: almost none of the document was re-read, and nothing left this computer." |
| 1:20-1:45 | Two more questions from different parts of the binder (chips) | "Page 4 or page 130, it doesn't matter. Every fact we hid in the document was found." |
| 1:45-2:10 | Load the second document, then switch back (about 0.5 s) | "Switching between documents takes half a second, because the app saves each document's memory to disk and restores it." |
| 2:10-2:40 | Open "For your IT team": layer strip, VRAM gauge, advisor slider to 12 GB, chart | "For your IT person: the model is Meta's Muse Glimmer 30B, 17 GB, on a 16 GB card. This panel measures how many layers fit on the GPU and recommends a setting for your hardware. We benchmarked eight configurations, and every answer was correct." |
| 2:40-3:00 | Back to hero headline, repo URL on screen | "Open weights, open source, MIT. Documents that can't leave the building finally get to be asked questions. Built with Meta's Muse Glimmer." |

## Claims to keep honest
- "Every fact was found": 6 planted single-fact questions per document, synthetic data, one machine. Say "synthetic" on screen at least once.
- "Nothing sent" is true by design (127.0.0.1 only, no outbound calls), not a live measurement.
- Timings: cold read ~138 s for 93K tokens, follow-up ~7 s, restore 0.5 s.

## Shot list for recording
1. Warm up: load sample, ask one question, confirm cache hit.
2. Turn Wi-Fi off, confirm the pill.
3. Record screen in one take per segment; cut the cold read.
4. Voiceover separately, or record live and trim.

# LocalLedger: ask your confidential documents questions, on your own computer

Some documents cannot go to a cloud chatbot: contracts, patient notes, audit files, operations manuals.
LocalLedger loads one of them (up to about 200 pages) into Meta's open-weight Muse Glimmer 30B, running on a
single 16 GB gaming graphics card, and answers questions about it in plain English. Nothing is uploaded,
there is no account, and it works with the internet switched off.

## What you get

- **Answers you can check.** Every answer quotes the sentence it came from and says roughly which page it is on.
  The app verifies the quote word for word against your document. If the model quotes something that is not
  really there, the app says so instead of showing a tick.
- **It reads all of it.** The whole document sits in the model's memory at once. There is no search index
  guessing which snippets matter, so questions that combine facts from different pages work.
- **Honest "I don't know".** If the document does not say, the answer is "not in the document".
- **Fast follow-ups.** The first read takes a couple of minutes on one gaming card. After that, answers take
  seconds, and switching between documents takes about half a second.
- **Quick or Careful.** Careful thinks longer before answering: about twice as slow, meant for harder questions.
- **Take it with you.** One click saves the questions, answers and quoted sources as a text file.
- **Clear failures.** A PDF, an empty file, a document that is too long for the memory, or a model that has
  crashed each gets a plain message saying what to do.

Files must be plain text (`.txt`, `.md`, `.log`, `.csv`) up to 2 MB. Save a PDF or Word file as text first.

## Try it

Requirements: Windows, an NVIDIA GPU with 16 GB, a CUDA build of `llama.cpp`, and the Muse Glimmer 30B
Q4_K_M GGUF (17 GB). Set `LLAMA_SERVER` and `GLIMMER_GGUF` to their paths. No other dependencies.

```
python -m localledger.app --ngl 48 --kv q8_0     # then open http://127.0.0.1:8802
python -m unittest discover tests                    # fast checks, no GPU needed
```

Click "Try a sample" for a synthetic 140-page operations binder with six facts hidden at known places.

## What we tested

All on one machine (RTX 5060 Ti 16 GB, 32 GB RAM, llama.cpp b11753, Q4_K_M), all with synthetic documents.

| Test | Result |
|---|---|
| Six hidden facts at 5% to 97% depth, documents of 8K to 91K tokens, four GPU settings | 96 of 96 correct |
| Question that needs two facts from different pages (who staffs the depot, and their extension) | correct |
| Sum of two figures found on different pages | correct |
| Questions the document does not answer (2 tried) | both answered "not in the document" |
| Quoted source sentence on 4 tried answers | all verified word for word; the fact planted 25% into the binder was reported on page 34 of 137 |
| A real document: RFC 7540 (HTTP/2, 26,700 words), 8 questions with answers checked against the text | 7 of 8 correct (the miss gave the right error name but left out the hex value asked for); both unanswerable questions got "not in the document"; every quoted source was verified word for word |
| Model killed mid-read | clear message, button re-enabled within 3 s |

Limits you should know about: the test documents are synthetic, most questions are single-fact lookups
(only a handful of multi-step and unanswerable ones were tried), and everything was measured on one machine.

## Does a smaller KV cache cost recall?

The KV cache is the model's working memory for the document. Storing it in 8-bit (`q8_0`) or 4-bit (`q4_0`) instead
of 16-bit (`f16`) saves video memory, so the question is whether it also loses facts. `python -m localledger.recall`
hides six random codes (like `QX-7316`) in a synthetic document, with look-alike decoys elsewhere in the text, at 5%
to 95% depth, then asks for each one. A trial passes only if the exact code is in the final answer. Settings:
`-ngl 44`, 131K context, temperature 0, two different documents per cell, so 12 questions per cell.

| KV cache | 32K tokens | 64K tokens | 124K tokens | Peak video memory |
|---|---|---|---|---|
| f16 | 12 / 12 | 12 / 12 | 12 / 12 | 15.2 GB |
| q8_0 | 12 / 12 | 12 / 12 | 12 / 12 | 14.4 GB |
| q4_0 | 12 / 12 | 12 / 12 | 12 / 12 | 14.0 GB |

All 108 trials were correct and none ran into the token limit (longest prompt: 116,992 tokens). What that does and
does not show: on single-code lookups in synthetic text, we found no recall loss from 8-bit or 4-bit caches, and they
used 0.8 and 1.2 GB less video memory. Because every cell is perfect, this test cannot rank the three; harder
questions (several facts combined, real documents) could still show a difference. Peak memory includes about 0.7 GB
the desktop was already using. Raw rows: `results/recall.jsonl`; logs: `results/recall_run.log`.

## For your IT person

Muse Glimmer 30B at Q4_K_M is a 17 GB file, so it does not fit in 16 GB of video memory together with a
100K-token cache. llama.cpp keeps some layers on the CPU. The question is how many, and what to do with the
KV cache. The app's "For your IT team" panel and the advisor answer that from measurements, not guesses.

| GPU layers (of 52) | KV cache | Answer speed, 8K to 91K tokens | Peak video memory |
|---|---|---|---|
| 48 | q8_0 | 13.9 to 9.7 tok/s | about 15.9 GB |
| 44 | q8_0 | 10.5 to 6.7 tok/s | about 14.7 GB |
| 44 | f16 | 10.9 to 5.7 tok/s | about 15.3 GB |
| 40 | f16 | 8.8 to 4.0 tok/s | about 14.1 GB |

A cold read of a 91K-token document takes about 130 s. For smaller cards we measured fewer layers on the GPU
(q8_0, 8K and 46K tokens, 48 of 48 correct). These runs used a 106K context on a 16 GB card, so peak memory
is a stand-in for a smaller card, not a test on one.

| GPU layers (of 52) | Answer speed, 8K to 46K tokens | Peak video memory | Cold read of 46K tokens |
|---|---|---|---|
| 36 | 6.9 to 5.2 tok/s | about 12.2 GB | 93 s |
| 28 | 4.9 to 3.7 tok/s | about 10.0 GB | 114 s |
| 20 | 4.1 to 3.0 tok/s | about 7.8 GB | 135 s |
| 12 | 3.6 to 2.6 tok/s | about 5.5 GB | 153 s |

The advisor interpolates between measured points and marks those rows with an asterisk.

**A real 12 GB profile.** The runs above kept the 106K context, so they only stand in for a smaller card.
To check a 12 GB card properly we re-ran with a 52K context (enough for a 46K-token document), q8_0 cache,
on the same machine. Peak memory is what a 12 GB card would have to hold (12,288 MiB minus about 250 MiB
headroom). We did not test on an actual 12 GB card.

| GPU layers (of 52) | Answer speed, 8K to 44K tokens | Peak video memory | Fits 12 GB? | Cold read of 44K tokens | Correct |
|---|---|---|---|---|---|
| 40 | 8.5 to 6.6 tok/s | 13.2 GB | no | 79 s | 12 of 12 |
| 36 | 7.1 to 5.1 tok/s | 12.1 GB | no, by about 40 MiB | 90 s | 12 of 12 |
| 32 | 5.6 to 4.2 tok/s | 10.9 GB | yes | 100 s | 12 of 12 |

So on a 12 GB card, a document of about 44K tokens (roughly 130 pages) runs at 32 GPU layers, at about
half the speed of the 16 GB setup. Raw rows are in `results/bench_12gb.jsonl`.

```
python -m localledger.advisor --doc-tokens 100000   # recommendation from results/bench.jsonl
python -m localledger.advisor probe                 # measure your own machine
python -m localledger.bench --help                  # custom sweeps
```

Raw benchmark rows are in `results/bench.jsonl`.

## Privacy

The app binds to 127.0.0.1 and makes no outbound requests. The only network call is a one-second socket
check that drives the "offline" indicator on the page. When you switch between documents, the model's cache
is parked in `results/slots/` and deleted when the app starts or stops. All test data is synthetic.

## Third-party components

- Muse Glimmer 30B (Meta, open weights), Q4_K_M GGUF. Meta's license terms for the model apply.
- llama.cpp (MIT), build b11753 with CUDA, runs the model.
- RFC 7540 (HTTP/2), a public document, used as one of the test documents (`results/rfc7540.txt`).
- Demo video narration only, not part of the app: Audio8-TTS-Realistic-0.6b (Apache 2.0), an AI voice cloned
  from the author's own voice. The video says so.

The app itself uses only the Python standard library.

## Contributors

- Bart Porcincula

Built with the help of Claude Code.

## License

MIT. See `LICENSE`.

## Built during the hackathon

This repository was started on October 27, 2026 (Manila time), after the event kickoff, and every file in it
was written during the building period. The idea came from earlier private experiments with the same model;
no code from those experiments is included.

import json
S = [
 ("s1", 0.3, [
  "Before this week: private speed tests and a command line demo of Meta's Muse Glimmer model on one gaming card.",
  "Built this week: this app, its answer checker, and a recall test."],
  [{"src": "card:title", "dur": 14}]),
 ("s2", 0.3, [
  "Wi-Fi is really off. LocalLedger loads a one hundred forty page binder into the model's memory, all of it at once."],
  [{"src": "take", "a": "hero", "b": "loaded+1.5"}]),
 ("s3", 0.3, [
  "The first question makes the model read every page, about two and a half minutes, so this part is sped up fifteen times.",
  "It answers locker nine, and quotes the sentence. The app checks the quote word for word against the document."],
  [{"src": "take", "a": "q1_asked", "b": "q1_answering", "dur": 10, "label": "Sped up 15x"},
   {"src": "take", "a": "q1_answering", "b": "q2_asked-1"}]),
 ("s4", 0.3, [
  "Follow-ups are fast, because the document is already in memory. Who may issue boil-water notices? Director Wray, with the extension, and the source is verified.",
  "Now a question the document cannot answer: the depot's annual budget. The app says: not in the document."],
  [{"src": "take", "a": "q2_asked", "b": "q2_done+1", "dur": 9, "label": "Sped up 2x"},
   {"src": "take", "a": "q3_asked", "b": "q3_done+1.5", "dur": 7, "label": "Sped up 2x"}]),
 ("s5", 0.3, [
  "Now a real document: R F C seven five four zero, the HTTP two specification, twenty-seven thousand words. Same flow: upload, read once, ask.",
  "The read takes about a hundred seconds, sped up twelve times here. It finds the error code named enhance your calm, with value hex b, quoted from the page."],
  [{"src": "take", "a": "upload_start", "b": "q4_answering+0.5"},
   {"src": "take", "a": "q4_answering+0.5", "b": "q4_done", "dur": 8, "label": "Sped up 12x"},
   {"src": "take", "a": "q4_done", "b": "q4_done+5"}]),
 ("s6", 0.3, [
  "Switching back to the water binder restores its memory from disk almost instantly, so there is no second long read.",
  "The answer arrives in twenty-two seconds: teal, with the quoted sentence verified."],
  [{"src": "take", "a": "switch_start", "b": "q5_asked", "dur": 5, "label": "Sped up 2x"},
   {"src": "take", "a": "q5_asked", "b": "q5_done+1", "dur": 15, "label": "Sped up 1.6x"}]),
 ("s7", 0.3, [
  "For whoever sets this up, there is an IT panel with numbers measured on this card.",
  "Tell the advisor you have a twelve gigabyte card and a long document, and it recommends thirty-four layers on the graphics card, from those measurements."],
  [{"src": "take", "a": "it_start", "b": "adv_start+1", "dur": 8, "label": "Sped up 1.5x"},
   {"src": "take", "a": "adv_start+1", "b": "adv_done+1.5", "dur": 7, "label": "Sped up 1.3x"},
   {"src": "take", "a": "chart", "b": "chart+3.5"}]),
 ("s8", 0.3, [
  "Does a smaller memory cost recall? We planted six random codes in documents up to one hundred twenty-four thousand tokens. One hundred eight questions, all answered correctly with all three cache types.",
  "That is a ceiling effect, so this test cannot rank them. It does show the smaller caches used up to one point two gigabytes less video memory."],
  [{"src": "card:stageb", "dur": 18}]),
 ("s9", 0.3, [
  "LocalLedger is open source under the MIT license. The repository address is on screen.",
  "This narration is an AI voice, cloned from my own."],
  [{"src": "card:close", "dur": 9}]),
]
sc = {"scenes": [{"id": i, "lead": l, "sentences": s, "video": v} for i, l, s, v in S]}
json.dump(sc, open("scenes.json", "w", encoding="utf-8"), indent=1)
json.dump([{"id": f"{i}_{k}", "text": t} for i, l, s, v in S for k, t in enumerate(s)], open("chunks.json", "w", encoding="utf-8"), indent=1)
print(sum(len(s) for _, _, s, _ in S), "chunks;", sum(len(t.split()) for _, _, s, _ in S for t in s), "words")

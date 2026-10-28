"""Fast checks that need no GPU and no model:  python -m unittest discover tests"""
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from localledger import advisor, app, corpus


def row(ngl, kv, tokens, peak, decode, prefill=900.0):
    q = {"decode_per_s": decode, "prompt_per_s": prefill, "ok": True}
    return {"ngl": ngl, "kv": kv, "doc_tokens": tokens, "peak_mib": peak, "ctx": 106496,
            "idle_mib": 600, "questions": [q, q]}


class Corpus(unittest.TestCase):
    def test_every_planted_fact_is_in_the_text(self):
        text, qs = corpus.build(3000, seed=1)
        for _, fact, _, expect in corpus.NEEDLES:
            self.assertIn(fact, text)
            self.assertIn(expect, fact)
        self.assertEqual(len(qs), len(corpus.NEEDLES))

    def test_same_seed_same_document(self):
        self.assertEqual(corpus.build(2000, seed=5)[0], corpus.build(2000, seed=5)[0])


class Advisor(unittest.TestCase):
    def setUp(self):
        rows = [row(48, "q8_0", 8000, 15000, 14.0), row(48, "q8_0", 90000, 15900, 9.7),
                row(36, "q8_0", 8000, 12000, 7.0), row(36, "q8_0", 90000, 12200, 5.2)]
        self.p = advisor.Profile(rows, [])

    def test_linfit_recovers_a_line(self):
        a, b = advisor.linfit([0, 10], [2, 22])
        self.assertAlmostEqual(a, 2)
        self.assertAlmostEqual(b, 2)

    def test_recommendation_never_exceeds_the_card(self):
        recs = advisor.recommend(self.p, 16 * 1024, 90000, 600, advisor.HEADROOM_MIB)
        self.assertTrue(recs)
        self.assertTrue(all(r["peak"] <= 16 * 1024 - advisor.HEADROOM_MIB for r in recs))

    def test_small_card_gets_fewer_layers_or_nothing(self):
        big = advisor.recommend(self.p, 16 * 1024, 90000, 600, advisor.HEADROOM_MIB)
        small = advisor.recommend(self.p, 12 * 1024, 90000, 600, advisor.HEADROOM_MIB)
        self.assertLess(max((r["ngl"] for r in small), default=0), max(r["ngl"] for r in big))

    def test_never_extrapolates_beyond_measured_layers(self):
        recs = advisor.recommend(self.p, 64 * 1024, 90000, 600, advisor.HEADROOM_MIB)
        self.assertLessEqual(max(r["ngl"] for r in recs), 48)


class Evidence(unittest.TestCase):
    DOC = "First line.\n\nThe annual drill is always held on the third Tuesday of May.\n\nLast line."

    def test_quote_found_even_across_line_breaks(self):
        ans = 'Third Tuesday.\n\nSource: "The annual drill is always held on the\nthird Tuesday of May."'
        ev = app.find_evidence(self.DOC, ans)
        self.assertTrue(ev["found"])
        self.assertEqual(ev["page"], 1)

    def test_invented_quote_is_flagged_not_trusted(self):
        ev = app.find_evidence(self.DOC, 'Friday.\nSource: "The drill is on Friday."')
        self.assertFalse(ev["found"])

    def test_no_source_line_means_no_evidence(self):
        self.assertIsNone(app.find_evidence(self.DOC, "not in the document"))


class Upload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}/api/upload"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def post(self, body):
        req = urllib.request.Request(self.url, body, {"X-Filename": "x.txt"})
        try:
            urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())["error"]

    def test_pdf_is_refused_with_advice(self):
        code, msg = self.post(b"%PDF-1.7 binary")
        self.assertEqual(code, 400)
        self.assertIn("plain text", msg)

    def test_office_file_is_refused(self):
        self.assertEqual(self.post(b"PK\x03\x04 zipped")[0], 400)

    def test_empty_file_is_refused(self):
        self.assertEqual(self.post(b"  \n")[0], 400)

    def test_oversize_is_refused(self):
        self.assertEqual(self.post(b"a" * (app.MAX_UPLOAD + 1))[0], 413)


if __name__ == "__main__":
    unittest.main()

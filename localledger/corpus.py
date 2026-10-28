"""Synthetic long document with facts planted at known depths. No real data."""
import random

SITES = ["Orrin", "Veldt", "Marrow", "Casket", "Tamsin", "Hollis", "Bracken", "Quill",
         "Sorrel", "Dunmore", "Elwick", "Fenwick", "Garrow", "Ilsey", "Jessop", "Kellam"]
ROLES = ["shift supervisor", "pump technician", "water-quality analyst", "field inspector",
         "dispatch coordinator", "valve crew lead", "telemetry engineer"]
VERBS = ["inspects", "calibrates", "logs", "reviews", "flushes", "samples", "audits"]
THINGS = ["the intake screen", "the chlorine dosing line", "the backup generator",
          "the pressure transducer", "the sludge pit", "the sample port", "the surge tank"]


def _para(rng):
    s = []
    for _ in range(rng.randint(4, 7)):
        site = rng.choice(SITES)
        kind = rng.randrange(4)
        if kind == 0:
            s.append(f"At {site} pump house the {rng.choice(ROLES)} {rng.choice(VERBS)} "
                     f"{rng.choice(THINGS)} every {rng.randint(2, 40)} days and records the "
                     f"reading in logbook {rng.choice('ABCDEFGH')}-{rng.randint(100, 999)}.")
        elif kind == 1:
            s.append(f"The maintenance passcode for {site} pump house is "
                     f"{rng.choice(['KT', 'MV', 'RN', 'DL'])}-{rng.randint(1000, 9999)} "
                     f"and is rotated each quarter.")
        elif kind == 2:
            s.append(f"{site} reservoir holds {rng.randint(300, 9000)} cubic metres and is "
                     f"drained for cleaning once every {rng.randint(2, 9)} years.")
        else:
            s.append(f"If {rng.choice(THINGS)} at {site} fails, call extension "
                     f"{rng.randint(2000, 2999)} and open ticket class {rng.choice('ABCD')}"
                     f"{rng.randint(1, 5)}.")
    return " ".join(s)


# (depth fraction, fact sentence, question, expected substring)
NEEDLES = [
    (0.05, "The maintenance passcode for Zephyrine pump house is QX-7316.",
     "What is the maintenance passcode for Zephyrine pump house?", "QX-7316"),
    (0.25, "Boil-water notices are authorised only by Director Anselm Wray, extension 2841-W.",
     "Who is authorised to issue boil-water notices, and which extension?", "Wray"),
    (0.50, "The emergency shutoff wheel at Ravensmoor Tank is painted teal and turns clockwise to close.",
     "What colour is the emergency shutoff wheel at Ravensmoor Tank?", "teal"),
    (0.70, "The 2019 audit found exactly 41 corroded hydrants in the Lowfield district.",
     "How many corroded hydrants did the 2019 audit find in the Lowfield district?", "41"),
    (0.88, "Spare chlorine cartridges are stored in locker 9 behind the Pellam depot office.",
     "Which locker holds the spare chlorine cartridges?", "9"),
    (0.97, "The annual drill is always held on the third Tuesday of May.",
     "On what day is the annual drill held?", "third Tuesday"),
]


def build(words, seed=11):
    """Return (text, questions). questions: list of dict(depth, question, expect)."""
    rng = random.Random(seed)
    paras, total = [], 0
    while total < words:
        p = _para(rng)
        paras.append(p)
        total += len(p.split())
    n = len(paras)
    qs = []
    for depth, fact, q, expect in NEEDLES:
        paras.insert(int(depth * n), fact)
        qs.append({"depth": depth, "question": q, "expect": expect})
    return "\n\n".join(paras), qs

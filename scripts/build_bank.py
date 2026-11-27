"""Fill the word bank ahead of time: write each word of a list `--variants` times with the
default settings, keep what passes the read-back (wordbank.py).

  python scripts/build_bank.py words.txt [--top 2000] [--variants 3] [--caps 150] [--style ...]

words.txt: one word per line, most frequent first. --caps also banks the capitalised form of
the first N words (sentence starts). Runs can be stopped and resumed: words that already have
enough variants are skipped.
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ductus.page import Options, Writer, load_style, _max_cer, _plain_letters, _letters, _style_sig  # noqa: E402
from ductus.synth import Synthesizer  # noqa: E402
from ductus.wordbank import Bank, signature  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("words")
ap.add_argument("--top", type=int, default=2000)
ap.add_argument("--variants", type=int, default=3)
ap.add_argument("--caps", type=int, default=150)
ap.add_argument("--style")
ap.add_argument("--chunk", type=int, default=120, help="words per synthesis call")
a = ap.parse_args()

ws = [w.strip() for w in Path(a.words).read_text().split() if w.strip().isalpha()]
ws = [w for w in ws if len(w) > 1 or w in ("a", "i")][:a.top]
ws += [w.capitalize() for w in ws[:a.caps]]
o = Options(style=a.style, look="natural")
refs, prof = load_style(o.style or "upright")
mc = _max_cer(o, prof)
W = Writer()
bank = Bank(signature(o, mc, _style_sig(o)))
todo = []
for w in ws:
    have = len(bank._load(w))
    todo += [w] * max(0, a.variants - have)
random.Random(0).shuffle(todo)
print(f"{len(ws)} words, {len(todo)} variants to write into {bank.dir}", flush=True)
t0 = time.time()
for i in range(0, len(todo), a.chunk):
    part = todo[i:i + a.chunk]
    prim, fb, pr = W.engines(o)
    rd, chk = W.readers(o)
    bank.used.clear()          # a build run may write and keep several variants of one word
    syn = Synthesizer(prim, refs[0], rd, checker=chk, bank=None, fallback=None, primary_rounds=pr, max_cer=mc,
                      loose=bool(prof), seed=1000 + i, log=lambda *x: None, strict=True, rounds=3,
                      plain=_plain_letters(o), letter_profile=_letters(o.style), templates=False, mess=0.3)
    out = syn.run(part)
    kept = 0
    for v in out:
        if (1.0 - v.score) <= mc:
            bank.add(v); kept += 1
    print(f"{i + len(part)}/{len(todo)} kept {kept}/{len(part)} {time.time() - t0:.0f}s", flush=True)

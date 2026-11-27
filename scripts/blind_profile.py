"""Style-matched blind test on the writer's own hand: the real side is the line crops of a
`ductus style` profile (styles/<name>/refs), the generated side is the same number of new
WikiText lines written with that profile (--style <dir> --look profile). Score the two folders
with blind_test.py. The writer's crops are private: keep OUT outside the repository.

  python scripts/blind_profile.py STYLE_DIR WIKITEXT OUT [lines]
"""
import random, shutil, sys
from pathlib import Path

import cv2

from ductus.page import Writer, Options, write_lines

style, corpus, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
refs = sorted((style / "refs").glob("*.png"))
n = int(sys.argv[4]) if len(sys.argv) > 4 else len(refs)
(out / "real").mkdir(parents=True, exist_ok=True)
(out / "fake").mkdir(parents=True, exist_ok=True)
for i, f in enumerate(refs):
    shutil.copy(f, out / "real" / f"me_{i:03d}.png")
texts = [t.strip() for t in corpus.read_text().splitlines() if 30 <= len(t.strip()) <= 50]
rng = random.Random(0)
lines = ["".join(c for c in t if c.isalnum() or c in " .,;:'!?-()") for t in rng.sample(texts, n)]
w = Writer(verify=True)
gen, st = write_lines(w, lines, Options(style=str(style), look="profile", seed=1))
for i, g in enumerate(gen):
    cv2.imwrite(str(out / "fake" / f"me_{i:03d}.png"), g)
print(st, flush=True)

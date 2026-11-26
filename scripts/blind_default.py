"""Generated side of the blind test for the default hand (no style image): the same number of
lines per IAM writer as proxy_blind_iam.py, the same WikiText selection, written with the
default engine, look and ink. Compare against the real lines with blind_test.py.

  python scripts/blind_default.py WIKITEXT OUT [writers] [lines_per_writer]
"""
import random, sys
from pathlib import Path

import cv2

from ductus.page import Writer, Options, write_lines

corpus, out = Path(sys.argv[1]), Path(sys.argv[2])
writers = sys.argv[3].split(",") if len(sys.argv) > 3 else ["173", "204", "209"]
per_writer = int(sys.argv[4]) if len(sys.argv) > 4 else 20
texts = [t.strip() for t in corpus.read_text().splitlines() if 30 <= len(t.strip()) <= 50]
rng = random.Random(0)
w = Writer(verify=True)
(out / "fake").mkdir(parents=True, exist_ok=True)
for wid in writers:
    lines = ["".join(c for c in t if c.isalnum() or c in " .,;:'!?-()") for t in rng.sample(texts, per_writer)]
    gen, st = write_lines(w, lines, Options(look="natural", seed=int(wid)))
    for i, g in enumerate(gen):
        cv2.imwrite(str(out / "fake" / f"{wid}_{i:03d}.png"), g)
    print(wid, st, flush=True)

"""Render one passage with every pen, reusing cached word images so only the ink changes.

  python scripts/ink_lab.py OUT_DIR [--text FILE] [--style quick] [--seed 3]
Writes OUT_DIR/pen-<name>.png (A4 crops of the passage), pens.png (stacked comparison),
flat-vs-physical.png and crops at 100% zoom.
"""
import argparse, pickle, sys
from pathlib import Path

import cv2
import numpy as np

from ductus.page import Writer, Options
from ductus import markup
from ductus.layout import Layouter, Look
from ductus.pens import PENS, get_pen, nib_kernel
from ductus.render import ink_coverage, compose, compose_physical, SS
from ductus.paper import make_paper
from ductus.layout import MM

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--text"); ap.add_argument("--style", default="quick")
ap.add_argument("--seed", type=int, default=3); ap.add_argument("--mode", default="notes")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
text = Path(a.text).read_text() if a.text else (
    "# The ink test\nBallpoint :: a small ball rolls the ink onto the paper\n"
    "Fountain pen -> wet ink, thick and thin lines\nNote: pencil is lighter and grainy\n"
    "1) Quick brown foxes jump over lazy dogs [v]\n2) {sum} x / n = 6 ((done))\n")
cache = out / "words.pkl"
if cache.exists():
    words, look_d = pickle.loads(cache.read_bytes())
else:
    w = Writer()
    r = w.write(text, Options(style=a.style, mode=a.mode, seed=a.seed, log=lambda *x: print(*x, file=sys.stderr)))
    words = r.words
    cache.write_bytes(pickle.dumps((words, {})))
blocks = markup.parse(text, a.mode)
deco = [wo.deco for b in blocks for wo in markup.block_words(b)]
look = Look.preset("natural")
L = Layouter(words, deco, look, seed=a.seed)
pg = L.layout(blocks)[0]
# crop region: bounding box of placed words
ys = [p.base for p in pg.words]; xs = [p.x for p in pg.words]
y0, y1 = int(min(ys) - 3 * L.xh), int(max(ys) + 2 * L.xh)
x0, x1 = int(L.left - 40), int(L.right + 40)
rows = []
for name in ["flat", *PENS]:
    rng = np.random.default_rng(a.seed)
    paper = make_paper("white", seed=1)
    if name == "flat":
        pen = get_pen(None)
        ink = ink_coverage(pg, words, look.mess, 1.0, max(2.0, pen.line_mm * MM), rng, uniform=False)
        img = compose(ink, "#1c2a6e", look.pressure, rng, paper)
    else:
        pen = get_pen(name)
        sw = pen.line_mm * MM
        shape, count = ink_coverage(pg, words, look.mess, 1.0, sw, rng, nib=nib_kernel(sw * SS, pen), with_count=True)
        img = compose_physical(shape, count, pen, look.pressure, rng, paper)
    crop = img[y0:y1, x0:x1][..., ::-1]
    cv2.imwrite(str(out / f"pen-{name}.png"), crop)
    lab = np.full((54, crop.shape[1], 3), 255, np.uint8)
    cv2.putText(lab, "old flat ink" if name == "flat" else name, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (90, 90, 90), 2)
    rows.append(np.vstack([lab, crop]))
    print(name, "ok", flush=True)
cv2.imwrite(str(out / "pens.png"), np.vstack(rows))

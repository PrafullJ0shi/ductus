"""Before/after crops for the plain-letterform check (f, r, looped ascenders).

  python scripts/letterform_compare.py OUT.png [--style STYLE ...]
Row "before": the old default (quick preset, no letter-shape check). Other rows: each given
style with the check on. Words with f and r, written once per row, same seeds."""
import argparse, sys

import cv2
import numpy as np

from ductus.letterforms import flourishes
from ductus.page import Writer, load_style
from ductus.synth import Synthesizer

WORDS = "frequent first from formula for after fresh rare river error number".split()


def row(w, style, plain, seed):
    refs, prof = load_style(style)
    syn = Synthesizer(w.engine, refs[0], w.reader, max_cer=0.5 if prof else 0.25, loose=bool(prof),
                      seed=seed, rounds=4, strict=not prof, plain=plain, log=lambda *a: print(*a, file=sys.stderr))
    ws = syn.run(WORDS)
    tiles, flagged = [], 0
    for wi in ws:
        fl = flourishes(wi.alpha, wi.text, wi.baseline, wi.xheight)
        flagged += bool(fl)
        a = wi.alpha
        s = 40.0 / max(wi.xheight, 1)
        im = (255 - 255 * np.clip(a, 0, 1)).astype(np.uint8)
        im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
        im = cv2.copyMakeBorder(im, 0, 0, 12, 12, cv2.BORDER_CONSTANT, value=255)
        tiles.append(im)
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=255) for t in tiles]
    return np.hstack(tiles), flagged, syn.stats


def label(img, text, W):
    bar = np.full((44, W), 255, np.uint8)
    cv2.putText(bar, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 90, 2)
    img = cv2.copyMakeBorder(img, 0, 0, 0, W - img.shape[1], cv2.BORDER_CONSTANT, value=255)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--style", nargs="*", default=["upright"])
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    w = Writer()
    rows = [("before: quick preset, no letter-shape check", *row(w, "quick", "", a.seed))]
    for st in a.style:
        rows.append((f"after: {st.rstrip('/').split('/')[-1]}, plain f/r check", *row(w, st, "abcdefghijklmnopqrstuvwxyz", a.seed)))
    W = max(r[1].shape[1] for r in rows)
    out = np.vstack([label(r[1], f"{r[0]}  (looped letters left: {r[2]}/{len(WORDS)} words)", W) for r in rows])
    cv2.imwrite(a.out, out)
    for r in rows:
        print(r[0], "flagged", r[2], r[3])


if __name__ == "__main__":
    main()

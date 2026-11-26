"""Word sheet for review: whole words cut from the writer's pages next to the same words
written by ductus.

  python scripts/word_sheet.py OUT_DIR --pages styles/my-hand/debug/page*_ink.png [--words the of and ...]
Each page ink map is split into lines (profile.line_crops) and each line into words at its
wide gaps; every word crop is read by the OCR model and kept when it reads as one of the
target words. ductus then writes the same words with the default engine. Writes
OUT_DIR/his/<word>/NNN.png, OUT_DIR/ours/<word>/NNN.png and OUT_DIR/word_sheet.png.
Contains the writer's handwriting: keep it private."""
import argparse, sys
from pathlib import Path

import cv2
import numpy as np

WORDS = ("the of and is to in a it for from first find real list value mean mode median process "
         "control software document data sample number total each with this that are").split()


def norm(t):
    return "".join(c for c in t.lower() if c.isalnum())


def word_spans(alpha, xh):
    cols = (alpha > 0.3).any(axis=0)
    spans, start, gap = [], None, 0
    for x, v in enumerate(cols):
        if v:
            if start is None:
                start = x
            gap = 0
            end = x
        elif start is not None:
            gap += 1
            if gap > 0.55 * xh:
                spans.append((start, end + 1)); start = None
    if start is not None:
        spans.append((start, end + 1))
    return [s for s in spans if s[1] - s[0] > 0.6 * xh]


def tile(gray, h=72):
    ys, xs = np.where(gray < 170)
    if len(ys) == 0:
        return np.full((h, h), 255, np.uint8)
    g = gray[max(0, ys.min() - 3):ys.max() + 4, max(0, xs.min() - 3):xs.max() + 4]
    s = (h - 10) / g.shape[0]
    g = cv2.resize(g, (max(1, int(g.shape[1] * s)), h - 10), interpolation=cv2.INTER_AREA)
    return cv2.copyMakeBorder(g, 5, 5, 8, 8, cv2.BORDER_CONSTANT, value=255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--pages", nargs="+", required=True)
    ap.add_argument("--words", nargs="*", default=WORDS); ap.add_argument("--per", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    targets = {norm(w): w for w in a.words}
    from ductus.profile import line_crops
    from ductus.segment import baseline_and_xheight
    from ductus.page import Writer, Options, load_style
    W = Writer()
    crops, keys = [], []
    for pf in a.pages:
        pg = cv2.imread(pf, 0)
        if pg is None:
            continue
        for ln in line_crops(1.0 - pg.astype(np.float32) / 255):
            c = ln["crop"].astype(np.float32)
            c = c / 255.0 if c.max() > 1.5 else c
            _, xh = baseline_and_xheight(c)
            for x0, x1 in word_spans(c, max(xh, 6)):
                crops.append((255 * (1 - np.clip(c[:, max(0, x0 - 4):x1 + 4], 0, 1))).astype(np.uint8))
    print(len(crops), "word crops", file=sys.stderr)
    his = {}
    for i in range(0, len(crops), 16):
        for crop, t in zip(crops[i:i + 16], W.reader.read(crops[i:i + 16])):
            k = norm(t)
            if k in targets and len(his.get(k, [])) < a.per:
                his.setdefault(k, []).append(crop)
    # ours: the same words, default engine and checks
    from ductus.synth import Synthesizer
    o = Options()
    prim, fb, pr = W.engines(o)
    syn = Synthesizer(prim, load_style("upright")[0][0], W.reader, max_cer=0.25, seed=17, strict=True,
                      rounds=pr + 2 if fb is not None and pr < 99 else 4, primary_rounds=pr, fallback=fb,
                      log=lambda *x: print(*x, file=sys.stderr))
    ours = {norm(w): (255 * (1 - np.clip(wi.alpha, 0, 1))).astype(np.uint8)
            for w, wi in zip(a.words, syn.run(list(a.words)))}
    rows = []
    for w in a.words:
        k = norm(w)
        for side, ims in (("his", his.get(k, [])), ("ours", [ours[k]] if k in ours else [])):
            d = out / side / k
            d.mkdir(parents=True, exist_ok=True)
            for j, im in enumerate(ims):
                cv2.imwrite(str(d / f"{j:03d}.png"), im)
        lab = np.full((72, 150), 255, np.uint8)
        cv2.putText(lab, w, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 60, 2)
        hs = [tile(im) for im in his.get(k, [])] or [np.full((72, 120), 240, np.uint8)]
        left = np.hstack(hs)
        left = cv2.copyMakeBorder(left, 0, 0, 0, max(0, 3 * 220 - left.shape[1]), cv2.BORDER_CONSTANT, value=255)[:, :3 * 220]
        sep = np.full((72, 20), 255, np.uint8); sep[:, 9:11] = 170
        right = tile(ours[k]) if k in ours else np.full((72, 120), 240, np.uint8)
        right = cv2.copyMakeBorder(right, 0, 0, 0, max(0, 260 - right.shape[1]), cv2.BORDER_CONSTANT, value=255)[:, :260]
        rows.append(np.hstack([lab, left, sep, right]))
    head = np.full((40, rows[0].shape[1]), 255, np.uint8)
    cv2.putText(head, "his pages (up to 3)", (160, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 60, 2)
    cv2.putText(head, "ductus", (150 + 660 + 30, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 60, 2)
    cv2.imwrite(str(out / "word_sheet.png"), np.vstack([head] + rows))
    print(out / "word_sheet.png", "his words found:", sum(1 for k in targets if k in his), "of", len(targets))


if __name__ == "__main__":
    main()

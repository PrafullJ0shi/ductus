"""Collect single-letter samples from line images and measure each letter's closed loops.

  python scripts/letter_glyphs.py OUT_DIR LINE.png [LINE.png ...]
  python scripts/letter_glyphs.py OUT_DIR --profile styles/my-hand        (uses its refs/)
  python scripts/letter_glyphs.py OUT_DIR --pages styles/my-hand/debug/page*_ink.png   (every line)

Each line is cut into ink components (i-dots and t-bars merged with their stem); every
component is read on its own by the OCR model, and kept when it reads as exactly one letter or
digit with good confidence. Writes OUT_DIR/glyphs/<char>/NNN.png and OUT_DIR/letters.json:
for each character, how often it has a closed loop above the x-height, inside it, or below
the baseline. `ductus.letterforms` uses letters.json (copied to a profile as letters.json)
to reject generated letters that use a loop the writer never uses.

The output contains crops of the writer's own handwriting: keep it private.
"""
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np

from ductus.letterforms import holes
from ductus.segment import baseline_and_xheight, ink_alpha


def components(alpha):
    ink = (alpha > 0.35).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    boxes = [list(st[i][:4]) for i in range(1, n) if st[i][4] >= 6]
    boxes.sort(key=lambda b: b[0])
    merged = []
    for b in boxes:                       # merge marks that overlap in x (dots, bars)
        if merged:
            m = merged[-1]
            ov = min(m[0] + m[2], b[0] + b[2]) - max(m[0], b[0])
            if ov > 0.5 * min(m[2], b[2]):
                x0, y0 = min(m[0], b[0]), min(m[1], b[1])
                x1, y1 = max(m[0] + m[2], b[0] + b[2]), max(m[1] + m[3], b[1] + b[3])
                merged[-1] = [x0, y0, x1 - x0, y1 - y0]
                continue
        merged.append(b)
    return merged


def zone_of(cy, bl, xh):
    if cy < bl - 1.3 * xh:
        return "asc"
    if cy > bl + 0.2 * xh:
        return "desc"
    return "x"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("lines", nargs="*"); ap.add_argument("--profile"); ap.add_argument("--pages", nargs="*", default=[])
    ap.add_argument("--min-conf", type=float, default=0.6)
    a = ap.parse_args()
    files = [Path(f) for f in a.lines]
    if a.profile:
        files += sorted((Path(a.profile) / "refs").glob("*.png"))
    from ductus.verify import Reader
    R = Reader()
    out = Path(a.out); (out / "glyphs").mkdir(parents=True, exist_ok=True)
    stats, count = {}, {}
    grays = [cv2.imread(str(f), 0) for f in files]
    if a.pages:
        from ductus.profile import line_crops
        for pf in a.pages:
            pg = cv2.imread(pf, 0)
            if pg is None:
                continue
            for ln in line_crops(1.0 - pg.astype(np.float32) / 255):
                c = ln["crop"]
                c = c.astype(np.float32) / (255.0 if c.max() > 1.5 else 1.0)
                grays.append((255 * (1 - np.clip(c, 0, 1))).astype(np.uint8))
    for g in grays:
        if g is None:
            continue
        alpha = ink_alpha(g)
        bl, xh = baseline_and_xheight(alpha)
        boxes = components(alpha)
        crops, meta = [], []
        for x, y, w, h in boxes:
            if w < 3 or h < 0.4 * xh:
                continue
            x0, x1 = max(0, x - 3), min(g.shape[1], x + w + 3)
            sub = alpha[:, x0:x1]
            crops.append((255 * (1 - np.clip(sub, 0, 1))).astype(np.uint8))
            meta.append((x0, sub))
        for i in range(0, len(crops), 16):
            texts, conf = R.read(crops[i:i + 16], with_conf=True)
            for (x0, sub), crop, t, c in zip(meta[i:i + 16], crops[i:i + 16], texts, conf):
                t = t.strip().rstrip(".")
                if len(t) != 1 or not t.isalnum() or c < a.min_conf:
                    continue
                k = count.get(t, 0); count[t] = k + 1
                d = out / "glyphs" / (t if t.islower() or t.isdigit() else f"cap_{t}")
                d.mkdir(exist_ok=True)
                cv2.imwrite(str(d / f"{k:03d}.png"), crop)
                zs = {zone_of(cy, bl, xh) for _, cy, _ in holes(sub)}
                s = stats.setdefault(t, {"n": 0, "asc": 0, "x": 0, "desc": 0})
                s["n"] += 1
                for z in zs:
                    s[z] += 1
    letters = {ch: {"n": s["n"], **{z: round(s[z] / s["n"], 3) for z in ("asc", "x", "desc")}}
               for ch, s in sorted(stats.items())}
    (out / "letters.json").write_text(json.dumps(letters, indent=1))
    print(f"{sum(count.values())} glyphs, {len(count)} characters ->", out / "letters.json")


if __name__ == "__main__":
    main()

"""Letter sheet for review: the writer's letters next to ductus's, one row per character.

  python scripts/letter_sheet.py HIS_DIR OURS_DIR OUT.png [--per 6]
HIS_DIR and OURS_DIR are outputs of letter_glyphs.py. Characters missing on one side are
shown with an empty cell. Contains the writer's handwriting: keep it private."""
import argparse
from pathlib import Path

import cv2
import numpy as np

CHARS = list("abcdefghijklmnopqrstuvwxyz") + [f"cap_{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"] + list("0123456789")
CELL = 56


def cell(p):
    g = cv2.imread(str(p), 0)
    ys, xs = np.where(g < 160)
    if len(ys) == 0:
        return np.full((CELL, CELL), 255, np.uint8)
    g = g[max(0, ys.min() - 2):ys.max() + 3, max(0, xs.min() - 2):xs.max() + 3]
    s = (CELL - 8) / max(g.shape)
    g = cv2.resize(g, (max(1, int(g.shape[1] * s)), max(1, int(g.shape[0] * s))), interpolation=cv2.INTER_AREA)
    c = np.full((CELL, CELL), 255, np.uint8)
    y, x = (CELL - g.shape[0]) // 2, (CELL - g.shape[1]) // 2
    c[y:y + g.shape[0], x:x + g.shape[1]] = g
    return c


def side(d, ch, per):
    fs = sorted((Path(d) / "glyphs" / ch).glob("*.png"))[:per]
    cells = [cell(f) for f in fs] + [np.full((CELL, CELL), 245, np.uint8)] * (per - len(fs))
    return np.hstack(cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("his"); ap.add_argument("ours"); ap.add_argument("out"); ap.add_argument("--per", type=int, default=6)
    a = ap.parse_args()
    rows = []
    for ch in CHARS:
        lab = np.full((CELL, 70), 255, np.uint8)
        cv2.putText(lab, ch.replace("cap_", ""), (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 60, 2)
        gap = np.full((CELL, 24), 255, np.uint8)
        gap[:, 11:13] = 180
        rows.append(np.hstack([lab, side(a.his, ch, a.per), gap, side(a.ours, ch, a.per)]))
    head = np.full((40, rows[0].shape[1]), 255, np.uint8)
    cv2.putText(head, "his pages", (80, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 60, 2)
    cv2.putText(head, "ductus", (70 + CELL * a.per + 34, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 60, 2)
    cv2.imwrite(a.out, np.vstack([head] + rows))
    print(a.out)


if __name__ == "__main__":
    main()

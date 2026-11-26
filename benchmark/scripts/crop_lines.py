"""Turn every system's raw output into one tight grayscale line crop per test text.
out/<system>/lines/NNN.png. Page outputs (ductus A4, text-to-handwriting page) are cropped to
the bounding box of ink pixels (gray < 150 after removing isolated specks); line outputs are
padded. Nothing else is changed (no denoising, no rescaling)."""
import sys, json
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
ids = (ROOT / "texts/ids.txt").read_text().split()


def crop(g, pad=12):
    m = (g < 150).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] >= 6:
            keep[lab == k] = 1
    ys, xs = np.where(keep)
    if len(xs) == 0:
        return g
    y0, y1, x0, x1 = max(0, ys.min() - pad), ys.max() + pad, max(0, xs.min() - pad), xs.max() + pad
    return g[y0:y1 + 1, x0:x1 + 1]


def src_for(system, i, name):
    d = ROOT / "out" / system
    for p in [d / f"{name}.png", d / f"{name}-1.png", d / f"{i:03d}_page.png", d / f"{i:03d}.png"]:
        if p.exists():
            return p


for system in sys.argv[1:]:
    od = ROOT / "out" / system / "lines"
    od.mkdir(exist_ok=True)
    for i, name in enumerate(ids):
        p = src_for(system, i, name)
        if p is None:
            print("missing", system, name); continue
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(od / f"{i:03d}.png"), crop(g))
    print("cropped", system)

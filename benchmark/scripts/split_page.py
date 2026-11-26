"""Cut the 7 test lines out of a ductus A4 page (blank line between inputs, so the gaps
between them are much larger than gaps inside a line). Row profile of ink pixels (gray < 150),
bands merged while the gap between them is under 0.9x the median band height; the 7 largest
gaps then separate the lines. Writes out/<system>/lines/NNN.png and saves a debug overlay."""
import sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sid in sys.argv[1:]:
    d = ROOT / "out" / sid
    p = d / "page.png" if (d / "page.png").exists() else d / "page-1.png"
    g = cv2.imread(str(p), 0)
    m = (g < 150).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    rows = np.where(m.sum(1) > 0)[0]
    runs, s = [], rows[0]
    for a, b in zip(rows[:-1], rows[1:]):
        if b - a > 1:
            runs.append([s, a]); s = b
    runs.append([s, rows[-1]])
    gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
    cut = sorted(np.argsort(gaps)[::-1][:6])          # 6 largest gaps -> 7 lines
    groups, start = [], 0
    for c in cut:
        groups.append((runs[start][0], runs[c][1])); start = c + 1
    groups.append((runs[start][0], runs[-1][1]))
    (d / "lines").mkdir(exist_ok=True)
    dbg = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for i, (y0, y1) in enumerate(groups):
        band = m[y0:y1 + 1]
        xs = np.where(band.sum(0) > 0)[0]
        x0, x1 = xs.min(), xs.max()
        cv2.imwrite(str(d / "lines" / f"{i:03d}.png"), g[max(0, y0 - 12):y1 + 13, max(0, x0 - 12):x1 + 13])
        cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.imwrite(str(d / "split_debug.png"), dbg)
    print(sid, len(runs), "bands ->", len(groups), "lines", groups)

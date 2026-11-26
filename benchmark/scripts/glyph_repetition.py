"""Glyph-repetition rate: how many ink shapes have an identical twin elsewhere in the output.

Label-free, so it works for joined and unjoined writing alike. For each system, every connected
ink component (gray < 128, 8-connected) in all 7 line crops is collected. Components shorter
than half the median component height of their line (dots, commas, specks) are ignored. Two
components are compared only if their bounding boxes agree within 5% (or 2 px) in width and
height; both are resized to the larger box and the normalised cross-correlation of the binary
masks is taken. A pair with NCC >= 0.95 is a twin. The rate is the share of components that
have at least one twin. A font repeats glyph images, so repeated letters or joined words become
twins; independently generated strokes should not.
  python glyph_repetition.py sys1 sys2 ...  -> out/repetition.json
"""
import json, sys, itertools
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
THR = 0.95


def comps(g):
    m = (g < 128).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    boxes = [(st[k, 0], st[k, 1], st[k, 2], st[k, 3], k) for k in range(1, n) if st[k, 4] >= 12]
    if not boxes:
        return []
    med = np.median([b[3] for b in boxes])
    return [(lab[y:y + h, x:x + w] == k).astype(np.float32) for x, y, w, h, k in boxes if h >= 0.5 * med]


def ncc(a, b):
    H, W = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA).ravel(); b = cv2.resize(b, (W, H), interpolation=cv2.INTER_AREA).ravel()
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def close(p, q):
    return abs(p - q) <= max(2, 0.05 * max(p, q))


outp = ROOT / "out/repetition.json"
res = json.loads(outp.read_text()) if outp.exists() else {}
for s in sys.argv[1:]:
    cs = []
    for p in sorted((ROOT / "out" / s / "lines").glob("*.png")):
        cs += comps(cv2.imread(str(p), 0))
    twin = np.zeros(len(cs), bool); pairs = 0; best = []
    for i, j in itertools.combinations(range(len(cs)), 2):
        a, b = cs[i], cs[j]
        if close(a.shape[0], b.shape[0]) and close(a.shape[1], b.shape[1]):
            pairs += 1
            v = ncc(a, b); best.append(v)
            if v >= THR:
                twin[i] = twin[j] = True
    res[s] = dict(components=len(cs), comparable_pairs=pairs, twin_components=int(twin.sum()),
                  twin_rate=round(float(twin.mean()), 3) if cs else None,
                  median_pair_ncc=round(float(np.median(best)), 3) if best else None)
    print(s, res[s], flush=True)
outp.write_text(json.dumps(res, indent=1))

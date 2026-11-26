"""Identical-copy rate: how often two occurrences of the same word look the same.

Each evaluation line is split into its words (widest gaps, known word count). Every pair of
occurrences of the same word (3+ letters) across the 40 lines is compared: both crops are
scaled to 48 px height, and the normalised cross-correlation of the ink maps is taken at the
best shift within +-3 px. Pairs whose widths differ by more than 15% are
skipped (usually a word-split error; skipping them can only raise a system's rate). A pair with NCC >= 0.95 counts as an identical copy.

  python scripts/repetition_probe.py sentences.txt EVAL_DIR sys1,sys2,...
"""
import itertools, json, re, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ductus.segment import ink_alpha, split_words, tight_bbox

src, root, systems = sys.argv[1], Path(sys.argv[2]), sys.argv[3].split(",")
lines = Path(src).read_text().splitlines()
H = 48


def norm(a):
    x0, y0, x1, y1 = tight_bbox(a, 0.2)
    a = a[y0:y1 + 1, x0:x1 + 1]
    w = max(4, int(round(a.shape[1] * H / max(1, a.shape[0]))))
    return cv2.resize(a, (w, H), interpolation=cv2.INTER_AREA)


def ncc(a, b):
    if abs(a.shape[1] - b.shape[1]) > 0.15 * max(a.shape[1], b.shape[1]):
        return None  # width disagrees: likely a split error, pair not counted
    if b.shape[1] > a.shape[1]:
        a, b = b, a
    pad = cv2.copyMakeBorder(a, 3, 3, 3, 3 + 0, cv2.BORDER_CONSTANT, value=0)
    r = cv2.matchTemplate(pad.astype(np.float32), b.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    return float(r.max())


res = {}
for s in systems:
    occ = defaultdict(list)
    for i, t in enumerate(lines):
        p = root / s / f"{i:03d}.png"
        if not p.exists():
            continue
        g = cv2.imread(str(p), 0)
        a = ink_alpha(g)
        words = t.split()
        spans = split_words(a, len(words))
        if spans is None:
            continue
        for w, (x0, x1) in zip(words, spans):
            k = w  # exact string: same case, same punctuation
            if len(k) >= 3:
                occ[k].append(norm(a[:, x0:x1 + 1]))
    scores = []
    for k, crops in occ.items():
        for a, b in itertools.combinations(crops[:8], 2):
            v = ncc(a, b)
            if v is not None:
                scores.append(v)
    scores = np.array(scores)
    res[s] = dict(pairs=int(len(scores)), identical_rate=round(float((scores >= 0.95).mean()), 3) if len(scores) else None,
                  near_rate_090=round(float((scores >= 0.90).mean()), 3) if len(scores) else None,
                  median_ncc=round(float(np.median(scores)), 3) if len(scores) else None)
    print(s, res[s], flush=True)
json.dump(res, open(root / "repetition.json", "w"), indent=1)

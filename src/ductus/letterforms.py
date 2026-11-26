"""Letter-shape check: reject calligraphic flourishes the target hand does not use.

An average print-ish hand writes
  f  as a stem with a crossbar (no loop below the baseline, no loop at the top),
  r  as a short stem with a small shoulder (no closed loop),
  b d h k l t  with plain ascenders (no closed loop above the x-height).
A closed counter (a hole in the ink) where the plain form has none is the signature of the
looped or stylish variant. Holes are found as background components that do not touch the
word's border; each one is assigned to a letter by its horizontal position and judged by
the zone (ascender, x-band, descender) its centre lies in.
"""
from __future__ import annotations

import cv2
import numpy as np

ASC_LOOP = set("bdhklt")
PLAIN_DEFAULT = set("fr") | ASC_LOOP


def holes(alpha: np.ndarray, thr: float = 0.35, min_area: int = 5):
    """(cx, cy, area) of closed counters in a word's ink map."""
    ink = (alpha > thr).astype(np.uint8)
    if ink.sum() == 0:
        return []
    bg = np.pad(1 - ink, 1, constant_values=1)
    n, lab, st, cen = cv2.connectedComponentsWithStats(bg, connectivity=4)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < min_area:
            continue
        if x == 0 or y == 0 or x + w >= bg.shape[1] or y + h >= bg.shape[0]:
            continue                      # the outside background, not a counter
        out.append((float(cen[i][0] - 1), float(cen[i][1] - 1), int(a)))
    return out


def flourishes(alpha: np.ndarray, text: str, baseline: float, xh: float,
               letters: set[str] = PLAIN_DEFAULT) -> list[str]:
    """Letters of ``text`` drawn with a closed loop the plain form does not have."""
    cols = np.where((alpha > 0.35).any(axis=0))[0]
    if len(cols) == 0 or not text:
        return []
    x0, x1 = cols[0], cols[-1] + 1
    n = len(text)
    cw = (x1 - x0) / n
    found = []
    for cx, cy, a in holes(alpha):
        i = int(np.clip((cx - x0) / cw, 0, n - 1))
        # the counter may sit between two letters: look at the nearest one or two
        cand = {i}
        frac = (cx - x0) / cw - i
        if frac < 0.3 and i > 0:
            cand.add(i - 1)
        if frac > 0.7 and i < n - 1:
            cand.add(i + 1)
        for j in cand:
            ch = text[j]
            if ch not in letters:
                continue
            asc = cy < baseline - 1.3 * xh
            desc = cy > baseline + 0.2 * xh
            xband = not asc and not desc
            if ch == "f" and (asc or desc):
                found.append("f")
            elif ch in ASC_LOOP and asc:
                found.append(ch)
            elif ch == "r" and xband and len(cand) == 1:
                found.append("r")
    return sorted(set(found))

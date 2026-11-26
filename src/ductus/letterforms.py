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
# where a plain print-like letter has no closed loop (counter). Letters with natural counters
# (a b d e g o p q, and capitals/digits) are only judged in zones where they have none.
NO_LOOP = {
    "asc": set("bdfhklt"),
    "desc": set("fjpqz"),                  # looped g and y are common in plain hands
    "x": set("cfhijklmnrstuvwxyz"),
}
ALL = set("abcdefghijklmnopqrstuvwxyz")
NEVER_LOOP = set("flr")       # the target hand writes these plain whatever the statistics say
PLAIN_DEFAULT = ALL


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
               letters: set[str] = PLAIN_DEFAULT, profile: dict | None = None) -> list[str]:
    """Letters of ``text`` drawn with a closed loop the plain form does not have.

    ``profile``: per-letter loop rates measured from the writer's own pages
    (scripts/letter_glyphs.py -> letters.json). With 5 or more samples of a letter (not f, l or r), a loop in a
    zone is allowed when the writer uses it in at least 20% of samples; everything else follows
    the plain-print rules above."""
    cols = np.where((alpha > 0.35).any(axis=0))[0]
    if len(cols) == 0 or not text:
        return []
    x0, x1 = cols[0], cols[-1] + 1
    n = len(text)
    cw = (x1 - x0) / n
    found = []
    for cx, cy, a in holes(alpha):
        i = int(np.clip((cx - x0) / cw, 0, n - 1))
        cand = {i}
        frac = (cx - x0) / cw - i
        if frac < 0.3 and i > 0:
            cand.add(i - 1)
        if frac > 0.7 and i < n - 1:
            cand.add(i + 1)
        zone = "asc" if cy < baseline - 1.3 * xh else ("desc" if cy > baseline + 0.2 * xh else "x")
        # a counter belongs to one letter: if any candidate may have a loop here, it is fine
        verdicts = [_forbidden(text[j], zone, profile) for j in cand if text[j] in letters]
        if not verdicts or not all(verdicts):
            continue
        if zone == "x" and len(cand) > 1:
            continue                       # between two letters in the x-band: too ambiguous
        found.extend(text[j] for j in cand if text[j] in letters)
    return sorted(set(found))


def _forbidden(ch: str, zone: str, profile: dict | None) -> bool:
    # the writer's own samples can only allow a loop (he uses it in at least 20% of samples);
    # they never forbid one the plain form has, since a missed counter in a small photo crop
    # would otherwise reject correct letters
    if (profile and ch not in NEVER_LOOP and ch in profile and profile[ch].get("n", 0) >= 5
            and profile[ch].get(zone, 0.0) >= 0.2):
        return False
    if not ch.isalpha() or not ch.islower():
        return False
    return ch in NO_LOOP[zone]

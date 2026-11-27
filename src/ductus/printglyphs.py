"""Plain hand-print glyphs for digits, maths symbols and capitals.

The RNN writer garbles numbers, symbols and acronyms ("4)" read as "1)", "BTC" as "BC",
"HTML" as "HTM"). Tokens made only of these characters are drawn from the templates here
instead, the way the r, f and the capitals M, R, H, E and N are redrawn in the RNN engine.
Each template is a list of strokes in x-height units: x to the right, y down, baseline at
y = 0, x-height line at y = -1. A stroke is ("s", points) for a smooth curve through the
points or ("c", points) for straight segments with softened corners. Every use gets its own
small scale, slant, baseline and point jitter, so no two characters are identical."""
from __future__ import annotations

import re

import numpy as np

D = 1.38          # digit height
C = 1.45          # capital height


def _oval(cx, cy, rx, ry, start=0.25, turn=1.08, n=14):
    t = np.linspace(0, turn, n) * 2 * np.pi + start * 2 * np.pi
    return [(cx + rx * np.cos(-a), cy + ry * np.sin(-a)) for a in t]


DOT = [("d", [(0.06, -0.03)])]
T: dict[str, list] = {
    # digits
    "0": [("s", _oval(0.36, -D / 2, 0.34, D / 2, start=0.2))],
    "1": [("c", [(0.08, -1.08), (0.30, -D), (0.30, 0.0)])],
    "2": [("s", [(0.04, -1.08), (0.16, -1.30), (0.36, -D), (0.56, -1.30), (0.60, -1.04), (0.46, -0.72),
                 (0.22, -0.34), (0.02, 0.0)]), ("c", [(0.02, 0.0), (0.68, 0.0)])],
    "3": [("s", [(0.04, -1.22), (0.26, -D), (0.50, -1.32), (0.58, -1.08), (0.44, -0.84), (0.24, -0.76)]),
          ("s", [(0.24, -0.76), (0.50, -0.66), (0.64, -0.38), (0.54, -0.10), (0.30, 0.0), (0.02, -0.14)])],
    "4": [("c", [(0.44, -D), (0.02, -0.42), (0.74, -0.42)]), ("c", [(0.54, -0.96), (0.54, 0.0)])],
    "5": [("c", [(0.62, -D), (0.12, -D), (0.08, -0.78)]),
          ("s", [(0.08, -0.78), (0.32, -0.88), (0.58, -0.76), (0.66, -0.42), (0.52, -0.10), (0.26, 0.0),
                 (0.02, -0.14)])],
    "6": [("s", [(0.58, -1.30), (0.38, -D), (0.16, -1.14), (0.05, -0.62), (0.10, -0.16), (0.34, 0.0),
                 (0.58, -0.12), (0.63, -0.44), (0.46, -0.70), (0.20, -0.66), (0.06, -0.42)])],
    "7": [("c", [(0.02, -D), (0.70, -D), (0.24, 0.0)])],
    "8": [("s", [(0.56, -1.18), (0.36, -D), (0.12, -1.26), (0.14, -0.96), (0.48, -0.66), (0.62, -0.34),
                 (0.44, -0.02), (0.14, -0.06), (0.06, -0.36), (0.24, -0.64), (0.52, -0.92), (0.56, -1.18)])],
    "9": [("s", [(0.60, -1.10), (0.46, -D), (0.20, -1.34), (0.06, -1.08), (0.16, -0.84), (0.42, -0.80),
                 (0.60, -1.02)]), ("s", [(0.61, -1.24), (0.60, -0.60), (0.52, 0.0)])],
    # punctuation and maths
    ".": DOT,
    ",": [("c", [(0.08, -0.06), (0.03, 0.22)])],
    ":": [("d", [(0.06, -0.72)]), ("d", [(0.06, -0.03)])],
    ";": [("d", [(0.08, -0.72)]), ("c", [(0.08, -0.06), (0.03, 0.22)])],
    "(": [("s", [(0.30, -1.50), (0.12, -1.06), (0.06, -0.60), (0.12, -0.10), (0.30, 0.26)])],
    ")": [("s", [(0.04, -1.50), (0.22, -1.06), (0.28, -0.60), (0.22, -0.10), (0.04, 0.26)])],
    "-": [("c", [(0.04, -0.62), (0.50, -0.60)])],
    "+": [("c", [(0.04, -0.62), (0.62, -0.60)]), ("c", [(0.33, -0.92), (0.33, -0.30)])],
    "=": [("c", [(0.04, -0.80), (0.62, -0.79)]), ("c", [(0.04, -0.42), (0.62, -0.41)])],
    "/": [("c", [(0.56, -1.48), (0.02, 0.10)])],
    "%": [("s", _oval(0.16, -1.16, 0.14, 0.22, n=10)), ("c", [(0.72, -D), (0.08, 0.0)]),
          ("s", _oval(0.64, -0.22, 0.14, 0.22, n=10))],
    "!": [("c", [(0.10, -D), (0.08, -0.36)]), ("d", [(0.08, -0.03)])],
    "?": [("s", [(0.02, -1.14), (0.14, -1.36), (0.38, -1.40), (0.56, -1.20), (0.46, -0.94), (0.28, -0.74),
                 (0.27, -0.40)]), ("d", [(0.27, -0.03)])],
    "#": [("c", [(0.30, -1.30), (0.18, 0.0)]), ("c", [(0.58, -1.30), (0.46, 0.0)]),
          ("c", [(0.04, -0.88), (0.74, -0.88)]), ("c", [(0.00, -0.42), (0.70, -0.42)])],
    "*": [("c", [(0.30, -1.30), (0.30, -0.62)]), ("c", [(0.04, -1.12), (0.56, -0.80)]),
          ("c", [(0.56, -1.12), (0.04, -0.80)])],
    "'": [("c", [(0.06, -1.46), (0.04, -1.12)])],
    '"': [("c", [(0.06, -1.46), (0.04, -1.12)]), ("c", [(0.24, -1.46), (0.22, -1.12)])],
    # capitals, plain print
    "A": [("c", [(0.00, 0.0), (0.37, -C), (0.74, 0.0)]), ("c", [(0.16, -0.50), (0.58, -0.50)])],
    "B": [("c", [(0.0, 0.0), (0.0, -C)]),
          ("s", [(0.0, -C), (0.34, -1.44), (0.54, -1.30), (0.54, -0.96), (0.30, -0.78), (0.0, -0.76)]),
          ("s", [(0.0, -0.76), (0.40, -0.74), (0.62, -0.54), (0.60, -0.18), (0.38, -0.01), (0.0, 0.0)])],
    "C": [("s", [(0.68, -1.20), (0.50, -1.42), (0.24, -1.40), (0.06, -1.10), (0.02, -0.70), (0.08, -0.26),
                 (0.30, -0.01), (0.54, -0.02), (0.70, -0.22)])],
    "D": [("c", [(0.0, 0.0), (0.0, -C)]),
          ("s", [(0.0, -C), (0.36, -1.40), (0.62, -1.12), (0.70, -0.72), (0.62, -0.30), (0.34, -0.03), (0.0, 0.0)])],
    "E": [("c", [(0.62, -C), (0.0, -C), (0.0, 0.0), (0.66, 0.0)]), ("c", [(0.0, -0.74), (0.50, -0.74)])],
    "F": [("c", [(0.62, -C), (0.0, -C), (0.0, 0.0)]), ("c", [(0.0, -0.76), (0.48, -0.76)])],
    "G": [("s", [(0.68, -1.20), (0.50, -1.42), (0.24, -1.40), (0.06, -1.10), (0.02, -0.70), (0.08, -0.26),
                 (0.30, -0.01), (0.56, -0.04), (0.70, -0.26), (0.70, -0.60)]), ("c", [(0.70, -0.60), (0.40, -0.60)])],
    "H": [("c", [(0.0, -C), (0.0, 0.0)]), ("c", [(0.72, -C), (0.72, 0.0)]), ("c", [(0.0, -0.75), (0.72, -0.76)])],
    "I": [("c", [(-0.16, -C), (0.26, -C)]), ("c", [(0.05, -C), (0.05, 0.0)]), ("c", [(-0.16, 0.0), (0.26, 0.0)])],
    "J": [("s", [(0.56, -C), (0.56, -0.36), (0.46, -0.06), (0.26, 0.0), (0.06, -0.20)])],
    "K": [("c", [(0.0, -C), (0.0, 0.0)]), ("c", [(0.64, -C), (0.02, -0.64), (0.68, 0.0)])],
    "L": [("c", [(0.0, -C), (0.0, 0.0), (0.58, 0.0)])],
    "M": [("c", [(0.0, 0.0), (0.03, -C), (0.46, -0.42), (0.88, -C), (0.93, 0.0)])],
    "N": [("c", [(0.0, 0.0), (0.0, -C), (0.72, 0.0), (0.72, -C)])],
    "O": [("s", _oval(0.40, -C / 2, 0.40, C / 2, start=0.2))],
    "P": [("c", [(0.0, 0.0), (0.0, -C)]),
          ("s", [(0.0, -C), (0.38, -1.44), (0.60, -1.26), (0.60, -0.96), (0.38, -0.74), (0.0, -0.72)])],
    "Q": [("s", _oval(0.40, -C / 2, 0.40, C / 2, start=0.2)), ("c", [(0.46, -0.36), (0.84, 0.08)])],
    "R": [("c", [(0.0, 0.0), (0.0, -C)]),
          ("s", [(0.0, -C), (0.36, -1.44), (0.58, -1.26), (0.58, -0.98), (0.36, -0.78), (0.0, -0.76)]),
          ("c", [(0.18, -0.76), (0.66, 0.0)])],
    "S": [("s", [(0.62, -1.24), (0.42, -1.42), (0.14, -1.38), (0.04, -1.10), (0.20, -0.82), (0.48, -0.62),
                 (0.64, -0.34), (0.54, -0.08), (0.28, 0.0), (0.02, -0.16)])],
    "T": [("c", [(0.0, -C), (0.74, -C)]), ("c", [(0.37, -C), (0.37, 0.0)])],
    "U": [("s", [(0.0, -C), (0.0, -0.44), (0.12, -0.08), (0.35, 0.0), (0.58, -0.08), (0.68, -0.44), (0.68, -C)])],
    "V": [("c", [(0.0, -C), (0.37, 0.0), (0.74, -C)])],
    "W": [("c", [(0.0, -C), (0.22, 0.0), (0.46, -1.0), (0.70, 0.0), (0.92, -C)])],
    "X": [("c", [(0.0, -C), (0.68, 0.0)]), ("c", [(0.68, -C), (0.0, 0.0)])],
    "Y": [("c", [(0.0, -C), (0.37, -0.70), (0.74, -C)]), ("c", [(0.37, -0.70), (0.37, 0.0)])],
    "Z": [("c", [(0.0, -C), (0.68, -C), (0.0, 0.0), (0.70, 0.0)])],
    # the two lowercase letters that stand alone in maths
    "x": [("c", [(0.0, -1.0), (0.60, 0.0)]), ("c", [(0.60, -1.0), (0.0, 0.0)])],
    "n": [("c", [(0.0, -1.0), (0.0, 0.0)]),
          ("s", [(0.0, -0.66), (0.18, -0.94), (0.42, -1.0), (0.58, -0.82), (0.60, -0.40), (0.60, 0.0)])],
}
CHARS = set(T)
_TOKEN = re.compile(r"^[0-9A-Z.,:;()\-+=/%!?#*'\"xn]+$")


def handles(word: str) -> bool:
    """True for tokens drawn from these templates: numbers, symbols and capitals-only tokens
    (acronyms, list markers, Q3., S.D.). Lowercase x and n only on their own or next to
    digits and brackets ("x", "(x", "2n"), never inside an ordinary word."""
    if not word or not _TOKEN.match(word):
        return False
    low = [c for c in word if c in "xn"]
    if low and (len(low) > 1 or any(c.isupper() for c in word)):
        return False
    if word in ("A", "I"):
        return False            # pronoun and article: the writer's own hand reads fine there
    return any(c.isalnum() for c in word) or word in ("=", "+", "-", "/", "%", "?", "!", "#", "*")


def _corners(t: np.ndarray, per: int = 10) -> np.ndarray:
    seg = [t[i] + (t[i + 1] - t[i]) * u[:, None] for i in range(len(t) - 1)
           for u in [np.linspace(0, 1, per, endpoint=i == len(t) - 2)]]
    p = np.vstack(seg)
    q = p.copy()
    q[1:-1] = 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]
    return q


def _smooth(t: np.ndarray, n: int = 40) -> np.ndarray:
    from scipy.interpolate import splprep, splev
    try:
        k = min(3, len(t) - 1)
        tck, _ = splprep([t[:, 0], t[:, 1]], s=0.0004 * len(t), k=k)
        x, y = splev(np.linspace(0, 1, n), tck)
        return np.c_[x, y]
    except Exception:
        return _corners(t)


def strokes(word: str, rng: np.random.Generator, mess: float = 0.3) -> list[np.ndarray]:
    """Strokes of a whole token in x-height units (baseline y = 0), characters side by side."""
    out = []
    x = 0.0
    shear = rng.normal(0.07, 0.03)                 # one slant per token, a little forward
    drift = 0.0
    for i, ch in enumerate(word):
        if ch == " ":
            x += 0.9
            continue
        tpl = T[ch]
        sx = rng.uniform(0.93, 1.07)
        sy = rng.uniform(0.95, 1.05) * (0.92 if ch in "0123456789" else 1.0)
        drift = 0.6 * drift + rng.normal(0, 0.025 + 0.02 * mess)   # baseline wander between characters
        allx = np.concatenate([np.array(p)[:, 0] for _, p in tpl])
        left = float(allx.min())
        decimal = ch in ".," and 0 < i < len(word) - 1 and word[i - 1].isdigit() and word[i + 1].isdigit()
        if ch in ".,:;" and i > 0 and not decimal:
            x -= 0.12                              # punctuation sits close to what it follows
        for kind, pts in tpl:
            t = np.array(pts, dtype=np.float64)
            t[:, 0] = (t[:, 0] - left) * sx
            t[:, 1] = t[:, 1] * sy
            t[:, 0] -= shear * t[:, 1]
            t += rng.normal(0, 0.016, t.shape)
            if kind == "d":
                p = t + np.array([[0.0, 0.0], [0.035, 0.02]])
            elif kind == "s":
                p = _smooth(t)
            else:
                p = _corners(t)
            p[:, 0] += x
            p[:, 1] += drift
            out.append(p)
        w = (float(allx.max()) - left) * sx
        x += w + rng.uniform(0.30, 0.42) + (0.10 if ch in ".,:;" else 0.0)
    return out


def render(word: str, rng: np.random.Generator, xh: float = 26.0, pen_px: float = 2.3, mess: float = 0.3):
    """(alpha 0..1, baseline row, x-height px) of the token at the RNN engine's line scale."""
    import cv2
    st = strokes(word, rng, mess)
    allp = np.concatenate(st)
    x0, y0 = allp[:, 0].min(), allp[:, 1].min()
    pad = 6
    ss = 4
    Wd = int((allp[:, 0].max() - x0) * xh + 2 * pad + 1)
    Hd = int((allp[:, 1].max() - y0) * xh + 2 * pad + 1)
    can = np.zeros((Hd * ss, Wd * ss), np.uint8)
    t = max(1, int(round(pen_px * ss)))
    for s_ in st:
        p = np.c_[((s_[:, 0] - x0) * xh + pad) * ss, ((s_[:, 1] - y0) * xh + pad) * ss].round().astype(np.int32)
        if np.ptp(s_, axis=0).max() < 0.1:
            c = p.mean(0).astype(int)
            cv2.circle(can, (int(c[0]), int(c[1])), int(t * 0.95), 255, -1, cv2.LINE_AA)
        else:
            cv2.polylines(can, [p], False, 255, t, cv2.LINE_AA)
    a = cv2.resize(can, (Wd, Hd), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    baseline = float((0 - y0) * xh + pad)
    return a, baseline, float(xh)

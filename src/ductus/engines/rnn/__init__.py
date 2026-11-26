"""Graves (2013) handwriting RNN as a ductus line engine.

The pretrained model of Grzego/handwriting-generation (MIT) samples pen strokes for a line of
text; this module renders them into the same 64 px grayscale line images the diffusion
engine produces, so carrier lines, word splitting, the OCR read-back, layout, pens and the
realism layers work unchanged. The RNN writes plain, print-like letters, reads well and is
fast on a CPU; it cannot copy a style from a photo (a few built-in writer styles only).

Setup: a checkout of github.com/Grzego/handwriting-generation (it ships data/ and
pretrained/) and a Python with TensorFlow 2.x. Point ductus at them:
  DUCTUS_RNN_DIR=/path/to/handwriting-generation  DUCTUS_RNN_PYTHON=/path/to/python
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

H = 120                # line image height (room for looped f above and below)
CHAR_W = 30.0          # mean character advance in px; large enough that small loops keep their hole
BASE_ROW = 68.0        # row of the line's median y


def model_dir() -> Path:
    d = os.environ.get("DUCTUS_RNN_DIR")
    if d:
        return Path(d)
    return Path.home() / ".cache" / "ductus" / "handwriting-generation"


def available() -> bool:
    return (model_dir() / "pretrained" / "model-29.meta").exists()


PLAIN_SHIFT = 0.4   # the attention's mean position runs ahead of the pen by about this much


def _downstrokes(seg, Ps, ci, xh, min_dy, need_climb, lo=0.35, hi=1.35):
    """(score, start, end) of the downstrokes in seg the model attended to as character ci."""
    out, q = [], 0
    while q < len(seg) - 1:
        r0 = q
        while q + 1 < len(seg) and seg[q + 1, 1] >= seg[q, 1] - 0.01 * xh and seg[q, 2] != 1.:
            q += 1
        dy = seg[q, 1] - seg[r0, 1]; dx = abs(seg[q, 0] - seg[r0, 0])
        ok = dy >= min_dy * xh and dx <= 0.6 * dy
        if ok and need_climb:              # an r retraces up its stem before the shoulder
            ok = False
            for u in range(q + 1, min(len(seg), q + 40)):
                if seg[u - 1, 2] == 1.:
                    break
                if seg[u, 1] <= seg[q, 1] - 0.3 * xh:
                    ok = seg[u, 0] <= seg[q, 0] + 0.35 * xh
                    break
        if ok:
            out.append((float(((Ps[r0:q + 1] >= ci + lo) & (Ps[r0:q + 1] < ci + hi)).sum()), r0, q))
        q += 1
    return out


def glyph_strokes(pts: np.ndarray, pos, text: str, mode: str = "cursive", rng=None) -> list[np.ndarray]:
    """Absolute points (x, y down, pen-up flag) -> strokes, with every lowercase r and f redrawn
    from the templates in glyphs.py ("cursive": the target hand's school-cursive r and looped f;
    "print": plain print forms; "off": unchanged). Each letter's own points are found from the
    model's attention plus the stroke geometry; when that is unclear the letter is left alone.
    The rest of the line moves right or left so the new letter has its own room."""
    from . import glyphs as G
    rng = rng or np.random.default_rng(len(text) * 7 + 1)
    pts = pts.copy()
    n = len(pts)
    keep = np.ones(n, bool)
    inserts = {}
    if mode == "off" or pos is None or len(pos) != n or not any(c in text for c in "rf"):
        inserts = {}
    else:
        P = np.asarray(pos)
        adv = max(1e-6, np.ptp(pts[:, 0])) / max(1, len(text.strip()))
        xh = 0.86 * adv
        taken = np.zeros(n, bool)
        for ci, ch in enumerate(text):
            if ch not in "rf":
                continue
            win = np.where((P >= ci - (0.1 if ch == "r" else 0.7)) & (P < ci + 1.45) & ~taken)[0]
            if len(win) < 4:
                continue
            runs = np.split(win, np.where(np.diff(win) > 1)[0] + 1)
            i0, i1 = max(runs, key=len)[[0, -1]]
            seg = pts[i0:i1 + 1]
            y = seg[:, 1]
            cands = _downstrokes(seg, P[i0:i1 + 1], ci, xh, 0.4 if ch == "r" else 1.05, ch == "r",
                                 *((0.35, 1.35) if ch == "r" else (-0.3, 1.0)))
            best, ist, ib = max(cands) if cands else (0.0, 0, 0)
            if best < 2:
                if ch != "r":
                    continue
                # no clear stem: take the points attended to as this r, if they look like one letter
                own = np.where((P >= ci + 0.35) & (P < ci + 1.35) & ~taken & keep)[0]
                if len(own) < 4:
                    continue
                rr = max(np.split(own, np.where(np.diff(own) > 1)[0] + 1), key=len)
                Rg = pts[rr[0]:rr[-1] + 1]
                if not (0.2 * xh <= np.ptp(Rg[:, 0]) <= 1.6 * xh and 0.3 * xh <= np.ptp(Rg[:, 1]) <= 1.6 * xh):
                    continue
                a0, a1 = int(rr[0]), int(rr[-1])
                h = float(np.clip(np.ptp(Rg[:, 1]), 0.75 * xh, 1.1 * xh))
                base, x_left = float(Rg[:, 1].max()), float(Rg[:, 0].min())
                new = G.draw(ch, mode, x_left, base, h, rng)
                dx = max(float(s_[:, 0].max()) for s_ in new) + 0.12 * h - float(Rg[:, 0].max())
                if mode == "cursive":
                    pts[a1 + 1:, 0] += max(dx, -0.2 * h)
                keep[a0:a1 + 1] = False; taken[a0:a1 + 1] = True
                inserts[a0] = new
                continue
            xs, bottom = float(seg[ib, 0]), float(seg[ib, 1])
            if ch == "r":
                ie, ytop = ib, bottom          # shoulder end, as for the print redraw
                for q in range(ib + 1, len(seg)):
                    if seg[q - 1, 2] == 1. or seg[q, 0] > xs + 0.75 * xh:
                        break
                    if seg[q, 0] < seg[q - 1, 0] - 0.03 * xh and seg[q, 0] > xs + 0.2 * xh:
                        break
                    ytop = min(ytop, seg[q, 1])
                    if seg[q, 1] > ytop + 0.22 * xh and seg[q, 0] > xs + 0.15 * xh:
                        break
                    ie = q
                if ie - ib < 2 or ie - ib > 60:
                    continue
                h = float(np.clip(bottom - min(float(y[ist:ib + 1].min()), ytop), 0.75 * xh, 1.1 * xh))
                base = bottom
                j0 = ist + int(np.argmin(y[ist:ib + 1]))
                if mode == "cursive":              # the cursive r starts with its own lead-in
                    j0 = ist
                    while j0 > 0 and seg[j0 - 1, 1] > seg[j0, 1] - 0.01 * xh and seg[j0 - 1, 2] != 1. \
                            and seg[j0 - 1, 0] > xs - 0.6 * xh:
                        j0 -= 1
                x_left = float(seg[j0:ie + 1, 0].min())
            else:
                # f: the long downstroke; take the climb into it and everything after it that
                # stays within the letter (lower loop, exit), until a pen lift or the next letter
                j0 = ist
                while j0 > 0 and seg[j0 - 1, 1] >= seg[j0, 1] - 0.01 * xh and seg[j0 - 1, 2] != 1. \
                        and seg[j0 - 1, 0] > xs - 0.9 * xh:
                    j0 -= 1
                while j0 > 0 and seg[j0 - 1, 1] > seg[j0, 1] and seg[j0 - 1, 2] != 1. and seg[j0 - 1, 0] > xs - 0.9 * xh:
                    j0 -= 1                    # the upstroke into the top of the f
                ytop_f = float(seg[ist, 1])
                while j0 > 0 and seg[j0 - 1, 2] != 1. and seg[j0 - 1, 1] < ytop_f + 0.5 * xh \
                        and abs(seg[j0 - 1, 0] - xs) < 0.9 * xh:
                    j0 -= 1                    # the hook at the top of a print f
                ie = ib
                for q in range(ib + 1, len(seg)):
                    if seg[q - 1, 2] == 1. or seg[q, 0] > xs + 0.55 * xh or P[i0 + q] >= ci + 1.6:
                        break
                    ie = q
                near = np.where((np.abs(pts[:, 0] - xs) < 2.5 * adv) & keep)[0]
                near = near[(near < i0 + j0) | (near > i0 + ie)]
                if len(near) < 8:
                    continue
                ny = pts[near, 1]
                base = float(np.percentile(ny, 80))
                h = float(np.clip(base - np.percentile(ny, 20), 0.7 * xh, 1.15 * xh))
                x_left = float(seg[j0:ie + 1, 0].min())
                # a crossbar drawn later as its own short stroke goes too
                ends = np.where(pts[:, 2] == 1.)[0]
                starts = np.r_[0, ends[:-1] + 1]
                for s0, s1 in zip(starts, ends):
                    if s0 <= i0 + ie or taken[s0]:
                        continue
                    S = pts[s0:s1 + 1]
                    if (np.ptp(S[:, 0]) < 1.4 * xh and np.ptp(S[:, 1]) < 0.45 * xh
                            and abs(S[:, 0].mean() - xs) < 0.8 * xh and np.all(np.abs(P[s0:s1 + 1] - ci - 1.0) < 1.8)):
                        keep[s0:s1 + 1] = False; taken[s0:s1 + 1] = True
                    elif (xs - 1.1 * xh < S[0, 0] < xs + 0.3 * xh and S[0, 1] < base + 0.15 * h and S[:, 0].max() > xs
                            and abs(P[s0] - ci - 1.0) < 1.8):
                        u = 0                      # a crossbar that runs on into the next letter
                        while u < len(S) and S[u, 0] < xs + 0.35 * xh:
                            u += 1
                        keep[s0:s0 + u] = False; taken[s0:s0 + u] = True
            a0, a1 = i0 + j0, i0 + ie
            new = G.draw(ch, mode, x_left, base, h, rng)
            right_new = max(float(s_[:, 0].max()) for s_ in new)
            dx = right_new + 0.12 * h - float(pts[a0:a1 + 1, 0].max())
            if mode == "cursive":
                pts[a1 + 1:, 0] += max(dx, -0.2 * h)  # room for the wider cursive letter
            keep[a0:a1 + 1] = False
            taken[a0:a1 + 1] = True
            inserts[a0] = new
    if __import__("os").environ.get("GLYPH_LOG"):
        import sys as _s
        print(f"glyphs {mode}: {len(inserts)} of {sum(text.count(c) for c in 'rf')} r/f redrawn in {text!r}", file=_s.stderr)
    strokes, cur = [], []
    for i in range(n):
        if i in inserts:
            if cur:
                strokes.append(np.array(cur)); cur = []
            strokes.extend(inserts[i])
        if keep[i]:
            cur.append(pts[i, :2])
        elif cur:
            strokes.append(np.array(cur)); cur = []
        if pts[i, 2] == 1. and cur:
            strokes.append(np.array(cur)); cur = []
    if cur:
        strokes.append(np.array(cur))
    return [s for s in strokes if len(s) > 0]


def plain_r_strokes(pts, pos, text, rng=None):
    """The plain print r and f (kept as an option: --glyphs print)."""
    return glyph_strokes(pts, pos, text, "print", rng)


def to_strokes(coords: np.ndarray, pos=None, text: str = "", glyphs: str = "cursive") -> list[np.ndarray]:
    """Pen offsets -> absolute strokes, deskewed, left-aligned, median y at 0. With the
    attention positions and the text, every r is redrawn plain (plain_r_strokes)."""
    pts = coords.copy()
    pts[:, :2] = np.cumsum(pts[:, :2], axis=0)
    strokes = glyph_strokes(pts, pos, text, glyphs)
    if len(strokes) > 1 and len(strokes[0]) == 1:
        strokes = strokes[1:]
    if not strokes:
        return []
    allp = np.concatenate(strokes)
    k = np.polyfit(allp[:, 0], allp[:, 1], 1)[0] if len(allp) > 10 else 0.
    strokes = [np.c_[s[:, 0], s[:, 1] - k * s[:, 0]] for s in strokes]
    allp = np.concatenate(strokes)
    x0, ymid = allp[:, 0].min(), np.median(allp[:, 1])
    return [np.c_[s[:, 0] - x0, s[:, 1] - ymid] for s in strokes]


def render_line(strokes: list[np.ndarray], n_chars: int, pen_px: float = 2.2, ss: int = 4) -> np.ndarray:
    """Draw strokes as dark ink on white, H px high, scaled to CHAR_W px per character."""
    import cv2
    if not strokes:
        return np.full((H, 64), 255, np.uint8)
    allp = np.concatenate(strokes)
    width = max(1e-3, allp[:, 0].max() - allp[:, 0].min())
    s = CHAR_W * max(1, n_chars) / width
    W = int(width * s + 24)
    can = np.zeros((H * ss, W * ss), np.uint8)
    t = max(1, int(round(pen_px * ss)))
    for st in strokes:
        p = np.c_[(st[:, 0] * s + 12) * ss, (st[:, 1] * s + BASE_ROW) * ss].round().astype(np.int32)
        if len(p) == 1 or np.ptp(st, axis=0).max() < 0.15:
            c = p.mean(0).astype(int)
            cv2.circle(can, (int(c[0]), int(c[1])), int(t * 0.8), 255, -1, cv2.LINE_AA)
            continue
        cv2.polylines(can, [p], False, 255, t, cv2.LINE_AA)
    a = cv2.resize(can, (W, H), interpolation=cv2.INTER_AREA)
    return (255 - a).astype(np.uint8)


def _find_python() -> str:
    """This interpreter if it has TensorFlow, else a venv next to the model checkout."""
    import importlib.util
    if importlib.util.find_spec("tensorflow") is not None:
        return sys.executable
    for c in (model_dir() / "venv" / "bin" / "python", model_dir() / ".venv" / "bin" / "python"):
        if c.exists():
            return str(c)
    return sys.executable


class RNNEngine:
    """Same interface as DiffBrushEngine: generate(texts, style_img, seed) -> line images."""
    name = "rnn"
    xh_per_cw = 0.86     # x-height per mean character advance (diffusion lines: 1.15)
    glyphs = "cursive"       # r and f redraw: cursive (school cursive) | print | off
    per_word_xh = True   # RNN letter size drifts within a line: measure each word

    def __init__(self, style: int | None = 1, bias: float = 1.5, python: str | None = None):
        self.style, self.bias = style, bias
        self.python = python or os.environ.get("DUCTUS_RNN_PYTHON") or _find_python()
        self.proc = None
        self.charset: set[str] = set()
        self.round = 0                 # set by the synthesizer: later rounds sample neater

    def _start(self):
        if self.proc is not None:
            return
        if not available():
            raise RuntimeError(f"RNN model not found in {model_dir()} (set DUCTUS_RNN_DIR; see engines/rnn)")
        self.proc = subprocess.Popen([self.python, str(Path(__file__).with_name("worker.py")), str(model_dir())],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, bufsize=1)
        hello = json.loads(self.proc.stdout.readline())
        self.charset = set(hello["charset"])

    def supported(self, text: str) -> bool:
        self._start()
        return all(c in self.charset or c == " " for c in text)

    def strokes(self, text: str, seed: int = 0, bias: float | None = None, style="default"):
        self._start()
        q = {"text": text, "bias": self.bias if bias is None else bias,
             "style": self.style if style == "default" else style, "seed": int(seed)}
        self.proc.stdin.write(json.dumps(q) + "\n")
        r = json.loads(self.proc.stdout.readline())
        return to_strokes(np.array(r["coords"], dtype=np.float64), r.get("pos"), r.get("text", text), self.glyphs), r["ok"]

    def generate(self, texts: list[str], style_img=None, seed: int = 0) -> list[np.ndarray]:
        out = []
        for i, t in enumerate(texts):
            st, _ = self.strokes(t, seed=seed * 101 + i, bias=self.bias + 0.3 * min(self.round, 4))
            out.append(render_line(st, len(t)))
        return out

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
            self.proc = None

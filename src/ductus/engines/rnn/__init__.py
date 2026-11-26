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

H = 64                 # line image height, as the diffusion engine
CHAR_W = 21.0          # target mean character advance in px (diffusion lines: about 22)
BASE_ROW = 40.0        # row of the line's median y


def model_dir() -> Path:
    d = os.environ.get("DUCTUS_RNN_DIR")
    if d:
        return Path(d)
    return Path.home() / ".cache" / "ductus" / "handwriting-generation"


def available() -> bool:
    return (model_dir() / "pretrained" / "model-29.meta").exists()


PLAIN_SHIFT = 0.4   # the attention's mean position runs ahead of the pen by about this much


def plain_r_strokes(pts: np.ndarray, pos, text: str, rng=None) -> list[np.ndarray]:
    """Absolute points (x, y down, pen-up flag) -> strokes, with every lowercase r redrawn as a
    plain print r: one straight stem, retraced up to just over half height, and one small
    shoulder arc. No lead-in, no top flick, no loop. The r's own points are found from the
    model's attention (mean attended position, shifted by PLAIN_SHIFT)."""
    rng = rng or np.random.default_rng(0)
    n = len(pts)
    keep = np.ones(n, bool)
    inserts = {}                                   # point index -> synthetic stroke drawn there
    if pos is not None and "r" in text and len(pos) == n:
        P = np.asarray(pos)
        span = max(1e-6, np.ptp(pts[:, 0]))
        adv = span / max(1, len(text.strip()))     # mean character advance
        xh = 0.86 * adv
        taken = np.zeros(n, bool)
        for ci, ch in enumerate(text):
            if ch != "r":
                continue
            win = np.where((P >= ci - 0.1) & (P < ci + 1.45) & ~taken)[0]
            if len(win) < 4:
                continue
            runs = np.split(win, np.where(np.diff(win) > 1)[0] + 1)
            i0, i1 = max(runs, key=len)[[0, -1]]
            seg = pts[i0:i1 + 1]
            y = seg[:, 1]
            # stem: the downstroke (>= 0.4 xh tall, mostly vertical) whose points the model
            # attended to as this r; its last point is the stem bottom
            Ps = P[i0:i1 + 1]
            best, ib = -1.0, -1
            q = 0
            while q < len(seg) - 1:
                r0 = q
                while q + 1 < len(seg) and seg[q + 1, 1] >= seg[q, 1] - 0.01 * xh and seg[q, 2] != 1.:
                    q += 1
                dy = seg[q, 1] - seg[r0, 1]; dx = abs(seg[q, 0] - seg[r0, 0])
                climbs = False                     # an r retraces up its stem before the shoulder
                for u in range(q + 1, min(len(seg), q + 40)):
                    if seg[u - 1, 2] == 1.:
                        break
                    if seg[u, 1] <= seg[q, 1] - 0.3 * xh:
                        climbs = seg[u, 0] <= seg[q, 0] + 0.35 * xh
                        break
                if dy >= 0.4 * xh and dx <= 0.6 * dy and climbs:
                    score = float(((Ps[r0:q + 1] >= ci + 0.35) & (Ps[r0:q + 1] < ci + 1.35)).sum())
                    if score > best:
                        best, ib, ist = score, q, r0
                q += 1
            if ib < 0 or best < 2:
                continue
            xs, bottom = float(seg[ib, 0]), float(seg[ib, 1])
            top_pre = float(y[ist:ib + 1].min())
            # shoulder: after the bottom the pen climbs and moves right; it ends at a pen lift,
            # when x turns back, when it drops 0.22 xh below its highest point, or 0.75 xh right
            ie, ytop = ib, bottom
            for q in range(ib + 1, len(seg)):
                if seg[q - 1, 2] == 1.:
                    break
                if seg[q, 0] > xs + 0.75 * xh:
                    break
                if seg[q, 0] < seg[q - 1, 0] - 0.03 * xh and seg[q, 0] > xs + 0.2 * xh:
                    break
                ytop = min(ytop, seg[q, 1])
                if seg[q, 1] > ytop + 0.22 * xh and seg[q, 0] > xs + 0.15 * xh:
                    break
                ie = q
            if ie - ib < 2 or (ie - ib) > 60:
                continue
            h = float(np.clip(bottom - min(top_pre, ytop), 0.75 * xh, 1.1 * xh))
            top = bottom - h
            w = h * rng.uniform(0.5, 0.62)
            jt = lambda v: v + rng.normal(0, 0.012 * h)
            stem = [(jt(xs), top + t * h) for t in np.linspace(0, 1, 7)]
            back = [(jt(xs + 0.02 * h), bottom - t * 0.55 * h) for t in np.linspace(0.1, 1, 5)]
            arc = []
            for t in np.linspace(0, 1, 9):         # quarter-ellipse shoulder, ends level, no hook
                an = t * np.pi / 2
                arc.append((xs + 0.02 * h + w * np.sin(an) * 0.92,
                            top + 0.45 * h - 0.43 * h * np.sin(an) + 0.20 * h * t ** 3))
            # drop the r's own points: the lead-in down to the stem top, the stem and the shoulder
            j0 = i0 + ist + int(np.argmin(y[ist:ib + 1]))
            keep[j0:i0 + ie + 1] = False
            taken[j0:i0 + ie + 1] = True
            inserts[j0] = np.array(stem + back + arc)
    strokes, cur = [], []
    for i in range(n):
        if i in inserts:
            if cur:
                strokes.append(np.array(cur)); cur = []
            strokes.append(inserts[i])
        if keep[i]:
            cur.append(pts[i, :2])
        elif cur:
            strokes.append(np.array(cur)); cur = []
        if pts[i, 2] == 1. and cur:
            strokes.append(np.array(cur)); cur = []
    if cur:
        strokes.append(np.array(cur))
    return [s for s in strokes if len(s) > 0]


def to_strokes(coords: np.ndarray, pos=None, text: str = "") -> list[np.ndarray]:
    """Pen offsets -> absolute strokes, deskewed, left-aligned, median y at 0. With the
    attention positions and the text, every r is redrawn plain (plain_r_strokes)."""
    pts = coords.copy()
    pts[:, :2] = np.cumsum(pts[:, :2], axis=0)
    strokes = plain_r_strokes(pts, pos, text)
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
    plain_r = True       # redraw every r as a plain print r
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
        return to_strokes(np.array(r["coords"], dtype=np.float64), r.get("pos") if self.plain_r else None, r.get("text", text)), r["ok"]

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

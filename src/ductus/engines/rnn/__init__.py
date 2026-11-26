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


def to_strokes(coords: np.ndarray) -> list[np.ndarray]:
    """Pen offsets -> absolute strokes, deskewed, left-aligned, median y at 0."""
    pts = coords.copy()
    pts[:, :2] = np.cumsum(pts[:, :2], axis=0)
    strokes, b = [], 0
    for i in range(len(pts)):
        if pts[i, 2] == 1.:
            strokes.append(pts[b:i + 1, :2].copy())
            b = i + 1
    strokes = [s for s in strokes if len(s) > 0]
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
    per_word_xh = True   # RNN letter size drifts within a line: measure each word

    def __init__(self, style: int | None = 4, bias: float = 1.5, python: str | None = None):
        self.style, self.bias = style, bias
        self.python = python or os.environ.get("DUCTUS_RNN_PYTHON") or _find_python()
        self.proc = None
        self.charset: set[str] = set()

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
        return to_strokes(np.array(r["coords"], dtype=np.float64)), r["ok"]

    def generate(self, texts: list[str], style_img=None, seed: int = 0) -> list[np.ndarray]:
        out = []
        for i, t in enumerate(texts):
            st, _ = self.strokes(t, seed=seed * 101 + i)
            out.append(render_line(st, len(t)))
        return out

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
            self.proc = None

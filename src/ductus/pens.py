"""Pens and the physical ink model.

A page is first rendered as stroke *shape* (where the pen went) and *count* (how many
strokes passed over each point). This module turns that into ink *concentration* the way
a real pen deposits it, and `compose` turns concentration into colour with the
Beer-Lambert law, so overlaps and blobs get darker on their own.

What each effect stands for:

* speed: straight, smooth stretches of a stroke are written fast and get less ink;
  curves, turns, starts and ends are slow and get more (structure-tensor coherence).
* blobs: ballpoints leave a small ink blob where the ball stops: stroke ends and
  junctions. Fountain pens pool there more strongly.
* skips: on fast stretches a ballpoint ball briefly runs dry: faint gaps.
* along-stroke variation: pressure changes slowly along a line of writing.
* edge roughness: paper fibres make the edge ragged at 600 dpi.
* bleed: ink wicks a little into the paper (more for gel and fountain ink).
* overlap: two strokes over the same point deposit twice the ink.
* pencil: graphite catches only the tops of the paper grain.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np


@dataclass(frozen=True)
class Pen:
    name: str
    ink: str                 # colour of a single, average stroke on white paper
    line_mm: float           # typical line width on paper
    kind: str                # ballpoint | gel | fountain | pencil
    density: float = 1.0     # mean ink concentration
    speed: float = 0.35      # how much faster (straighter) strokes lighten
    blob: float = 0.45       # extra ink at ends and junctions
    skips: float = 0.25      # share of fast stretches with faint gaps
    rough: float = 0.35      # edge roughness in px at 600 dpi
    bleed: float = 0.6       # wicking blur in px at 600 dpi
    nib_ratio: float = 1.0   # fountain: thin/thick ratio of the nib
    nib_angle: float = 40.0  # fountain: nib angle, degrees
    grain: float = 0.0       # pencil: paper-tooth texture strength


# line widths are what the tip leaves on paper, not the tip size printed on the pen
TIPS = {0.5: 0.30, 0.7: 0.38, 1.0: 0.50}

PENS = {
    "blue-ballpoint": Pen("blue-ballpoint", "#2c3a86", 0.38, "ballpoint"),
    "black-ballpoint": Pen("black-ballpoint", "#2a2a30", 0.38, "ballpoint", density=1.05),
    "blue-gel": Pen("blue-gel", "#1f3fc4", 0.48, "gel", density=1.25, speed=0.08, blob=0.15,
                    skips=0.0, rough=0.2, bleed=0.9),
    "black-gel": Pen("black-gel", "#141418", 0.48, "gel", density=1.3, speed=0.08, blob=0.15,
                     skips=0.0, rough=0.2, bleed=0.9),
    "fountain": Pen("fountain", "#2a3a7e", 0.42, "fountain", density=0.8, speed=0.2, blob=0.5,
                    skips=0.0, rough=0.25, bleed=1.2, nib_ratio=0.38, nib_angle=40.0),
    "pencil": Pen("pencil", "#55565c", 0.42, "pencil", density=0.75, speed=0.3, blob=0.1,
                  skips=0.0, rough=0.6, bleed=0.2, grain=0.75),
}
DEFAULT = "blue-ballpoint"


def get_pen(name: str | None, tip: float | None = None, ink: str | None = None) -> Pen:
    p = PENS[name or DEFAULT]
    if tip:
        line = TIPS.get(float(tip), 0.42 * float(tip) + 0.08)
        if p.kind in ("gel", "fountain"):
            line *= 1.2
        p = replace(p, line_mm=line)
    if ink:
        p = replace(p, ink=ink)
    return p


def nib_kernel(width_px: float, pen: Pen) -> np.ndarray | None:
    """Fountain nib footprint: a thin rotated ellipse. Dilating a centre line with it makes
    the line thick in one direction and thin in the other."""
    if pen.kind != "fountain":
        return None
    a = max(1.0, width_px * 0.75)
    b = max(0.8, a * pen.nib_ratio)
    n = int(2 * a + 3) | 1
    yy, xx = np.mgrid[:n, :n] - n // 2
    t = np.radians(pen.nib_angle)
    u = xx * np.cos(t) + yy * np.sin(t)
    v = -xx * np.sin(t) + yy * np.cos(t)
    return ((u / a) ** 2 + (v / b) ** 2 <= 1).astype(np.uint8)


def _noise(shape, rng, scale, octaves=1):
    h, w = shape
    out = np.zeros(shape, np.float32)
    amp = 1.0
    for o in range(octaves):
        s = max(1.0, scale / (2 ** o))
        gh, gw = max(2, int(h / s) + 2), max(2, int(w / s) + 2)
        g = rng.standard_normal((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        amp *= 0.5
    return out / max(1e-6, float(out.std()))


def concentration(shape: np.ndarray, count: np.ndarray, pen: Pen, rng, px_per_mm: float) -> np.ndarray:
    """shape: stroke coverage 0..1 (union), count: overlapping strokes, both at render scale.
    Returns ink concentration (0 = paper, 1 = one normal stroke)."""
    m = shape > 0.35
    if not m.any():
        return np.zeros_like(shape)
    ys, xs = np.where(m)
    y0, y1 = max(0, ys.min() - 8), min(m.shape[0], ys.max() + 9)
    x0, x1 = max(0, xs.min() - 8), min(m.shape[1], xs.max() + 9)
    S = shape[y0:y1, x0:x1]
    C = count[y0:y1, x0:x1]
    M = m[y0:y1, x0:x1]

    from skimage.morphology import skeletonize
    sk = skeletonize(M).astype(np.float32)
    # endpoints and junctions of the centre line
    nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.float32)) - sk
    ends = (sk > 0) & (nb <= 1.01)
    junc = (sk > 0) & (nb >= 3)
    # orientation coherence of the stroke: 1 on straight stretches, low at turns
    gx = cv2.Sobel(S, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(S, cv2.CV_32F, 0, 1, ksize=3)
    sig = 0.9 * px_per_mm
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sig)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sig)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sig)
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    coh = np.clip(coh, 0, 1)

    blob_r = 0.35 * px_per_mm
    blobs = cv2.GaussianBlur(ends.astype(np.float32) + 0.6 * junc.astype(np.float32), (0, 0), blob_r)
    blobs = blobs / max(1e-6, float(np.percentile(blobs[M], 99.5)) if M.sum() > 20 else 1.0)
    blobs = np.clip(blobs, 0, 1.5)

    slow = 1.0 - coh                                          # 0 fast .. 1 slow
    along = 0.12 * _noise(S.shape, rng, 6 * px_per_mm)       # slow pressure change along lines
    c = pen.density * (1.0 - pen.speed * (coh - 0.5) + pen.blob * blobs + along)

    if pen.skips > 0:
        # faint gaps: elongated low-ink patches, only on fast stretches
        # anisotropic noise: short streaks rather than round holes
        n = _noise((S.shape[0] // 3 + 1, S.shape[1]), rng, 0.12 * px_per_mm)
        n = cv2.resize(n, (S.shape[1], S.shape[0]), interpolation=cv2.INTER_LINEAR)
        gap = (n > 2.2 - 0.8 * pen.skips) & (coh > 0.8)
        gap = cv2.GaussianBlur(gap.astype(np.float32), (0, 0), 0.8)
        c = c * (1 - 0.4 * gap)

    # ink deposits inside the stroke shape with a rough edge
    edge_n = _noise(S.shape, rng, 0.06 * px_per_mm)
    edge = np.clip(S + pen.rough * 0.25 * edge_n * (S > 0.05) * (S < 0.95), 0, 1)
    if pen.kind == "fountain":
        # wet ink dries darker at the edges of the line and pools at the ends
        inner = cv2.erode(M.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)
        rim = (M.astype(np.float32) - inner) * 0.25
        c = c + rim + 0.1 * slow
    if pen.kind == "pencil" and pen.grain > 0:
        tooth = _noise(S.shape, rng, 0.05 * px_per_mm, octaves=2)
        keep = np.clip(0.55 + 0.35 * tooth, 0, 1) ** (1 + pen.grain)
        c = c * (1 - pen.grain + pen.grain * keep * 1.6)
    c = np.clip(c, 0.15, 2.2) * edge
    # overlapping strokes add ink
    c = c * np.clip(1 + 0.75 * (C - 1), 1, 2.2)
    if pen.bleed > 0:
        c = np.maximum(c, cv2.GaussianBlur(c, (0, 0), pen.bleed) * 0.9)
    out = np.zeros_like(shape)
    out[y0:y1, x0:x1] = c
    return out


def to_rgb(conc: np.ndarray, pen: Pen, paper: np.ndarray) -> np.ndarray:
    """Beer-Lambert: transmittance = T1 ** concentration, T1 = colour of one stroke."""
    from .render import hex_rgb
    t1 = np.clip(hex_rgb(pen.ink) / 255.0, 0.02, 0.98)
    a = np.clip(conc, 0, 3)[..., None]
    trans = np.power(t1[None, None, :], a)
    return np.clip(paper.astype(np.float32) * trans, 0, 255).astype(np.uint8)

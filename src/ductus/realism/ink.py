"""Physically motivated pen-ink rendering from stroke polylines.

Input
-----
``strokes``: list of (N, 3) float arrays ``x, y, pressure`` in page pixels at ``dpi``.
Points are assumed to be sampled at a constant time step (tablet / online handwriting /
RNN output), so the spacing between points is the pen speed. If the spacing is uniform
(a resampled path with no timing), speed is inferred from curvature with the two-thirds
power law (v ~ k^-1/3, Lacquaniti, Terzuolo & Viviani 1983). Pressure is 0..1; pass a
constant 0.6 if you have none.

Output
------
``render_strokes`` returns an RGBA uint8 layer. Over white paper, alpha compositing the
layer gives exactly the Beer-Lambert colour; over any other paper use ``composite`` which
multiplies the transmittance (ink is a subtractive filter, not paint).

Model (per stroke, vectorised in stroke coordinates s = arc length, d = signed offset)
------------------------------------------------------------------------------------
* line width from the tip size (0.5 / 0.7 / 1.0 mm tips leave ~0.30 / 0.38 / 0.50 mm lines
  for ballpoints) and pressure;
* ink deposit grows with pen-tip force and falls with speed (Franke & Rose 2004 ink
  deposition model; Franke 2005);
* downstrokes carry more force than upstrokes;
* ballpoint striations: thin light tracks parallel to the stroke from the ball;
* skips (ink gaps) on fast stretches, mostly in the middle tracks;
* ink depots (blobs) where the ball rested: stroke starts after a pause, sharp turns, ends;
* flying starts and ends: fast strokes taper in and out;
* along-stroke opacity variation (slow 1-D noise), plus per-stroke ink-flow variation;
* rough edges from paper fibres and anisotropic fibre bleed;
* overlap darkening: concentrations of separate strokes add before Beer-Lambert;
* pen kinds: ballpoint, gel (even, saturated, coffee-ring edge), fountain (nib-angle width,
  pooling, feathering), pencil (graphite on the paper tooth, sheen-free grey).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import cv2
import numpy as np

# line width on paper (mm) for a ballpoint of a given nominal tip size (mm)
BALLPOINT_LINE_MM = {0.5: 0.30, 0.7: 0.38, 1.0: 0.50}


@dataclass(frozen=True)
class Pen:
    name: str
    kind: str                      # ballpoint | gel | fountain | pencil
    rgb: tuple                     # colour of ONE average stroke on white paper (sRGB 0-255)
    width_scale: float = 1.0       # multiplier on the ballpoint line width for this tip
    density: float = 1.0           # mean ink concentration (1 = the colour above)
    pressure_gain: float = 0.55    # how much pressure changes the deposit
    speed_fade: float = 0.35       # how much fast stretches lighten
    down_gain: float = 0.12        # extra ink on downstrokes
    flow_var: float = 0.10         # slow along-stroke opacity variation (std)
    stroke_var: float = 0.07       # per-stroke ink-flow variation (std)
    striation: float = 0.22        # depth of ball-track light lines (0..1)
    skips: float = 0.25            # skip rate on fast stretches (0..1)
    blob: float = 0.55             # extra ink at depots
    blob_rate: float = 0.35        # share of strokes that start with a depot
    taper: float = 0.6             # flying start/end strength on fast strokes
    rough: float = 0.05            # edge roughness as share of line width
    bleed_px: float = 0.5          # wicking blur at 600 dpi
    edge_dark: float = 0.0         # gel/fountain: darker rim where the ink dried
    grain: float = 0.0             # pencil: paper-tooth texture
    nib_ratio: float = 1.0         # fountain: thin/thick
    nib_angle: float = 40.0        # fountain: nib angle in degrees


PENS = {
    # colour measured as a single-pass blue ballpoint line on white; calibrate.py replaces it
    "ballpoint-blue": Pen("ballpoint-blue", "ballpoint", (52, 66, 150)),
    "ballpoint-black": Pen("ballpoint-black", "ballpoint", (48, 48, 56), density=1.05),
    "gel-blue": Pen("gel-blue", "gel", (30, 62, 196), width_scale=1.25, density=1.3, speed_fade=0.06,
                    down_gain=0.04, flow_var=0.04, striation=0.0, skips=0.02, blob=0.15,
                    blob_rate=0.1, taper=0.25, rough=0.06, bleed_px=1.0, edge_dark=0.25),
    "gel-black": Pen("gel-black", "gel", (22, 22, 28), width_scale=1.25, density=1.35, speed_fade=0.06,
                     down_gain=0.04, flow_var=0.04, striation=0.0, skips=0.02, blob=0.15,
                     blob_rate=0.1, taper=0.25, rough=0.06, bleed_px=1.0, edge_dark=0.25),
    "fountain": Pen("fountain", "fountain", (36, 50, 118), width_scale=1.3, density=1.1,
                    speed_fade=0.18, down_gain=0.05, flow_var=0.08, striation=0.0, skips=0.0,
                    blob=0.8, blob_rate=0.6, taper=0.2, rough=0.08, bleed_px=1.4, edge_dark=0.35,
                    nib_ratio=0.35, nib_angle=40.0),
    "pencil": Pen("pencil", "pencil", (88, 88, 94), width_scale=1.2, density=0.8,
                  pressure_gain=0.9, speed_fade=0.25, down_gain=0.1, flow_var=0.06,
                  striation=0.0, skips=0.0, blob=0.05, blob_rate=0.0, taper=0.4, rough=0.15,
                  bleed_px=0.2, grain=0.6),
}
DEFAULT_PEN = "ballpoint-blue"


def get_pen(name: str | Pen = DEFAULT_PEN, rgb=None, **overrides) -> Pen:
    p = name if isinstance(name, Pen) else PENS[name]
    if rgb is not None:
        overrides["rgb"] = tuple(int(v) for v in rgb)
    return replace(p, **overrides) if overrides else p


def line_width_px(tip_mm: float, dpi: float, pen: Pen | str = DEFAULT_PEN) -> float:
    """Typical line width in px that a pen with this nominal tip leaves on paper."""
    pen = get_pen(pen)
    base = BALLPOINT_LINE_MM.get(round(float(tip_mm), 2), 0.42 * float(tip_mm) + 0.09)
    return base * pen.width_scale * dpi / 25.4


# ----------------------------------------------------------------------------- noise
def _noise1d(n: int, rng, corr: float) -> np.ndarray:
    """Smooth 1-D noise with unit std and correlation length ``corr`` samples."""
    corr = max(1.0, corr)
    m = int(n / corr) + 5
    g = rng.standard_normal(m)
    x = np.linspace(0, m - 4, n)
    x = np.minimum(x, m - 4 - 1e-6)
    i = np.floor(x).astype(int); f = x - i
    # cubic (Catmull-Rom) interpolation
    p0, p1, p2, p3 = g[i], g[i + 1], g[i + 2], g[i + 3]
    v = 0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f ** 2
               + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)
    s = v.std()
    return v / s if s > 1e-6 else v


def _noise2d(shape, rng, cell: float) -> np.ndarray:
    h, w = shape
    cell = max(1.0, cell)
    g = rng.standard_normal((int(h / cell) + 3, int(w / cell) + 3)).astype(np.float32)
    n = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    s = float(n.std())
    return n / s if s > 1e-6 else n


def fibre_field(shape, rng, scale_px: float) -> np.ndarray:
    """Paper fibre texture: short oriented streaks, values ~N(0,1). Used for bleed and edges."""
    h, w = shape
    out = np.zeros((h, w), np.float32)
    for ang in rng.uniform(0, 180, 3):
        n = rng.standard_normal((h, w)).astype(np.float32)
        k = max(3, int(scale_px * 4) | 1)
        ker = np.zeros((k, k), np.float32)
        c = k // 2
        dx, dy = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        for t in np.linspace(-c, c, 2 * k):
            ker[int(round(c + t * dy)) % k, int(round(c + t * dx)) % k] = 1
        ker /= ker.sum()
        out += cv2.filter2D(n, -1, ker)
    out = cv2.GaussianBlur(out, (0, 0), max(0.5, scale_px * 0.3))
    return out / max(1e-6, float(out.std()))


# ----------------------------------------------------------------------------- geometry
def _resample(pts: np.ndarray, step: float):
    """Resample a polyline at constant arc-length step. Returns points, s, and the source
    index (fractional) of each sample so per-point attributes can be interpolated."""
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(pts[:, :2], axis=0).T))]
    L = d[-1]
    n = max(2, int(np.ceil(L / step)) + 1)
    s = np.linspace(0, L, n)
    idx = np.interp(s, d, np.arange(len(d))) if L > 0 else np.zeros(n)
    xy = np.c_[np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])] if L > 0 else np.repeat(pts[:1, :2], n, 0)
    return xy, s, idx


def stroke_speed(pts: np.ndarray) -> np.ndarray:
    """Relative speed per input point (median of the stroke = 1)."""
    if len(pts) < 3:
        return np.ones(len(pts))
    seg = np.hypot(*np.diff(pts[:, :2], axis=0).T)
    v = np.r_[seg[0], (seg[:-1] + seg[1:]) / 2, seg[-1]]
    med = np.median(v)
    if med <= 0:
        return np.ones(len(pts))
    if v.std() / max(1e-9, v.mean()) < 0.05:  # uniform spacing: no timing, use the 2/3 power law
        k = _curvature(pts[:, :2])
        v = (k + 1e-3) ** (-1 / 3)
        med = np.median(v)
    v = v / med
    return np.clip(cv2.GaussianBlur(v.reshape(-1, 1).astype(np.float32), (1, 5), 1.0).ravel(), 0.05, 6)


def _curvature(xy):
    if len(xy) < 3:
        return np.zeros(len(xy))
    d1 = np.gradient(xy, axis=0); d2 = np.gradient(d1, axis=0)
    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    den = (d1 ** 2).sum(1) ** 1.5 + 1e-9
    return num / den


# ----------------------------------------------------------------------------- core
def stroke_concentration(stroke: np.ndarray, pen: Pen, width_px: float, rng, ss: int = 1,
                         speed: np.ndarray | None = None):
    """Ink concentration of one stroke in a local crop. Returns (conc, x0, y0) where conc is
    float32 at the working resolution and (x0, y0) its top-left offset in working px."""
    pts = np.asarray(stroke, np.float64).copy()
    if pts.ndim != 2 or len(pts) == 0:
        return None
    if pts.shape[1] == 2:
        pts = np.c_[pts, np.full(len(pts), 0.6)]
    spd = stroke_speed(pts) if speed is None else np.asarray(speed, float)
    pts[:, :2] *= ss
    w = width_px * ss
    if len(pts) == 1 or np.hypot(*np.ptp(pts[:, :2], axis=0)) < 1.5 * w:  # a dot or a tiny flick
        c = pts[:, :2].mean(0)
        p0 = float(np.clip(pts[:, 2].max() + 0.15, 0, 1))
        pts = np.array([[c[0] - 0.2 * w, c[1] + 0.05 * w, p0], [c[0] + 0.2 * w, c[1] - 0.05 * w, p0]])
        spd = np.array([0.3, 0.3])
    xy, s, idx = _resample(pts, step=max(0.35, 0.25 * w / 2))
    L = s[-1]
    ds = max(0.35, s[1] - s[0]) if len(s) > 1 else 1.0
    pr = np.interp(idx, np.arange(len(pts)), pts[:, 2])
    v = np.interp(idx, np.arange(len(spd)), spd)
    tang = np.gradient(xy, axis=0)
    tn = np.hypot(tang[:, 0], tang[:, 1]) + 1e-9
    tang = tang / tn[:, None]
    nrm = np.c_[-tang[:, 1], tang[:, 0]]

    # half-width along the stroke
    hw = 0.5 * w * (0.78 + 0.4 * np.clip(pr, 0, 1))
    if pen.kind == "fountain":
        a = np.radians(pen.nib_angle)
        cross = np.abs(tang[:, 0] * np.sin(a) - tang[:, 1] * np.cos(a))  # stroke vs nib edge
        hw = 0.5 * w * (pen.nib_ratio + (1 - pen.nib_ratio) * cross) * (0.85 + 0.3 * pr)
    # flying start/end: fast strokes taper, slow strokes are blunt
    tl = max(1.0, 1.2 * w)
    short = float(np.clip(L / (3 * tl), 0, 1))        # dots and tiny ticks do not taper
    fast_start = np.clip(v[0] - 0.8, 0, 1.5) / 1.5
    fast_end = np.clip(v[-1] - 0.8, 0, 1.5) / 1.5
    st = 1 - short * pen.taper * fast_start * np.clip(1 - s / tl, 0, 1)
    en = 1 - short * pen.taper * (0.4 + 0.6 * fast_end) * np.clip(1 - (L - s) / tl, 0, 1)
    hw = hw * np.clip(np.minimum(st, en), 0.35, 1)

    # deposit along the stroke
    down = np.clip(tang[:, 1], 0, 1)  # image y grows downward
    flow = 1 + pen.flow_var * _noise1d(len(s), rng, corr=max(2, 6 * w / ds))
    dep = pen.density * (1 + pen.stroke_var * rng.standard_normal())
    dep = dep * (1 + pen.pressure_gain * (pr - 0.6)) * (1 + pen.down_gain * (down - 0.3))
    dep = dep * (1 - pen.speed_fade * np.clip(np.log(np.maximum(v, 1e-3)), -1.0, 1.2) * 0.8) * flow
    dep = dep * np.minimum(st, en) ** 0.7
    # ink depots: start after a pause, sharp turns, end
    blob = np.zeros_like(s)
    r = max(1.0, 0.9 * w)
    if rng.random() < pen.blob_rate:
        blob += np.exp(-0.5 * (s / r) ** 2) * rng.uniform(0.6, 1.2)
    if rng.random() < pen.blob_rate * 0.6:
        blob += np.exp(-0.5 * ((L - s) / r) ** 2) * rng.uniform(0.4, 1.0)
    turn = np.abs(np.r_[0, np.diff(np.unwrap(np.arctan2(tang[:, 1], tang[:, 0])))])
    turn = cv2.GaussianBlur(turn.reshape(-1, 1).astype(np.float32), (1, 0), max(1.0, r / ds)).ravel()
    blob += np.clip(turn * 4 - 0.3, 0, 1) * (v < 0.9) * 0.5
    dep = dep + pen.blob * pen.density * blob
    hw = hw * (1 + 0.18 * np.clip(blob, 0, 1.2))

    # rasterise in stroke coordinates via nearest centre-line sample
    m = int(np.ceil(hw.max() + 3 + 2 * pen.bleed_px * ss))
    x0, y0 = int(np.floor(xy[:, 0].min())) - m, int(np.floor(xy[:, 1].min())) - m
    x1, y1 = int(np.ceil(xy[:, 0].max())) + m, int(np.ceil(xy[:, 1].max())) + m
    H, W = y1 - y0 + 1, x1 - x0 + 1
    seed = np.ones((H, W), np.uint8)
    pix = np.round(xy - [x0, y0]).astype(int)
    seed[pix[:, 1], pix[:, 0]] = 0
    dist, lab = cv2.distanceTransformWithLabels(seed, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lab_to_i = np.zeros(lab.max() + 1, np.int64)
    lab_to_i[lab[pix[:, 1], pix[:, 0]]] = np.arange(len(pix))
    near = lab_to_i[lab]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = xx + x0 - xy[near, 0]; dy = yy + y0 - xy[near, 1]
    d = dx * nrm[near, 0] + dy * nrm[near, 1]          # signed offset across the stroke
    along = dx * tang[near, 0] + dy * tang[near, 1]
    radial = np.hypot(dx, dy)                           # distance to the centre line
    hwp = hw[near]
    # rough edge: independent 1-D noise on each side of the stroke
    rough_l = _noise1d(len(s), rng, corr=max(1.5, 1.2 * w / ds))
    rough_r = _noise1d(len(s), rng, corr=max(1.5, 1.2 * w / ds))
    edge = hwp + pen.rough * w * np.where(d >= 0, rough_l[near], rough_r[near])
    cov = np.clip(edge - radial + 0.5, 0, 1).astype(np.float32)   # anti-aliased coverage
    if not cov.any():
        return None
    conc = cov * dep[near].astype(np.float32)

    u = np.clip(d / np.maximum(hwp, 1e-3), -1.2, 1.2)                # -1..1 across
    if pen.striation > 0:
        ntr = 3 + int(rng.integers(0, 3))
        phase = rng.uniform(0, np.pi)
        track = 0.5 + 0.5 * np.cos(np.pi * ntr * u + phase)            # light lines across
        tvar = 0.5 + 0.5 * _noise1d(len(s), rng, corr=8)[near]
        conc *= 1 - pen.striation * track * np.clip(tvar, 0, 1).astype(np.float32)
    if pen.skips > 0:
        fast = np.clip((v[near] - 1.1) / 1.2, 0, 1)
        g = _noise1d(len(s), rng, corr=max(2.0, 1.5 * w / ds))[near]
        hole = np.clip((g - (2.2 - 1.6 * pen.skips)) * 1.5, 0, 1) * fast
        centre = np.exp(-0.5 * (u / 0.55) ** 2)
        conc *= (1 - 0.85 * hole * (0.35 + 0.65 * centre)).astype(np.float32)
    if pen.edge_dark > 0:
        conc *= (1 + pen.edge_dark * np.clip(np.abs(u), 0, 1) ** 4).astype(np.float32)
    return conc, x0, y0


def render_strokes(strokes: Sequence[np.ndarray], size: tuple, dpi: float = 300, pen: str | Pen = DEFAULT_PEN,
                   tip_mm: float = 0.5, seed: int = 0, ss: int = 2, rgb=None,
                   return_concentration: bool = False):
    """Render strokes to an RGBA layer of ``size=(height, width)`` px at ``dpi``.

    ``ss`` is the supersampling factor (2 is enough at 300 dpi). With
    ``return_concentration=True`` the float32 concentration map is returned instead
    (for callers that composite themselves, e.g. with paper texture)."""
    pen = get_pen(pen, rgb=rgb)
    rng = np.random.default_rng(seed)
    H, W = size
    w = line_width_px(tip_mm, dpi, pen)
    C = np.zeros((H * ss, W * ss), np.float32)
    for st in strokes:
        st = np.asarray(st, np.float64)
        if len(st) == 0:
            continue
        r = stroke_concentration(st, pen, w, rng, ss=ss)
        if r is None:
            continue
        c, x0, y0 = r
        _paste_add(C, c, x0, y0)
    C = _finish(C, pen, dpi * ss / 600.0, rng)
    if ss > 1:
        C = cv2.resize(C, (W, H), interpolation=cv2.INTER_AREA)
    if return_concentration:
        return C
    return concentration_to_rgba(C, pen)


def _paste_add(C, c, x0, y0):
    H, W = C.shape
    h, w = c.shape
    ax0, ay0 = max(0, x0), max(0, y0)
    ax1, ay1 = min(W, x0 + w), min(H, y0 + h)
    if ax1 <= ax0 or ay1 <= ay0:
        return
    C[ay0:ay1, ax0:ax1] += c[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]


def _finish(C, pen: Pen, px_per_600, rng):
    """Page-level effects: overlap saturation, fibre bleed, pencil grain."""
    if not C.any():
        return C
    ys, xs = np.nonzero(C > 0)
    y0, y1 = max(0, ys.min() - 8), min(C.shape[0], ys.max() + 9)
    x0, x1 = max(0, xs.min() - 8), min(C.shape[1], xs.max() + 9)
    c = C[y0:y1, x0:x1]
    # overlapping strokes add ink but paste saturates (paper absorbs less on wet ink)
    c = np.where(c > 1.4, 1.4 + 0.6 * np.tanh((c - 1.4) / 0.6), c)
    if pen.bleed_px > 0:
        b = pen.bleed_px * px_per_600 * 2
        fib = fibre_field(c.shape, rng, max(1.0, 1.5 * px_per_600 * 2))
        spread = cv2.GaussianBlur(c, (0, 0), b)
        wick = np.clip(0.55 + 0.25 * fib, 0, 1) * spread
        c = np.maximum(c, wick * 0.8)
    if pen.grain > 0:
        tooth = _noise2d(c.shape, rng, 1.2 * px_per_600 * 2) * 0.6 + _noise2d(c.shape, rng, 4 * px_per_600 * 2) * 0.4
        keep = 1 / (1 + np.exp(-(tooth * 2.2 + 0.8)))
        c = c * (1 - pen.grain + pen.grain * keep * 1.25)
    C[y0:y1, x0:x1] = c
    return C


def transmittance(C: np.ndarray, pen: Pen | str) -> np.ndarray:
    """Per-channel transmittance (H, W, 3) in 0..1 from a concentration map (Beer-Lambert).
    The single-stroke colour is defined in sRGB; the law is applied in linear light."""
    pen = get_pen(pen)
    t1 = srgb_to_linear(np.clip(np.asarray(pen.rgb, np.float32) / 255.0, 0.01, 0.99))
    a = np.clip(C, 0, 4)[..., None]
    return np.power(t1[None, None, :], a)


def concentration_to_rgba(C: np.ndarray, pen: Pen | str) -> np.ndarray:
    """RGBA layer that composites over white to the Beer-Lambert colour."""
    T = linear_to_srgb(transmittance(C, pen))          # what the ink looks like on white, sRGB
    a = 1 - T.min(axis=2)
    col = np.where(a[..., None] > 1e-4, (T - (1 - a[..., None])) / np.maximum(a[..., None], 1e-4), 0)
    out = np.dstack([np.clip(col, 0, 1), a])
    return (out * 255 + 0.5).astype(np.uint8)


def composite(paper_rgb: np.ndarray, layer_rgba: np.ndarray) -> np.ndarray:
    """Multiply an ink layer onto paper (works for any paper colour, grid lines included)."""
    a = layer_rgba[..., 3:4].astype(np.float32) / 255
    col = layer_rgba[..., :3].astype(np.float32) / 255
    T_srgb = 1 - a * (1 - col)                           # colour on white
    T = srgb_to_linear(T_srgb)
    P = srgb_to_linear(paper_rgb.astype(np.float32) / 255)
    return (linear_to_srgb(P * T) * 255 + 0.5).clip(0, 255).astype(np.uint8)


def srgb_to_linear(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x):
    x = np.clip(np.asarray(x, np.float32), 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


# ----------------------------------------------------------------------------- raster input
def ink_from_mask(coverage: np.ndarray, dpi: float = 300, pen: str | Pen = DEFAULT_PEN, seed: int = 0,
                  rgb=None) -> np.ndarray:
    """Re-ink a raster word image (e.g. diffusion output): ``coverage`` is 0..1 (1 = ink).
    Stroke direction comes from the structure tensor; striations, skips and flow variation
    are drawn along it, blobs at skeleton end points. Returns RGBA."""
    pen = get_pen(pen, rgb=rgb)
    rng = np.random.default_rng(seed)
    cov = np.clip(coverage.astype(np.float32), 0, 1)
    if not cov.any():
        return concentration_to_rgba(np.zeros_like(cov), pen)
    px600 = dpi / 600.0
    gx = cv2.Sobel(cov, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(cov, cv2.CV_32F, 0, 1, ksize=3)
    sig = max(1.0, 3 * px600)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sig); jyy = cv2.GaussianBlur(gy * gy, (0, 0), sig)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sig)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2       # stroke direction
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    # oriented noise: blur white noise along 8 directions, pick by theta
    n = rng.standard_normal(cov.shape).astype(np.float32)
    L = max(5, int(14 * px600) | 1)
    bank = []
    angs = np.linspace(0, np.pi, 8, endpoint=False)
    for a in angs:
        k = np.zeros((L, L), np.float32); c = L // 2
        for t in np.linspace(-c, c, 3 * L):
            k[int(round(c + t * np.sin(a))), int(round(c + t * np.cos(a)))] = 1
        bank.append(cv2.filter2D(n, -1, k / k.sum()))
    bank = np.stack(bank)
    sel = np.round(((theta % np.pi) / np.pi) * 8).astype(int) % 8
    streak = np.take_along_axis(bank, sel[None], 0)[0]
    streak /= max(1e-6, float(streak.std()))
    flow = _noise2d(cov.shape, rng, 40 * px600)
    from . import knobs
    conc = pen.density * (1 + pen.flow_var * 1.5 * flow) * (1 - pen.speed_fade * 0.5 * (coh - 0.5))
    press = float(knobs.get("press", 0.0))
    if press:
        # pressure: downstrokes are pressed harder than upstrokes and joins, and pressure
        # drifts slowly along the line
        vert = np.abs(np.sin(theta)) * np.clip(coh * 1.5, 0, 1)
        slow = _noise2d(cov.shape, rng, 90 * px600)
        conc *= 1 + press * (0.6 * (vert - 0.5) + 0.35 * slow)
    conc *= 1 - pen.striation * 0.8 * np.clip(streak * 0.5 + 0.2, 0, 1) * (pen.striation > 0)
    skips = pen.skips * float(knobs.get("skips_x", 1.0))
    if skips > 0:
        gap = np.clip(streak - (2.6 - 1.2 * skips), 0, 1) * (coh > 0.75)
        conc *= 1 - 0.7 * cv2.GaussianBlur(gap, (0, 0), 0.7)
    # blobs at end points of a crude skeleton
    bw = (cov > 0.5).astype(np.uint8)
    sk = _skeleton(bw)
    nb = cv2.filter2D(sk.astype(np.float32), -1, np.ones((3, 3), np.float32)) * sk
    ends = (nb == 2).astype(np.float32)
    taper = float(knobs.get("taper", 0.0))
    if taper:
        # flying starts and ends: ink thins out over the last stretch of a free stroke end
        tl = max(2.0, 6 * px600)
        near = cv2.GaussianBlur(ends, (0, 0), tl)
        near /= max(1e-6, float(near.max()))
        conc *= 1 - taper * 0.6 * np.clip(near * 2.5, 0, 1)
    ends *= rng.random(ends.shape) < pen.blob_rate * 1.5 * float(knobs.get("blob_rate_x", 1.0))
    bl = cv2.GaussianBlur(ends, (0, 0), max(1.0, 2.5 * px600))
    bl /= max(1e-6, float(bl.max()))
    conc = conc + pen.blob * float(knobs.get("blob_x", 1.0)) * bl
    conc = np.clip(conc, 0.1, 3) * cov
    conc = _finish(conc, pen, px600, rng)
    return concentration_to_rgba(conc, pen)


def _skeleton(bw: np.ndarray) -> np.ndarray:
    sk = np.zeros_like(bw); el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)); img = bw.copy()
    for _ in range(64):
        er = cv2.erode(img, el)
        sk |= img - cv2.dilate(er, el)
        img = er
        if not img.any():
            break
    return sk


def flat_layer(strokes, size, dpi=300, tip_mm=0.5, rgb=(52, 66, 150)) -> np.ndarray:
    """Baseline for comparisons: constant-width, constant-colour anti-aliased polylines
    (what most text-to-handwriting tools draw)."""
    H, W = size
    ss = 4
    m = np.zeros((H * ss, W * ss), np.uint8)
    w = max(1, int(round(line_width_px(tip_mm, dpi) * ss)))
    for st in strokes:
        p = np.round(np.asarray(st)[:, :2] * ss).astype(np.int32)
        if len(p) == 1:
            cv2.circle(m, tuple(p[0]), w // 2, 255, -1, cv2.LINE_AA)
        else:
            cv2.polylines(m, [p], False, 255, w, cv2.LINE_AA)
    a = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA)
    out = np.zeros((H, W, 4), np.uint8)
    out[..., :3] = rgb
    out[..., 3] = a
    return out


def strokes_from_offsets(arr: np.ndarray, scale: float = 1.0, origin=(0.0, 0.0), pressure=None,
                         seed: int = 0) -> list:
    """Convert (N,3) absolute x, y, end_of_stroke arrays (Graves/IAM-OnDB style) to the
    stroke list format. A plausible pressure profile is synthesised if none is given:
    higher on slow stretches and downstrokes, lower at stroke ends (Franke 2005)."""
    rng = np.random.default_rng(seed)
    out, b = [], 0
    for i in range(len(arr)):
        if arr[i, 2] >= 0.5 or i == len(arr) - 1:
            seg = arr[b:i + 1, :2].astype(np.float64) * scale + origin
            b = i + 1
            if len(seg) == 0:
                continue
            if pressure is None:
                v = stroke_speed(np.c_[seg, np.zeros(len(seg))])
                tang = np.gradient(seg, axis=0)[:, 1] if len(seg) > 1 else np.zeros(1)
                p = 0.62 - 0.12 * np.clip(np.log(v), -1, 1) + 0.08 * np.sign(tang)
                n = len(seg)
                ramp = np.minimum(1, np.minimum(np.arange(n) + 1, n - np.arange(n)) / 3.0)
                p = p * (0.75 + 0.25 * ramp) + 0.05 * rng.standard_normal()
                p = np.clip(p, 0.15, 1.0)
            else:
                p = np.full(len(seg), float(pressure))
            out.append(np.c_[seg, p])
    return out

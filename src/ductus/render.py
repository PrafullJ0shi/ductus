"""Raster, PDF and SVG output for laid-out pages.

Ink model: every word crop and pen stroke adds coverage to one ink layer; overlapping
ink gets a little darker, density follows a smooth pressure field, and the ink colour
is multiplied onto the paper the way a ballpoint sits on paper.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .layout import A4, Page

SS = 2  # supersampling factor for anti-aliasing


def hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def sharpen_ink(a: np.ndarray, mess: float) -> np.ndarray:
    """Upscaled diffusion ink is soft; restore a ballpoint edge but keep some tone."""
    lo, hi = 0.18, 0.62
    t = np.clip((a - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def word_layer(alpha: np.ndarray, scale: float, angle: float, mess: float, pen: float, baseline: float = 0.0):
    """Scale + rotate one word crop. Returns (layer, anchor) where anchor maps the
    model-space origin (0,0) of the crop into layer coordinates."""
    s = scale * SS
    h, w = alpha.shape
    big = cv2.resize(alpha.astype(np.float32), (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                     interpolation=cv2.INTER_CUBIC)
    big = np.clip(big, 0, 1)
    if pen != 1.0:
        k = int(round(abs(pen - 1.0) * s * 1.5))
        if k >= 1:
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            big = cv2.dilate(big, ker) if pen > 1 else cv2.erode(big, ker)
    big = sharpen_ink(big, mess)
    pad = int(0.25 * max(big.shape))
    big = np.pad(big, pad)
    H, W = big.shape
    M = cv2.getRotationMatrix2D((pad, pad + baseline * s), angle, 1.0)  # rotate about the baseline start
    rot = cv2.warpAffine(big, M, (W, H), flags=cv2.INTER_LINEAR)
    return rot, (pad, pad)


def pressure_field(shape, rng, amount, cell=180):
    h, w = shape
    gh, gw = max(2, h // cell), max(2, w // cell)
    g = rng.normal(0, 1, (gh, gw)).astype(np.float32)
    f = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    f = cv2.GaussianBlur(f, (0, 0), cell / 3)
    f /= max(1e-6, np.abs(f).max())
    return 1.0 - amount * (0.5 + 0.5 * f)


def draw_stroke(layer: np.ndarray, pts: np.ndarray, width: float, rng):
    """Ballpoint-like stroke: slight taper at the ends and gentle width change."""
    p = (np.asarray(pts) * SS)
    if len(p) < 2:
        return
    n = len(p)
    t = np.linspace(0, 1, n)
    taper = np.clip(np.minimum(t, 1 - t) * 8, 0.55, 1.0)
    wv = width * SS * taper * (1 + 0.12 * np.sin(2 * math.pi * (t * rng.uniform(0.5, 1.5) + rng.random())))
    tmp = np.zeros_like(layer, dtype=np.uint8)
    # draw in short chunks so the width can change along the stroke
    step = max(2, n // 12)
    for i in range(0, n - 1, step):
        seg = np.round(p[i:min(n, i + step + 1)] * 4).astype(np.int32)
        th = max(1, int(round(wv[i])))
        cv2.polylines(tmp, [seg.reshape(-1, 1, 2)], False, 255, th, cv2.LINE_AA, shift=2)
    a = tmp.astype(np.float32) / 255.0
    a *= rng.uniform(0.82, 0.97)
    np.maximum(layer, a, out=layer)


def ink_coverage(page: Page, words, mess: float, pen: float, pen_px: float, rng, size=A4):
    W, H = size[0] * SS, size[1] * SS
    ink = np.zeros((H, W), np.float32)
    for pl in page.words:
        wi = words[pl.idx]
        lay, (ax, ay) = word_layer(wi.alpha, pl.scale, pl.angle, mess, pen, wi.baseline)
        # crop origin (top-left) sits at baseline - wi.baseline*scale
        ox = int(round(pl.x * SS - ax))
        oy = int(round((pl.base - wi.baseline * pl.scale) * SS - ay))
        y0, x0 = max(0, oy), max(0, ox)
        y1, x1 = min(H, oy + lay.shape[0]), min(W, ox + lay.shape[1])
        if y1 <= y0 or x1 <= x0:
            continue
        sub = lay[y0 - oy:y1 - oy, x0 - ox:x1 - ox] * np.clip(pl.ink, 0.75, 1.1)
        region = ink[y0:y1, x0:x1]
        ink[y0:y1, x0:x1] = 1 - (1 - region) * (1 - np.clip(sub, 0, 1))
    strokes_layer = np.zeros_like(ink)
    for pts, wf in page.strokes:
        draw_stroke(strokes_layer, pts, pen_px * wf * pen, rng)
    ink = 1 - (1 - ink) * (1 - strokes_layer)
    return ink


def compose(ink: np.ndarray, ink_hex: str, pressure: float, rng, paper: np.ndarray | None = None) -> np.ndarray:
    """Coverage -> RGB page at final resolution."""
    field = pressure_field(ink.shape, rng, pressure)
    a = np.clip(ink * field, 0, 1)
    a = cv2.GaussianBlur(a, (0, 0), 0.6)  # tiny bleed into the paper fibres
    a = cv2.resize(a, (ink.shape[1] // SS, ink.shape[0] // SS), interpolation=cv2.INTER_AREA)
    if paper is None:
        paper = np.full(a.shape + (3,), 255, np.float32)
    col = hex_rgb(ink_hex)
    # multiply blend: ink absorbs light, darker where coverage stacks
    tint = 1 - a[..., None] * (1 - col / 255.0)
    out = paper.astype(np.float32) * tint
    return np.clip(out, 0, 255).astype(np.uint8)


def save_pdf(pngs: list[str], path: str):
    from reportlab.lib.pagesizes import A4 as RA4
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(path, pagesize=RA4)
    for p in pngs:
        c.drawImage(p, 0, 0, width=RA4[0], height=RA4[1])
        c.showPage()
    c.save()


# -- SVG -----------------------------------------------------------------------------
def _contours_path(alpha: np.ndarray, up: int = 4) -> str:
    big = cv2.resize(alpha.astype(np.float32), (alpha.shape[1] * up, alpha.shape[0] * up), interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (0, 0), up * 0.5)
    m = (big > 0.42).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    parts = []
    for c in cs:
        c = c.reshape(-1, 2) / up
        if len(c) < 3:
            continue
        parts.append("M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in c) + "Z")
    return "".join(parts)


def save_svg(page: Page, words, path: str, ink_hex: str, pen_px: float, animate: bool = False,
             size=A4, seconds_per_px: float = 0.0016, title: str = "Handwritten page"):
    """Vector page. With animate=True, words are revealed left to right and pen marks
    are drawn with stroke-dash animation, in writing order."""
    W, H = size
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="210mm" height="297mm" '
           f'role="img" aria-labelledby="page-title"><title id="page-title">{title}</title>',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>', '<defs>']
    body = []
    t = 0.3
    defs = []
    events = []  # (y, x, kind, payload) for writing order
    for k, pl in enumerate(page.words):
        events.append((pl.base, pl.x, "w", k))
    for k, (pts, wf) in enumerate(page.strokes):
        events.append((float(np.median(pts[:, 1])) + 1e-3, float(pts[0, 0]), "s", k))
    # writing order: line by line (rows of ~1 line pitch), then left to right
    events.sort(key=lambda e: (round(e[0] / 60), e[1]))
    for _, _, kind, k in events:
        if kind == "w":
            pl = page.words[k]
            wi = words[pl.idx]
            d = _contours_path(wi.alpha)
            tx, ty = pl.x, pl.base - wi.baseline * pl.scale
            tr = f'translate({tx:.1f},{ty:.1f}) rotate({-pl.angle:.2f},0,{wi.baseline * pl.scale:.1f}) scale({pl.scale:.4f})'
            if animate:
                w = wi.alpha.shape[1]
                dur = max(0.15, w * pl.scale * seconds_per_px)
                defs.append(f'<clipPath id="c{k}"><rect x="-2" y="-20" height="{wi.alpha.shape[0] + 40}" width="0">'
                            f'<animate attributeName="width" from="0" to="{w + 4}" begin="{t:.2f}s" dur="{dur:.2f}s" fill="freeze"/></rect></clipPath>')
                body.append(f'<g transform="{tr}"><path d="{d}" fill="{ink_hex}" fill-rule="evenodd" clip-path="url(#c{k})"/></g>')
                t += dur + 0.05
            else:
                body.append(f'<path transform="{tr}" d="{d}" fill="{ink_hex}" fill-rule="evenodd"/>')
        else:
            pts, wf = page.strokes[k]
            d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            sw = pen_px * wf
            if animate:
                seg = np.diff(pts, axis=0)
                L = float(np.hypot(seg[:, 0], seg[:, 1]).sum()) + 1
                dur = max(0.12, L * seconds_per_px)
                body.append(f'<path d="{d}" fill="none" stroke="{ink_hex}" stroke-width="{sw:.1f}" stroke-linecap="round" '
                            f'stroke-linejoin="round" stroke-dasharray="{L:.0f}" stroke-dashoffset="{L:.0f}">'
                            f'<animate attributeName="stroke-dashoffset" from="{L:.0f}" to="0" begin="{t:.2f}s" dur="{dur:.2f}s" fill="freeze"/></path>')
                t += dur + 0.03
            else:
                body.append(f'<path d="{d}" fill="none" stroke="{ink_hex}" stroke-width="{sw:.1f}" stroke-linecap="round" stroke-linejoin="round"/>')
    out += defs + ['</defs>'] + body + ['</svg>']
    Path(path).write_text("\n".join(out))

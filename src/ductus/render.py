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


def word_layer(alpha: np.ndarray, scale: float, angle: float, mess: float, pen: float, baseline: float = 0.0,
               stroke_px: float | None = None, nib: np.ndarray | None = None, sx: float = 1.0):
    """Scale + rotate one word crop. Returns (layer, anchor) where anchor maps the
    model-space origin (0,0) of the crop into layer coordinates."""
    s = scale * SS
    h, w = alpha.shape
    big = cv2.resize(alpha.astype(np.float32), (max(1, int(round(w * s * sx))), max(1, int(round(h * s)))),
                     interpolation=cv2.INTER_CUBIC)
    big = np.clip(big, 0, 1)
    if stroke_px:
        # re-ink along the centre line with one ballpoint width, so every word on the page
        # has the same stroke weight whatever carrier it came from
        from skimage.morphology import skeletonize
        m = big > 0.42
        if m.sum() > 8:
            sk = skeletonize(m)
            r = 0.5 * stroke_px * SS
            pad = int(r) + 2
            d = cv2.distanceTransform(np.pad((~sk).astype(np.uint8), pad, constant_values=1), cv2.DIST_L2, 5)
            if nib is not None:
                big = cv2.dilate(np.pad(sk, pad).astype(np.uint8), nib).astype(np.float32)
                big = cv2.GaussianBlur(big, (0, 0), 0.5)
            else:
                big = np.clip(r + 0.5 - d, 0, 1)
            # crop back to the original frame (the pad keeps round stroke ends)
            big = big[pad:-pad, pad:-pad]
            pad2 = int(0.25 * max(big.shape))
            big = np.pad(big, pad2)
            H, W = big.shape
            M = cv2.getRotationMatrix2D((pad2, pad2 + baseline * s), angle, 1.0)
            return cv2.warpAffine(big, M, (W, H), flags=cv2.INTER_LINEAR), (pad2, pad2)
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


def ink_coverage(page: Page, words, mess: float, pen: float, pen_px: float, rng, size=A4, uniform: bool = True,
                 nib: np.ndarray | None = None, with_count: bool = False):
    """uniform=True: words are re-inked at pen_px (one ballpoint width for the whole page)."""
    W, H = size[0] * SS, size[1] * SS
    ink = np.zeros((H, W), np.float32)
    count = np.zeros((H, W), np.float32) if with_count else None
    for pl in page.words:
        wi = words[pl.idx]
        lay, (ax, ay) = word_layer(wi.alpha, pl.scale, pl.angle, mess, pen, wi.baseline,
                                   stroke_px=pen_px * pen * float(np.clip(pl.ink, 0.9, 1.1)) if uniform else None,
                                   nib=nib, sx=getattr(pl, "sx", 1.0))
        clip = getattr(pl, "clip", None)
        if clip is not None:
            wpx = wi.alpha.shape[1] * pl.scale * getattr(pl, "sx", 1.0) * SS
            c0, c1 = int(ax + clip[0] * wpx) - 2, int(ax + clip[1] * wpx) + 2
            lay = lay.copy()
            lay[:, :max(0, c0)] = 0
            lay[:, max(0, c1):] = 0
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
        if count is not None:
            count[y0:y1, x0:x1] += (sub > 0.35)
    strokes_layer = np.zeros_like(ink)
    for pts, wf in page.strokes:
        if count is not None:
            one = np.zeros_like(ink)
            draw_stroke(one, pts, pen_px * wf * pen * 0.95, rng)
            count += one > 0.35
            np.maximum(strokes_layer, one, out=strokes_layer)
        else:
            draw_stroke(strokes_layer, pts, pen_px * wf * pen * 0.95, rng)
    ink = 1 - (1 - ink) * (1 - strokes_layer)
    if count is not None:
        return ink, np.maximum(count, (ink > 0.35).astype(np.float32))
    return ink


def smudge(c: np.ndarray, rng, n_mean: float, px_per_mm: float) -> np.ndarray:
    """Hand smudges: the side of the hand drags fresh ink a few mm, leaving a faint smear."""
    n = rng.poisson(n_mean) if n_mean > 0 else 0
    ys, xs = np.where(c > 0.6)
    for _ in range(n):
        if len(ys) == 0:
            break
        i = rng.integers(len(ys))
        cy, cx = ys[i], xs[i]
        hh, ww = int(3 * px_per_mm), int(6 * px_per_mm)
        y0, y1 = max(0, cy - hh), min(c.shape[0], cy + hh)
        x0, x1 = max(0, cx - ww), min(c.shape[1], cx + ww)
        patch = c[y0:y1, x0:x1]
        L = int(rng.uniform(1.5, 4) * px_per_mm)
        k = np.zeros((3, L), np.float32); k[1, :] = 1.0 / L
        ang = rng.uniform(-25, 10)
        M = cv2.getRotationMatrix2D((L / 2, 1), ang, 1.0)
        k = cv2.warpAffine(k, M, (L, 3))
        sm = cv2.filter2D(patch, -1, k / max(1e-6, k.sum()))
        fade = np.outer(np.hanning(patch.shape[0]), np.hanning(patch.shape[1])).astype(np.float32)
        c[y0:y1, x0:x1] = np.maximum(patch, sm * rng.uniform(0.12, 0.25) * fade)
    return c


def compose_physical(shape: np.ndarray, count: np.ndarray, pen, pressure: float, rng,
                     paper: np.ndarray | None = None, smudges: float = 0.0) -> np.ndarray:
    """Stroke shape -> ink concentration (pens.concentration) -> RGB at 300 dpi."""
    from .pens import concentration, to_rgb
    from .layout import MM
    c = concentration(shape, count, pen, rng, MM * SS)
    c = c * pressure_field(shape.shape, rng, pressure)
    c = smudge(c, rng, smudges, MM * SS)
    c = cv2.resize(c, (shape.shape[1] // SS, shape.shape[0] // SS), interpolation=cv2.INTER_AREA)
    if paper is None:
        paper = np.full(c.shape + (3,), 255, np.float32)
    return to_rgb(c, pen, paper)


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


REALISM_PENS = {"blue-ballpoint": "ballpoint-blue", "black-ballpoint": "ballpoint-black", "blue-gel": "gel-blue",
                "black-gel": "gel-black", "fountain": "fountain", "pencil": "pencil"}


def pressure(cov: np.ndarray, seed: int) -> np.ndarray:
    """Pen pressure on a stroke-shape map: a slow pressure field (scale of a letter or two)
    makes strokes wider and darker where the pen was pressed and thinner and fainter where it
    was light, the way a ballpoint line varies along a word. Width changes by re-thresholding
    a blurred copy of the shape at a level that follows the field."""
    from .realism import knobs as KN
    w_amp, l_amp = float(KN.get("press_w", 0.0)), float(KN.get("press_l", 0.0))
    if not (w_amp or l_amp) or not cov.any():
        return cov
    rng = np.random.default_rng(seed + 991)
    cell = float(KN.get("press_cell", 70.0))          # px at the 600 dpi working scale
    h, w = cov.shape
    gh, gw = max(2, int(h / cell) + 2), max(2, int(w / cell) + 2)
    f = cv2.resize(rng.standard_normal((gh, gw)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    f = np.clip(f / max(1e-6, float(f.std())), -2.5, 2.5)
    out = cov
    if w_amp:
        b = cv2.GaussianBlur(cov, (0, 0), 2.0)
        t = np.clip(0.42 + 0.16 * w_amp * f, 0.18, 0.72)
        out = np.clip((b - t) / 0.18 + 0.5, 0, 1) * (cov > 0.02).astype(np.float32) + \
              np.clip((b - t) / 0.18 + 0.5, 0, 1) * (cov <= 0.02) * (t < 0.42)
    if l_amp:
        out = out * np.clip(1 - 0.25 * l_amp * np.clip(-f, 0, None), 0.45, 1)
    return out.astype(np.float32)


def compose_realism(shape: np.ndarray, count: np.ndarray, pen, seed: int, paper: np.ndarray | None = None,
                    rgb=None, smudges: float = 0.0) -> np.ndarray:
    """Default ink model (realism.ink): the page's stroke shape at 600 dpi is re-inked with
    oriented striations, skips, flow variation and blobs; strokes that pass twice are inked
    twice (overlap darkening). Composited in linear light onto the paper, then 300 dpi."""
    from .realism import ink as RI
    name = REALISM_PENS.get(getattr(pen, "name", pen), "ballpoint-blue")
    H, W = shape.shape
    if paper is None:
        paper = np.full((H // SS, W // SS, 3), 255, np.uint8)
    big = cv2.resize(paper.astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
    m = shape > 0.02
    if m.any():
        ys, xs = np.where(m)
        y0, y1 = max(0, ys.min() - 16), min(H, ys.max() + 17)
        x0, x1 = max(0, xs.min() - 16), min(W, xs.max() + 17)
        cov = shape[y0:y1, x0:x1]
        cov = pressure(cov, seed)
        if smudges:
            from .layout import MM
            cov = smudge(cov.copy(), np.random.default_rng(seed + 7), smudges, MM * SS)
        layer = RI.ink_from_mask(cov, dpi=300 * SS, pen=name, seed=seed, rgb=rgb)
        sub = RI.composite(big[y0:y1, x0:x1], layer)
        twice = cov * (count[y0:y1, x0:x1] >= 2)
        if twice.any():
            sub = RI.composite(sub, RI.ink_from_mask(twice, dpi=300 * SS, pen=name, seed=seed + 1, rgb=rgb))
        big[y0:y1, x0:x1] = sub
    return cv2.resize(big, (W // SS, H // SS), interpolation=cv2.INTER_AREA)

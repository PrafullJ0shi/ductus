"""Paper backgrounds and the optional phone-photo look."""
from __future__ import annotations

import cv2
import numpy as np

from .layout import A4, MM


def make_paper(kind: str = "white", size=A4, seed: int = 0) -> np.ndarray:
    W, H = size
    rng = np.random.default_rng(seed)
    p = np.full((H, W, 3), 255, np.float32)
    if kind == "white":
        return p
    # notebook paper is very slightly warm with faint fibre noise
    p[:] = (252, 251, 247)
    p += rng.normal(0, 1.2, (H, W, 1)).astype(np.float32)
    line_col = np.array((176, 190, 214), np.float32)   # pale blue-grey print
    if kind == "ruled":
        y = 34 * MM
        while y < H - 12 * MM:
            yi = int(round(y))
            p[yi:yi + 2] = p[yi:yi + 2] * 0.35 + line_col * 0.65
            y += 8 * MM
        x = int(round(28 * MM))
        p[:, x:x + 2] = p[:, x:x + 2] * 0.4 + np.array((215, 150, 150), np.float32) * 0.6
    elif kind == "grid":
        step = 5 * MM
        for v in np.arange(step / 2, H, step):
            vi = int(round(v))
            p[vi:vi + 1] = p[vi:vi + 1] * 0.55 + line_col * 0.45
        for v in np.arange(step / 2, W, step):
            vi = int(round(v))
            p[:, vi:vi + 1] = p[:, vi:vi + 1] * 0.55 + line_col * 0.45
    return np.clip(p, 0, 255)


def phone_photo(img: np.ndarray, seed: int = 0) -> np.ndarray:
    """Make a flat page look photographed: slight perspective, uneven light, soft focus,
    sensor noise and JPEG compression."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    pad = int(0.04 * w)
    canvas = np.full((h + 2 * pad, w + 2 * pad, 3), (88, 80, 72), np.uint8)  # desk
    canvas[pad:pad + h, pad:pad + w] = img
    H, W = canvas.shape[:2]
    src = np.float32([[pad, pad], [pad + w, pad], [pad + w, pad + h], [pad, pad + h]])
    j = lambda: rng.uniform(-0.018, 0.018) * w
    dst = np.float32([[pad + j(), pad + j()], [pad + w + j(), pad + j()],
                      [pad + w + j(), pad + h + j()], [pad + j(), pad + h + j()]])
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(canvas, M, (W, H), borderValue=(88, 80, 72)).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = rng.uniform(0.3, 0.7) * W, rng.uniform(0.2, 0.5) * H
    light = 1.0 - 0.16 * (((xx - cx) / W) ** 2 + ((yy - cy) / H) ** 2) * 2.2
    light *= 1.0 - 0.06 * (xx / W)   # a soft side shadow
    out *= light[..., None]
    out *= np.array((1.0, 0.985, 0.95), np.float32)  # warm indoor light
    out = cv2.GaussianBlur(out, (0, 0), 0.9)
    out += rng.normal(0, 2.5, out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 82])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]

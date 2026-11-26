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


def phone_photo(img: np.ndarray, seed: int = 0, **kw) -> np.ndarray:
    """Phone capture of the page; see photo.py for the camera model."""
    from .photo import Capture, capture
    return capture(img, seed, Capture(**kw))

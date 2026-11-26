"""Image utilities: ink masks, word splitting, baselines, stroke width, line finding."""
from __future__ import annotations

import numpy as np


def ink_alpha(gray: np.ndarray, lo: float = 90, hi: float = 215) -> np.ndarray:
    """Map a grayscale crop (dark ink on light paper) to ink coverage in 0..1."""
    g = gray.astype(np.float32)
    return np.clip((hi - g) / (hi - lo), 0.0, 1.0)


def clean_line(gray: np.ndarray) -> np.ndarray:
    """Remove the faint grey haze diffusion leaves on the paper and return coverage 0..1."""
    a = ink_alpha(gray)
    a[a < 0.12] = 0.0
    return a


def column_runs(mask_cols: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) of runs of True in a 1-D boolean array (end exclusive)."""
    runs, start = [], None
    for i, v in enumerate(mask_cols):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask_cols)))
    return runs


def split_words(alpha: np.ndarray, n_words: int, min_gap: int = 4) -> list[tuple[int, int]] | None:
    """Split a single line into exactly n_words horizontal spans.

    Uses the n_words-1 widest empty column gaps. Returns None when the line does
    not contain enough clear gaps (the caller regenerates)."""
    cols = alpha.max(axis=0) > 0.25
    runs = column_runs(cols)
    if not runs:
        return None
    # merge specks: ignore ink runs narrower than 2 px that sit far from others
    x0, x1 = runs[0][0], runs[-1][1]
    if n_words == 1:
        return [(x0, x1)]
    gaps = [(runs[i + 1][0] - runs[i][1], runs[i][1], runs[i + 1][0]) for i in range(len(runs) - 1)]
    gaps = [g for g in gaps if g[0] >= min_gap]
    if len(gaps) < n_words - 1:
        return None
    chosen = sorted(sorted(gaps, reverse=True)[: n_words - 1], key=lambda g: g[1])
    # a chosen gap that is much smaller than the median chosen gap suggests a wrong split
    spans, left = [], x0
    for _, a, b in chosen:
        spans.append((left, a))
        left = b
    spans.append((left, x1))
    return spans


def baseline_and_xheight(alpha: np.ndarray) -> tuple[float, float]:
    """Estimate baseline row and x-height (px) of a word or line crop."""
    rows = alpha.sum(axis=1)
    if rows.max() <= 0:
        return alpha.shape[0] * 0.7, alpha.shape[0] * 0.3
    band = np.where(rows > 0.35 * rows.max())[0]
    top, bot = band.min(), band.max()
    return float(bot + 1), float(max(4, bot + 1 - top))


def stroke_width(alpha: np.ndarray) -> float:
    """Median stroke thickness in px via the distance transform on the skeleton."""
    import cv2
    from skimage.morphology import skeletonize
    m = (alpha > 0.4).astype(np.uint8)
    if m.sum() < 10:
        return 2.0
    d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    sk = skeletonize(m > 0)
    v = d[sk]
    return float(2 * np.median(v)) if len(v) else 2.0


def tight_bbox(alpha: np.ndarray, thr: float = 0.15):
    ys, xs = np.where(alpha > thr)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def find_text_lines(alpha: np.ndarray, min_h: int = 8) -> list[tuple[int, int]]:
    """Row spans of text lines in a deskewed page-level ink map (projection profile)."""
    import cv2
    rows = alpha.sum(axis=1)
    k = max(3, alpha.shape[0] // 300)
    sm = cv2.GaussianBlur(rows.reshape(-1, 1).astype(np.float32), (1, 2 * k + 1), 0).ravel()
    on = sm > 0.08 * sm.max()
    return [(a, b) for a, b in column_runs(on) if b - a >= min_h]


def skew_angle(alpha: np.ndarray, search: float = 8.0, step: float = 0.25) -> float:
    """Angle (degrees) that maximises horizontal projection sharpness."""
    import cv2
    h, w = alpha.shape
    small = cv2.resize(alpha, (w // 2, h // 2)) if max(h, w) > 1600 else alpha
    best, best_a = -1.0, 0.0
    c = (small.shape[1] / 2, small.shape[0] / 2)
    for a in np.arange(-search, search + 1e-6, step):
        M = cv2.getRotationMatrix2D(c, a, 1.0)
        r = cv2.warpAffine(small, M, (small.shape[1], small.shape[0]))
        s = np.var(r.sum(axis=1))
        if s > best:
            best, best_a = s, a
    return float(-best_a)  # skew of the content; rotate by -angle to straighten


def ref_habits(gray: np.ndarray) -> dict:
    """Spacing and stroke habits of a reference line image: word gap and stroke width,
    both in x-heights. The generator copies letter shapes well but not spacing or weight,
    so the layout and renderer take these from the reference directly."""
    a = clean_line(gray)
    bb = tight_bbox(a, 0.3)
    if bb is None:
        return {}
    x0, y0, x1, y1 = bb
    a = a[y0:y1, x0:x1]
    _, xh = baseline_and_xheight(a)
    runs = column_runs(a.max(axis=0) > 0.25)
    gaps = np.array([runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)], float)
    word_gaps = gaps[gaps > 0.45 * xh]
    out = {"stroke_xh": stroke_width(a) / xh}
    if len(word_gaps) >= 2:
        out["gap_xh"] = float(np.clip(np.median(word_gaps) / xh, 0.5, 2.2))
    return out

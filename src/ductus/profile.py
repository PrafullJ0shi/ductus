"""Build a personal style profile from phone photos of handwritten pages.

Steps per photo: normalise lighting -> find ink (dark, saturated) -> remove printed grid
or ruling -> pick the rotation (0/90/180/270) and deskew -> split into text lines ->
keep clean, wide lines as one-shot style references -> measure layout habits
(x-height, line pitch, baseline drift, word gaps, ink colour, pen weight).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .segment import find_text_lines, skew_angle, baseline_and_xheight, stroke_width, column_runs


def load_photo(path: str, max_side: int = 2200) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    k = max_side / max(h, w)
    if k < 1:
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    return img[..., ::-1].copy()  # RGB


def ink_map(rgb: np.ndarray) -> np.ndarray:
    """Ink coverage 0..1 with lighting removed. Pen ink is darker than the local paper
    and (for blue/black pens) more saturated than printed grid lines."""
    f = rgb.astype(np.float32)
    gray = f.mean(axis=2)
    bg = cv2.medianBlur(cv2.resize(gray, (gray.shape[1] // 4, gray.shape[0] // 4)).astype(np.uint8), 21)
    bg = cv2.resize(bg, (gray.shape[1], gray.shape[0])).astype(np.float32)
    dark = np.clip((bg - gray) / np.maximum(bg, 1), 0, 1)       # relative darkness
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[..., 1] / 255.0
    score = dark * (0.6 + 0.8 * sat)
    t = max(0.10, float(np.percentile(score, 99.5)) * 0.35)
    a = np.clip((score - t * 0.6) / (t * 0.8), 0, 1)
    return a


def paper_mask(rgb: np.ndarray) -> np.ndarray:
    """The page in a photo: the largest bright, low-saturation region, holes filled and the
    border pulled in a little, so the desk, the spiral binding and the page edge are not read
    as ink."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v = cv2.GaussianBlur(hsv[..., 2], (0, 0), 3)
    sat = cv2.GaussianBlur(hsv[..., 1], (0, 0), 3)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ((v > t) & (sat < 90)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return np.ones(rgb.shape[:2], np.float32)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    m = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    m = np.zeros_like(m)
    cv2.fillPoly(m, [hull], 1)
    e = max(5, int(0.025 * min(m.shape)))
    m = cv2.erode(m, np.ones((e, e), np.uint8))
    if m.mean() < 0.2:                     # no clear page (a scan, or a crop): keep everything
        return np.ones(rgb.shape[:2], np.float32)
    return m.astype(np.float32)


def remove_grid(a: np.ndarray) -> np.ndarray:
    """Remove long thin straight lines (printed grid or ruling) from an ink map."""
    m = (a > 0.35).astype(np.uint8)
    h, w = m.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 12), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 12)))
    lines = cv2.morphologyEx(m, cv2.MORPH_OPEN, hk) | cv2.morphologyEx(m, cv2.MORPH_OPEN, vk)
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    out = a.copy()
    out[lines > 0] = 0
    return out


def grid_period(img: np.ndarray) -> float | None:
    """Spacing in px of a printed grid. The page is perspective-warped and has text on it, so
    the period is read from the spectrum of row and column profiles in several 500 px windows
    and the median is kept. `img` is ink-positive (high = dark)."""
    g = img.astype(np.float64)
    H, W = g.shape
    if min(H, W) < 700:
        return None
    est = []
    for fy in (0.25, 0.4, 0.55):
        for fx in (0.25, 0.4, 0.55):
            y, x = int(fy * H), int(fx * W)
            c = g[y:y + 500, x:x + 500]
            for prof in (c.mean(axis=0), c.mean(axis=1)):
                prof = prof - np.convolve(prof, np.ones(41) / 41, "same")
                prof = prof[41:-41] * np.hanning(len(prof) - 82)
                F = np.abs(np.fft.rfft(prof, 4096))
                fr = np.fft.rfftfreq(4096)
                m = (fr > 1 / 120) & (fr < 1 / 15)
                i = int(np.argmax(F * m))
                if F[i] > 3 * np.median(F[m]):
                    est.append(1 / fr[i])
    return float(np.median(est)) if len(est) >= 4 else None


def orient(a: np.ndarray, reader=None) -> tuple[np.ndarray, int]:
    """Choose 0/90 from projection sharpness, then 0/180 by OCR confidence when available."""
    def sharp(x):
        return np.var(x.sum(axis=1)) / max(1e-9, x.sum() ** 2)
    rot = 0
    if sharp(np.rot90(a)) > sharp(a) * 1.15:
        a, rot = np.rot90(a, -1).copy(), 90
    if reader is not None:
        lines = find_text_lines(a)
        if lines:
            y0, y1 = max(lines, key=lambda r: a[r[0]:r[1]].sum())
            crop = (255 * (1 - a[y0:y1])).astype(np.uint8)
            _, c0 = reader.read([crop], with_conf=True)
            _, c1 = reader.read([crop[::-1, ::-1].copy()], with_conf=True)
            if c1[0] > c0[0] + 0.05:
                a, rot = a[::-1, ::-1].copy(), rot + 180
    return a, rot


def drop_blobs(a: np.ndarray) -> np.ndarray:
    """Remove ink blobs far thicker than pen strokes: spiral-binding rings, shadows, page
    edges. A stroke's thickness is the largest distance-transform value inside it."""
    m = (a > 0.3).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 3:
        return a
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    thick = np.zeros(n, np.float32)
    np.maximum.at(thick, lab.ravel(), dt.ravel())
    big = st[:, cv2.CC_STAT_AREA] > 20
    big[0] = False
    if big.sum() < 5:
        return a
    med = float(np.median(thick[big]))
    w, h = st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]
    bad = (thick > 2.6 * med) | (np.maximum(w, h) > 0.5 * max(a.shape))
    bad[0] = False
    out = a.copy()
    mask = bad[lab]
    # also clear the faint halo around a dropped blob
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    out[mask] = 0
    return out


def orient_deskew(a: np.ndarray, reader=None, search: float = 15.0) -> tuple[np.ndarray, int, float]:
    """Upright, deskewed page ink map: 0/90 by the sharper deskewed row projection, then
    0/180 by OCR confidence over the longest few lines (trimmed to their ink)."""
    from .segment import skew_angle
    best = None
    for r in (0, 90):
        b = np.rot90(a, -1).copy() if r else a
        ang = skew_angle(b, search=search)
        b = rotate(b, -ang)
        # line structure only: remove the slow envelope of the text block before scoring,
        # so a narrow, tall block is not mistaken for sideways text
        pr = b.sum(axis=1).astype(np.float64)
        env = cv2.GaussianBlur(pr.reshape(-1, 1), (1, 0), sigmaX=0.1, sigmaY=max(8.0, len(pr) / 25)).ravel()
        sh = np.var(pr - env) / max(1e-9, float(pr.mean()) ** 2)
        if best is None or sh > best[0]:
            best = (sh, b, r, ang)
    _, a, rot, ang = best
    if reader is not None:
        spans = sorted(find_text_lines(a), key=lambda r: -a[r[0]:r[1]].sum())[:8]
        crops = []
        for y0, y1 in spans:
            sub = a[y0:y1]
            cols = np.where(sub.max(axis=0) > 0.3)[0]
            if len(cols) > 10:
                crops.append((255 * (1 - sub[:, cols.min():cols.max() + 1])).astype(np.uint8))
        if crops:
            _, c0 = reader.read(crops, with_conf=True)
            _, c1 = reader.read([c[::-1, ::-1].copy() for c in crops], with_conf=True)
            orient_deskew.last = (float(np.mean(c0)), float(np.mean(c1)))
            # a sideways page is as likely one way as the other; an upright-looking page is
            # rarely upside down (phones rotate photos), so it needs a clear OCR margin
            need = 0.02 if rot == 90 else 0.15
            if float(np.mean(c1)) > float(np.mean(c0)) + need:
                a, rot = a[::-1, ::-1].copy(), rot + 180
    return a, rot, ang


def rotate(a: np.ndarray, deg: float) -> np.ndarray:
    h, w = a.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR)


def _ncomp(crop: np.ndarray) -> int:
    m = (crop > 0.3).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return int((st[1:, cv2.CC_STAT_AREA] > 15).sum())


def cc_lines(a: np.ndarray) -> list[dict]:
    """Text lines as chains of ink marks: each mark links to the nearest mark to its right at
    about the same height, so lines that curve, tilt or sit close together (descenders of
    one line reaching the next) are kept apart. Small marks (dots, commas) join the line
    whose body is nearest. Returns, per line, the ink of that line's marks only, cut out and
    straightened along the line, with its fitted slope in degrees."""
    m = (a > 0.3).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 3:
        return []
    area = st[:, cv2.CC_STAT_AREA]
    hgt = st[:, cv2.CC_STAT_HEIGHT].astype(float)
    big = [i for i in range(1, n) if area[i] >= 25]
    if len(big) < 3:
        return []
    h = float(np.median(hgt[big]))
    small = [i for i in range(1, n) if 4 <= area[i] < 25 or hgt[i] < 0.35 * h]
    body = [i for i in big if hgt[i] >= 0.35 * h]
    # vertical centre of a mark's main body (tall letters are pulled toward the x-band)
    cy = {i: st[i, 1] + min(st[i, 3], 1.4 * h) - 0.5 * min(st[i, 3], 1.2 * h) for i in body}
    parent = {i: i for i in body}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    order = sorted(body, key=lambda i: st[i, 0])
    xs = np.array([st[i, 0] for i in order])
    for k, i in enumerate(order):
        x1 = st[i, 0] + st[i, 2]
        best, bc = None, 1e9
        for j in order[k + 1:k + 60]:
            gap = st[j, 0] - x1
            if gap > 3.0 * h:
                if st[j, 0] > x1 + 3.0 * h and st[j, 0] - st[i, 0] > 6 * h:
                    break
                continue
            dy = abs(cy[j] - cy[i])
            if dy > 0.6 * h:
                continue
            c = max(gap, 0) + 2.5 * dy
            if c < bc:
                best, bc = j, c
        if best is not None:
            parent[find(best)] = find(i)
    groups: dict[int, list[int]] = {}
    for i in body:
        groups.setdefault(find(i), []).append(i)
    lines = [g for g in groups.values() if len(g) >= 2]
    def fit_of(g):
        x = np.array([st[i, 0] + st[i, 2] / 2 for i in g]); y = np.array([cy[i] for i in g])
        return np.polyfit(x, y, 1) if np.ptp(x) > 2 * h else np.array([0.0, y.mean()])
    # pieces of one line split at a wide gap: merge pieces that do not overlap and sit on
    # the same fitted line
    merged = True
    while merged and len(lines) > 1:
        merged = False
        fits = [fit_of(g) for g in lines]
        ext = [(min(st[i, 0] for i in g), max(st[i, 0] + st[i, 2] for i in g)) for g in lines]
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                (a0, a1), (b0, b1) = ext[i], ext[j]
                if min(a1, b1) - max(a0, b0) > 0.5 * h:
                    continue
                mid_j = (b0 + b1) / 2; mid_i = (a0 + a1) / 2
                yj = np.mean([cy[k] for k in lines[j]]); yi = np.mean([cy[k] for k in lines[i]])
                if abs(np.polyval(fits[i], mid_j) - yj) < 0.45 * h and abs(np.polyval(fits[j], mid_i) - yi) < 0.45 * h \
                        and max(a0, b0) - min(a1, b1) < 25 * h:
                    lines[i] = lines[i] + lines[j]
                    del lines[j]
                    merged = True
                    break
            if merged:
                break
    lines = [g for g in lines if len(g) >= 3]
    if not lines:
        return []
    fits = [fit_of(g) for g in lines]
    for i in small:
        x = st[i, 0] + st[i, 2] / 2
        best, bd = None, 1e9
        for k, g in enumerate(lines):
            gx0 = min(st[j, 0] for j in g) - 1.5 * h; gx1 = max(st[j, 0] + st[j, 2] for j in g) + 1.5 * h
            if not gx0 <= x <= gx1:
                continue
            d = abs(cen[i, 1] - np.polyval(fits[k], x))
            if d < bd:
                best, bd = k, d
        if best is not None and bd < 1.3 * h:
            lines[best].append(i)
    out = []
    for g, fit in zip(lines, fits):
        sel = np.zeros(n, bool); sel[g] = True
        x0 = int(min(st[i, 0] for i in g)); x1 = int(max(st[i, 0] + st[i, 2] for i in g))
        y0 = int(min(st[i, 1] for i in g)); y1 = int(max(st[i, 1] + st[i, 3] for i in g))
        pad = int(0.3 * h) + 4
        ya, yb, xa, xb = max(0, y0 - pad), min(a.shape[0], y1 + pad), max(0, x0 - 4), min(a.shape[1], x1 + 4)
        crop = a[ya:yb, xa:xb] * sel[lab[ya:yb, xa:xb]]
        ang = float(np.degrees(np.arctan(fit[0])))
        if abs(ang) > 0.3:
            M = cv2.getRotationMatrix2D((crop.shape[1] / 2, crop.shape[0] / 2), ang, 1.0)
            ch = int(crop.shape[0] + abs(np.tan(np.radians(ang))) * crop.shape[1]) + 2
            M[1, 2] += (ch - crop.shape[0]) / 2
            crop = cv2.warpAffine(crop, M, (crop.shape[1], ch), flags=cv2.INTER_LINEAR)
            rows = np.where(crop.max(axis=1) > 0.3)[0]
            if len(rows):
                crop = crop[max(0, rows.min() - pad):rows.max() + pad + 1]
        out.append(dict(crop=crop, y=(ya, yb), x=(x0, x1), slope=ang, cy=float(np.polyval(fit, (x0 + x1) / 2))))
    out.sort(key=lambda d: d["cy"])
    return out


def line_crops(a: np.ndarray) -> list[dict]:
    """Split a deskewed page ink map into line crops with measurements."""
    out = []
    for ln in cc_lines(a):
        crop = ln["crop"]
        ya, yb = ln["y"]
        x0, x1 = ln["x"]
        bl, xh = baseline_and_xheight(crop)
        runs = column_runs(crop.max(axis=0) > 0.3)
        gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
        word_gaps = [g for g in gaps if g > 0.5 * xh]
        # baseline drift: lowest ink of each of 6 column bands, linear fit
        bands = np.array_split(np.arange(crop.shape[1]), 6)
        pts = []
        for b in bands:
            sub = crop[:, b]
            rows = np.where(sub.max(axis=1) > 0.3)[0]
            if len(rows):
                prof = sub.sum(axis=1)
                band = np.where(prof > 0.35 * prof.max())[0]
                pts.append((b.mean(), band.max()))
        slope = float(np.tan(np.radians(-ln["slope"])))      # drift of the line before straightening
        resid = float(np.std(np.array([p[1] for p in pts]) - np.polyval(np.polyfit(*zip(*pts), 1), [p[0] for p in pts]))) if len(pts) >= 3 else 0.0
        out.append(dict(y=(ya, yb), x=(x0, x1), crop=crop, xh=xh, width=x1 - x0,
                        gaps=word_gaps, slope_deg=float(np.degrees(np.arctan(slope))), resid=resid,
                        sw=stroke_width(crop), ncomp=_ncomp(crop)))
    return out


def build(photos: list[str], out_dir: str, name: str = "my-hand", reader=None, max_refs: int = 12,
          log=print) -> dict:
    out = Path(out_dir)
    (out / "refs").mkdir(parents=True, exist_ok=True)
    (out / "debug").mkdir(exist_ok=True)
    all_lines, inks, periods = [], [], []
    for pi, p in enumerate(photos):
        rgb = load_photo(p)
        raw = ink_map(rgb) * paper_mask(rgb)
        a, rot, ang = orient_deskew(drop_blobs(remove_grid(raw)), reader)
        per = grid_period(255.0 - cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        if per:
            periods.append(per)
        cv2.imwrite(str(out / "debug" / f"page{pi:02d}_ink.png"), (255 * (1 - a)).astype(np.uint8))
        # ink colour from the strongest ink pixels of the original photo
        m = raw > 0.8
        if m.sum() > 50:
            inks.append(np.median(rgb[m], axis=0))
        lines = line_crops(a)
        for ln in lines:
            ln["page"] = pi
            ln["grid"] = per
        all_lines += lines
        log(f"{Path(p).name}: rotated {rot} deg, deskewed {ang:+.2f} deg, {len(lines)} lines, grid {per and round(per,1)} px")
    if not all_lines:
        raise RuntimeError("no text lines found")
    xh = float(np.median([l["xh"] for l in all_lines]))
    # good references: wide single lines with average x-height
    # real writing has many marks along the line; rules, stray bars and blobs do not
    cand = [l for l in all_lines if l["width"] > 8 * xh and 0.6 * xh < l["xh"] < 1.6 * xh
            and l["ncomp"] >= 0.35 * l["width"] / max(l["xh"], 1)]
    cand.sort(key=lambda l: -l["width"])
    refs = []
    for i, l in enumerate(cand[:max_refs]):
        g = (255 * (1 - np.clip(l["crop"], 0, 1))).astype(np.uint8)
        f = out / "refs" / f"ref_{i:02d}.png"
        cv2.imwrite(str(f), g)
        refs.append(f.name)
    grid_px = float(np.median(periods)) if periods else None
    mm_per_px = (5.0 / grid_px) if grid_px else None    # 5 mm squares are standard
    gaps = [g for l in all_lines for g in l["gaps"]]
    # short pieces give noisy slopes: measure drift on lines of some length
    slopes = np.array([l["slope_deg"] for l in all_lines if l["width"] > 8 * xh] or
                      [l["slope_deg"] for l in all_lines])
    pitch = None
    ys = sorted([(l["page"], (l["y"][0] + l["y"][1]) / 2) for l in all_lines])
    d = [b[1] - a[1] for a, b in zip(ys, ys[1:]) if a[0] == b[0] and 0.8 * xh < b[1] - a[1] < 6 * xh]
    if d:
        pitch = float(np.median(d)) / xh
    ink = np.median(np.array(inks), axis=0) if inks else np.array([30, 50, 140])
    look = {
        "xh_mm": round(xh * mm_per_px, 2) if mm_per_px else 2.6,
        "pitch": round(pitch, 2) if pitch else 3.6,
        "slope_deg": round(float(np.median(-slopes)), 2),
        "slope_sd": round(float(np.std(slopes)), 2),
        "gap_xh": round(float(np.median(gaps) / xh), 2) if gaps else 0.8,
        "gap_sd_xh": round(float(np.std(gaps) / xh), 2) if gaps else 0.3,
        "baseline_resid_xh": round(float(np.median([l["resid"] for l in all_lines]) / xh), 3),
        "stroke_xh": round(float(np.median([l["sw"] for l in all_lines]) / xh), 3),
        "ink": "#%02x%02x%02x" % tuple(int(v) for v in ink),
    }
    # messiness estimate from baseline wobble and drift spread (0 tidy .. 1 hurried)
    mess = np.clip(look["baseline_resid_xh"] * 4 + look["slope_sd"] / 3 + look["gap_sd_xh"] / 2, 0.1, 1.0)
    look["mess"] = round(float(mess), 2)
    prof = {"name": name, "refs": refs, "look": look, "pages": len(photos), "lines_found": len(all_lines),
            "grid_px": grid_px, "ocr_mode": "loose", "habits": measure_habits(all_lines, xh)}
    (out / "profile.json").write_text(json.dumps(prof, indent=2))
    return prof


def measure_habits(lines: list[dict], xh: float) -> dict:
    """Page-level habits from the segmented lines of all pages: growth of the writing and
    tightening of the lines toward the page end, left-margin spread, how often and how much
    larger headings are, and the space left between sections."""
    import numpy as np
    by_page: dict[int, list] = {}
    for l in lines:
        by_page.setdefault(l["page"], []).append(l)
    grow, tight, margins, sections = [], [], [], []
    margin_sd = []
    for pls in by_page.values():
        # pieces of one written row (split at wide gaps) count as one row: leftmost start,
        # median size
        pls = sorted(pls, key=lambda l: (l["y"][0] + l["y"][1]) / 2)
        rows_: list[list] = []
        for l in pls:
            c = (l["y"][0] + l["y"][1]) / 2
            if rows_ and abs(c - np.mean([(q["y"][0] + q["y"][1]) / 2 for q in rows_[-1]])) < 0.6 * xh:
                rows_[-1].append(l)
            else:
                rows_.append([l])
        pls = [dict(y=(min(q["y"][0] for q in r), max(q["y"][1] for q in r)),
                    x=(min(q["x"][0] for q in r), max(q["x"][1] for q in r)),
                    xh=float(np.median([q["xh"] for q in r]))) for r in rows_]
        if len(pls) < 6:
            continue
        t = np.linspace(0, 1, len(pls))
        sizes = np.array([l["xh"] for l in pls], float)
        body = sizes < 1.3 * np.median(sizes)
        if body.sum() >= 4:
            k = np.polyfit(t[body], sizes[body] / np.median(sizes[body]), 1)[0]
            grow.append(k)
        ys = np.array([(l["y"][0] + l["y"][1]) / 2 for l in pls])
        d = np.diff(ys) / xh
        ok = d < 6
        if ok.sum() >= 4:
            half = len(d) // 2
            top, bot = np.median(d[:half][ok[:half]]) if ok[:half].any() else None, \
                np.median(d[half:][ok[half:]]) if ok[half:].any() else None
            if top and bot:
                tight.append(1 - bot / top)
        sections += list(d[(d >= 4.5) & (d < 12)])
        m_ = np.array([l["x"][0] for l in pls]) / xh
        margins += list(m_)
        # spread around the page's own margin; indented items (lists, working) are habits too,
        # but a row that starts mid-page is a second column, not the margin
        m_ = m_[m_ < np.percentile(m_, 25) + 8]
        if len(m_) >= 4:
            margin_sd.append(float(np.std(m_)))
    sizes = np.array([l["xh"] for l in lines], float)
    heads = sizes[sizes > 1.3 * np.median(sizes)]
    out = {
        "fatigue": round(float(np.clip(np.median(grow), 0, 0.12)), 3) if grow else 0.06,
        "tighten": round(float(np.clip(np.median(tight), 0, 0.4)), 3) if tight else 0.1,
        "margin_sd_xh": round(float(np.median(margin_sd)), 2) if margin_sd else None,
        "heading_share": round(float(len(heads) / max(1, len(sizes))), 3),
        "heading_scale": round(float(np.median(heads) / np.median(sizes)), 2) if len(heads) else 1.45,
        "section_gap_xh": round(float(np.median(sections)), 2) if sections else None,
    }
    return out

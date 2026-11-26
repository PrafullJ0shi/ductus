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


def grid_period(a_raw: np.ndarray) -> float | None:
    """Spacing in px of a printed grid, from the autocorrelation of the column profile."""
    prof = a_raw.mean(axis=0) - a_raw.mean()
    if len(prof) < 200:
        return None
    ac = np.correlate(prof, prof, "full")[len(prof) - 1:]
    ac /= max(ac[0], 1e-9)
    lo = 12
    pk = lo + int(np.argmax(ac[lo:len(ac) // 4]))
    return float(pk) if ac[pk] > 0.2 else None


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


def rotate(a: np.ndarray, deg: float) -> np.ndarray:
    h, w = a.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR)


def line_crops(a: np.ndarray) -> list[dict]:
    """Split a deskewed page ink map into line crops with measurements."""
    out = []
    spans = find_text_lines(a)
    if not spans:
        return out
    hs = np.array([b - a_ for a_, b in spans])
    med = float(np.median(hs))
    for (y0, y1) in spans:
        if not (0.45 * med <= y1 - y0 <= 2.2 * med):
            continue
        pad = int(0.25 * (y1 - y0))
        ya, yb = max(0, y0 - pad), min(a.shape[0], y1 + pad)
        crop = a[ya:yb]
        cols = np.where(crop.max(axis=0) > 0.3)[0]
        if len(cols) < 10:
            continue
        x0, x1 = cols.min(), cols.max() + 1
        crop = crop[:, max(0, x0 - 4):x1 + 4]
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
        slope = float(np.polyfit(*zip(*pts), 1)[0]) if len(pts) >= 3 else 0.0
        resid = float(np.std(np.array([p[1] for p in pts]) - np.polyval(np.polyfit(*zip(*pts), 1), [p[0] for p in pts]))) if len(pts) >= 3 else 0.0
        out.append(dict(y=(ya, yb), x=(x0, x1), crop=crop, xh=xh, width=x1 - x0,
                        gaps=word_gaps, slope_deg=float(np.degrees(np.arctan(slope))), resid=resid,
                        sw=stroke_width(crop)))
    return out


def build(photos: list[str], out_dir: str, name: str = "my-hand", reader=None, max_refs: int = 12,
          log=print) -> dict:
    out = Path(out_dir)
    (out / "refs").mkdir(parents=True, exist_ok=True)
    (out / "debug").mkdir(exist_ok=True)
    all_lines, inks, periods = [], [], []
    for pi, p in enumerate(photos):
        rgb = load_photo(p)
        raw = ink_map(rgb)
        a, rot = orient(remove_grid(raw), reader)
        raw_o = np.rot90(raw, -(rot // 90) % 4) if rot % 180 == 90 else (raw[::-1, ::-1] if rot == 180 else raw)
        per = grid_period(raw_o)
        if per:
            periods.append(per)
        ang = skew_angle(a)
        a = rotate(a, -ang)
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
    cand = [l for l in all_lines if l["width"] > 8 * xh and 0.6 * xh < l["xh"] < 1.6 * xh]
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
    slopes = np.array([l["slope_deg"] for l in all_lines])
    pitch = None
    ys = sorted([(l["page"], (l["y"][0] + l["y"][1]) / 2) for l in all_lines])
    d = [b[1] - a[1] for a, b in zip(ys, ys[1:]) if a[0] == b[0] and 0 < b[1] - a[1] < 6 * xh]
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
    for pls in by_page.values():
        pls = sorted(pls, key=lambda l: l["y"][0])
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
        margins += list(np.array([l["x"][0] for l in pls]) / xh)
    sizes = np.array([l["xh"] for l in lines], float)
    heads = sizes[sizes > 1.3 * np.median(sizes)]
    out = {
        "fatigue": round(float(np.clip(np.median(grow), 0, 0.3)), 3) if grow else 0.06,
        "tighten": round(float(np.clip(np.median(tight), 0, 0.4)), 3) if tight else 0.1,
        "margin_sd_xh": round(float(np.std(margins)), 2) if margins else None,
        "heading_share": round(float(len(heads) / max(1, len(sizes))), 3),
        "heading_scale": round(float(np.median(heads) / np.median(sizes)), 2) if len(heads) else 1.45,
        "section_gap_xh": round(float(np.median(sections)), 2) if sections else None,
    }
    return out

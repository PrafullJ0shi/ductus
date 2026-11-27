"""Which measurable cues separate real line crops from generated ones?

For each crop (the same crops blind_test.py uses): stroke width and its spread, ink darkness
and its spread, edge sharpness, blur, x-height, slant and its spread, word-gap regularity,
baseline wobble and ink-row texture. Prints the mean of each cue for both sets and the effect
size (Cohen's d), largest first. A cue with |d| > 1 is a strong giveaway.

  python scripts/blind_cues.py REAL_DIR FAKE_DIR
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from ductus.segment import ink_alpha, column_runs, baseline_and_xheight


def cues(gray):
    a = ink_alpha(gray.astype(np.float32) if gray.dtype != np.uint8 else gray)
    m = a > 0.5
    if m.sum() < 50:
        return None
    rows = np.where(m.any(1))[0]
    a, m = a[rows.min():rows.max() + 1], m[rows.min():rows.max() + 1]
    bl, xh = baseline_and_xheight(a)
    xh = max(float(xh), 4.0)
    dt = ndimage.distance_transform_edt(m)
    sk = dt[(dt >= ndimage.maximum_filter(dt, 3)) & m]
    sw = 2 * sk
    gx, gy = cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1)
    g = np.hypot(gx, gy)
    edge = g[(a > 0.2) & (a < 0.8)]
    ang = np.degrees(np.arctan2(gy, gx))[g > 0.3]
    # slant from near-vertical stroke gradients (gradient is horizontal on vertical strokes)
    vert = ang[(np.abs(np.abs(ang) - 0) < 40) | (np.abs(np.abs(ang) - 180) < 40)]
    runs = column_runs(m.any(0))
    gaps = np.array([runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)], float)
    wg = gaps[gaps > 0.5 * xh] / xh
    cols = np.where(m.any(0))[0]
    bands = np.array_split(cols, 8)
    lows = [np.where(m[:, b].any(1))[0].max() for b in bands if len(b)]
    dark = a[m]
    return {
        "stroke_w/xh": float(np.median(sw)) / xh,
        "stroke_w_cv": float(np.std(sw) / max(np.mean(sw), 1e-6)),
        "ink_mean": float(dark.mean()),
        "ink_sd": float(dark.std()),
        "partial_ink_share": float(((a > 0.1) & (a < 0.6)).sum() / max(1, (a > 0.1).sum())),
        "edge_sharpness": float(np.median(edge)) if len(edge) else 0.0,
        "xh_px": xh,
        "slant_sd": float(np.std(vert)) if len(vert) > 10 else 0.0,
        "gap_mean/xh": float(wg.mean()) if len(wg) else 0.0,
        "gap_cv": float(wg.std() / wg.mean()) if len(wg) > 1 else 0.0,
        "baseline_wobble/xh": float(np.std(lows) / xh) if len(lows) > 3 else 0.0,
        "ink_density": float(m.mean()),
        "height/xh": a.shape[0] / xh,
    }


def load(d):
    out = []
    for f in sorted(Path(d).glob("*.png")):
        c = cues(cv2.imread(str(f), 0))
        if c:
            out.append(c)
    return out


R, F = load(sys.argv[1]), load(sys.argv[2])
rows = []
for k in R[0]:
    r, f = np.array([x[k] for x in R]), np.array([x[k] for x in F])
    d = (f.mean() - r.mean()) / max(np.sqrt((r.var() + f.var()) / 2), 1e-9)
    rows.append((abs(d), k, r.mean(), f.mean(), d))
print(f"real n={len(R)}  generated n={len(F)}")
print(f"{'cue':22s} {'real':>9s} {'generated':>10s} {'d':>7s}")
for _, k, rm, fm, d in sorted(rows, reverse=True):
    print(f"{k:22s} {rm:9.3f} {fm:10.3f} {d:7.2f}")

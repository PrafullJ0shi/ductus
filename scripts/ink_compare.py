"""Compare real ink with generated ink at 100% zoom.

  python scripts/ink_compare.py --real photo1.jpg [photo2.jpg ...] --fake page.png -o ink_compare.png
Cuts 6 ink-dense 240x120 px patches from each source (no scaling), stacks real above
generated, and prints ink statistics for both: median ink colour, darkness percentiles
and the share of light pixels inside strokes (texture). Run it on your own machine: the
real photos never need to leave it.
"""
import argparse, json

import cv2
import numpy as np


def patches(img, n, rng, size=(120, 240)):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.medianBlur(g.astype(np.uint8), 41).astype(np.float32)
    ink = np.clip((bg - g) / np.maximum(bg, 1), 0, 1)
    h, w = size
    out, tries = [], 0
    while len(out) < n and tries < 5000:
        tries += 1
        y, x = rng.integers(0, img.shape[0] - h), rng.integers(0, img.shape[1] - w)
        if (ink[y:y + h, x:x + w] > 0.3).mean() > 0.06:
            out.append((img[y:y + h, x:x + w], ink[y:y + h, x:x + w]))
    return out


def stats(ps, imgs):
    px = np.concatenate([im.reshape(-1, 3)[(ik.reshape(-1) > 0.3)] for im, ik in ps])
    ik = np.concatenate([ik.reshape(-1)[ik.reshape(-1) > 0.3] for _, ik in ps])
    return {"median_ink_bgr": np.median(px, 0).round(1).tolist(),
            "darkness_p10_p50_p90": np.percentile(ik, [10, 50, 90]).round(3).tolist(),
            "light_share_in_strokes": round(float((ik < 0.5).mean()), 3)}


ap = argparse.ArgumentParser()
ap.add_argument("--real", nargs="+", required=True); ap.add_argument("--fake", nargs="+", required=True)
ap.add_argument("-o", "--out", default="ink_compare.png")
a = ap.parse_args()
rng = np.random.default_rng(0)
res, rows = {}, []
for name, files in (("real", a.real), ("generated", a.fake)):
    ps = []
    for f in files:
        ps += patches(cv2.imread(f), max(1, 6 // len(files)), rng)
    res[name] = stats(ps, None)
    rows.append(np.hstack([p for p, _ in ps[:6]]))
W = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 8), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
cv2.imwrite(a.out, np.vstack(rows))
print(json.dumps(res, indent=1))

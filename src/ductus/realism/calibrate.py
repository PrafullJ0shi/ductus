"""Measure real phone photos of handwritten notebook pages and write calibration.json.

    python -m ductus.realism.calibrate photo1.jpg photo2.jpg ... -o calibration.json [--crops DIR]

Per photo it measures (all robust statistics, no manual labels):
* JPEG quality, estimated from the luminance quantisation table (IJG scaling), and chroma
  subsampling; EXIF make/model/ISO/exposure/focal length when present;
* paper colour, its correlated colour temperature (McCamy 1992) and the white-balance
  gains that would make it neutral;
* ink colour (median of the darkest ink pixels and of all ink pixels), single-pass ink
  transmittance relative to the paper, and line width in px;
* sensor noise (std of the high-pass residual in flat paper areas, luma and chroma);
* blur: edge spread sigma across ink edges (for a Gaussian-blurred step the maximum gradient
  is C / (sigma sqrt(2 pi))) and variance of the Laplacian;
* tilt: dominant angle of grid / text lines (Hough) and convergence of near-vertical lines
  (keystone), page coverage of the frame;
* lighting: luminance fall-off across the page (plane fit of the paper background) and the
  radial vignetting term.
The aggregate (median over photos) is mapped to ``CaptureParams`` and ink colours, and
written with the per-photo numbers so the source of every default is traceable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

# IJG standard luminance quantisation table (quality 50)
_STD_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                      14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                      49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], np.float64)


def jpeg_quality(path: str) -> dict:
    """Estimate IJG-equivalent quality from the first quantisation table."""
    try:
        im = Image.open(path)
        q = getattr(im, "quantization", None)
        if not q:
            return {"quality": None}
        t = np.array(q[0], np.float64)
        if t.size != 64:
            return {"quality": None}
        # PIL gives tables in zig-zag order on some versions; compare sorted ratios (order-free)
        scale = np.median(t / _STD_LUMA) * 100 if np.all(t > 0) else None
        best, berr = None, 1e18
        for qq in range(1, 101):
            s = 5000 / qq if qq < 50 else 200 - 2 * qq
            ref = np.clip(np.floor((_STD_LUMA * s + 50) / 100), 1, 255)
            err = np.abs(np.sort(ref) - np.sort(t)).sum()
            if err < berr:
                best, berr = qq, err
        sub = None
        try:
            from PIL import JpegImagePlugin
            sub = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}.get(JpegImagePlugin.get_sampling(im))
        except Exception:
            pass
        return {"quality": int(best), "fit_error": float(berr), "subsampling": sub,
                "scale_pct": None if scale is None else float(scale)}
    except Exception as e:  # not a JPEG
        return {"quality": None, "error": str(e)}


def exif(path: str) -> dict:
    try:
        im = Image.open(path)
        ex = im.getexif()
        out = {}
        tags = {271: "make", 272: "model", 305: "software"}
        for k, n in tags.items():
            if k in ex:
                out[n] = str(ex[k])
        sub = ex.get_ifd(0x8769) if hasattr(ex, "get_ifd") else {}
        for k, n in {33434: "exposure_s", 34855: "iso", 37386: "focal_mm", 41989: "focal_35mm", 33437: "f_number"}.items():
            if k in sub:
                v = sub[k]
                try:
                    v = float(v)
                except Exception:
                    v = str(v)
                out[n] = v
        return out
    except Exception:
        return {}


def mccamy_cct(rgb_srgb01) -> float:
    """Correlated colour temperature of an sRGB colour (McCamy 1992 approximation)."""
    lin = np.where(rgb_srgb01 <= 0.04045, rgb_srgb01 / 12.92, ((rgb_srgb01 + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    X, Y, Z = M @ lin
    s = X + Y + Z
    if s <= 0:
        return float("nan")
    x, y = X / s, Y / s
    n = (x - 0.3320) / (0.1858 - y)
    return float(449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33)


def _load(path, max_side=2400):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    im = im[..., ::-1]
    h, w = im.shape[:2]
    k = max_side / max(h, w)
    if k < 1:
        im = cv2.resize(im, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(im)


def background(gray: np.ndarray) -> np.ndarray:
    """Paper background (ink and grid removed) by a large max filter + blur."""
    k = max(15, int(min(gray.shape) / 40) | 1)
    bg = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.GaussianBlur(bg, (0, 0), k)


def measure_photo(path: str, crop_dir: str | None = None) -> dict:
    img = _load(path)
    H, W = img.shape[:2]
    f = img.astype(np.float32)
    gray = f.mean(-1)
    bg = background(gray)
    rel = gray / np.maximum(bg, 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    page = page_mask(img)
    inner = cv2.erode(page.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    # paper: near the local background, low saturation, inside the sheet
    paper = inner & (rel > 0.93) & (hsv[..., 1] < 70)
    page_cov = float(page.mean())
    prgb = np.median(f[paper], axis=0) if paper.any() else np.array([np.nan] * 3)
    # ink: much darker than local paper and bluish (or just dark for black pens)
    dark = rel < 0.72
    blue = (f[..., 2] > f[..., 0] + 8)
    ink = inner & dark & (blue | (rel < 0.5))
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    res = {"file": os.path.basename(path), "size": [int(W), int(H)]}
    res["jpeg"] = jpeg_quality(path)
    res["exif"] = exif(path)
    res["paper_rgb"] = [float(v) for v in prgb]
    res["paper_cct_K"] = mccamy_cct(np.clip(prgb / 255, 0, 1)) if paper.any() else None
    if paper.any():
        lin = np.where(prgb / 255 <= 0.04045, prgb / 255 / 12.92, ((prgb / 255 + 0.055) / 1.055) ** 2.4)
        res["wb_gains_observed"] = [float(v) for v in lin / lin[1]]
    res["page_coverage"] = page_cov
    if ink.sum() > 50:
        ip = f[ink]
        lum = ip.mean(-1)
        core = ip[lum <= np.percentile(lum, 30)]
        res["ink_rgb_core"] = [float(v) for v in np.median(core, axis=0)]
        res["ink_rgb_mean"] = [float(v) for v in np.median(ip, axis=0)]
        bgc = np.stack([cv2.dilate(f[..., c], np.ones((15, 15), np.uint8)) for c in range(3)], -1)[ink]
        ratio = ip / np.maximum(bgc, 1)
        t = np.median(ratio[lum <= np.percentile(lum, 30)], axis=0)   # stroke cores (full coverage)
        res["ink_transmittance"] = [float(v) for v in t]
        # the single-stroke colour on white paper (what ink.Pen.rgb means)
        res["ink_on_white_rgb"] = [float(v) for v in np.clip(t * 255, 0, 255)]
        # mean line width = ink area / centre-line length (soft coverage counted fractionally)
        cov = np.clip((1 - rel) / max(1e-3, 1 - float(np.median(rel[ink]) if ink.any() else 0.5)), 0, 1)
        cov = cov * cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8))
        from .ink import _skeleton
        sk = _skeleton(ink.astype(np.uint8))
        res["ink_width_px"] = float(cov.sum() / max(1, sk.sum() * 1.12))
        res["ink_pixel_share"] = float(ink.mean())
        res["ink_lum_iqr"] = float(np.subtract(*np.percentile(lum, [75, 25])))
        if crop_dir:
            ys, xs = np.nonzero(ink)
            k = np.argmax(cv2.blur(ink.astype(np.float32), (151, 151))[ys, xs])
            cy, cx = ys[k], xs[k]
            y0, x0 = max(0, cy - 150), max(0, cx - 250)
            os.makedirs(crop_dir, exist_ok=True)
            cv2.imwrite(os.path.join(crop_dir, f"ink_crop_{os.path.splitext(res['file'])[0]}.png"),
                        img[y0:y0 + 300, x0:x0 + 500][..., ::-1])
    # noise: high-pass residual in flat paper areas
    flat = paper & (cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 1.0), cv2.CV_32F) ** 2 < 4)
    flat = cv2.erode(flat.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if flat.sum() > 500:
        ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        out = {}
        for i, n in enumerate(("Y", "Cr", "Cb")):
            hp = ycc[..., i] - cv2.GaussianBlur(ycc[..., i], (0, 0), 2.0)
            v = hp[flat]
            out[n] = float(np.std(np.clip(v, -4 * v.std(), 4 * v.std())) * 1.08)
        res["noise_sigma"] = out
    # blur: edge spread across strong ink edges
    gs = cv2.GaussianBlur(gray, (0, 0), 0.3)
    gx = cv2.Sobel(gs, cv2.CV_32F, 1, 0, ksize=3) / 8; gy = cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3) / 8
    gm = np.hypot(gx, gy)
    edge = (gm > np.percentile(gm, 99)) & (cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    if edge.sum() > 50:
        contrast = np.maximum(bg - cv2.erode(gray, np.ones((7, 7), np.uint8)), 1)
        sig = contrast[edge] / (np.sqrt(2 * np.pi) * gm[edge])
        res["edge_sigma_px"] = float(np.median(sig))
    res["laplacian_var"] = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    # tilt: projection-profile skew (Postl 1986) on the central part of the sheet, where grid
    # and text lines dominate; the vertical-line angle near the left and right side of the
    # sheet differs under keystone, so it is measured on both halves
    # printed grid / ruling is the reliable reference: faint, long, straight. Ink is clipped
    # so handwriting (whose baselines drift) does not dominate the profile.
    darkness = np.clip(1 - rel, 0, 0.2) * inner * (~cv2.dilate(ink.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool))
    res["rotation_deg"] = skew_angle(darkness, H, W, (0.2, 0.8, 0.15, 0.85))
    vl = skew_angle(darkness.T, W, H, (0.15, 0.85, 0.1, 0.45))
    vr = skew_angle(darkness.T, W, H, (0.15, 0.85, 0.55, 0.9))
    if vl is not None and vr is not None:
        res["vertical_convergence_deg"] = float(vr - vl)
    # lighting: plane + radial fit on the paper background
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    m = paper & (bg > np.percentile(bg, 20))
    if m.sum() > 1000:
        sel = np.flatnonzero(m.ravel())[:: max(1, int(m.sum() // 20000))]
        u = (xx.ravel()[sel] / W - 0.5); v = (yy.ravel()[sel] / H - 0.5)
        r2 = (u ** 2 + v ** 2) / 0.5
        A = np.c_[np.ones_like(u), u, v, r2]
        coef, *_ = np.linalg.lstsq(A, bg.ravel()[sel], rcond=None)
        res["light_gradient"] = float(np.hypot(coef[1], coef[2]) / coef[0])
        res["light_direction_deg"] = float(np.degrees(np.arctan2(coef[2], coef[1])))
        res["vignette"] = float(-coef[3] / coef[0])
        res["paper_lum_p5_p95"] = [float(np.percentile(bg[m], 5)), float(np.percentile(bg[m], 95))]
    return res


def page_mask(img: np.ndarray) -> np.ndarray:
    """The sheet: largest bright, low-saturation connected region, holes filled."""
    small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    v = cv2.GaussianBlur(hsv[..., 2], (0, 0), 3)
    sat = cv2.GaussianBlur(hsv[..., 1], (0, 0), 3)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ((v >= t) & (sat < 80)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.ones(img.shape[:2], bool)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    comp = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    return cv2.resize(filled, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def skew_angle(dark: np.ndarray, H: int, W: int, box, max_deg: float = 10.0, step: float = 0.1):
    """Angle (deg, positive = lines fall to the right in image coordinates) maximising the
    variance of the row-sum profile of ``dark`` inside box=(y0,y1,x0,x1) fractions."""
    y0, y1, x0, x1 = int(box[0] * H), int(box[1] * H), int(box[2] * W), int(box[3] * W)
    crop = dark[y0:y1, x0:x1]
    if crop.size == 0 or crop.sum() < 10:
        return None
    crop = crop.astype(np.float32)
    # band-pass: removes JPEG 8x8 block edges (they sit at exactly 0 deg) and lighting
    crop = cv2.GaussianBlur(crop, (0, 0), 2.0)
    crop = np.clip(crop - cv2.GaussianBlur(crop, (0, 0), 7), 0, None)
    k = 1000 / max(crop.shape)
    if k < 1:
        crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    h, w = crop.shape
    best, bv = None, -1
    xs = np.arange(w) - w / 2
    for a in np.arange(-max_deg, max_deg + 1e-9, step):
        M = np.float32([[1, 0, 0], [-np.tan(np.radians(a)), 1, 0]])
        r = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_LINEAR)
        prof = r[int(0.1 * h):int(0.9 * h)].sum(1)
        v = float(np.var(prof))
        if v > bv:
            best, bv = float(a), v
    return best


def _med(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.median(v)) if v else None


def aggregate(per: list) -> dict:
    g = lambda k, sub=None: [((p.get(k) or {}).get(sub) if sub else p.get(k)) for p in per]
    agg = {
        "jpeg_quality": _med(g("jpeg", "quality")),
        "paper_rgb": [_med([p["paper_rgb"][i] for p in per if "paper_rgb" in p]) for i in range(3)],
        "paper_cct_K": _med(g("paper_cct_K")),
        "ink_rgb_core": [_med([p["ink_rgb_core"][i] for p in per if "ink_rgb_core" in p]) for i in range(3)],
        "ink_on_white_rgb": [_med([p["ink_on_white_rgb"][i] for p in per if "ink_on_white_rgb" in p]) for i in range(3)],
        "ink_width_px": _med(g("ink_width_px")),
        "noise_sigma_Y": _med(g("noise_sigma", "Y")),
        "noise_sigma_C": _med([_med([(p.get("noise_sigma") or {}).get("Cr"), (p.get("noise_sigma") or {}).get("Cb")]) for p in per]),
        "edge_sigma_px": _med(g("edge_sigma_px")),
        "rotation_deg_abs": _med([abs(x) for x in g("rotation_deg") if x is not None]),
        "vertical_convergence_deg_abs": _med([abs(x) for x in g("vertical_convergence_deg") if x is not None]),
        "light_gradient": _med(g("light_gradient")),
        "vignette": _med(g("vignette")),
        "page_coverage": _med(g("page_coverage")),
        "size": per[0]["size"] if per else None,
    }
    return agg


def to_capture_params(agg: dict) -> dict:
    """Map measured aggregate numbers onto phone_capture.CaptureParams fields."""
    cp = {}
    if agg.get("jpeg_quality"):
        cp["jpeg_quality"] = int(round(agg["jpeg_quality"]))
    if agg.get("size"):
        w, h = agg["size"]
        cp["out_size"] = [int(max(w, h)), int(min(w, h))] if h >= w else [int(h), int(w)]
    if agg.get("rotation_deg_abs") is not None:
        cp["roll_deg"] = round(max(0.3, 1.25 * agg["rotation_deg_abs"]), 2)   # |N(0,s)| median = 0.67 s
    if agg.get("vertical_convergence_deg_abs") is not None:
        cp["pitch_deg"] = round(max(1.0, 1.5 * agg["vertical_convergence_deg_abs"] * 2), 2)
    if agg.get("light_gradient") is not None:
        cp["light_gradient"] = round(float(np.clip(agg["light_gradient"], 0.02, 0.6)), 3)
    if agg.get("vignette") is not None:
        cp["vignette"] = round(float(np.clip(agg["vignette"], 0.0, 0.5)), 3)
    if agg.get("noise_sigma_Y") is not None:
        # sigma (8-bit, sRGB, after ISP) -> linear shot noise term at paper level ~0.8
        s = agg["noise_sigma_Y"] / 255
        lin_sd = s * (0.8 ** (1 - 1 / 2.4)) * 2.4 / 1.055   # d(lin)/d(srgb) at 0.8 linear
        cp["noise_shot"] = float(lin_sd ** 2 / 0.8 * 0.9)
        cp["noise_read"] = float(lin_sd ** 2 * 0.02)
    if agg.get("edge_sigma_px") is not None:
        cp["base_blur"] = round(float(np.clip(np.sqrt(max(0.0, agg["edge_sigma_px"] ** 2 - 0.35 ** 2)), 0.2, 2.5)), 3)
    if agg.get("paper_rgb") and agg["paper_rgb"][0] is not None:
        prgb = np.array(agg["paper_rgb"]) / 255
        lin = np.where(prgb <= 0.04045, prgb / 12.92, ((prgb + 0.055) / 1.055) ** 2.4)
        cp["wb_gains"] = [round(float(v), 4) for v in lin / lin[1] * (0.957, 1.0, 1.089)]  # rel. to paper stock
        cp["exposure"] = round(float(lin[1]), 3)
    if agg.get("page_coverage") is not None:
        cp["page_fill"] = round(float(np.clip(np.sqrt(agg["page_coverage"]) * 1.02, 0.6, 1.0)), 3)
    return cp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+")
    ap.add_argument("-o", "--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json"))
    ap.add_argument("--crops", default=None, help="write an ink crop per photo here (for side-by-side checks)")
    a = ap.parse_args(argv)
    per = []
    for p in a.photos:
        try:
            per.append(measure_photo(p, a.crops))
            print("measured", p, file=sys.stderr)
        except Exception as e:
            print("skip", p, e, file=sys.stderr)
    agg = aggregate(per)
    doc = {"status": "measured", "n_photos": len(per), "aggregate": agg,
           "capture_params": to_capture_params(agg),
           "ink": {"ballpoint-blue_rgb": agg["ink_on_white_rgb"], "ink_width_px_at_photo": agg["ink_width_px"]},
           "per_photo": per}
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=2)
    print(json.dumps(doc["capture_params"], indent=2))


if __name__ == "__main__":
    main()

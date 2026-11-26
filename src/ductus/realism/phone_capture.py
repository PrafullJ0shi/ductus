"""Turn a clean, flat page image into a realistic phone photo of that page.

The pipeline follows the physical order of image formation used by document-image
synthesis work (DocCreator, Journet et al. 2017; Augraphy, Groleau et al. 2023; Doc3D /
DewarpNet, Das et al. 2019): reflectance -> sheet geometry -> illumination -> optics ->
sensor -> ISP -> JPEG. Light is multiplied in linear RGB, noise is Poisson-Gaussian on the
linear signal (Foi et al. 2008), the ISP then denoises chroma, applies a tone curve and
sharpening, and the result is JPEG-compressed with 4:2:0 chroma subsampling.

Main entry points
-----------------
``make_paper(height, width, dpi, kind="grid")``   blank grid / ruled / plain sheet (RGB)
``add_show_through(page, back=None, strength)``    faint mirrored writing from the back
``capture(page, seed, params=None, **overrides)``  -> (photo RGB uint8, info dict)
``CaptureParams.from_calibration(path)``           defaults measured from real photos

All randomness comes from ``seed``; every effect can be turned off by setting its strength
to 0.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, replace

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CALIBRATION = os.path.join(HERE, "calibration.json")


def srgb_to_linear(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x):
    x = np.clip(np.asarray(x, np.float32), 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def _smooth(h, w, rng, cell):
    g = rng.standard_normal((max(2, int(h / cell) + 3), max(2, int(w / cell) + 3))).astype(np.float32)
    n = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    return n / max(1e-6, float(n.std()))


# ----------------------------------------------------------------------------- parameters
@dataclass
class CaptureParams:
    out_size: tuple = (1920, 1440)       # (height, width) of the photo, portrait 3:4
    page_fill: float = 0.86              # share of the frame height the page spans
    # geometry
    pitch_deg: float = 7.0               # camera tilt top/bottom (keystone), sd
    yaw_deg: float = 4.0                 # camera tilt left/right, sd
    roll_deg: float = 1.6                # in-plane rotation, sd
    focal: float = 0.78                  # focal length / frame width (26 mm-equiv main camera)
    curl: float = 0.018                  # page lift near the binding, share of page width
    curl_width: float = 0.14             # how far from the binding the curl reaches
    binding: str = "left"                # left | top | none | loose (single sheet: corner/edge lift)
    spiral: bool = True
    page_edge: bool = True               # visible page edge, thickness shadow, pages below
    desk: bool = True
    desk_rgb: tuple | None = None        # None = random from a palette
    # light
    light_gradient: float = 0.22         # fall-off across the page (0..1)
    shadow: float = 0.28                 # hand / phone shadow darkness (0 = none)
    shadow_softness: float = 0.10        # penumbra as share of frame width
    exposure: float = 0.80               # linear paper level after auto exposure
    wb_gains: tuple = (1.04, 1.0, 0.90)  # residual white-balance error (R, G, B) in linear light
    wb_jitter: float = 0.025
    vignette: float = 0.22               # corner fall-off (cos^4-like)
    # optics / sensor / ISP
    defocus: float = 1.1                 # blur sigma (px at output) at the farthest depth
    base_blur: float = 0.55              # lens + demosaic softness everywhere (px)
    chroma_aberration: float = 0.0008    # lateral CA, relative scale of R vs B
    noise_shot: float = 0.0009           # Poisson-Gaussian: var = a*x + b (linear 0..1)
    noise_read: float = 0.000012
    chroma_denoise: float = 2.0          # ISP chroma smoothing sigma (px)
    sharpen: float = 0.55                # unsharp-mask amount
    sharpen_radius: float = 1.3
    tone_contrast: float = 0.12          # S-curve strength on top of sRGB gamma
    jpeg_quality: int = 88
    show_through: float = 0.03
    extra: dict = field(default_factory=dict)

    @classmethod
    def surface(cls, name: str = "a4", path: str | None = None) -> "CaptureParams":
        """Capture presets on top of the calibration: ``notebook`` (spiral-bound pad, pages
        below, curl at the binding) or ``a4`` (one loose sheet on a desk, corner lift)."""
        P = cls.from_calibration(path)
        if name == "notebook":
            return replace(P, binding="left", spiral=True, page_edge=True, page_fill=min(P.page_fill, 0.86))
        if name == "a4":
            return replace(P, binding="loose", spiral=False, page_edge=False, curl=0.02, curl_width=0.25,
                           page_fill=min(P.page_fill, 0.80), pitch_deg=min(P.pitch_deg, 7.0), show_through=0.0)
        raise ValueError(f"unknown surface {name!r} (use a4 or notebook)")

    @classmethod
    def from_calibration(cls, path: str | None = None) -> "CaptureParams":
        path = path or CALIBRATION
        p = cls()
        if not os.path.exists(path):
            return p
        with open(path) as f:
            cal = json.load(f)
        cap = cal.get("capture_params", {})
        names = {f.name for f in fields(cls)}
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cap.items() if k in names}
        return replace(p, **kw)


# ----------------------------------------------------------------------------- paper
def make_paper(height: int, width: int, dpi: float = 300, kind: str = "grid", seed: int = 0,
               paper_rgb=(244, 243, 236), line_rgb=(178, 196, 214), pitch_mm: float = 5.0,
               line_mm: float = 0.12, margin_rgb=(222, 150, 160), margin_mm: float | None = None,
               top_band_mm: float = 0.0) -> np.ndarray:
    """A blank notebook sheet: ``kind`` in grid | ruled | plain | dotted. Lines are printed
    (slightly uneven ink, soft edges) and the paper has low-frequency cloudiness and fibres."""
    rng = np.random.default_rng(seed)
    pxmm = dpi / 25.4
    img = np.ones((height, width, 3), np.float32) * np.asarray(paper_rgb, np.float32)
    cloud = 0.010 * _smooth(height, width, rng, 30 * pxmm) + 0.004 * _smooth(height, width, rng, 2 * pxmm)
    fib = rng.standard_normal((height, width)).astype(np.float32)
    fib = cv2.GaussianBlur(fib, (0, 0), 0.6) * 0.006
    img *= (1 + cloud + fib)[..., None]
    if kind in ("grid", "ruled", "dotted"):
        m = np.zeros((height, width), np.float32)
        lw = max(1, int(round(line_mm * pxmm)))
        step = pitch_mm * pxmm if kind != "ruled" else 7.1 * pxmm
        off = rng.uniform(0, step)
        y = off + top_band_mm * pxmm
        while y < height:
            if kind == "dotted":
                x = off
                while x < width:
                    cv2.circle(m, (int(x), int(y)), lw, 1.0, -1, cv2.LINE_AA); x += step
            else:
                cv2.line(m, (0, int(round(y))), (width, int(round(y))), 1.0, lw, cv2.LINE_AA)
            y += step
        if kind == "grid":
            x = rng.uniform(0, step)
            while x < width:
                cv2.line(m, (int(round(x)), 0), (int(round(x)), height), 1.0, lw, cv2.LINE_AA); x += step
        m = cv2.GaussianBlur(m, (0, 0), 0.5) * (0.85 + 0.15 * _smooth(height, width, rng, 15 * pxmm)).clip(0.5, 1.2)
        line = np.asarray(line_rgb, np.float32) / np.asarray(paper_rgb, np.float32)
        img *= 1 - m[..., None] * (1 - line)[None, None, :]
        if kind == "ruled" or margin_mm:
            mx = int((margin_mm or 25) * pxmm)
            mm = np.zeros((height, width), np.float32)
            cv2.line(mm, (mx, 0), (mx, height), 1.0, lw + 1, cv2.LINE_AA)
            mc = np.asarray(margin_rgb, np.float32) / np.asarray(paper_rgb, np.float32)
            img *= 1 - mm[..., None] * (1 - mc)[None, None, :]
    return np.clip(img, 0, 255).astype(np.uint8)


def add_show_through(page: np.ndarray, back: np.ndarray | None = None, strength: float = 0.03,
                     seed: int = 0, blur_px: float = 2.5) -> np.ndarray:
    """Writing on the back of the sheet seen through the paper: mirrored, blurred, faint.
    ``back`` is the other side (same size); if None the front is reused with a shift."""
    if strength <= 0:
        return page
    rng = np.random.default_rng(seed)
    src = page if back is None else back
    g = src.astype(np.float32).mean(axis=2)
    ink = np.clip(1 - g / max(1.0, np.percentile(g, 90)), 0, 1)
    ink = cv2.GaussianBlur(ink[:, ::-1], (0, 0), blur_px)
    if back is None:
        ink = np.roll(ink, int(rng.integers(-page.shape[0] // 6, page.shape[0] // 6)), axis=0)
    out = page.astype(np.float32) * (1 - strength * 4 * ink[..., None] * np.array([1.0, 0.97, 0.9]))
    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- desk & binding
DESK_PALETTE = [(120, 92, 68), (150, 126, 100), (74, 72, 76), (196, 188, 176), (96, 66, 46), (176, 170, 160)]


def _desk(h, w, rng, rgb=None):
    """Table top: low-frequency tone changes, fine grain and (for wood) gentle wavy grain lines."""
    base = np.asarray(rgb if rgb is not None else DESK_PALETTE[int(rng.integers(len(DESK_PALETTE)))], np.float32)
    grain = 0.4 * _smooth(h, w, rng, 2) + 0.6 * _smooth(h, w, rng, 12) + _smooth(h, w, rng, 120)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    warp = 25 * _smooth(h, w, rng, 200)
    wood = np.sin((xx + warp) / rng.uniform(9, 16)) * _smooth(h, w, rng, 80).clip(0, None) * 0.5
    d = base * (1 + 0.025 * grain[..., None] + 0.03 * wood[..., None] * (rng.random() < 0.6))
    return srgb_to_linear(np.clip(d, 0, 255) / 255)


def _spiral(canvas_lin, mask, pad, h, w, binding, rng, scale):
    """Punched holes on the sheet and wire coils crossing the binding edge (in flat page space)."""
    n = int((h if binding == "left" else w) / (8.5 * scale)) if scale > 0 else 20
    n = max(8, n)
    hole_r = 1.6 * scale
    for i in range(n):
        t = (i + 0.5) / n
        if binding == "left":
            cx, cy = pad + 5.0 * scale, pad + t * h
        else:
            cx, cy = pad + t * w, pad + 5.0 * scale
        # hole: desk shows through, but in shadow
        cv2.circle(mask, (int(cx), int(cy)), int(hole_r), 0.0, -1, cv2.LINE_AA)
        # coil: a loop from the hole out over the edge
        if binding == "left":
            p0, p1 = (cx, cy), (pad - 5 * scale, cy - 2.4 * scale)
        else:
            p0, p1 = (cx, cy), (cx - 2.4 * scale, pad - 5 * scale)
        thick = max(2, int(0.9 * scale))
        for k, shade in ((1.25, 0.05), (1.0, 0.32), (0.45, 0.65)):
            cv2.line(canvas_lin, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                     (shade, shade, shade * 1.03), max(1, int(thick * k)), cv2.LINE_AA)
            cv2.line(mask, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 2.0, max(1, int(thick * k)), cv2.LINE_AA)


# ----------------------------------------------------------------------------- capture
def _rot(pitch, yaw, roll):
    a, b, c = np.radians([pitch, yaw, roll])
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def capture(page: np.ndarray, seed: int = 0, params: CaptureParams | None = None, back: np.ndarray | None = None,
            **overrides):
    """Photograph ``page`` (RGB uint8, the whole sheet) with a phone. Returns (photo, info)."""
    P = params or CaptureParams.from_calibration()
    if overrides:
        P = replace(P, **overrides)
    rng = np.random.default_rng(seed)
    OH, OW = P.out_size
    h0, w0 = page.shape[:2]

    # 1. reflectance at roughly the output scale (x1.5 to keep detail through the warp)
    s = min(1.0, 1.25 * P.page_fill * OH / h0)
    pg = cv2.resize(page, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA) if s < 1 else page.copy()
    if P.show_through > 0:
        bk = None if back is None else cv2.resize(back, (pg.shape[1], pg.shape[0]), interpolation=cv2.INTER_AREA)
        pg = add_show_through(pg, bk, P.show_through, seed)
    h, w = pg.shape[:2]
    refl = srgb_to_linear(pg.astype(np.float32) / 255)

    # 2. sheet on the desk (flat page space with padding)
    pad = int(0.45 * w)
    H, W = h + 2 * pad, w + 2 * pad
    canvas = _desk(H, W, rng, P.desk_rgb) if P.desk else np.full((H, W, 3), 0.9, np.float32)
    mask = np.zeros((H, W), np.float32)
    if P.binding == "loose":
        # a single sheet lying on the desk: only a thin contact shadow along its edges
        sh = np.zeros((H, W), np.float32)
        cv2.rectangle(sh, (pad, pad), (pad + w, pad + h), 1.0, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), w / 400)
        canvas *= (1 - 0.22 * sh)[..., None]
    elif P.page_edge:
        # pages underneath peek out on two sides, and the sheet casts a thin shadow
        dx, dy = rng.uniform(1, 3, 2) * w / 600
        under = np.zeros((H, W), np.float32)
        cv2.rectangle(under, (int(pad + dx), int(pad + dy)), (int(pad + w + dx), int(pad + h + dy)), 1.0, -1)
        sh = cv2.GaussianBlur(under, (0, 0), w / 250)
        canvas *= (1 - 0.35 * sh)[..., None]
        edge_col = srgb_to_linear(np.array([0.90, 0.89, 0.86], np.float32))
        canvas = np.where(under[..., None] > 0.5, edge_col * 0.92, canvas)
    canvas[pad:pad + h, pad:pad + w] = refl
    mask[pad:pad + h, pad:pad + w] = 1
    scale = w / 210.0  # px per mm on this sheet (A4 width 210 mm)
    if P.spiral and P.binding in ("left", "top"):
        hole_mask = mask.copy()
        _spiral(canvas, hole_mask, pad, h, w, P.binding, rng, scale)
        holes = (mask > 0.5) & (hole_mask < 0.5)
        canvas[holes] *= 0.35
        mask = np.where(hole_mask > 1.5, 1.0, mask)

    # 3. curl near the binding: the sheet lifts; displacement plus a shading band
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    depth = np.zeros((H, W), np.float32)
    shade = np.ones((H, W), np.float32)
    if P.binding in ("left", "top") and P.curl > 0:
        u = (xx - pad) / w if P.binding == "left" else (yy - pad) / h
        lift = np.clip(1 - u / max(1e-3, P.curl_width), 0, 1) ** 2 * (u > -0.02)
        amp = P.curl * w * rng.uniform(0.6, 1.3)
        wav = 1 + 0.25 * np.sin((yy if P.binding == "left" else xx) / (h if P.binding == "left" else w) * np.pi * rng.uniform(1, 3))
        if P.binding == "left":
            mx, my = xx + amp * lift * wav, yy
        else:
            mx, my = xx, yy + amp * lift * wav
        canvas = cv2.remap(canvas, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = cv2.remap(mask, mx, my, cv2.INTER_LINEAR)
        slope = np.gradient(lift, axis=1 if P.binding == "left" else 0)
        shade = 1 - 0.25 * lift + 6.0 * slope * rng.uniform(-1, 1)
        depth = -amp * lift
        canvas *= np.where(mask[..., None] > 0.5, np.clip(shade, 0.6, 1.1)[..., None], 1.0)

    if P.binding == "loose" and P.curl > 0:
        # loose sheet: one or two corners lift off the desk (paper memory), plus a gentle bow
        cxp, cyp = pad + w / 2, pad + h / 2
        nx, ny = (xx - cxp) / (w / 2), (yy - cyp) / (h / 2)          # -1..1 over the sheet
        amp_tot = np.zeros((H, W), np.float32)
        corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        for k in rng.choice(4, size=int(rng.integers(1, 3)), replace=False):
            sx_, sy_ = corners[k]
            d = np.sqrt(((nx - sx_) / 2) ** 2 + ((ny - sy_) / 2) ** 2)    # 0 at the corner
            reach = P.curl_width * rng.uniform(1.0, 2.0)
            amp_tot += (np.clip(1 - d / reach, 0, 1) ** 2) * P.curl * w * rng.uniform(1.0, 2.2)
        bow = P.curl * w * 0.25 * rng.uniform(0, 1) * (1 - np.clip(nx, -1, 1) ** 2)
        lift = amp_tot + bow
        # a lifted part seen from above shifts away from the sheet centre (perspective of height)
        k_par = 0.10
        mx, my = xx - k_par * lift * np.sign(nx) * np.abs(nx), yy - k_par * lift * np.sign(ny) * np.abs(ny)
        # the lifted corner casts a soft shadow on the desk next to it
        ins = ((np.abs(nx) <= 1) & (np.abs(ny) <= 1)).astype(np.float32)
        cast = cv2.GaussianBlur(ins * amp_tot / max(1e-6, float(amp_tot.max() or 1)), (0, 0), w / 60)
        canvas *= np.where(ins[..., None] > 0.5, 1.0, (1 - 0.35 * cast)[..., None])
        canvas = cv2.remap(canvas, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        mask = cv2.remap(mask, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR)
        gy, gx = np.gradient(lift / w)
        shade = 1 + 4.0 * (gx * rng.uniform(-1, 1) + gy * rng.uniform(-1, 1)) - 0.6 * lift / w
        depth = -lift
        canvas *= np.where(mask[..., None] > 0.5, np.clip(shade, 0.75, 1.08)[..., None], 1.0)

    # 4. perspective camera
    pitch = rng.normal(0, P.pitch_deg); yaw = rng.normal(0, P.yaw_deg); roll = rng.normal(0, P.roll_deg)
    if rng.random() < 0.5:
        roll += 90 * 0  # portrait page in a portrait frame
    R = _rot(pitch, yaw, roll)
    f = P.focal * OW
    Z0 = f * h / (P.page_fill * OH)
    cx_, cy_ = W / 2 + rng.normal(0, 0.02) * w, H / 2 + rng.normal(0, 0.02) * h
    def project(px, py, pz=0.0):
        X = np.stack([px - cx_, py - cy_, np.broadcast_to(pz, np.shape(px))], -1) @ R.T
        Z = X[..., 2] + Z0
        return np.stack([f * X[..., 0] / Z + OW / 2, f * X[..., 1] / Z + OH / 2], -1), Z
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst, _ = project(src[:, 0], src[:, 1])
    Hm = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    img = cv2.warpPerspective(canvas, Hm, (OW, OH), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    pmask = cv2.warpPerspective(mask, Hm, (OW, OH), flags=cv2.INTER_LINEAR)
    # depth at each output pixel (for defocus): z of the plane + curl lift
    corners, Zc = project(np.float32([pad, pad + w, pad + w, pad]), np.float32([pad, pad, pad + h, pad + h]))
    Hinv = np.linalg.inv(Hm)
    oy, ox = np.mgrid[0:OH, 0:OW].astype(np.float32)
    q = np.stack([ox, oy, np.ones_like(ox)], -1) @ Hinv.T
    sx, sy = (q[..., 0] / q[..., 2]).astype(np.float32), (q[..., 1] / q[..., 2]).astype(np.float32)
    _, Zmap = project(sx, sy)
    if P.curl > 0 and P.binding in ("left", "top", "loose"):
        Zmap = Zmap + cv2.remap(depth, sx, sy, cv2.INTER_LINEAR)
    page_pix = pmask > 0.5
    z_focus = float(np.median(Zmap[page_pix])) if page_pix.any() else Z0
    dz = np.abs(Zmap - z_focus) / max(1e-6, float(np.abs(Zmap - z_focus).max()))

    # 5. illumination (linear light)
    ang = rng.uniform(0, 2 * np.pi)
    g = ((ox / OW - 0.5) * np.cos(ang) + (oy / OH - 0.5) * np.sin(ang))
    light = 1 - P.light_gradient * (g + 0.5) * 0.9 + 0.03 * _smooth(OH, OW, rng, 300)
    if P.shadow > 0:
        sh = np.zeros((OH // 4, OW // 4), np.float32)
        side = int(rng.integers(4))
        pts = _shadow_polygon(OW // 4, OH // 4, side, rng)
        cv2.fillPoly(sh, [pts], 1.0, cv2.LINE_AA)
        sh = cv2.GaussianBlur(sh, (0, 0), P.shadow_softness * OW / 4)
        sh = cv2.resize(sh, (OW, OH), interpolation=cv2.INTER_CUBIC)
        light *= 1 - P.shadow * np.clip(sh, 0, 1)
    r2 = ((ox - OW / 2) ** 2 + (oy - OH / 2) ** 2) / ((OW / 2) ** 2 + (OH / 2) ** 2)
    light *= (1 - P.vignette * r2) ** 1.5
    img = img * light[..., None]

    # 6. exposure and white balance (auto exposure puts the paper at a fixed level)
    paper_level = float(np.percentile(img[page_pix].mean(-1), 75)) if page_pix.any() else 0.8
    img = img * (P.exposure / max(1e-4, paper_level))
    gains = np.asarray(P.wb_gains, np.float32) * (1 + rng.normal(0, P.wb_jitter, 3)).astype(np.float32)
    img = img * gains[None, None, :]

    # 7. optics: base softness, depth defocus, lateral chromatic aberration
    if P.base_blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), P.base_blur)
    if P.defocus > 0:
        b1 = cv2.GaussianBlur(img, (0, 0), P.defocus * 0.5)
        b2 = cv2.GaussianBlur(img, (0, 0), P.defocus)
        k = np.clip(dz * 1.0 + 0.6 * r2 - 0.25, 0, 1)[..., None]
        img = np.where(k < 0.5, img * (1 - 2 * k) + b1 * 2 * k, b1 * (2 - 2 * k) + b2 * (2 * k - 1))
    if P.chroma_aberration > 0:
        for ch, sgn in ((0, 1), (2, -1)):
            sc = 1 + sgn * P.chroma_aberration
            M = np.float32([[sc, 0, (1 - sc) * OW / 2], [0, sc, (1 - sc) * OH / 2]])
            img[..., ch] = cv2.warpAffine(np.ascontiguousarray(img[..., ch]), M, (OW, OH), borderMode=cv2.BORDER_REFLECT)

    # 8. sensor noise, Poisson-Gaussian in linear light, slightly correlated (demosaic)
    x = np.clip(img, 0, 1)
    sd = np.sqrt(P.noise_shot * x + P.noise_read)
    n = rng.standard_normal(x.shape).astype(np.float32)
    n = 0.7 * n + 0.3 * cv2.GaussianBlur(n, (0, 0), 0.7)
    x = x + sd * n

    # 9. ISP: chroma denoise, tone curve, sharpening
    srgb = linear_to_srgb(np.clip(x, 0, 1))
    ycc = cv2.cvtColor(srgb, cv2.COLOR_RGB2YCrCb)
    if P.chroma_denoise > 0:
        ycc[..., 1] = cv2.GaussianBlur(ycc[..., 1], (0, 0), P.chroma_denoise)
        ycc[..., 2] = cv2.GaussianBlur(ycc[..., 2], (0, 0), P.chroma_denoise)
    Y = ycc[..., 0]
    if P.tone_contrast > 0:
        Y = Y + P.tone_contrast * (Y - 0.5) * (1 - np.abs(2 * Y - 1)) * 2 * 0.5
    if P.sharpen > 0:
        Y = Y + P.sharpen * (Y - cv2.GaussianBlur(Y, (0, 0), P.sharpen_radius))
    ycc[..., 0] = np.clip(Y, 0, 1)
    out = np.clip(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB), 0, 1)
    out8 = (out * 255 + 0.5).astype(np.uint8)

    # 10. JPEG (4:2:0)
    if P.jpeg_quality and P.jpeg_quality < 100:
        ok, enc = cv2.imencode(".jpg", out8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(P.jpeg_quality),
                                                          cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420])
        out8 = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].copy()
    c0 = np.array([[pad + w / 2 - w / 4, pad + h / 2, 1], [pad + w / 2 + w / 4, pad + h / 2, 1]]) @ Hm.T
    c0 = c0[:, :2] / c0[:, 2:]
    line_ang = float(np.degrees(np.arctan2(c0[1, 1] - c0[0, 1], c0[1, 0] - c0[0, 0])))
    info = dict(centre_line_angle_deg=line_ang, pitch_deg=float(pitch), yaw_deg=float(yaw), roll_deg=float(roll), homography=Hm.tolist(),
                page_corners=corners.tolist(), page_scale=float(s), wb_gains=gains.tolist(), params=_jsonable(asdict(P)))
    return out8, info


def _shadow_polygon(W, H, side, rng):
    """A blob entering from one side of the frame: a hand, an arm or the phone itself."""
    c = np.array([W / 2, H / 2])
    if side == 0:   # from the bottom
        base = [(rng.uniform(0.1, 0.5) * W, H * 1.1), (rng.uniform(0.6, 1.1) * W, H * 1.1)]
        tip = (rng.uniform(0.3, 0.9) * W, rng.uniform(0.75, 0.95) * H)
    elif side == 1:  # right
        base = [(W * 1.1, rng.uniform(0.3, 0.6) * H), (W * 1.1, rng.uniform(0.8, 1.1) * H)]
        tip = (rng.uniform(0.75, 0.95) * W, rng.uniform(0.5, 0.9) * H)
    elif side == 2:  # left
        base = [(-0.1 * W, rng.uniform(0.4, 0.7) * H), (-0.1 * W, rng.uniform(0.8, 1.1) * H)]
        tip = (rng.uniform(0.05, 0.25) * W, rng.uniform(0.6, 0.9) * H)
    else:  # top corner (phone held above)
        base = [(rng.uniform(0.5, 0.8) * W, -0.1 * H), (W * 1.1, -0.1 * H)]
        tip = (rng.uniform(0.75, 0.95) * W, rng.uniform(0.05, 0.2) * H)
    pts = np.array([base[0], tip, base[1]], np.float32)
    # round it out
    t = np.linspace(0, 1, 24)
    curve = ((1 - t) ** 2)[:, None] * pts[0] + (2 * t * (1 - t))[:, None] * (2 * pts[1] - (pts[0] + pts[2]) / 2) + (t ** 2)[:, None] * pts[2]
    return np.round(curve).astype(np.int32)


def _jsonable(d):
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}

"""Phone-photo capture of a rendered page.

The page is treated as a sheet on a desk photographed with a phone, following the
reflectance x shading decomposition used to synthesise document photos (DocAligner,
Doc3D, Augraphy): geometry first (curl, keystone, rotation), then light (gradient, a soft
hand or phone shadow, white balance, vignetting), then the camera (edge defocus, noise,
sharpening, JPEG). Every effect is sampled per seed and can be switched off.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Capture:
    keystone: float = 0.022       # corner jitter as a share of page width
    rotate_deg: float = 1.5       # max in-plane rotation
    curl: float = 0.012           # page lift near the binding, share of width
    binding: str = "left"         # left | top | none (spiral or stapled edge)
    spiral_holes: bool = True
    desk: bool = True
    margin: float = 0.06          # desk visible around the page
    shadow: float = 0.22          # darkness of the hand/phone shadow (0 = none)
    light: float = 0.18           # strength of the uneven light
    tint: float = 0.035           # white balance error
    vignette: float = 0.18
    defocus: float = 1.6          # blur (px) at the far corners
    noise: float = 2.2
    show_through: float = 0.035   # faint mirrored writing from the back of the page
    finger: bool = False
    jpeg: int = 84


def _smooth_noise(h, w, rng, cell):
    g = rng.standard_normal((max(2, h // cell + 2), max(2, w // cell + 2))).astype(np.float32)
    n = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    return n / max(1e-6, float(np.abs(n).max()))


def _desk(h, w, rng):
    base = np.array(rng.choice([(118, 92, 70), (150, 128, 104), (70, 70, 74), (196, 190, 180), (92, 62, 44)]),
                    np.float32)
    grain = _smooth_noise(h, w, rng, 7) * 0.5 + _smooth_noise(h, w, rng, 90)
    # wood-like streaks along x
    streak = cv2.resize(rng.standard_normal((h // 3 + 2, 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    d = base[None, None, :] * (1 + 0.06 * grain[..., None] + 0.05 * streak[..., None])
    return np.clip(d, 0, 255)


def capture(page: np.ndarray, seed: int = 0, c: Capture | None = None) -> np.ndarray:
    c = c or Capture()
    rng = np.random.default_rng(seed)
    img = page.astype(np.float32)
    h, w = img.shape[:2]

    # back of the page shows through, mirrored and blurred
    if c.show_through > 0:
        back = cv2.GaussianBlur(255 - img.mean(axis=2), (0, 0), 3.0)[:, ::-1]
        back = np.roll(back, int(rng.integers(-h // 8, h // 8)), axis=0)
        img = img * (1 - c.show_through * back[..., None] / 255.0)

    # paper albedo: fibres and slight unevenness
    img *= (1 + 0.012 * _smooth_noise(h, w, rng, 40) + 0.006 * _smooth_noise(h, w, rng, 3))[..., None]

    if c.spiral_holes and c.binding in ("left", "top"):
        n = 22 if c.binding == "left" else 16
        for i in range(n):
            t = (i + 0.5) / n
            x, y = (int(0.018 * w), int(t * h)) if c.binding == "left" else (int(t * w), int(0.018 * w))
            cv2.ellipse(img, (x, y), (int(0.006 * w), int(0.0045 * w)), 0, 0, 360, (60, 58, 56), -1, cv2.LINE_AA)

    # page on the desk
    pad = int(c.margin * w) if c.desk else 0
    H, W = h + 2 * pad, w + 2 * pad
    canvas = _desk(H, W, rng) if c.desk else np.full((H, W, 3), 255, np.float32)
    mask = np.zeros((H, W), np.float32)
    canvas[pad:pad + h, pad:pad + w] = img
    mask[pad:pad + h, pad:pad + w] = 1

    # curl: the sheet lifts towards the binding -> shift and a shading band
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    mx, my = xx.copy(), yy.copy()
    shade = np.ones((H, W), np.float32)
    if c.binding in ("left", "top") and c.curl > 0:
        u = ((xx - pad) / w) if c.binding == "left" else ((yy - pad) / h)
        lift = np.clip(1 - u / 0.12, 0, 1) ** 2
        amp = c.curl * w
        if c.binding == "left":
            mx = xx + amp * lift
            my = yy + 0.25 * amp * lift * ((yy - H / 2) / H)
        else:
            my = yy + amp * lift
        shade *= 1 - 0.18 * lift
    canvas = cv2.remap(canvas, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = cv2.remap(mask, mx, my, cv2.INTER_LINEAR)
    canvas *= np.where(mask[..., None] > 0.5, shade[..., None], 1.0)

    # keystone + rotation
    j = lambda: rng.uniform(-c.keystone, c.keystone) * w
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = np.float32([[j(), j()], [W + j(), j()], [W + j(), H + j()], [j(), H + j()]])
    M = cv2.getPerspectiveTransform(src, dst)
    R = np.vstack([cv2.getRotationMatrix2D((W / 2, H / 2), rng.uniform(-c.rotate_deg, c.rotate_deg), 1.0), [0, 0, 1]])
    M = R @ M
    desk_fill = tuple(float(v) for v in canvas[2, 2])
    out = cv2.warpPerspective(canvas, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # light: broad gradient, then a soft shadow of the hand / phone from one side
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    grad = ((xx / W - 0.5) * np.cos(ang) + (yy / H - 0.5) * np.sin(ang))
    light = 1 - c.light * (grad + 0.5) * 0.8 + 0.04 * _smooth_noise(H, W, rng, 400)
    if c.shadow > 0:
        sh = np.zeros((H // 8, W // 8), np.float32)
        side = rng.integers(4)
        cx = [rng.uniform(0.6, 1.1), rng.uniform(-0.1, 0.4), rng.uniform(0.3, 0.8), rng.uniform(0.3, 0.8)][side] * W / 8
        cy = [rng.uniform(0.5, 1.1), rng.uniform(0.5, 1.1), rng.uniform(0.85, 1.15), rng.uniform(-0.15, 0.1)][side] * H / 8
        cv2.ellipse(sh, (int(cx), int(cy)), (int(W / 8 * rng.uniform(0.25, 0.45)), int(H / 8 * rng.uniform(0.18, 0.35))),
                    rng.uniform(0, 180), 0, 360, 1.0, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), W / 8 * 0.08)
        light *= 1 - c.shadow * cv2.resize(sh, (W, H))
    r2 = ((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2) / 0.5
    light *= 1 - c.vignette * r2
    out *= light[..., None]
    tint = 1 + rng.uniform(-c.tint, c.tint, 3).astype(np.float32)
    tint[2] -= abs(rng.normal(0, c.tint))          # indoor light is usually warm
    out *= tint[None, None, :]

    if c.finger:
        f = np.zeros((H, W), np.float32)
        x = int(rng.uniform(0.1, 0.9) * W)
        cv2.ellipse(f, (x, H + int(0.02 * H)), (int(0.05 * W), int(0.06 * H)), 0, 0, 360, 1.0, -1)
        f = cv2.GaussianBlur(f, (0, 0), 0.012 * W)[..., None]
        out = out * (1 - f) + np.array((196, 150, 128), np.float32) * f

    # camera: edge defocus, noise (more in shadows), phone sharpening, JPEG
    if c.defocus > 0:
        blur = cv2.GaussianBlur(out, (0, 0), c.defocus)
        k = np.clip(r2 * 1.2, 0, 1)[..., None]
        out = out * (1 - k) + blur * k
        out = cv2.GaussianBlur(out, (0, 0), 0.5)
    lum = out.mean(axis=2, keepdims=True)
    out += rng.normal(0, 1, out.shape).astype(np.float32) * c.noise * (1.4 - lum / 255.0)
    out += rng.normal(0, 1, out.shape[:2] + (1,)).astype(np.float32) * c.noise * 0.6
    soft = cv2.GaussianBlur(out, (0, 0), 1.2)
    out = out + 0.35 * (out - soft)
    out = np.clip(out, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(c.jpeg)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]

"""Hand-drawn pen marks: arrows, ticks, circles, rules, fraction bars, radicals, sigma...

Every mark is built from a nominal shape plus low-frequency wobble and random shape
parameters, so no two marks on a page are identical. Units are page pixels; `xh` is
the x-height of the writing, which sets the size of every mark.
"""
from __future__ import annotations

import math

import numpy as np


def wobble(pts: np.ndarray, rng: np.random.Generator, amp: float, freq: float = 1.0) -> np.ndarray:
    """Add smooth perpendicular noise to a polyline."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2 or amp <= 0:
        return pts
    seg = np.diff(pts, axis=0)
    L = np.r_[0, np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))]
    total = max(L[-1], 1e-6)
    t = L / total
    n = np.zeros(len(pts))
    for k in range(1, 4):
        n += rng.normal(0, 1) / k * np.sin(2 * math.pi * (k * freq * t + rng.random()))
    d = np.gradient(pts, axis=0)
    nrm = np.c_[-d[:, 1], d[:, 0]]
    nrm /= np.maximum(np.hypot(nrm[:, 0], nrm[:, 1]), 1e-9)[:, None]
    return pts + nrm * (n * amp)[:, None]


def resample(pts, step: float = 2.0) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    seg = np.diff(pts, axis=0)
    L = np.r_[0, np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))]
    if L[-1] < step:
        return pts
    s = np.arange(0, L[-1], step)
    return np.c_[np.interp(s, L, pts[:, 0]), np.interp(s, L, pts[:, 1])]


def bezier(p0, p1, p2, n=24):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = map(np.asarray, (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


class Marks:
    """Factory of stroke lists. Each method returns list[np.ndarray (N,2)] in page px,
    relative to an origin (x, baseline) and the width it occupies."""

    def __init__(self, rng: np.random.Generator, xh: float, mess: float = 0.4):
        self.r = rng
        self.xh = xh
        self.mess = mess

    def j(self, s=1.0):
        """jitter factor around 1"""
        return 1.0 + self.r.normal(0, 0.06 + 0.10 * self.mess) * s

    def line(self, p0, p1, amp=None, freq=1.0):
        amp = self.xh * (0.02 + 0.05 * self.mess) if amp is None else amp
        pts = resample([p0, p1], 3.0)
        return wobble(pts, self.r, amp, freq)

    # -- arrows ------------------------------------------------------------------
    def arrow(self, x, base, direction=1, double=False):
        xh, r = self.xh, self.r
        L = xh * (1.9 + 0.6 * r.random()) * self.j()
        y = base - xh * (0.45 + 0.08 * r.normal())
        tilt = r.normal(0, 0.05 + 0.05 * self.mess) * xh
        x0, x1 = (x, x + L) if direction > 0 else (x + L, x)
        strokes = []
        if double:
            g = xh * 0.16
            strokes.append(self.line((x0, y - g), (x1 - direction * xh * 0.15, y - g + tilt)))
            strokes.append(self.line((x0, y + g), (x1 - direction * xh * 0.15, y + g + tilt)))
        else:
            strokes.append(self.line((x0, y), (x1, y + tilt)))
        h = xh * (0.42 + 0.12 * r.random())
        a1 = math.radians(150 + r.normal(0, 8))
        a2 = math.radians(210 + r.normal(0, 8))
        tipx, tipy = x1, y + tilt
        d = direction
        head = [np.array([[tipx + d * h * math.cos(a1), tipy + h * math.sin(a1)], [tipx, tipy],
                          [tipx + d * h * math.cos(a2), tipy + h * math.sin(a2)]])]
        if r.random() < 0.5:   # sometimes the head is two separate strokes
            head = [head[0][:2], head[0][1:]]
        strokes += [wobble(resample(hd, 2.0), r, xh * 0.02) for hd in head]
        return strokes, L

    def down_arrow(self, x, top, length):
        xh, r = self.xh, self.r
        s = self.line((x, top), (x + r.normal(0, xh * 0.1), top + length))
        tip = s[-1]
        h = xh * 0.4
        head = np.array([[tip[0] - h * 0.6, tip[1] - h], tip, [tip[0] + h * 0.6, tip[1] - h]])
        return [s, wobble(resample(head, 2.0), r, xh * 0.02)]

    # -- marks -------------------------------------------------------------------
    def tick(self, x, base):
        xh, r = self.xh, self.r
        s = xh * (1.1 + 0.3 * r.random())
        p0 = (x, base - s * 0.45)
        p1 = (x + s * (0.30 + 0.05 * r.normal()), base + s * 0.05)
        p2 = (x + s * (0.95 + 0.1 * r.normal()), base - s * (1.15 + 0.15 * r.random()))
        pts = np.r_[bezier(p0, (p0[0] + s * 0.15, p1[1] - s * 0.1), p1, 10),
                    bezier(p1, (p1[0] + s * 0.2, p1[1] - s * 0.5), p2, 16)[1:]]
        return [wobble(pts, r, xh * 0.02)], s

    def cross(self, x, base):
        xh, r = self.xh, self.r
        s = xh * (0.9 + 0.2 * r.random())
        a = self.line((x, base - s), (x + s, base))
        b = self.line((x + s * (1 + 0.1 * r.normal()), base - s * 1.05), (x - s * 0.05, base + s * 0.05))
        return [a, b], s

    def eq(self, x, base):
        xh, r = self.xh, self.r
        w = xh * (0.9 + 0.25 * r.random())
        y1 = base - xh * (0.62 + 0.05 * r.normal())
        y2 = base - xh * (0.22 + 0.05 * r.normal())
        return [self.line((x, y1), (x + w, y1 + r.normal(0, xh * 0.04))),
                self.line((x + r.normal(0, xh * 0.05), y2), (x + w * self.j(), y2 + r.normal(0, xh * 0.04)))], w

    def lt(self, x, base, flip=False):
        xh = self.xh
        w = xh * 0.8
        pts = [(x + w, base - xh), (x, base - xh * 0.5), (x + w, base)]
        if flip:
            pts = [(x, base - xh), (x + w, base - xh * 0.5), (x, base)]
        return [wobble(resample(pts, 2), self.r, xh * 0.02)], w

    def pct(self, x, base):
        xh, r = self.xh, self.r
        w = xh * 1.0
        o1 = self._ellipse(x + w * 0.2, base - xh * 0.85, xh * 0.17, xh * 0.2, 1.0)
        o2 = self._ellipse(x + w * 0.8, base - xh * 0.15, xh * 0.17, xh * 0.2, 1.0)
        sl = self.line((x + w, base - xh * 1.05), (x, base + xh * 0.05))
        return [o1, sl, o2], w

    def tilde(self, x, base):
        xh = self.xh
        w = xh * 1.0
        t = np.linspace(0, 1, 30)
        pts = np.c_[x + w * t, base - xh * 0.5 - xh * 0.15 * np.sin(2 * math.pi * t)]
        return [wobble(pts, self.r, xh * 0.015)], w

    def sigma(self, x, base):
        xh, r = self.xh, self.r
        h = xh * (1.7 + 0.2 * r.random())
        w = h * (0.75 + 0.08 * r.normal())
        top = base - h * 0.85
        pts = [(x + w, top + r.normal(0, xh * 0.05)), (x, top), (x + w * 0.5, top + h * 0.5),
               (x, top + h), (x + w * self.j(), top + h + r.normal(0, xh * 0.05))]
        return [wobble(resample(pts, 2), r, xh * 0.03)], w

    def bullet(self, x, base):
        xh = self.xh
        w = xh * 0.55
        return [self.line((x, base - xh * 0.5), (x + w, base - xh * 0.52))], w

    # -- decorations around things ---------------------------------------------------
    def _ellipse(self, cx, cy, rx, ry, turns=1.12, start=None):
        r = self.r
        a0 = r.uniform(0, 2 * math.pi) if start is None else start
        t = np.linspace(a0, a0 + 2 * math.pi * turns, max(24, int(60 * turns)))
        rr = 1 + 0.04 * np.sin(3 * t + r.random() * 6) * (1 + self.mess)
        pts = np.c_[cx + rx * rr * np.cos(t), cy + ry * rr * np.sin(t)]
        return wobble(pts, r, max(rx, ry) * 0.02)

    def circle_around(self, x0, y0, x1, y1):
        xh, r = self.xh, self.r
        cx, cy = (x0 + x1) / 2 + r.normal(0, xh * 0.1), (y0 + y1) / 2 + r.normal(0, xh * 0.08)
        rx = (x1 - x0) / 2 + xh * (0.45 + 0.2 * r.random())
        ry = (y1 - y0) / 2 + xh * (0.35 + 0.15 * r.random())
        tilt = math.radians(r.normal(0, 3))
        e = self._ellipse(0, 0, rx, ry, 1.08 + 0.12 * r.random())
        c, s = math.cos(tilt), math.sin(tilt)
        return [np.c_[cx + e[:, 0] * c - e[:, 1] * s, cy + e[:, 0] * s + e[:, 1] * c]]

    def underline(self, x0, x1, base, wavy=False):
        xh, r = self.xh, self.r
        y = base + xh * (0.28 + 0.08 * r.random())
        x0 -= xh * 0.2 * r.random()
        x1 += xh * (0.1 + 0.4 * r.random())
        s = self.line((x0, y), (x1, y + r.normal(0, xh * 0.12)), amp=xh * (0.03 + 0.04 * self.mess))
        out = [s]
        if wavy or r.random() < 0.15 * self.mess:   # occasional double underline
            out.append(self.line((x0 + xh * 0.3, y + xh * 0.22), (x1 - xh * 0.2, y + xh * 0.25)))
        return out

    def strike(self, x0, x1, base, scribble=False):
        xh, r = self.xh, self.r
        y = base - xh * (0.45 + 0.1 * r.normal())
        out = [self.line((x0 - xh * 0.15, y + r.normal(0, xh * 0.08)), (x1 + xh * 0.15, y + r.normal(0, xh * 0.12)))]
        if scribble:
            n = max(3, int((x1 - x0) / (xh * 0.45)))
            xs = np.linspace(x0, x1, n)
            ys = np.where(np.arange(n) % 2 == 0, base - xh * 0.95, base + xh * 0.05)
            out = [wobble(resample(np.c_[xs, ys], 2), r, xh * 0.04)]
            if r.random() < 0.6:
                out.append(self.line((x0 - xh * 0.1, base - xh * 0.5), (x1 + xh * 0.1, base - xh * 0.4)))
        elif r.random() < 0.35:
            out.append(self.line((x0, y + xh * 0.18), (x1, y + xh * 0.16)))
        return out

    def rule(self, x0, x1, y):
        """A long separator: often drawn in two or three overlapping pulls of the pen."""
        r = self.r
        pieces = 1 if r.random() < 0.5 else 2
        cuts = np.sort(r.uniform(x0 + (x1 - x0) * 0.3, x1 - (x1 - x0) * 0.3, pieces - 1))
        xs = np.r_[x0, cuts, x1]
        out = []
        yy = y
        for a, b in zip(xs[:-1], xs[1:]):
            nb = yy + r.normal(0, self.xh * 0.12)
            out.append(self.line((a - self.xh * 0.2 * r.random(), yy), (b + self.xh * 0.2 * r.random(), nb),
                                 amp=self.xh * 0.05, freq=1.5))
            yy = nb
        return out

    def frac_bar(self, x0, x1, y):
        return [self.line((x0, y), (x1, y + self.r.normal(0, self.xh * 0.05)))]

    def radical(self, x, base, top, inner_w):
        xh, r = self.xh, self.r
        pts = [(x, base - xh * 0.45), (x + xh * 0.18, base - xh * 0.55), (x + xh * 0.42, base + xh * 0.05),
               (x + xh * 0.85, top), (x + xh * 0.95 + inner_w + xh * 0.2 * r.random(), top + r.normal(0, xh * 0.06))]
        return [wobble(resample(pts, 2), r, xh * 0.02)], xh * 0.95

    def vrule(self, x, y0, y1):
        return [self.line((x, y0), (x + self.r.normal(0, self.xh * 0.12), y1), amp=self.xh * 0.04)]

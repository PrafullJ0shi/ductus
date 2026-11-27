"""Hand-ruled page border and margin line, like a student draws in a project file.

Lines are drawn with the pen of the writing, into the same stroke-shape and overlap-count
layers the words use (`render.ink_coverage`), so they go through the ink model and, with
`--photo`, through the phone capture. Nothing here is a flat vector overlay.

Each line is one pull of the pen: a slow perpendicular wobble, a small bow, pressure that
changes along the line (width and density), a lifted pen at the end, and corners that
overshoot or stop short of each other by up to `CORNER_MM`. The four sides are slightly
different distances from the paper edge on every page. Some lines are gone over a second
time, a little to one side, which darkens the overlap in the ink model.

The text layout stays clear of all of it: `BorderSpec.apply` shrinks the writing area of a
`layout.Layouter` by the worst case of the random offsets plus a gap.

Modes (`--border`): none | box | box-margin | margin. `--border-inset` is the distance of the
box from the paper edge in mm. `--border-header` adds a horizontal line under a heading area
that many mm from the top edge (0 = none).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MODES = ("none", "box", "box-margin", "margin")
JITTER_MM = 2.0          # largest random offset of one side from its nominal position
CORNER_MM = 2.0          # corners overshoot or stop short by 0 to this much
GAP_MM = 4.5             # clear space between a line and the nearest ink
WALK_MM = 2.0            # largest sideways wander of the left text edge (layout margin walk)
MARGIN_MM = 28.0         # nominal distance of the margin line from the left paper edge
HALF_LINE_MM = 0.35


@dataclass(frozen=True)
class BorderSpec:
    mode: str = "box"
    inset_mm: float = 12.0
    header_mm: float = 0.0

    @property
    def box(self) -> bool:
        return self.mode in ("box", "box-margin")

    @property
    def margin(self) -> bool:
        return self.mode in ("box-margin", "margin")

    @property
    def margin_x_mm(self) -> float:
        """Nominal margin-line distance from the left paper edge."""
        return max(MARGIN_MM, self.inset_mm + 14.0) if self.box else MARGIN_MM

    @property
    def header_y_mm(self) -> float:
        """Nominal header line distance from the top paper edge (0 = none)."""
        if self.header_mm <= 0:
            return 0.0
        floor = self.inset_mm + 10.0 if self.box else 14.0
        return max(self.header_mm, floor)

    # -- layout --------------------------------------------------------------------
    def limits_mm(self) -> tuple[float, float, float, float]:
        """Closest the text may come to (left, top, right, bottom) paper edges, worst case."""
        pad = JITTER_MM + HALF_LINE_MM + GAP_MM
        left = top = right = bottom = 0.0
        if self.box:
            left = top = right = self.inset_mm + pad
            bottom = self.inset_mm + pad + 3.5        # descenders and the slope of the last line
            top += 2.0                                # ascenders of the first line
        if self.margin:
            left = max(left, self.margin_x_mm + pad)
        if self.header_y_mm:
            top = max(top, self.header_y_mm + pad + 2.0)
        return left, top, right, bottom

    def apply(self, L) -> None:
        """Shrink the writing area of a `layout.Layouter` (call at the end of `start()`)."""
        from .layout import MM
        l, t, r, b = self.limits_mm()
        L.left = max(L.left, (l + WALK_MM) * MM)
        L.top = max(L.top, t * MM)
        L.bottom = min(L.bottom, L.H - b * MM) if b else L.bottom
        if self.box:
            cap = L.W - r * MM
            L.right = min(L.right, cap - 1.2 * L.xh)       # a squeezed last word may use 1.2 x-heights
            L.right_cap = cap
        L.walk_cap = WALK_MM * MM
        L.y = L.top + L.xh * 1.6

    def text_box_px(self, W: int, H: int, default_margin_mm=(22, 20, 18, 22)) -> tuple[float, float, float, float]:
        """(left, top, right, bottom) of the writing area in page px, for tests and docs."""
        from .layout import MM
        l, t, r, b = self.limits_mm()
        dl, dt, dr, db = default_margin_mm
        return (max(dl, l + WALK_MM) * MM, max(dt, t) * MM, W - max(dr, r) * MM, H - max(db, b) * MM)

    # -- geometry ------------------------------------------------------------------
    def lines(self, W: int, H: int, seed: int, mess: float = 0.4) -> list[dict]:
        """The pen pulls for one page: dicts with `pts` (N, 2) px, `w` (N,) width factor,
        `a` (N,) density 0..1 and `kind` (side, margin, header, retrace). Deterministic for (seed, page)."""
        from .layout import MM
        rng = np.random.default_rng(seed)
        amp = (0.12 + 0.30 * mess) * MM          # wobble amplitude
        out: list[dict] = []

        def j(mm=JITTER_MM):
            return float(np.clip(rng.normal(0, mm * 0.45), -mm, mm)) * MM

        def corner():
            return float(rng.uniform(-CORNER_MM, CORNER_MM)) * MM   # + overshoot, - stops short

        def pull(p0, p1, kind, retrace_ok=True):
            out.append(dict(_pull(rng, np.asarray(p0, float), np.asarray(p1, float), amp), kind=kind))
            if retrace_ok and rng.random() < 0.3:
                q0 = np.asarray(p0, float) + (np.asarray(p1, float) - np.asarray(p0, float)) * rng.uniform(0.0, 0.2)
                q1 = np.asarray(p0, float) + (np.asarray(p1, float) - np.asarray(p0, float)) * rng.uniform(0.7, 1.0)
                nrm = np.array([-(q1 - q0)[1], (q1 - q0)[0]]); nrm /= max(np.hypot(*nrm), 1e-6)
                off = nrm * rng.uniform(-0.4, 0.4) * MM
                d = _pull(rng, q0 + off, q1 + off, amp * 0.8)
                d["w"] = d["w"] * 0.9
                out.append(dict(d, kind="retrace"))

        if self.box:
            ins = self.inset_mm * MM
            xl, xr = ins + j(), W - ins - j()
            yt, yb = ins + j(), H - ins - j()
            # slightly off-square: each corner is its own point
            tl = (xl + j(0.7), yt + j(0.7)); tr = (xr + j(0.7), yt + j(0.7))
            br = (xr + j(0.7), yb + j(0.7)); bl = (xl + j(0.7), yb + j(0.7))

            def ext(a, b, ea, eb):
                a, b = np.asarray(a, float), np.asarray(b, float)
                u = (b - a) / max(np.hypot(*(b - a)), 1e-6)
                return a - u * ea, b + u * eb
            pull(*ext(tl, tr, corner(), corner()), "side")       # top, left to right
            pull(*ext(tr, br, corner(), corner()), "side")       # right, downwards
            pull(*ext(br, bl, corner(), corner()), "side")       # bottom, right to left
            pull(*ext(bl, tl, corner(), corner()), "side")       # left, upwards
            top_y, bot_y = min(tl[1], tr[1]), max(bl[1], br[1])
            left_x, right_x = min(tl[0], bl[0]), max(tr[0], br[0])
        else:
            top_y, bot_y = 8 * MM + j(1.5), H - 8 * MM - j(1.5)
            left_x, right_x = 0.0, W - 10 * MM
        mx = None
        if self.margin:
            mx = self.margin_x_mm * MM + j()
            skew = float(rng.normal(0, 0.5)) * MM
            # inside a box the margin line meets the box lines (a hair over or short); alone it
            # starts and ends a little off the paper edges
            c0, c1 = (corner() * 0.6, corner() * 0.6) if self.box else (abs(corner()), abs(corner()))
            pull((mx, top_y - c0), (mx + skew, bot_y + c1), "margin")
        hy = self.header_y_mm
        if hy:
            y = hy * MM + j()
            x0 = (mx if mx is not None else left_x) - abs(corner())
            x1 = right_x + (corner() if self.box else 0.0)
            pull((x0, y), (x1, y + float(rng.normal(0, 0.5)) * MM), "header")
        return out

    # -- ink -----------------------------------------------------------------------
    def draw(self, shape: np.ndarray, count: np.ndarray | None, W: int, H: int, pen_px: float,
             seed: int, mess: float = 0.4) -> tuple[np.ndarray, np.ndarray | None]:
        """Add the border to the page's stroke-shape (and overlap-count) layers, which are
        at `render.SS` times page resolution. Returns the updated layers."""
        from .render import SS
        for ln in self.lines(W, H, seed, mess):
            layer = _raster(ln, shape.shape, pen_px, SS)
            shape = 1 - (1 - shape) * (1 - layer)
            if count is not None:
                count = count + (layer > 0.35)
        if count is not None:
            count = np.maximum(count, (shape > 0.35).astype(np.float32))
        return shape, count


def _pull(rng, p0: np.ndarray, p1: np.ndarray, amp: float) -> dict:
    """One pull of the pen from p0 to p1: wobble, bow, pressure and a lifted end."""
    from .layout import MM
    d = p1 - p0
    length = float(np.hypot(*d)) or 1.0
    n = max(12, int(length / (1.5 * MM)))
    t = np.linspace(0, 1, n)
    u = d / length
    nrm = np.array([-u[1], u[0]])
    ph = rng.random(6)
    wob = amp * (0.55 * np.sin(2 * math.pi * (0.8 * t + ph[0])) + 0.30 * np.sin(2 * math.pi * (2.1 * t + ph[1]))
                 + 0.15 * np.sin(2 * math.pi * (5.3 * t + ph[2])))
    wob += rng.normal(0, 0.9) * amp * np.sin(math.pi * t)               # a bow
    pts = p0[None, :] + t[:, None] * d[None, :] + wob[:, None] * nrm[None, :]
    lift = np.clip(np.minimum(t * length / (1.2 * MM), (1 - t) * length / (3.0 * MM)), 0.5, 1.0)
    w = (1 + 0.10 * np.sin(2 * math.pi * (1.2 * t + ph[3])) + 0.06 * np.sin(2 * math.pi * (3.7 * t + ph[4]))) * lift
    w *= 1.0 + 0.10 * np.clip(1 - t * 6, 0, 1)                           # pressed in at the start
    a = np.clip(0.90 + 0.07 * np.sin(2 * math.pi * (0.9 * t + ph[5])) - 0.12 * (1 - lift), 0.55, 0.98)
    return {"pts": pts, "w": w, "a": a}


def _raster(ln: dict, shape: tuple, pen_px: float, SS: int) -> np.ndarray:
    import cv2
    tmp = np.zeros(shape, np.uint8)
    p = ln["pts"] * SS
    for i in range(len(p) - 1):
        th = max(1, int(round(pen_px * SS * float(ln["w"][i]))))
        col = int(round(255 * float(ln["a"][i])))
        seg = np.round(p[i:i + 2] * 4).astype(np.int32)
        cv2.polylines(tmp, [seg.reshape(-1, 1, 2)], False, col, th, cv2.LINE_AA, shift=2)
    return tmp.astype(np.float32) / 255.0


def parse(mode: str | None, inset_mm: float = 12.0, header_mm: float = 0.0) -> BorderSpec | None:
    """None for 'none' (or empty); raises ValueError for an unknown mode."""
    if not mode or mode == "none":
        return None
    if mode not in MODES:
        raise ValueError(f"border must be one of {', '.join(MODES)}, not {mode!r}")
    return BorderSpec(mode, float(inset_mm), float(header_mm))

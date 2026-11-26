"""Letter templates the RNN output is corrected with.

Each template is one pen path in x-height units: x to the right, y down, baseline at y = 0,
x-height line at y = -1. "cursive" are the default r and f of the target hand (school cursive),
fitted by eye to crops of the writer's pages, which stay private; "print" are the plain print forms.
Every use gets a small random affine and point jitter, so no two letters are identical."""
import numpy as np

from scipy.interpolate import splprep, splev

# school-cursive r: lead-in from the baseline, tiny loop at the x-height, short dip, then a
# shoulder that runs right and down to the baseline and leaves toward the next letter
R_HIS = [(-0.12, 0.00), (0.04, -0.45), (0.19, -0.84), (0.29, -1.04), (0.20, -1.12), (0.07, -1.00),
         (0.15, -0.85), (0.32, -0.83), (0.42, -0.74), (0.52, -0.86), (0.62, -0.93), (0.70, -0.83),
         (0.72, -0.52), (0.75, -0.17), (0.86, -0.01), (1.04, -0.03), (1.22, -0.13), (1.36, -0.26)]
# looped f: up to an upper loop at the ascender, a long stem below the baseline, a lower
# loop that closes at the baseline, and the exit at the x-height that joins the next letter
F_HIS = [(0.00, 0.00), (0.18, -0.50), (0.36, -1.05), (0.50, -1.45), (0.54, -1.68), (0.44, -1.78),
         (0.30, -1.66), (0.27, -1.20), (0.28, -0.55), (0.28, 0.00), (0.27, 0.50), (0.25, 0.84),
         (0.12, 0.98), (-0.04, 0.84), (0.02, 0.42), (0.20, 0.08), (0.36, -0.28), (0.52, -0.60),
         (0.70, -0.76), (0.86, -0.70), (0.92, -0.58)]
# plain print forms (the redraw from before): stem, retrace, one shoulder
R_PRINT = [(0.00, -1.00), (0.00, 0.00), (0.02, -0.55), (0.14, -0.86), (0.32, -0.99), (0.52, -0.84)]
F_PRINT = [(0.62, -1.62), (0.40, -1.72), (0.22, -1.55), (0.20, -1.00), (0.20, 0.00)]

TEMPLATES = {"cursive": {"r": R_HIS, "f": F_HIS}, "print": {"r": R_PRINT, "f": F_PRINT}}
F_PRINT_BAR = [(0.00, -0.92), (0.50, -0.95)]


def width(ch: str, mode: str) -> float:
    t = np.array(TEMPLATES[mode][ch])
    return float(t[:, 0].max() - t[:, 0].min())


def draw(ch: str, mode: str, x0: float, baseline: float, h: float, rng) -> list[np.ndarray]:
    """Strokes of one letter, x0 = left edge, scaled by the x-height h."""
    t = np.array(TEMPLATES[mode][ch], dtype=np.float64)
    t[:, 0] *= rng.uniform(0.92, 1.08)
    shear = rng.normal(0.06, 0.04)                       # a little forward slant, varied
    t[:, 0] -= shear * t[:, 1]
    t += rng.normal(0, 0.018, t.shape)
    pts = _smooth(t)
    pts[:, 0] = pts[:, 0] * h + x0
    pts[:, 1] = pts[:, 1] * h * rng.uniform(0.95, 1.05) + baseline
    out = [pts]
    if mode == "print" and ch == "f":
        b = np.array(F_PRINT_BAR) + rng.normal(0, 0.02, (2, 2))
        out.append(np.c_[b[:, 0] * h + x0, b[:, 1] * h + baseline])
    return out


def _smooth(t: np.ndarray, n: int = 48) -> np.ndarray:
    try:
        k = min(3, len(t) - 1)
        tck, _ = splprep([t[:, 0], t[:, 1]], s=0.0004 * len(t), k=k)
        x, y = splev(np.linspace(0, 1, n), tck)
        return np.c_[x, y]
    except Exception:
        return t.copy()

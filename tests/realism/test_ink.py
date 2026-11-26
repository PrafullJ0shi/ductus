import numpy as np
import pytest
from ductus.realism import ink


def line(x0, y0, x1, y1, n=40, p=0.6, speed=1.0):
    t = np.linspace(0, 1, int(n / speed))
    return np.c_[x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, np.full(len(t), p)]


def test_shape_dtype_and_empty_alpha():
    lay = ink.render_strokes([line(10, 10, 90, 12)], (40, 100), seed=0)
    assert lay.shape == (40, 100, 4) and lay.dtype == np.uint8
    assert lay[35:, :, 3].max() == 0          # nothing far from the stroke
    assert lay[8:14, 20:80, 3].max() > 150    # the stroke is there


def test_deterministic_per_seed():
    s = [line(10, 10, 90, 30)]
    a = ink.render_strokes(s, (40, 100), seed=3)
    b = ink.render_strokes(s, (40, 100), seed=3)
    c = ink.render_strokes(s, (40, 100), seed=4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


@pytest.mark.parametrize("pen", sorted(ink.PENS))
def test_every_pen_renders(pen):
    lay = ink.render_strokes([line(10, 20, 90, 20), np.array([[50, 30, 0.6]])], (40, 100), pen=pen, seed=1)
    assert lay[..., 3].max() > 60


def test_tip_size_sets_width():
    widths = []
    for tip in (0.5, 0.7, 1.0):
        C = ink.render_strokes([line(10, 20, 190, 20, n=120)], (40, 200), tip_mm=tip, seed=0, return_concentration=True)
        widths.append((C[:, 50:150] > 0.3).sum(0).mean())
    assert widths[0] < widths[1] < widths[2]
    assert 2.5 < widths[0] < 5.5          # ~0.30 mm at 300 dpi = 3.5 px


def test_fast_stroke_is_lighter_than_slow():
    slow = ink.render_strokes([line(10, 20, 190, 20, n=400)], (40, 200), seed=0, return_concentration=True)
    fast = ink.render_strokes([line(10, 20, 190, 20, n=60)], (40, 200), seed=0, return_concentration=True)
    # same path; constant spacing means no timing so speed comes from curvature -> equal
    assert abs(slow[:, 60:140].sum() - fast[:, 60:140].sum()) / slow[:, 60:140].sum() < 0.25
    # accelerating stroke: the fast end gets less ink than the slow start
    t = np.linspace(0, 1, 80) ** 2
    acc = np.c_[10 + 180 * t, np.full(80, 20.0), np.full(80, 0.6)]
    C = ink.render_strokes([acc], (40, 200), seed=0, return_concentration=True)
    assert C[:, 30:60].sum() > C[:, 140:170].sum()


def test_overlap_darkens():
    one = ink.render_strokes([line(10, 20, 90, 20)], (40, 100), seed=0, return_concentration=True)
    two = ink.render_strokes([line(10, 20, 90, 20), line(50, 5, 50, 35)], (40, 100), seed=0, return_concentration=True)
    assert two[18:23, 48:53].mean() > one[18:23, 48:53].mean() * 1.3


def test_composite_white_matches_layer():
    lay = ink.render_strokes([line(10, 20, 90, 20)], (40, 100), seed=0)
    white = np.full((40, 100, 3), 255, np.uint8)
    out = ink.composite(white, lay).astype(int)
    a = lay[..., 3:4] / 255.0
    alpha_blend = (lay[..., :3] * a + 255 * (1 - a)).astype(int)
    assert np.abs(out - alpha_blend).max() <= 3


def test_colour_is_blue_and_dark():
    lay = ink.render_strokes([line(10, 20, 190, 20, n=200, p=0.8)], (40, 200), seed=0)
    out = ink.composite(np.full((40, 200, 3), 245, np.uint8), lay)
    core = out[19:22, 40:160].reshape(-1, 3).mean(0)
    assert core[2] > core[0] + 30 and core.mean() < 140


def test_ink_from_mask():
    m = np.zeros((60, 120), np.float32)
    m[28:32, 10:110] = 1
    m[10:50, 58:62] = 1
    lay = ink.ink_from_mask(m, dpi=300, seed=0)
    assert lay.shape == (60, 120, 4)
    assert lay[29:31, 20:100, 3].mean() > 100 and lay[0:5, :, 3].max() < 30


def test_strokes_from_offsets_splits_on_pen_up():
    arr = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 1], [5, 0, 0], [6, 1, 1]], float)
    st = ink.strokes_from_offsets(arr, scale=10)
    assert len(st) == 2 and st[0].shape == (3, 3) and np.all((st[0][:, 2] > 0) & (st[0][:, 2] <= 1))


def test_line_width_px():
    assert abs(ink.line_width_px(0.5, 300) - 0.30 * 300 / 25.4) < 1e-6
    assert ink.line_width_px(1.0, 600) > ink.line_width_px(0.5, 600)

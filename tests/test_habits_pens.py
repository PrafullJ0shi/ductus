import numpy as np

from ductus import habits as H
from ductus.pens import PENS, get_pen, nib_kernel, concentration, to_rgb
from ductus.photo import capture, Capture


def test_habit_rewrites():
    hb = {"spellings": {"software": "softwear"}, "abbreviations": {"explains": "->"}, "apply_rate": 1.0}
    assert H.apply_text("Software explains it", hb) == "Softwear -> it"


def test_habit_rate_zero_keeps_text():
    hb = {"spellings": {"software": "softwear"}, "apply_rate": 0.0}
    assert H.apply_text("software", hb) == "software"


def test_pens_and_tips():
    assert set(PENS) >= {"blue-ballpoint", "black-ballpoint", "blue-gel", "black-gel", "fountain", "pencil"}
    assert get_pen("blue-ballpoint", 1.0).line_mm > get_pen("blue-ballpoint", 0.5).line_mm
    assert nib_kernel(10, PENS["fountain"]) is not None
    assert nib_kernel(10, PENS["blue-ballpoint"]) is None


def test_ink_overlap_is_darker():
    shape = np.zeros((80, 200), np.float32)
    shape[30:40, 20:180] = 1
    count = shape.copy()
    count[30:40, 90:110] = 2
    rng = np.random.default_rng(0)
    pen = PENS["black-gel"]
    c = concentration(shape, count, pen, rng, 23.6)
    assert c[35, 100] > c[35, 50]
    rgb = to_rgb(c, pen, np.full((80, 200, 3), 255, np.float32))
    assert rgb[35, 100].mean() < rgb[35, 50].mean() < 255
    assert rgb[5, 5].mean() > 250


def test_phone_capture_shape():
    page = np.full((350, 248, 3), 255, np.uint8)
    out = capture(page, 1, Capture(desk=True))
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.shape[0] > page.shape[0]

import json
import os

import cv2
import numpy as np
import pytest

from ductus.realism import calibrate as cal
from ductus.realism import phone_capture as pc


@pytest.fixture(scope="module")
def page():
    p = pc.make_paper(1754, 1240, dpi=150, kind="grid", seed=0)
    cv2.putText(p, "Mean = sum / n", (150, 400), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (40, 50, 140), 4, cv2.LINE_AA)
    return p


SMALL = dict(out_size=(800, 600))


def test_make_paper_kinds():
    for k in ("grid", "ruled", "plain", "dotted"):
        p = pc.make_paper(200, 150, dpi=100, kind=k)
        assert p.shape == (200, 150, 3) and p.dtype == np.uint8 and p.mean() > 200


def test_capture_shape_and_determinism(page):
    a, info = pc.capture(page, seed=1, **SMALL)
    b, _ = pc.capture(page, seed=1, **SMALL)
    c, _ = pc.capture(page, seed=2, **SMALL)
    assert a.shape == (800, 600, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert len(info["page_corners"]) == 4 and "homography" in info


def test_capture_page_is_brighter_than_desk(page):
    a, info = pc.capture(page, seed=3, **SMALL)
    poly = np.array(info["page_corners"], np.int32)
    m = np.zeros(a.shape[:2], np.uint8); cv2.fillPoly(m, [poly], 1)
    assert a[m == 1].mean() > a[m == 0].mean() + 20
    assert 0.45 < m.mean() < 0.95


def test_noise_parameter_increases_noise(page):
    q, _ = pc.capture(page, seed=4, noise_shot=0.0, noise_read=0.0, jpeg_quality=100, **SMALL)
    n, _ = pc.capture(page, seed=4, noise_shot=0.004, noise_read=0.0001, jpeg_quality=100, **SMALL)
    hp = lambda x: (x.astype(np.float32) - cv2.GaussianBlur(x.astype(np.float32), (0, 0), 2)).std()
    assert hp(n) > hp(q) * 1.1


def test_effects_off_keeps_page_readable(page):
    a, _ = pc.capture(page, seed=0, pitch_deg=0, yaw_deg=0, roll_deg=0, curl=0, shadow=0, light_gradient=0,
                      vignette=0, defocus=0, noise_shot=0, noise_read=0, desk=False, spiral=False, page_edge=False,
                      show_through=0, jpeg_quality=100, wb_jitter=0, **SMALL)
    assert a.std() > 5


def test_show_through_is_faint_and_mirrored(page):
    back = np.full_like(page, 245); cv2.rectangle(back, (100, 100), (300, 300), (30, 30, 30), -1)
    out = pc.add_show_through(np.full_like(page, 245), back, strength=0.04)
    w = page.shape[1]
    assert out[200, w - 200].mean() < 240 and out[200, 200].mean() > 243


def test_calibration_round_trip(tmp_path, page):
    ph, info = pc.capture(page, seed=5, jpeg_quality=83, out_size=(1600, 1200))
    f = str(tmp_path / "p.jpg")
    cv2.imwrite(f, ph[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 83])
    m = cal.measure_photo(f)
    assert m["jpeg"]["quality"] == 83
    assert m["noise_sigma"]["Y"] > 0.3
    assert abs(m["rotation_deg"] - info["centre_line_angle_deg"]) < 2.5
    assert m["ink_rgb_core"][2] > m["ink_rgb_core"][0]        # blue ink found
    assert 0.3 < m["page_coverage"] < 0.95
    out = str(tmp_path / "cal.json")
    cal.main([f, "-o", out])
    doc = json.load(open(out))
    assert doc["capture_params"]["jpeg_quality"] == 83
    P = pc.CaptureParams.from_calibration(out)
    assert P.jpeg_quality == 83


def test_default_calibration_loads():
    P = pc.CaptureParams.from_calibration()
    assert 50 <= P.jpeg_quality <= 100

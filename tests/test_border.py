import cv2
import numpy as np
import pytest

from ductus import border as B
from ductus import markup
from ductus.layout import Layouter, Look, MM
from ductus.render import SS, ink_coverage
from ductus.synth import WordImage

PAGE = (1240, 1754)          # A5 at 300 dpi: small enough for fast tests, big enough for a box


def fake_words(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        a = np.zeros((64, int(rng.integers(30, 90))), np.float32)
        a[24:40, 2:-2] = 0.9
        a[10:24, 4:8] = 0.9                       # an ascender
        a[40:52, 10:14] = 0.9                     # a descender
        out.append(WordImage(f"w{i}", a, 40.0, 16.0))
    return out


DOC = "# Heading words\n" + " ".join(f"word{i}" for i in range(45)) + "\n---\nTerm :: " + " ".join(f"def{i}" for i in range(40)) + "\n" + " ".join(f"tail{i}" for i in range(30))


def layout(border, look="natural", seed=1):
    blocks = markup.parse(DOC, "notes")
    ws = fake_words(sum(1 for b in blocks for _ in markup.block_words(b)), seed)
    deco = [w.deco for b in blocks for w in markup.block_words(b)]
    L = Layouter(ws, deco, Look.preset(look), seed=seed, page_size=PAGE, border=border)
    return L, L.layout(blocks), ws


def test_parse():
    assert B.parse("none") is None and B.parse(None) is None and B.parse("") is None
    assert B.parse("box", 15).inset_mm == 15
    for m in ("box", "box-margin", "margin"):
        assert B.parse(m).mode == m
    with pytest.raises(ValueError):
        B.parse("circle")


@pytest.mark.parametrize("mode", ["box", "box-margin", "margin"])
def test_text_box_shrinks_and_pages_grow(mode):
    L0, p0, _ = layout(None)
    spec = B.parse(mode)
    L1, p1, _ = layout(spec)
    assert L1.left >= L0.left and L1.top >= L0.top
    if spec.box:
        assert L1.right < L0.right and L1.bottom < L0.bottom and L1.top > L0.top
        assert L1.right_cap is not None and L1.right_cap > L1.right
    if spec.margin:
        assert L1.left >= (spec.margin_x_mm + B.GAP_MM) * MM          # clear of the margin line
    assert len(p1) >= len(p0)
    assert L1.walk_cap == B.WALK_MM * MM


def test_default_layout_untouched_without_border():
    L, pages, _ = layout(None)
    assert L.right_cap is None and L.walk_cap is None
    assert L.left == Look().margin_mm[0] * MM


@pytest.mark.parametrize("mode,header", [("box", 0), ("box-margin", 0), ("margin", 0), ("box-margin", 40)])
def test_no_ink_overlap_and_gap(mode, header):
    spec = B.BorderSpec(mode, 12.0, float(header))
    L, pages, ws = layout(spec)
    assert len(pages) >= 2
    for pi, pg in enumerate(pages):
        rng = np.random.default_rng(0)
        text = ink_coverage(pg, ws, 0.4, 1.0, 0.4 * MM, rng, size=PAGE)
        line, _ = spec.draw(np.zeros_like(text), None, *PAGE, 0.4 * MM, seed=pi, mess=0.4)
        assert line.max() > 0.5 and (line > 0.05).sum() > 2000      # something was drawn
        t, b = text > 0.02, line > 0.02
        assert not (t & b).any()                                   # never drawn over text
        dist = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
        assert dist[t].min() >= 1.5 * MM * SS                       # and a visible gap everywhere


def test_text_box_px_matches_layout():
    spec = B.parse("box-margin")
    L, _, _ = layout(spec)
    l, t, r, b = spec.text_box_px(*PAGE)
    assert abs(L.left - l) < 1 and abs(L.top - t) < 1 and abs(L.bottom - b) < 1


def test_lines_vary_per_page_and_corners_are_loose():
    spec = B.parse("box-margin")
    a = spec.lines(*PAGE, seed=1)
    b = spec.lines(*PAGE, seed=2)
    assert len(a) >= 5                                              # 4 sides, margin, sometimes retraces
    assert not np.allclose(a[0]["pts"][0], b[0]["pts"][0])        # inset differs per page
    assert np.allclose(a[0]["pts"], spec.lines(*PAGE, seed=1)[0]["pts"])   # but is reproducible
    ln = a[0]
    assert ln["w"].std() > 0.02 and ln["a"].std() > 0.01            # pressure changes along the line
    # corners: sides overshoot or stop short, never by much; the top line start vs the left line end
    ends = []
    for s in range(60):
        sides = [l for l in spec.lines(*PAGE, seed=s) if l["kind"] == "side"]
        assert len(sides) == 4
        ends.append(np.hypot(*(sides[0]["pts"][0] - sides[3]["pts"][-1])) / MM)
    assert max(ends) < 2 * B.CORNER_MM + 2 * 0.7 * 1.4143 + 0.5     # corner slack plus the corner jitter
    assert len({round(e, 1) for e in ends}) > 15                    # and it differs from page to page


def test_border_goes_through_the_ink_model():
    from ductus.paper import make_paper
    from ductus.pens import get_pen
    from ductus.render import compose_realism
    spec = B.parse("box")
    shape = np.zeros((PAGE[1] * SS, PAGE[0] * SS), np.float32)
    count = np.zeros_like(shape)
    shape, count = spec.draw(shape, count, *PAGE, 0.4 * MM, seed=3, mess=0.4)
    img = compose_realism(shape, count, get_pen("blue-ballpoint"), 1, make_paper("white", size=PAGE))
    assert img.shape == (PAGE[1], PAGE[0], 3)
    ys, xs = np.where(shape[::SS, ::SS] > 0.3)
    px = img[ys, xs].astype(int)
    assert px[:, 2].mean() > px[:, 0].mean() + 20                  # blue ink, not black or grey
    assert len(np.unique(px[:, 2])) > 15                           # texture, not one flat colour

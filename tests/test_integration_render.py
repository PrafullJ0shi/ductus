import numpy as np

from ductus import markup
from ductus.humanize_bridge import humanize_page
from ductus.layout import Layouter, Look, MM
from ductus.paper import make_paper, phone_photo
from ductus.pens import get_pen
from ductus.render import compose_realism, ink_coverage
from ductus.synth import WordImage


def fake_word(text):
    a = np.zeros((64, 14 * len(text) + 6), np.float32)
    for i in range(len(text)):
        a[22:42, 4 + 14 * i:4 + 14 * i + 3] = 1.0      # one vertical stroke per letter
        a[38:42, 4 + 14 * i:4 + 14 * i + 10] = 1.0
    return WordImage(text, a, 42.0, 18.0)


def test_page_renders_with_realism_ink_and_humanize():
    text = "Range :: largest minus smallest\n1) Mean deviation [v]\nNote: spread matters"
    blocks = markup.parse(text, "notes")
    ws = [fake_word(w.text) for b in blocks for w in markup.block_words(b)]
    deco = [w.deco for b in blocks for w in markup.block_words(b)]
    L = Layouter(ws, deco, Look.preset("natural"), seed=1, page_size=(1240, 900))
    pg = L.layout(blocks)[0]
    n0 = len(pg.words)
    humanize_page(pg, ws, L, 1.0, 3)
    assert len(pg.words) >= n0
    rng = np.random.default_rng(0)
    shape, count = ink_coverage(pg, ws, 0.4, 1.0, 0.38 * MM, rng, size=(1240, 900), with_count=True)
    img = compose_realism(shape, count, get_pen("blue-ballpoint"), 1, make_paper("white", size=(1240, 900)))
    assert img.shape == (900, 1240, 3)
    assert img.min() < 150 and img.max() > 240
    photo = phone_photo(img, seed=2)
    assert photo.ndim == 3


def test_photo_surfaces():
    from ductus.page import Options, _paper, _surface
    from ductus.realism.phone_capture import CaptureParams, capture
    assert _surface(True) == "a4" and _surface("notebook") == "notebook"
    assert _paper(Options(photo="notebook")) == "grid" and _paper(Options(photo="a4")) == "white"
    assert _paper(Options(photo="notebook", paper="ruled")) == "ruled"
    page = np.full((700, 495, 3), 250, np.uint8)
    page[100:110, 50:400] = 40
    for s in ("a4", "notebook"):
        P = CaptureParams.surface(s)
        assert (P.binding == "loose") == (s == "a4") and P.spiral == (s == "notebook")
        out, _ = capture(page, 1, replace_out(P))
        assert out.shape[:2] == (480, 360)


def replace_out(P):
    from dataclasses import replace
    return replace(P, out_size=(480, 360))

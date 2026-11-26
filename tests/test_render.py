import numpy as np

from ductus.layout import Layouter, Look
from ductus.markup import parse, block_words
from ductus.render import ink_coverage, compose, save_svg
from ductus.paper import make_paper, phone_photo
from ductus.synth import WordImage


def fake_words(n):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        a = np.zeros((64, int(rng.integers(30, 90))), np.float32)
        a[24:40, 2:-2] = 0.9
        out.append(WordImage(f"w{i}", a, 40.0, 16.0))
    return out


def test_full_layout_and_render(tmp_path):
    doc = "# Heading\nTerm :: a long explanation that keeps going\n1) one [v]\n= {a/b} sqrt(x) ((7))\n| a | b |\n| c | d |\n---"
    blocks = parse(doc, mode="maths")
    n = sum(1 for b in blocks for _ in block_words(b))
    words = fake_words(n)
    L = Layouter(words, [frozenset()] * n, Look(), seed=1)
    pages = L.layout(blocks)
    assert len(pages) == 1 and len(pages[0].words) == n and len(pages[0].strokes) > 5
    ink = ink_coverage(pages[0], words, 0.4, 1.0, 3.0, np.random.default_rng(0), size=(2480, 3508))
    img = compose(ink, "#1c2a6e", 0.2, np.random.default_rng(0))
    assert img.shape == (3508, 2480, 3) and img.min() < 120 and img.max() == 255
    save_svg(pages[0], words, str(tmp_path / "p.svg"), "#1c2a6e", 3.0, animate=True)
    assert "<animate" in (tmp_path / "p.svg").read_text()


def test_paper_kinds():
    for k in ("white", "ruled", "grid"):
        p = make_paper(k, size=(300, 400))
        assert p.shape == (400, 300, 3)
    assert make_paper("white", size=(10, 10)).min() == 255


def test_phone_photo_shape():
    img = np.full((400, 300, 3), 255, np.uint8)
    out = phone_photo(img, seed=1)
    assert out.ndim == 3 and out.shape[0] > 400


def test_long_text_breaks_pages():
    blocks = parse("\n".join(["word " * 12] * 120))
    n = sum(1 for b in blocks for _ in block_words(b))
    words = fake_words(n)
    pages = Layouter(words, [frozenset()] * n, Look(), seed=0).layout(blocks)
    assert len(pages) >= 2

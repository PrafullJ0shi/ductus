import numpy as np

from ductus import printglyphs as P


def test_handles_numbers_symbols_and_acronyms():
    for w in ["4)", "1.41", "30/5", "-2", "Q3.", "S.D.", "BTC", "HTML", "2026", "x", "(x", "6)", "=", "(", ";", "&"]:
        assert P.handles(w), w
    for w in ["Data:", "mean", "Find", "I", "A", "x-ray", "nx", "", "?!", "and"]:
        assert not P.handles(w), w


def test_render_every_template_and_no_two_alike():
    rng = np.random.default_rng(0)
    for ch in P.T:
        a, bl, xh = P.render(ch, rng)
        assert a.max() > 0.5 and bl > 0 and xh > 0
    a1, _, _ = P.render("2026", rng)
    a2, _, _ = P.render("2026", rng)
    assert a1.shape != a2.shape or np.abs(a1 - a2).mean() > 0.01


def test_retraced_word_moves_with_its_word():
    from ductus.layout import Placed, Page
    from ductus.humanize_bridge import humanize_page

    class W:
        def __init__(self, t):
            self.text, self.alpha = t, np.ones((30, 60))

    class L:
        xh, W, H = 30.0, 2480, 3508

    pg = Page()
    pg.words = [Placed(0, 100, 500, 1.0, 0.0), Placed(0, 100.3, 500.2, 1.0, 0.0, of=0), Placed(1, 200, 500, 1.0, 0.0)]
    humanize_page(pg, [W("like"), W("to")], L, 1.0, 3)
    assert abs(pg.words[1].x - pg.words[0].x) < 2.0

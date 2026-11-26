import copy
import numpy as np
from ductus.realism import human_errors as he

TEXT = ("the quick brown fox jumps over the lazy dog while writing notes about statistics and "
        "variance with little time left in the exam hall today " * 6)


def page(n_text=TEXT):
    p = he.Page(2480, 3508, 300, 300, 150, 250, 180)
    return he.layout_text(n_text, lambda t: 30.0 * len(t), p, x_height=30, line_gap=170, space=33)


def test_layout_fills_lines_within_margins():
    p = page()
    assert len(p.lines) > 5
    for ln in p.lines:
        assert ln.words[-1].right <= p.right_edge + 1e-6


def test_functions_are_pure():
    p = page(); snap = copy.deepcopy(p)
    for f in (he.drift_size_slant, he.apply_fatigue, he.tighten_bottom, he.uneven_margins, he.squeeze_line_ends,
              he.add_strike_corrections, he.add_caret_insertions, he.add_overwrites, he.delayed_strokes, he.humanize):
        f(p, seed=1)
    assert p == snap


def test_seeded():
    a = he.humanize(page(), seed=5); b = he.humanize(page(), seed=5)
    assert a == b


def test_rate_zero_changes_nothing_in_geometry():
    p = page()
    q = he.drift_size_slant(p, seed=1, rate=0)
    assert q == p


def test_fatigue_grows_size_down_the_page():
    q = he.apply_fatigue(page(), seed=0, rate=1.0)
    first = np.mean([w.scale for w in q.lines[0].words]); last = np.mean([w.scale for w in q.lines[-1].words])
    assert last > first * 1.04


def test_tighten_bottom_reduces_gaps():
    p = page(TEXT * 3)
    q = he.tighten_bottom(p, rate=1.0)
    g = np.diff([ln.baseline_y for ln in q.lines])
    assert g[-1] < g[0] * 0.95
    assert all(w.y == ln.baseline_y for ln in q.lines for w in ln.words)


def test_strike_corrections_insert_struck_slips():
    q = he.add_strike_corrections(page(), seed=2, rate=1.0, per_100_words=20)
    slips = [w for _, _, w in he.all_words(q) if w.struck]
    assert slips and all(w.role == "slip" and w.text != w.meta["for"] for w in slips)
    assert sum(m.kind in ("strike", "scribble") for m in q.marks) == len(slips)
    # each slip is followed by its correct word
    for ln in q.lines:
        for a, b in zip(ln.words, ln.words[1:]):
            if a.struck:
                assert b.text == a.meta["for"] and b.x >= a.right


def test_make_slip_is_plausible():
    rng = np.random.default_rng(0)
    for w in ["variance", "statistics", "population", "sample"]:
        for _ in range(20):
            s = make = he.make_slip(w, rng)
            assert s != w and abs(len(s) - len(w)) <= len(w) - 2


def test_caret_insertion_moves_word_above():
    q = he.add_caret_insertions(page(), seed=3, rate=1.0, per_100_words=30)
    ins = [(li, w) for li, _, w in he.all_words(q) if w.role == "insert"]
    assert ins
    for li, w in ins:
        assert w.dy < -w.height * 0.5 and w.scale < 1
    assert sum(m.kind == "caret" for m in q.marks) == len(ins)


def test_delayed_strokes_cover_i_j_t():
    p = page("tilt it in time")
    q = he.delayed_strokes(p, seed=0, rate=1.0, missing_dot=0.0)
    n_t = sum(m.kind == "tcross" for m in q.marks)
    n_i = sum(m.kind == "idot" for m in q.marks)
    assert n_t == "tilt it in time".count("t") and n_i == "tilt it in time".count("i")
    # dots sit above the x-height and mostly to the right of the stem
    for m in q.marks:
        if m.kind == "idot":
            assert m.points[0][0][1] < q.lines[0].baseline_y - 30


def test_squeeze_line_ends():
    p = page()
    p.lines[0].words[-1].width += 200   # overflow the margin
    q = he.squeeze_line_ends(p, seed=0, rate=1.0)
    w = q.lines[0].words[-1]
    assert w.scale_x < 1 and w.meta.get("squeezed")
    assert w.right <= q.right_edge + 0.02 * q.width


def test_marks_to_strokes():
    q = he.humanize(page(), seed=1, rate=1.0, delayed=True)
    st = he.marks_to_strokes(q)
    assert st and all(s.ndim == 2 and s.shape[1] == 3 for s in st)


def test_uneven_margins_wander():
    q = he.uneven_margins(page(TEXT * 2), seed=4, rate=1.0)
    xs = [ln.words[0].x for ln in q.lines]
    assert np.std(xs) > 3


def test_rewrap_keeps_words_on_the_page_after_growth():
    p = page(TEXT * 2)
    q = he.apply_fatigue(he.drift_size_slant(p, seed=2, rate=1.5), seed=3, rate=1.5)
    r = he.rewrap(q)
    texts = [w.text for _, _, w in he.all_words(r)]
    assert texts == [w.text for _, _, w in he.all_words(q)]          # same words, same order
    for ln in r.lines:
        assert ln.words[-1].right <= r.right_edge + 0.012 * r.width + 1e-6 or len(ln.words) == 1


def test_humanize_marks_follow_their_words():
    q = he.humanize(page(TEXT * 2), seed=11, rate=1.0, corrections_per_100=10)
    by_uid = {w.meta["uid"]: w for _, _, w in he.all_words(q)}
    for m in q.marks:
        if m.kind in ("strike", "scribble"):
            w = by_uid[m.uid]
            xs = [pt[0] for st in m.points for pt in st]
            assert w.x - w.height <= min(xs) <= w.x + w.height

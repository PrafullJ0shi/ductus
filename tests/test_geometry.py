import numpy as np

from ductus.segment import split_words, baseline_and_xheight, column_runs, tight_bbox, skew_angle
from ductus.verify import cer, levenshtein, normalise
from ductus.synth import align_words, Synthesizer
from ductus.pen import Marks, wobble, resample
from ductus.page import typo
import random


def fake_line(widths, gap=12, h=64):
    a = np.zeros((h, sum(widths) + gap * len(widths) + 10), np.float32)
    x = 5
    for w in widths:
        a[20:44, x:x + w] = 1.0
        # a small inner gap inside each word that must not be chosen as a word gap
        a[20:44, x + w // 2:x + w // 2 + 3] = 0
        x += w + gap
    return a


def test_split_words_picks_widest_gaps():
    a = fake_line([40, 60, 30])
    spans = split_words(a, 3)
    assert spans is not None and len(spans) == 3
    assert spans[0][0] == 5 and abs(spans[1][0] - 57) <= 1


def test_split_words_fails_cleanly():
    a = fake_line([40])
    assert split_words(a, 3) is None


def test_baseline_xheight():
    a = np.zeros((64, 50), np.float32)
    a[20:40, 5:45] = 1
    bl, xh = baseline_and_xheight(a)
    assert bl == 40 and xh == 20


def test_runs_and_bbox():
    assert column_runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]
    a = np.zeros((10, 10)); a[2:5, 3:7] = 1
    assert tight_bbox(a) == (3, 2, 7, 5)


def test_cer_and_normalise():
    assert levenshtein("kitten", "sitting") == 3
    assert cer("abc", "abc") == 0
    assert normalise("Hello, World!") == "hello world"


def test_align_words_handles_drops():
    out = align_words(["the", "mean", "value"], ["the", "value"])
    assert out[0] == "the" and out[2] == "value" and out[1] == ""


def test_carriers_fixed_length():
    s = Synthesizer(engine=None, style_img=None)
    words = list(enumerate("a short list of words that needs packing into several full lines ok".split()))
    car = s._carriers(words, random.Random(0))
    assert sum(len(g) for g, _ in car) == len(words)
    for group, full in car:
        assert full.startswith(" ".join(w for _, w in group))
        assert len(full) <= 46


def test_marks_never_identical():
    m = Marks(np.random.default_rng(0), xh=30, mess=0.5)
    a, _ = m.arrow(0, 0)
    b, _ = m.arrow(0, 0)
    assert not np.allclose(a[0][: min(len(a[0]), len(b[0]))], b[0][: min(len(a[0]), len(b[0]))])


def test_wobble_keeps_shape():
    pts = resample([(0, 0), (100, 0)], 2)
    w = wobble(pts, np.random.default_rng(1), 2.0)
    assert np.abs(w[:, 1]).max() < 10 and len(w) == len(pts)


def test_typo_changes_word():
    r = random.Random(3)
    assert typo("variance", r) != "variance"


def test_skew_angle_recovers_rotation():
    import cv2
    a = np.zeros((400, 600), np.float32)
    for y in range(60, 360, 50):
        a[y:y + 12, 40:560] = 1
    M = cv2.getRotationMatrix2D((300, 200), 3.0, 1)
    r = cv2.warpAffine(a, M, (600, 400))
    assert abs(skew_angle(r) - 3.0) <= 0.5

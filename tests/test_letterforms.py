import numpy as np

from ductus.letterforms import flourishes, holes


def canvas():
    return np.zeros((64, 40), np.float32)


def test_plain_f_has_no_loop():
    a = canvas()
    a[6:44, 18:21] = 1          # stem from ascender to baseline (baseline row 44)
    a[26:29, 12:28] = 1         # crossbar
    a[6:9, 18:28] = 1           # hook at the top
    assert holes(a) == []
    assert flourishes(a, "f", 44, 14) == []


def test_looped_f_is_flagged():
    a = canvas()
    a[6:58, 18:21] = 1          # stem down into the descender
    a[26:29, 12:28] = 1
    a[48:58, 21:30] = 0
    a[48:51, 18:30] = 1         # lower loop: a closed box below the baseline
    a[55:58, 18:30] = 1
    a[48:58, 27:30] = 1
    assert flourishes(a, "f", 44, 14) == ["f"]


def test_o_counter_is_not_a_flourish():
    a = canvas()
    a[30:44, 10:30] = 1
    a[33:41, 13:27] = 0         # an o: hole in the x-band
    assert len(holes(a)) == 1
    assert flourishes(a, "o", 44, 14) == []

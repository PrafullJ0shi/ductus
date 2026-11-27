import random

import numpy as np

from ductus import wordbank as WB
from ductus.synth import WordImage


def word(text, k=0):
    a = np.zeros((40, 30 + 10 * len(text)), np.float32)
    a[10:30, 5 + k:20 + k] = 1.0
    return WordImage(text, a, 30.0, 14.0, 1.0, text)


def test_variant_used_once_per_run_and_retired(tmp_path, monkeypatch):
    monkeypatch.setenv("DUCTUS_CACHE", str(tmp_path))
    b = WB.Bank("sig", per_word=3, max_uses=2)
    for k in range(2):
        b.add(word("the", k))
    run = WB.Bank("sig", per_word=3, max_uses=2, seed=1)
    got = [run.take("the") for _ in range(3)]
    assert got[0] is not None and got[1] is not None and got[2] is None   # two variants, each once
    run2 = WB.Bank("sig", per_word=3, max_uses=2, seed=2)
    assert run2.take("the") is not None and run2.take("the") is not None
    run3 = WB.Bank("sig", per_word=3, max_uses=2, seed=3)
    assert run3.take("the") is None                                      # both used twice: retired


def test_words_written_in_a_run_are_not_taken_in_it(tmp_path, monkeypatch):
    monkeypatch.setenv("DUCTUS_CACHE", str(tmp_path))
    b = WB.Bank("sig")
    b.add(word("model"))
    assert b.take("model") is None


def test_jitter_keeps_ink_and_baseline():
    w = word("data")
    j = WB.jitter(w, random.Random(3))
    assert j.alpha.sum() > 0.8 * w.alpha.sum()
    assert abs(j.baseline - w.baseline) < 0.05 * w.baseline + 1
    assert j.text == "data"

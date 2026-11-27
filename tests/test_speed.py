"""Speed plumbing: thread choice, the OCR cascade, token caps, the batched RNN sampler."""
import os

import numpy as np
import pytest

from ductus import speed
from ductus.synth import Synthesizer, _cap


def test_threads_follow_env(monkeypatch):
    monkeypatch.setenv("DUCTUS_THREADS", "3")
    assert speed.n_threads() == 3
    monkeypatch.delenv("DUCTUS_THREADS")
    assert 1 <= speed.n_threads() <= 8
    assert speed.free_cores() >= 1


def test_token_cap_covers_the_target():
    assert _cap("word") >= len("word") + 1
    assert _cap("a longer carrier line of text") > len("a longer carrier line of text")


class BlockEngine:
    """Writes each word as a row of letter blocks with clear gaps (splittable, no OCR needed)."""
    name = "fake"

    def generate(self, texts, style_img=None, seed=0):
        out = []
        for t in texts:
            words = t.split()
            W = 20 + sum(14 * len(w) + 30 for w in words)
            img = np.full((64, W), 255, np.uint8)
            x = 10
            for w in words:
                for _ in w:
                    img[22:42, x:x + 9] = 0
                    x += 14
                x += 30
            out.append(img)
        return out


class FakeReader:
    def __init__(self, good: bool):
        self.good, self.calls = good, 0
        self.truth = []

    def read(self, grays, with_conf=False, max_tokens=None):
        self.calls += len(grays)
        return [self.truth.pop(0) if self.good else "zzz" for _ in grays]


def _run(first, checker, words):
    syn = Synthesizer(BlockEngine(), None, first, checker=checker, max_cer=0.1, strict=True,
                      rounds=1, plain="", log=lambda *a: None)
    return syn, syn.run(words)


def test_cascade_rechecks_only_rejects():
    words = ["alpha", "bravo"]
    # the fast reader reads everything wrong; the checker reads the line, then each word
    first = FakeReader(good=False)
    chk = FakeReader(good=True)
    chk.truth = ["alpha bravo", "alpha", "bravo"]
    syn, out = _run(first, chk, words)
    assert [w.score for w in out] == [1.0, 1.0]
    assert syn.stats["rechecked"] == 2
    assert chk.calls == 3            # one line read + two word reads


def test_no_recheck_when_first_reader_passes():
    words = ["alpha", "bravo"]
    first = FakeReader(good=True)
    first.truth = ["alpha bravo", "alpha", "bravo"]
    chk = FakeReader(good=False)
    syn, out = _run(first, chk, words)
    assert [w.score for w in out] == [1.0, 1.0]
    assert chk.calls == 0


@pytest.mark.skipif(not os.environ.get("DUCTUS_SLOW_TESTS"), reason="needs the RNN model and its TensorFlow worker")
def test_batched_sampler_matches_worker():
    import json
    import subprocess
    from pathlib import Path
    from ductus.engines import rnn as R
    from ductus.engines.rnn.graves import Graves
    eng = R.RNNEngine()
    g = Graves(R.model_dir(), eng.python)
    reqs = [{"text": "the quick brown fox", "bias": 1.5, "style": 1, "seed": 5},
            {"text": "and a much longer line of words", "bias": 1.8, "style": 1, "seed": 9}]
    ours = g.lines(reqs)
    p = subprocess.Popen([eng.python, str(Path(R.__file__).with_name("worker.py")), str(R.model_dir())],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    json.loads(p.stdout.readline())
    for q, o in zip(reqs, ours):
        p.stdin.write(json.dumps(q) + "\n"); p.stdin.flush()
        ref = json.loads(p.stdout.readline())
        a, b = np.array(ref["coords"]), np.array(o["coords"])
        assert a.shape == b.shape
        assert np.abs(a - b).max() < 1e-2
    p.stdin.close(); p.wait(timeout=30)

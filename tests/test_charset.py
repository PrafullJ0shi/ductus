import cv2
import numpy as np

from ductus import charset as C
from ductus import printglyphs as PG


def test_trace_ring_and_bar():
    img = np.zeros((60, 60), np.float32)
    cv2.circle(img, (30, 30), 20, 1.0, 3)
    lines = C._trace(img, min_len=4)
    assert len(lines) >= 1
    span = np.ptp(np.vstack(lines), axis=0)
    assert span.min() > 30
    bar = np.zeros((20, 60), np.float32)
    cv2.line(bar, (5, 10), (55, 10), 1.0, 3)
    (q,) = C._trace(bar, min_len=4)
    assert np.ptp(q[:, 0]) > 40 and np.ptp(q[:, 1]) < 3


def test_personal_table_is_used():
    tab = {"7": [("c", [(0.0, -1.0), (0.5, -1.0), (0.1, 0.0)])]}
    rng = np.random.default_rng(0)
    a, bl, xh = PG.render("7", rng, table=tab)
    b, _, _ = PG.render("7", np.random.default_rng(0))
    assert a.shape != b.shape or not np.allclose(a, b)


def test_l_t_swap_is_rejected():
    from ductus.verify import confused
    assert confused("population", "poputation")
    assert confused("small", "smalt")
    assert not confused("little", "little")
    assert not confused("help", "hep")

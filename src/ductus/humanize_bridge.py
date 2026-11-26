"""Run realism.human_errors on a laid-out ductus page.

ductus's own layout already applies drift, fatigue, tightening, margins and squeezed line
ends (layout.py). From realism.human_errors this bridge takes the per-word behaviours that do
not move words away from the pen marks drawn next to them:

* drift_size_slant: slow AR(1) drift of size, baseline and rotation across words;
* add_overwrites: a letter written over a second time, slightly offset (darker, doubled).

Text-changing behaviours (corrections, caret insertions) need new word images and are done
at generation time instead (`{oops:word}`, `--corrections`).
"""
from __future__ import annotations

from .layout import Placed
from .realism import human_errors as he


def humanize_page(pg, words, L, rate: float, seed: int) -> None:
    if rate <= 0 or not pg.words:
        return
    # group placements into lines by baseline
    order = sorted(range(len(pg.words)), key=lambda i: (pg.words[i].base, pg.words[i].x))
    lines, cur, last = [], [], None
    for i in order:
        p = pg.words[i]
        if last is not None and abs(p.base - last) > 0.8 * L.xh:
            lines.append(cur); cur = []
        cur.append(i); last = p.base
    if cur:
        lines.append(cur)
    hp = he.Page(L.W, L.H, 300)
    for li in lines:
        li.sort(key=lambda i: pg.words[i].x)
        ws = []
        for i in li:
            p = pg.words[i]
            wi = words[p.idx]
            ws.append(he.Word(wi.text, p.x, p.base, wi.alpha.shape[1] * p.scale * p.sx, L.xh, meta={"uid": i}))
        hp.lines.append(he.Line(ws, pg.words[li[0]].base, pg.words[li[0]].x))
    hp = he.drift_size_slant(hp, seed, rate)
    hp = he.add_overwrites(hp, seed + 3, rate)
    over = {}
    for m in hp.marks:
        if m.kind == "overwrite":
            over.setdefault(m.uid, []).append(m.points[0][0])
    new = []
    for ln in hp.lines:
        shift, prev_right = 0.0, None
        for w in ln.words:
            i = w.meta["uid"]
            p = pg.words[i]
            p.x += shift
            p.scale *= w.scale
            p.base += w.dy
            p.angle += w.rotate
            # size drift grows words in place: keep the word gap (otherwise words run together)
            if prev_right is not None and p.x < prev_right + 0.9 * L.xh:
                d = prev_right + 0.9 * L.xh - p.x
                p.x += d
                shift += d
            prev_right = p.x + words[p.idx].alpha.shape[1] * p.scale * p.sx
            for ci, d in zip(w.meta.get("overwrite_chars", []), over.get(i, [])):
                n = max(1, len(w.text))
                new.append(Placed(p.idx, p.x + d[0], p.base + d[1], p.scale, p.angle, p.ink * 0.9, p.sx,
                                  clip=(ci / n, (ci + 1) / n)))
    pg.words.extend(new)

"""A writer's own capitals, digits and symbols from one photographed character sheet.

The sheet holds each character once, in a known order, one row of the order per written row
(see ROWS). The page is found and deskewed, ink is split into rows and the marks in a row are
grouped into characters by horizontal overlap, then each character is traced to centre-line
strokes (skeleton -> polylines) in x-height units, the format printglyphs uses. The result is
saved as glyphs.json in the style folder; printglyphs then draws these shapes, with its usual
per-use jitter, in place of its plain-print templates."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROWS = ["abcdefghijklmn", "opqrstuvwxyz", "ABCDEFGHIJKLMNOP", "QRSTUVWXYZ", "0123456789",
        "=+-x÷/()%.,:;?!√", "²\"'"]
LOWER_XH = "acemnorsuvwxz"        # letters whose height is the x-height


def page_ink(path: str) -> np.ndarray:
    from . import profile as P
    rgb = P.load_photo(path)
    a = P.drop_blobs(P.remove_grid(P.ink_map(rgb) * P.paper_mask(rgb)))
    from .segment import skew_angle
    ang = skew_angle(a, search=6.0)
    a = P.rotate(a, -ang)
    # refine on the character rows themselves: slope of mark bottoms along each row
    rows, _ = rows_of(a)
    sl = []
    for _, comps in rows:
        if len(comps) >= 8:
            x = np.array([s[0] + s[2] / 2 for _, s in comps]); y = np.array([s[1] + s[3] for _, s in comps])
            b = np.polyfit(x, y, 1)[0]
            r = y - np.polyval([b, np.polyval(np.polyfit(x, y, 1), 0)], x)
            ok = np.abs(r) < 2 * np.median(np.abs(r)) + 1
            if ok.sum() >= 5:
                sl.append(np.polyfit(x[ok], y[ok], 1)[0])
    if sl:
        a = P.rotate(a, float(np.degrees(np.arctan(np.median(sl)))))
    return a


def _components(a):
    m = (a > 0.3).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return [(i, st[i]) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 4], lab


def _group(boxes, tol, st=None, vgap=None):
    """Merge marks whose x-ranges overlap (i, j, =, ÷, :, ;, %, ?, !). With stats `st` and
    `vgap`, marks merge only if they are also vertically close."""
    boxes = sorted(boxes, key=lambda b: b[0])
    groups = []
    for b in boxes:
        best = None
        for g in groups:
            x0, x1 = min(c[0] for c in g), max(c[1] for c in g)
            if min(x1, b[1]) - max(x0, b[0]) <= -tol:
                continue
            if st is not None:
                gy0 = min(st[c[2]][1] for c in g); gy1 = max(st[c[2]][1] + st[c[2]][3] for c in g)
                by0, by1 = st[b[2]][1], st[b[2]][1] + st[b[2]][3]
                if max(gy0 - by1, by0 - gy1) > vgap:
                    continue
            best = g
        if best is not None:
            best.append(b)
        else:
            groups.append([b])
    return sorted(groups, key=lambda g: min(c[0] for c in g))


def rows_of(a: np.ndarray):
    from .segment import find_text_lines
    comps, lab = _components(a)
    if not comps:
        return [], lab
    hs = np.array([s[cv2.CC_STAT_HEIGHT] for _, s in comps])
    unit = float(np.median(hs))
    # keep ordinary marks: no long rules or tall bars from the page edge
    comps = [(i, s) for i, s in comps if s[cv2.CC_STAT_WIDTH] < 8 * unit and s[cv2.CC_STAT_HEIGHT] < 6 * unit]
    keep = np.isin(lab, [i for i, _ in comps])
    spans = find_text_lines(a * keep, min_h=max(4, int(0.4 * unit)))
    # marks outside every span (a thin bar, a dot) join the nearest span within one mark height
    def _near(s):
        c = s[1] + s[3] / 2
        d = [(max(a_ - c, c - b, 0), k) for k, (a_, b) in enumerate(spans)]
        return min(d) if d else (1e9, -1)
    grow = {}
    for i, s in comps:
        dist, k = _near(s)
        if 0 < dist <= unit:
            a_, b = spans[k]
            c = int(s[1] + s[3] / 2)
            grow[k] = (min(grow.get(k, (a_, b))[0], c), max(grow.get(k, (a_, b))[1], c + 1))
    spans = [grow.get(k, sp) for k, sp in enumerate(spans)]
    rows = []
    hmed = float(np.median([b - a_ for a_, b in spans])) if spans else 0
    for y0, y1 in spans:
        mine = [(i, s) for i, s in comps
                if y0 <= s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT] / 2 < y1]
        if not mine:
            continue
        if y1 - y0 > 1.5 * hmed and len(mine) > 3:
            # two written rows that drift into each other: chain characters left to right,
            # each joining the row whose last character sits closest in height
            st = dict(mine)
            chars = _group([(s[0], s[0] + s[2], i) for i, s in mine], tol=0.05 * unit, st=st, vgap=1.0 * unit)
            def cy(g):
                return np.mean([st[b[2]][1] + st[b[2]][3] / 2 for b in g])
            chains = []
            for g in sorted(chars, key=lambda g: min(b[0] for b in g)):
                c, x = cy(g), min(b[0] for b in g)
                # nearest row end in 2D: height difference plus some of the horizontal gap
                cost = lambda ch: abs(ch[-1][0] - c) + 0.3 * max(0, x - ch[-1][2])
                best = min(chains, key=cost, default=None)
                if best is not None and abs(best[-1][0] - c) < 0.7 * hmed:
                    best.append((c, g, max(b[1] for b in g)))
                else:
                    chains.append([(c, g, max(b[1] for b in g))])
            chains.sort(key=lambda ch: np.mean([c for c, _, _ in ch]))
            for ch in chains:
                ids = [b[2] for _, g, _ in ch for b in g]
                cs = [c for c, _, _ in ch]
                rows.append(((int(min(cs)), int(max(cs)) + 1), [(i, st[i]) for i in ids]))
            continue
        rows.append(((y0, y1), mine))
    return rows, lab


def segment(path: str, rows=ROWS, log=print):
    a = page_ink(path)
    found, lab = rows_of(a)
    # rows with the most marks, in reading order, are the character rows
    unit = np.median([s[cv2.CC_STAT_HEIGHT] for _, c in found for _, s in c])
    out = []
    lab_h = {i: s[cv2.CC_STAT_HEIGHT] for _, c in found for i, s in c}
    for (y0, y1), comps in found:
        bx = [(s[0], s[0] + s[2], i) for i, s in comps]
        g = _group(bx, tol=0.05 * unit, st=dict(comps), vgap=1.3 * unit)
        out.append(((y0, y1), g))
    out = [r for r in out if len(r[1]) >= 2]
    # page specks and paper-edge smears: rows whose marks are much smaller than characters
    sizes = []
    for _, g in out:
        hs = [max(lab_h[b[2]] for b in grp) for grp in g]
        sizes.append(float(np.median(hs)))
    big = np.median(sizes) if sizes else 0
    out = [r for r, sz in zip(out, sizes) if sz >= 0.45 * big]
    log(f"rows found: {[len(g) for _, g in out]}")
    return a, lab, out


def assign(path: str, rows=ROWS, log=print) -> dict:
    """{char: (crop alpha, (x0, y0) of the crop on the page)} for every character found, plus
    '_rows': baseline and x-height per sheet row. Rows are matched to the expected order by
    count; extra marks (page specks) are dropped smallest first."""
    a, lab, found = segment(path, rows, log)
    ink = lambda g: float(sum(a[lab == b[2]].sum() for b in g))
    # pair expected rows with found rows in order, skipping found rows that fit no expected row
    out, k = {}, 0
    for (y0, y1), groups in found:
        if k >= len(rows):
            break
        want = rows[k]
        if len(groups) < len(want) or len(groups) > len(want) + 3:
            log(f"skip row y={y0}: {len(groups)} marks, next expected {len(want)} ({want[:4]}...)")
            continue
        while len(groups) > len(want):
            groups = [g for g in groups if g is not min(groups, key=ink)]
        for ch, g in zip(want, groups):
            ids = [b[2] for b in g]
            m = np.isin(lab, ids)
            ys, xs = np.where(m)
            y_0, y_1, x_0, x_1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            out[ch] = (a[y_0:y_1, x_0:x_1] * m[y_0:y_1, x_0:x_1], (int(x_0), int(y_0)), k)
        k += 1
    log(f"characters: {len(out)} of {sum(map(len, rows))}")
    return out


def _trace(crop: np.ndarray, min_len: float, dots: bool = True) -> list[np.ndarray]:
    """Centre-line polylines of a glyph crop (pixel coordinates): skeleton pixels walked from
    end points and junctions, short spurs dropped, then smoothed."""
    from skimage.morphology import skeletonize
    m = cv2.GaussianBlur(crop.astype(np.float32), (0, 0), 0.8) > 0.15
    m = np.pad(m, 2)
    sk = skeletonize(m)
    pts = set(zip(*np.where(sk)))
    nb = lambda p: [(p[0] + dy, p[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if (dy or dx) and (p[0] + dy, p[1] + dx) in pts]
    deg = {p: len(nb(p)) for p in pts}
    nodes = {p for p in pts if deg[p] != 2}
    used = set()
    paths = []

    def walk(a, b):
        path = [a, b]
        used.add(frozenset((a, b)))
        while path[-1] not in nodes:
            nxt = [q for q in nb(path[-1]) if frozenset((path[-1], q)) not in used and q != path[-2]]
            if not nxt:
                break
            used.add(frozenset((path[-1], nxt[0])))
            path.append(nxt[0])
        return path

    for a in sorted(nodes):
        for b in nb(a):
            if frozenset((a, b)) not in used:
                paths.append(walk(a, b))
    # loops with no node (an O): start anywhere
    rest = [p for p in pts if p not in nodes and not any(frozenset((p, q)) in used for q in nb(p))]
    while rest:
        a = rest[0]
        n = nb(a)
        if not n:
            rest = rest[1:]
            continue
        path = walk(a, n[0])
        paths.append(path + [a])
        done = set(path)
        rest = [p for p in rest if p not in done]
    out = []
    for p in paths:
        q = np.array([(x - 2.0, y - 2.0) for y, x in p], np.float64)
        L = float(np.sum(np.hypot(*np.diff(q, axis=0).T))) if len(q) > 1 else 0.0
        touches_end = deg.get(p[0], 0) == 1 or deg.get(p[-1], 0) == 1
        if L < 0.25 * min_len:
            continue                      # links inside a junction blob
        if L < min_len and touches_end and len(paths) > 1:
            continue                      # skeleton spur
        if len(q) >= 5:
            k = np.ones(3) / 3
            q[1:-1] = np.c_[np.convolve(q[:, 0], k, "valid"), np.convolve(q[:, 1], k, "valid")]
        out.append(q)
    # separate small marks (dots of i, j, :, ÷, .) that kept no path: one dot each
    n, lab, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    for i in range(1, n):
        x, y, w, h = st[i, :4]
        inside = any(((o[:, 0] + 2 >= x - 1) & (o[:, 0] + 2 <= x + w) & (o[:, 1] + 2 >= y - 1)
                      & (o[:, 1] + 2 <= y + h)).any() for o in out)
        if not inside and (dots or not out):
            out.append(np.array([[cen[i, 0] - 2, cen[i, 1] - 2]]))
    return out


DOTTED = "ij.,:;!?÷%"
BASE_ROW_CHARS = {0: "acemnorsuvwxz", 1: "acemnorsuvwxz", 2: None, 3: None, 4: None, 5: "x!?.:"}


def build_glyphs(path: str, out_json: str | None = None, rows=ROWS, log=print) -> dict:
    """Trace the character sheet into printglyphs templates (x-height units, baseline y = 0)."""
    chars = assign(path, rows, log)
    lower = [c for c in LOWER_XH if c in chars]
    xh = float(np.median([chars[c][0].shape[0] for c in lower])) if lower else 20.0
    # baseline per sheet row: median bottom of the characters that sit on it
    base = {}
    for r in range(len(rows)):
        sel = BASE_ROW_CHARS.get(r)
        mine = [(c, v) for c, v in chars.items() if v[2] == r and (sel is None or c in sel)]
        if r in (2, 3):
            mine = [(c, v) for c, v in mine if c not in "QJ"]
        if mine:
            xs = np.array([v[1][0] + v[0].shape[1] / 2 for _, v in mine])
            ys = np.array([v[1][1] + v[0].shape[0] for _, v in mine], float)
            if len(mine) >= 4:
                # rows drift as they are written: a robust line through the bottoms
                fit = np.polyfit(xs, ys, 1)
                ok = np.abs(ys - np.polyval(fit, xs)) < max(2.0, 2 * np.median(np.abs(ys - np.polyval(fit, xs))))
                fit = np.polyfit(xs[ok], ys[ok], 1) if ok.sum() >= 3 else fit
                base[r] = fit
            else:
                base[r] = np.array([0.0, float(np.median(ys))])
    glyphs = {}
    for ch, (crop, (x0, y0), r) in chars.items():
        if r not in base:
            continue
        lines = _trace(crop, min_len=0.18 * xh, dots=ch in DOTTED)
        b = float(np.polyval(base[r], x0 + crop.shape[1] / 2))
        tpl = []
        for q in lines:
            u = np.c_[q[:, 0] / xh, (q[:, 1] + y0 - b) / xh]
            length = float(np.sum(np.hypot(*np.diff(u, axis=0).T))) if len(u) > 1 else 0.0
            if len(u) > 8 and np.hypot(*(u[-1] - u[0])) < max(0.22, 0.12 * length) and ch not in "CcGJUu(){}":
                u = np.vstack([u, u[:1]])          # a closed loop (O, 0, D): close it
                closed = True
            else:
                closed = False
            if len(u) == 1 or np.ptp(u, axis=0).max() < 0.1:
                if ch not in DOTTED and len(lines) > 1:
                    continue
                tpl.append(["d", [[round(float(u[:, 0].mean()), 3), round(float(u[:, 1].mean()), 3)]]])
            else:
                eps = 0.03
                d = u[-1] - u[0]
                n_ = np.hypot(*d)
                if n_ > 0.2:
                    dev = np.abs((u[:, 0] - u[0, 0]) * d[1] - (u[:, 1] - u[0, 1]) * d[0]) / n_
                    if dev.max() < 0.09:          # a straight bar: keep it straight
                        u = u[[0, -1]]
                ap = cv2.approxPolyDP((u * 1000).astype(np.float32).reshape(-1, 1, 2), eps * 1000, False)
                ap = ap.reshape(-1, 2) / 1000.0
                if closed and np.hypot(*(ap[-1] - ap[0])) > 1e-3:
                    ap = np.vstack([ap, ap[:1]])
                tpl.append(["s" if len(ap) > 3 else "c", [[round(float(x), 3), round(float(y), 3)] for x, y in ap]])
        if tpl:
            glyphs[ch] = tpl
    res = {"source": "character sheet", "xh_px": round(xh, 2), "glyphs": glyphs}
    if out_json:
        Path(out_json).write_text(json.dumps(res, ensure_ascii=False))
    log(f"traced {len(glyphs)} characters, x-height {xh:.1f} px")
    return res

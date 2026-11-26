"""Text- and layout-level human behaviours for handwritten pages.

Data model (plain dataclasses, all lengths in page pixels, y grows downward)
---------------------------------------------------------------------------
``Page(width, height, dpi, margin_left, margin_right, margin_top, margin_bottom, lines)``
``Line(words, baseline_y, x_start)``
``Word(text, x, y, width, height, ...)``
    x, y      left end of the word on its baseline (y = baseline)
    width     advance width of the written word
    height    x-height of the writing (the body of lowercase letters)
    scale     overall size multiplier (1 = as generated)
    scale_x   extra horizontal compression (squeezed words)
    slant     extra slant in degrees (positive leans right)
    rotate    rotation of the word about its left baseline point, degrees
    dy        vertical offset from the line baseline (baseline wobble)
    struck    word is a mistake that was crossed out
    role      "word" | "slip" (wrong word, struck) | "insert" (added above a caret)
    char_x    optional list of the x centre of each character, relative to x; if missing,
              characters are assumed to be evenly spread over ``width``
``Mark(kind, points, word_index, line_index)`` extra pen marks (strokes) in page px:
    strike, scribble, caret, idot, tcross, overwrite

Every function is pure: it takes a Page and returns a new Page (deep copy), driven by an
explicit ``seed`` and a ``rate`` (0 disables, 1 = typical student notes). ``humanize``
runs them in a sensible order. ``marks_to_strokes`` turns marks into (N,3) stroke arrays
for ``ink.render_strokes``; the caller renders word glyphs (with the word's scale, slant,
rotation and dy applied) from its own engine.

Evidence (see NOTES.md): slips of the pen (Ellis 1979; Wing & Baddeley 1980: spelling slips
cluster in the middle of words), delayed strokes for i-dots and t-crosses (Bilgin Tasdemir &
Yanikoglu 2019), size growth with fatigue (Garnacho-Castano et al. 2020; Kushki et al. 2011),
natural variation of size, slant and alignment (Huber & Headrick 1999).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ----------------------------------------------------------------------------- data model
@dataclass
class Word:
    text: str
    x: float
    y: float
    width: float
    height: float
    scale: float = 1.0
    scale_x: float = 1.0
    slant: float = 0.0
    rotate: float = 0.0
    dy: float = 0.0
    struck: bool = False
    role: str = "word"
    char_x: Optional[list] = None
    meta: dict = field(default_factory=dict)

    @property
    def right(self) -> float:
        return self.x + self.width * self.scale * self.scale_x

    def char_centre(self, i: int) -> float:
        n = max(1, len(self.text))
        k = self.scale * self.scale_x
        if self.char_x is not None and i < len(self.char_x):
            return self.x + self.char_x[i] * k
        return self.x + self.width * k * (i + 0.5) / n


@dataclass
class Mark:
    kind: str
    points: list              # list of strokes, each a list of (x, y) or (x, y, pressure)
    line_index: int = -1
    word_index: int = -1
    style: str = ""           # strike style: line | double | scribble
    uid: int = -1             # Word.meta["uid"] of the word the mark belongs to


@dataclass
class Line:
    words: list
    baseline_y: float
    x_start: float


@dataclass
class Page:
    width: float
    height: float
    dpi: float = 300
    margin_left: float = 0.0
    margin_right: float = 0.0
    margin_top: float = 0.0
    margin_bottom: float = 0.0
    lines: list = field(default_factory=list)
    marks: list = field(default_factory=list)

    @property
    def right_edge(self) -> float:
        return self.width - self.margin_right


def layout_text(text: str, width_of: Callable[[str], float], page: Page, x_height: float,
                line_gap: float, space: float) -> Page:
    """Minimal greedy layout to build a Page from text (paragraphs split on blank lines)."""
    p = copy.deepcopy(page)
    p.lines = []
    y = p.margin_top + line_gap
    uid = 0
    for pi, para in enumerate(text.split("\n\n")):
        words = para.split()
        x = p.margin_left
        cur = []
        for t in words:
            w = width_of(t)
            if cur and x + w > p.right_edge:
                p.lines.append(Line(cur, y, p.margin_left)); y += line_gap; x = p.margin_left; cur = []
            cur.append(Word(t, x, y, w, x_height, meta={"uid": uid, "para": pi})); x += w + space
            uid += 1
        if cur:
            p.lines.append(Line(cur, y, p.margin_left)); y += line_gap
        y += line_gap * 0.4
    return p


def _rng(seed):
    return np.random.default_rng(seed)


def _copy(page: Page) -> Page:
    return copy.deepcopy(page)


def _ar1(n, rng, phi=0.85, sigma=1.0):
    """Slow random walk (AR(1)) with unit stationary std."""
    out = np.zeros(n)
    s = sigma * np.sqrt(1 - phi ** 2)
    v = rng.standard_normal() * sigma
    for i in range(n):
        v = phi * v + s * rng.standard_normal()
        out[i] = v
    return out


def _gaps(line: Line) -> list:
    return [max(0.0, b.x - a.right) for a, b in zip(line.words, line.words[1:])]


def _reflow_line(line: Line, gaps: list | None = None, start: float | None = None, gap_scale: float = 1.0):
    """Re-place words after a change in size, keeping the word gaps measured before it."""
    if not line.words:
        return
    x = line.words[0].x if start is None else start
    gaps = _gaps(line) if gaps is None else gaps
    line.words[0].x = x
    for i, w in enumerate(line.words[1:]):
        prev = line.words[i]
        w.x = prev.right + gaps[i] * gap_scale


# ----------------------------------------------------------------------------- drift & fatigue
def drift_size_slant(page: Page, seed: int = 0, rate: float = 1.0, size_sd: float = 0.035,
                     slant_sd: float = 2.0, baseline_sd: float = 0.06, rotate_sd: float = 0.8) -> Page:
    """Slow drift of size and slant across words (AR(1)), baseline wobble and tiny rotations.
    Writers are never constant: size and slant wander within a person's range of variation."""
    p = _copy(page); rng = _rng(seed)
    words = [w for ln in p.lines for w in ln.words]
    if not words or rate <= 0:
        return p
    n = len(words)
    G = [_gaps(ln) for ln in p.lines]
    sz = _ar1(n, rng, 0.9) * size_sd * rate
    sl = _ar1(n, rng, 0.93) * slant_sd * rate
    for i, w in enumerate(words):
        w.scale *= float(np.exp(sz[i] + 0.02 * rate * rng.standard_normal()))
        w.slant += float(sl[i] + 0.6 * rate * rng.standard_normal())
        w.dy += float(baseline_sd * rate * w.height * rng.standard_normal())
        w.rotate += float(rotate_sd * rate * rng.standard_normal())
        w.meta["gap_after"] = float(np.exp(0.12 * rate * rng.standard_normal()))
    for ln, g in zip(p.lines, G):
        _reflow_line(ln, [gi * ln.words[i].meta.get("gap_after", 1) for i, gi in enumerate(g)])
    return p


def apply_fatigue(page: Page, seed: int = 0, rate: float = 1.0, size_gain: float = 0.08,
                  width_gain: float = 0.05, slant_gain: float = 2.5, wobble_gain: float = 0.8,
                  rise_deg: float = 0.6) -> Page:
    """Down the page writing gets larger and wider, slightly more slanted and less steady
    (Garnacho-Castano et al. 2020: vertical size grew in >90 % of writers under fatigue,
    width in 77 %). Lines also tend to rise towards the right as the hand moves faster."""
    p = _copy(page); rng = _rng(seed)
    if rate <= 0 or not p.lines:
        return p
    n = len(p.lines)
    for li, ln in enumerate(p.lines):
        t = li / max(1, n - 1)
        f = rate * t ** 1.3
        g = _gaps(ln)
        rise = np.radians(rise_deg * rate * (0.4 + 0.6 * t) * rng.uniform(0.3, 1.2))
        for w in ln.words:
            w.scale *= 1 + size_gain * f
            w.scale_x *= 1 + width_gain * f
            w.slant += slant_gain * f
            w.dy += wobble_gain * f * 0.05 * w.height * rng.standard_normal()
        _reflow_line(ln, g, gap_scale=1 + 0.08 * f)
        for w in ln.words:
            w.meta["rise"] = float(np.tan(rise))
            w.dy -= w.meta["rise"] * (w.x - ln.x_start)
    return p


def tighten_bottom(page: Page, seed: int = 0, rate: float = 1.0, start: float = 0.78,
                   squeeze: float = 0.22) -> Page:
    """Near the bottom of the page people try to fit the rest in: line spacing shrinks and
    the writing gets a little smaller on the last lines."""
    p = _copy(page)
    if rate <= 0 or len(p.lines) < 3:
        return p
    top = p.lines[0].baseline_y
    bottom = p.height - p.margin_bottom
    span = bottom - top
    gaps = np.diff([ln.baseline_y for ln in p.lines])
    y = p.lines[0].baseline_y
    for i, ln in enumerate(p.lines):
        if i > 0:
            t = (p.lines[i - 1].baseline_y - top) / max(1, span)
            k = 1 - squeeze * rate * np.clip((t - start) / (1 - start), 0, 1)
            y += gaps[i - 1] * k
        shift = y - ln.baseline_y
        ln.baseline_y = y
        g = _gaps(ln)
        tt = (y - top) / max(1, span)
        shrink = 1 - 0.08 * rate * np.clip((tt - start) / (1 - start), 0, 1)
        for w in ln.words:
            w.y += shift
            w.scale *= shrink
        _reflow_line(ln, g)
    return p


def uneven_margins(page: Page, seed: int = 0, rate: float = 1.0, creep: float = 0.012,
                   jitter: float = 0.004) -> Page:
    """Left margin wanders: a slow drift (often widening down the page) plus per-line jitter;
    the right edge is ragged because words are not hyphenated."""
    p = _copy(page); rng = _rng(seed)
    if rate <= 0:
        return p
    n = len(p.lines)
    walk = _ar1(n, rng, 0.8) * jitter * p.width
    trend = rng.uniform(0.3, 1.0) * creep * p.width
    for i, ln in enumerate(p.lines):
        dx = rate * (trend * i / max(1, n - 1) + walk[i])
        ln.x_start += dx
        for w in ln.words:
            w.x += dx
    return p


def squeeze_line_ends(page: Page, seed: int = 0, rate: float = 1.0, max_squeeze: float = 0.72) -> Page:
    """Words that would cross the right margin are squeezed (narrower, smaller gaps, a slight
    downward or upward curl) instead of being moved to the next line; past a limit they
    wrap. People plan line ends late, so the last word often runs into the margin."""
    p = _copy(page); rng = _rng(seed)
    if rate <= 0:
        return p
    edge = p.right_edge + 0.012 * p.width * rate   # writing may run slightly into the margin
    for ln in p.lines:
        if not ln.words:
            continue
        last = ln.words[-1]
        over = last.right - edge
        if over <= 0:
            continue
        avail = edge - last.x
        need = last.width * last.scale * last.scale_x
        k = avail / need
        if k >= max_squeeze:
            last.scale_x *= k
            last.scale *= 0.96
            last.rotate += float(rng.choice([-1, 1]) * rng.uniform(0.5, 2.0))
            last.meta["squeezed"] = True
            if len(ln.words) > 1:   # also close the last gap a bit
                prev = ln.words[-2]
                gap = last.x - prev.right
                last.x -= 0.35 * gap
    return p


# ----------------------------------------------------------------------------- slips of the pen
_KEY_NEIGHBOURS = "aeiou"


def make_slip(word: str, rng) -> str:
    """A plausible handwriting slip for ``word``: letter omission, doubling, transposition,
    anticipation (a later letter written early) or a vowel swap. Positions favour the middle
    of the word (Wing & Baddeley 1980)."""
    letters = [c for c in word]
    n = len(letters)
    if n < 3:
        return word[:-1] if n > 1 else word + word
    # beta(2,2) favours the middle
    i = int(np.clip(round(rng.beta(2.2, 2.2) * (n - 1)), 1, n - 2))
    kind = rng.choice(["omit", "double", "swap", "anticipate", "vowel", "stop"], p=[.28, .12, .2, .15, .1, .15])
    if kind == "omit":
        s = letters[:i] + letters[i + 1:]
    elif kind == "double":
        s = letters[:i + 1] + [letters[i]] + letters[i + 1:]
    elif kind == "swap":
        s = letters[:i] + [letters[i + 1], letters[i]] + letters[i + 2:]
    elif kind == "anticipate" and i + 2 < n:
        s = letters[:i] + [letters[i + 2]] + letters[i + 1:]
    elif kind == "vowel" and letters[i].lower() in _KEY_NEIGHBOURS:
        s = letters[:i] + [rng.choice([v for v in _KEY_NEIGHBOURS if v != letters[i].lower()])] + letters[i + 1:]
    else:  # stopped part-way, noticed, crossed out
        s = letters[:max(2, i)]
    s = "".join(s)
    return s if s != word else word[:max(2, i)]


def _strike_points(w: Word, rng, style: str):
    x0, x1 = w.x - 0.05 * w.height, w.right + 0.08 * w.height
    yc = w.y + w.dy - 0.5 * w.height * w.scale
    if style == "line":
        n = 12
        t = np.linspace(0, 1, n)
        y = yc + (rng.normal(0, 0.06) + 0.08 * np.sin(np.pi * t)) * w.height - 0.1 * w.height * t
        return [np.c_[x0 + (x1 - x0) * t, y, 0.75 + 0.1 * t].tolist()]
    if style == "double":
        a = _strike_points(w, rng, "line")[0]
        b = [[x, y + 0.28 * w.height * w.scale, p] for x, y, p in _strike_points(w, rng, "line")[0]]
        return [a, b[::-1]]
    # scribble: tight zig-zag loops across the word
    k = max(4, int((x1 - x0) / (0.35 * w.height)))
    xs = np.linspace(x0, x1, 2 * k)
    ys = np.where(np.arange(2 * k) % 2 == 0, yc - 0.55 * w.height, yc + 0.45 * w.height)
    ys = ys + rng.normal(0, 0.08 * w.height, len(ys))
    return [np.c_[xs, ys, np.full(len(xs), 0.85)].tolist()]


def add_strike_corrections(page: Page, seed: int = 0, rate: float = 1.0, per_100_words: float = 1.2,
                           width_of: Callable[[str], float] | None = None,
                           styles=("line", "line", "double", "scribble")) -> Page:
    """Insert a slip before some words, cross it out, then write the right word after it.
    ``width_of(text)`` estimates the width of a new word (default: proportional to the
    neighbour's width per character). Returns new Word objects with role="slip", struck=True
    and Marks of kind "strike"/"scribble"."""
    p = _copy(page); rng = _rng(seed)
    prob = rate * per_100_words / 100
    if prob <= 0:
        return p
    for li, ln in enumerate(p.lines):
        out = []
        for w in ln.words:
            if len(w.text) >= 4 and w.text.isalpha() and rng.random() < prob:
                slip = make_slip(w.text, rng)
                ww = width_of(slip) if width_of else w.width * len(slip) / max(1, len(w.text))
                s = Word(slip, w.x, w.y, ww, w.height, scale=w.scale, scale_x=w.scale_x, slant=w.slant,
                         rotate=w.rotate, dy=w.dy, struck=True, role="slip",
                         meta={"for": w.text, "uid": 10_000_000 + int(rng.integers(1 << 30)), "para": w.meta.get("para", 0)})
                out.append(s)
                style = str(rng.choice(styles))
                p.marks.append(Mark("scribble" if style == "scribble" else "strike",
                                    _strike_points(s, rng, style), li, len(out) - 1, style, s.meta["uid"]))
            out.append(w)
        if len(out) != len(ln.words):
            gap = 1.1 * (out[0].height if out else 0)
            ln.words = out
            # re-flow with standard gaps where slips were added
            for a, b in zip(out, out[1:]):
                if b.x < a.right + gap:
                    shift = a.right + gap - b.x
                    for c in out[out.index(b):]:
                        c.x += shift
            _refresh_marks(p, li)
    return p


def _refresh_marks(p: Page, li: int | None = None):
    """Re-attach marks to their words (by uid) after words moved, changed size or wrapped,
    and redraw strikes and carets at the new place."""
    where = {w.meta.get("uid"): (l, i, w) for l, i, w in all_words(p) if "uid" in w.meta}
    for k, m in enumerate(p.marks):
        if m.uid not in where:
            continue
        l, i, w = where[m.uid]
        if li is not None and l != li:
            continue
        m.line_index, m.word_index = l, i
        if m.kind in ("strike", "scribble"):
            m.points = _strike_points(w, np.random.default_rng(7 * k + 13), m.style or "line")
        elif m.kind == "caret":
            ln = p.lines[l]
            prev = ln.words[i - 1] if i > 0 else None
            gx = (prev.right + w.meta.get("caret_dx", 0.25 * w.height)) if prev else w.x
            h = w.height * w.scale / 0.68
            base = ln.baseline_y + w.meta.get("caret_dy", 0.0) + (w.y - ln.baseline_y)
            m.points = [[[gx - 0.33 * h, base + 0.75 * h, 0.6], [gx, base + 0.05 * h, 0.75],
                         [gx + 0.30 * h, base + 0.70 * h, 0.5]]]
            w.x = gx - 0.5 * w.width * w.scale * w.scale_x


def rewrap(page: Page, tolerance: float = 0.012) -> Page:
    """Re-break paragraphs after words grew or shrank, the way a writer would: a word that
    does not fit goes to the next line; following lines move down by the line pitch.
    Inserted (caret) words ride along with the word before them."""
    p = _copy(page)
    if not p.lines:
        return p
    pitch = float(np.median(np.diff([ln.baseline_y for ln in p.lines]))) if len(p.lines) > 1 else 0
    edge = p.right_edge + tolerance * p.width
    out, i = [], 0
    shift = 0.0
    while i < len(p.lines):
        para = p.lines[i].words[0].meta.get("para") if p.lines[i].words else None
        group = [p.lines[i]]
        i += 1
        while i < len(p.lines) and p.lines[i].words and p.lines[i].words[0].meta.get("para") == para and para is not None:
            group.append(p.lines[i]); i += 1
        words = [w for ln in group for w in ln.words]
        for ln in group:                      # take the line rise out; it is re-applied below
            for w in ln.words:
                w.dy += w.meta.get("rise", 0.0) * (w.x - ln.x_start)
        gaps = []
        for ln in group:
            gaps += _gaps(ln) + [None]
        base_gap = float(np.median([g for g in gaps if g])) if any(gaps) else 0
        new_lines = []
        cur, x = [], None
        k = 0
        for w, g in zip(words, gaps):
            ref = group[min(len(new_lines), len(group) - 1)]
            if x is None:
                x = ref.x_start
            wid = w.width * w.scale * w.scale_x
            if w.role == "insert":
                cur.append(w); continue
            if cur and x + wid > edge:
                new_lines.append(cur); cur = []
                ref = group[min(len(new_lines), len(group) - 1)]
                x = ref.x_start
            w.x = x
            cur.append(w)
            x = x + wid + max(g if g else base_gap, 0.6 * base_gap)
        if cur:
            new_lines.append(cur)
        for j, ws in enumerate(new_lines):
            if j < len(group):
                y = group[j].baseline_y + shift; xs = group[j].x_start
            else:
                shift += pitch
                y = group[-1].baseline_y + shift; xs = group[-1].x_start
            for w in ws:
                w.y = y
                w.dy -= w.meta.get("rise", 0.0) * (w.x - xs)
            out.append(Line(ws, y, xs))
    p.lines = out
    _refresh_marks(p)
    return p


def add_caret_insertions(page: Page, seed: int = 0, rate: float = 1.0, per_100_words: float = 0.6) -> Page:
    """Simulate an omitted word that was added later: the word is removed from the line, a
    caret (^) is drawn on the baseline at the gap and the word is written smaller above it."""
    p = _copy(page); rng = _rng(seed)
    prob = rate * per_100_words / 100
    for li, ln in enumerate(p.lines):
        if len(ln.words) < 3:
            continue
        for i in range(1, len(ln.words) - 1):
            w = ln.words[i]
            if w.role != "word" or rng.random() >= prob or len(w.text) > 7:
                continue
            prev, nxt = ln.words[i - 1], ln.words[i + 1]
            gap_x = (prev.right + nxt.x) / 2
            # close the gap: everything after moves left by the word's width
            shift = nxt.x - prev.right - 0.5 * w.height
            for c in ln.words[i + 1:]:
                c.x -= shift
            gap_x = prev.right + 0.25 * w.height
            h = w.height * w.scale
            base = ln.baseline_y + w.dy
            caret = [[gap_x - 0.33 * h, base + 0.75 * h, 0.6], [gap_x, base + 0.05 * h, 0.75],
                     [gap_x + 0.30 * h, base + 0.70 * h, 0.5]]
            p.marks.append(Mark("caret", [caret], li, i, uid=w.meta.get("uid", -1)))
            w.meta["caret_dy"] = w.dy
            w.meta["caret_dx"] = gap_x - prev.right
            w.role = "insert"
            w.scale *= 0.68
            w.x = gap_x - 0.5 * w.width * w.scale * w.scale_x
            w.dy -= 2.5 * h + rng.normal(0, 0.1 * h)   # above the ascenders of the line
            w.rotate += float(rng.normal(-1.5, 1.0))
            break   # at most one per line
    return p


def add_overwrites(page: Page, seed: int = 0, rate: float = 1.0, per_100_words: float = 1.5) -> Page:
    """A letter that was fixed by writing over it: a Mark("overwrite") with the character
    index; the renderer should draw that glyph a second time, offset by ``points`` (one
    point: dx, dy in px), which makes it darker and doubled like a real retouch."""
    p = _copy(page); rng = _rng(seed)
    prob = rate * per_100_words / 100
    for li, ln in enumerate(p.lines):
        for wi, w in enumerate(ln.words):
            if w.role == "word" and len(w.text) >= 3 and rng.random() < prob:
                ci = int(rng.integers(0, len(w.text)))
                d = rng.normal(0, 0.06 * w.height, 2)
                m = Mark("overwrite", [[[float(d[0]), float(d[1])]]], li, wi, uid=w.meta.get("uid", -1))
                w.meta.setdefault("overwrite_chars", []).append(ci)
                p.marks.append(m)
    return p


def delayed_strokes(page: Page, seed: int = 0, rate: float = 1.0, dot_offset=(0.25, 0.18),
                    missing_dot: float = 0.06, dash_dot: float = 0.15, cross_offset=(0.18, 0.15),
                    cross_rise_deg: float = 6.0) -> Page:
    """Produce i/j dots and t crosses as separate delayed strokes. Writers add them after the
    word, so they land to the right of the stem (in the writing direction), sometimes high,
    sometimes as a short dash, occasionally missing; t-crosses lean up and can reach the next
    letter. Offsets are in x-heights (mean, sd). Use when your glyph engine omits these
    strokes or set ``Word.meta['has_delayed']=True`` to skip a word."""
    p = _copy(page); rng = _rng(seed)
    if rate <= 0:
        return p
    for li, ln in enumerate(p.lines):
        for wi, w in enumerate(ln.words):
            if w.meta.get("has_delayed"):
                continue
            h = w.height * w.scale
            base = ln.baseline_y + w.dy if w.y == ln.baseline_y else w.y + w.dy
            sl = np.tan(np.radians(w.slant))
            for ci, ch in enumerate(w.text):
                cx = w.char_centre(ci)
                if ch in "ij":
                    if rng.random() < missing_dot * rate:
                        continue
                    dx = rng.normal(dot_offset[0], dot_offset[1]) * h * rate
                    hy = (1.55 + abs(rng.normal(0, 0.25 * rate))) * h
                    x, y = cx + dx + sl * hy, base - hy
                    if rng.random() < dash_dot * rate:
                        a = np.radians(rng.normal(25, 12))
                        L = 0.22 * h
                        pts = [[x - L * np.cos(a) / 2, y + L * np.sin(a) / 2, 0.55],
                               [x + L * np.cos(a) / 2, y - L * np.sin(a) / 2, 0.45]]
                    else:
                        pts = [[x, y, 0.7], [x + 0.03 * h, y + 0.02 * h, 0.7]]
                    p.marks.append(Mark("idot", [pts], li, wi))
                elif ch == "t":
                    dx = rng.normal(cross_offset[0], cross_offset[1]) * h * rate
                    hy = (0.95 + rng.normal(0, 0.12)) * h
                    L = (0.75 + abs(rng.normal(0, 0.25))) * h
                    a = np.radians(rng.normal(cross_rise_deg, 4) * min(1.0, rate + 0.3))
                    x0 = cx + dx - 0.42 * L + sl * hy
                    y0 = base - hy
                    n = 6
                    t = np.linspace(0, 1, n)
                    pts = np.c_[x0 + L * t * np.cos(a), y0 - L * t * np.sin(a) + 0.04 * h * np.sin(np.pi * t),
                                0.65 - 0.2 * t]
                    p.marks.append(Mark("tcross", [pts.tolist()], li, wi))
    return p


def humanize(page: Page, seed: int = 0, rate: float = 1.0, *, corrections: bool = True,
             carets: bool = True, overwrites: bool = True, delayed: bool = False,
             width_of: Callable[[str], float] | None = None, corrections_per_100: float = 1.2,
             carets_per_100: float = 0.6, overwrites_per_100: float = 1.5) -> Page:
    """All behaviours in order: corrections and insertions change the text, then size and
    slant drift, fatigue, bottom tightening, margins and squeezed line ends. ``delayed`` is
    off by default because most glyph engines draw dots and crosses themselves."""
    p = page
    if corrections:
        p = add_strike_corrections(p, seed + 1, rate, corrections_per_100, width_of=width_of)
    if carets:
        p = add_caret_insertions(p, seed + 2, rate, carets_per_100)
    if overwrites:
        p = add_overwrites(p, seed + 3, rate, overwrites_per_100)
    p = drift_size_slant(p, seed + 4, rate)
    p = apply_fatigue(p, seed + 5, rate)
    p = rewrap(p)
    p = tighten_bottom(p, seed + 6, rate)
    p = uneven_margins(p, seed + 7, rate)
    p = squeeze_line_ends(p, seed + 8, rate)
    _refresh_marks(p)
    if delayed:
        p = delayed_strokes(p, seed + 9, rate)
    return p


def marks_to_strokes(page: Page, pressure: float = 0.7, kinds=None) -> list:
    """Marks as (N,3) float arrays (x, y, pressure) for ink.render_strokes. Marks are densified
    so that point spacing is even (strike-throughs are fast, constant-speed strokes)."""
    out = []
    for m in page.marks:
        if kinds is not None and m.kind not in kinds:
            continue
        if m.kind == "overwrite":
            continue
        for st in m.points:
            a = np.asarray(st, float)
            if a.shape[1] == 2:
                a = np.c_[a, np.full(len(a), pressure)]
            if len(a) >= 2:
                d = np.r_[0, np.cumsum(np.hypot(*np.diff(a[:, :2], axis=0).T))]
                n = max(2, int(d[-1] / 2.5) + 1)
                s = np.linspace(0, d[-1], n)
                a = np.c_[[np.interp(s, d, a[:, k]) for k in range(3)]].T
            out.append(a)
    return out


def all_words(page: Page):
    """(line_index, word_index, Word) for every word, in reading order."""
    return [(li, wi, w) for li, ln in enumerate(page.lines) for wi, w in enumerate(ln.words)]

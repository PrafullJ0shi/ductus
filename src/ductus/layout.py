"""Page layout with human habits: drifting baselines, wandering margins, uneven gaps,
per-word size and angle jitter, notebook structures (definitions, lists, maths, tables).

The layout works in page pixels at 300 dpi. Text comes in as WordImage crops and pen
marks come from `pen.Marks`; the output is a list of Page objects holding word
placements and pen strokes, which `render.py` turns into PNG, PDF and SVG.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .markup import Block, Sym, Word
from .pen import Marks

DPI = 300
A4 = (2480, 3508)
MM = DPI / 25.4


@dataclass
class Look:
    """Human-variation knobs. `mess` 0 = tidy, 1 = hurried."""
    mess: float = 0.45
    xh_mm: float = 2.8             # x-height of the writing
    pitch: float = 3.6             # line pitch in x-heights
    slope_deg: float = -0.5        # mean baseline drift (negative = drifts upward to the right)
    margin_mm: tuple = (22, 20, 18, 22)   # left, top, right, bottom
    corrections: float = 0.0       # share of words written wrong, struck out and rewritten
    pressure: float = 0.18         # ink density variation
    ink: str = "#1c2a6e"
    pen: float = 1.0               # stroke weight multiplier
    pen_mm: float = 0.42           # ballpoint line width (measured from the reference when there is one)
    # human-error layer (see docs/REALISM.md); all scale with these rates
    fatigue: float = 0.06          # writing grows this much (and gets messier) by the page end
    tighten: float = 0.10          # line pitch shrinks near the bottom of the page
    squeeze: float = 0.14          # a word overflowing the line by up to this share is squeezed in
    hesitation: float = 0.02       # share of words with a small pen-rest dot before them
    retrace: float = 0.012         # share of words partly written over a second time
    punct_gap: float = 0.35        # extra, uneven gap after , . ; :
    session: float = 0.5           # chance that a section rule starts a new "session" (slope, pressure)
    smudges: float = 0.6           # mean hand smudges per page
    gap_xh: float = 0.75           # word gap in x-heights (measured from the reference when there is one)

    @staticmethod
    def preset(name: str) -> "Look":
        if name == "neat":
            return Look(mess=0.12, slope_deg=-0.15, pressure=0.10, fatigue=0.02, tighten=0.04, squeeze=0.08,
                        hesitation=0.0, retrace=0.0, punct_gap=0.15, session=0.2, smudges=0.0)
        if name == "messy":
            return Look(mess=0.9, slope_deg=-1.1, corrections=0.03, pressure=0.26, xh_mm=2.9, pitch=3.3,
                        fatigue=0.12, tighten=0.18, squeeze=0.2, hesitation=0.04, retrace=0.025, smudges=1.5)
        return Look()


@dataclass
class Placed:
    idx: int            # index into word images
    x: float            # left edge (page px)
    base: float         # baseline y (page px)
    scale: float        # page px per model px
    angle: float        # degrees, positive = counter-clockwise
    ink: float = 1.0    # density factor
    sx: float = 1.0     # horizontal squeeze (cramped word at a line end)


@dataclass
class Page:
    words: list = field(default_factory=list)      # list[Placed]
    strokes: list = field(default_factory=list)    # list[(np.ndarray, width_factor)]


class Box:
    width: float = 0.0
    def place(self, L: "Layouter", x: float, base: float, scale: float) -> None: ...


class WordBox(Box):
    def __init__(self, L, idx, k=1.0):
        self.idx, self.k = idx, k
        w = L.words[idx]
        # carriers come out at slightly different sizes; pull each word most of the way to
        # the page x-height, then let size wander slowly along the line like a real hand
        norm = float(np.clip(L.model_xh / max(w.xheight, 4.0), 0.7, 1.45))
        L.size_walk = 0.8 * L.size_walk + L.rng.normal(0, 0.012 + 0.03 * L.look.mess)
        self.jit = norm * (1 + L.size_walk + L.rng.normal(0, 0.015 + 0.03 * L.look.mess)) * L.fatigue_k()
        self.sx = 1.0
        self.width = w.alpha.shape[1] * L.s * k * self.jit
        self.deco = L.word_deco[idx]
        self.text = w.text
        if "oops" in self.deco:
            self.width += 0  # strike is drawn over the same span

    def place(self, L, x, base, scale=1.0):
        w = L.words[self.idx]
        s = L.s * self.k * self.jit
        # model baseline -> page baseline
        dy = L.rng.normal(0, L.xh * (0.03 + 0.07 * L.look.mess))
        ang = L.line_angle + L.rng.normal(0, 0.4 + 1.2 * L.look.mess)
        ink = (1 + L.rng.normal(0, L.look.pressure * 0.5)) * L.session_ink
        L.page.words.append(Placed(self.idx, x, base + dy, s, ang, ink, self.sx))
        if L.rng.random() < L.look.retrace:
            # part of the word gone over again: a second, slightly offset pass
            L.page.words.append(Placed(self.idx, x + L.rng.normal(0, 0.6), base + dy + L.rng.normal(0, 0.6), s,
                                       ang + L.rng.normal(0, 0.3), ink * 0.8, self.sx))
        if L.rng.random() < L.look.hesitation:
            # pen rested before starting the word
            cx, cy = x - L.xh * 0.15, base + dy - L.xh * L.rng.uniform(0.3, 0.8)
            L.marks_out([np.array([[cx, cy], [cx + 0.8, cy + 0.6]])])
        top = base + dy - (w.baseline - 0) * s
        bot = top + w.alpha.shape[0] * s
        x1 = x + self.width * self.sx
        if "strike" in self.deco or "oops" in self.deco:
            L.marks_out(L.mk.strike(x, x1, base + dy, scribble="oops" in self.deco and L.rng.random() < 0.5))
        if "underline" in self.deco:
            L.marks_out(L.mk.underline(x, x1, base + dy))


class MarkBox(Box):
    def __init__(self, L, kind):
        self.kind = kind
        fn = {"arrow": lambda: L.mk.arrow(0, 0), "larrow": lambda: L.mk.arrow(0, 0, -1),
              "darrow": lambda: L.mk.arrow(0, 0, 1, True), "tick": lambda: L.mk.tick(0, 0),
              "cross": lambda: L.mk.cross(0, 0), "eq": lambda: L.mk.eq(0, 0), "lt": lambda: L.mk.lt(0, 0),
              "gt": lambda: L.mk.lt(0, 0, True), "pct": lambda: L.mk.pct(0, 0),
              "tilde": lambda: L.mk.tilde(0, 0), "sigma": lambda: L.mk.sigma(0, 0),
              "bullet": lambda: L.mk.bullet(0, 0)}[kind]
        self.strokes, self.width = fn()

    def place(self, L, x, base, scale=1.0):
        dy = L.rng.normal(0, L.xh * 0.04)
        L.marks_out([s + np.array([x, base + dy]) for s in self.strokes])


class HBox(Box):
    def __init__(self, L, children, k=1.0):
        self.children = children
        self.gaps = [L.gap(k) for _ in children]
        self.width = sum(c.width for c in children) + sum(self.gaps[:-1]) if children else 0

    def place(self, L, x, base, scale=1.0):
        for c, g in zip(self.children, self.gaps):
            c.place(L, x, base + L.slope_dy(c.width), scale)
            x += c.width + g


class FracBox(Box):
    def __init__(self, L, num, den):
        self.num, self.den = num, den
        self.width = max(num.width, den.width) + L.xh * 0.6

    def place(self, L, x, base, scale=1.0):
        bar = base - L.xh * 0.5
        L.marks_out(L.mk.frac_bar(x, x + self.width, bar))
        self.num.place(L, x + (self.width - self.num.width) / 2, bar - L.xh * 0.35, scale)
        self.den.place(L, x + (self.width - self.den.width) / 2, bar + L.xh * 1.35, scale)


class SqrtBox(Box):
    def __init__(self, L, inner):
        self.inner = inner
        self.width = inner.width + L.xh * 1.2

    def place(self, L, x, base, scale=1.0):
        st, w = L.mk.radical(x, base, base - L.xh * 1.75, self.inner.width)
        L.marks_out(st)
        self.inner.place(L, x + w + L.xh * 0.1, base, scale)


class CircleBox(Box):
    def __init__(self, L, inner):
        self.inner = inner
        self.width = inner.width + L.xh * 0.9

    def place(self, L, x, base, scale=1.0):
        self.inner.place(L, x + L.xh * 0.45, base, scale)
        L.marks_out(L.mk.circle_around(x + L.xh * 0.45, base - L.xh * 1.4, x + L.xh * 0.45 + self.inner.width, base + L.xh * 0.25))


class SupBox(Box):
    def __init__(self, L, inner):
        self.inner = inner
        self.width = inner.width

    def place(self, L, x, base, scale=1.0):
        self.inner.place(L, x, base - L.xh * 0.95, scale)


class Layouter:
    def __init__(self, words, word_deco, look: Look, seed: int = 0, page_size=A4):
        self.words = words
        self.word_deco = word_deco
        self.look = look
        self.rng = np.random.default_rng(seed)
        self.W, self.H = page_size
        self.xh = look.xh_mm * MM
        model_xh = float(np.median([w.xheight for w in words])) if words else 20.0
        self.model_xh = max(model_xh, 4.0)
        self.s = self.xh / self.model_xh
        self.size_walk = 0.0
        self.session_ink = 1.0
        self.slope_offset = 0.0
        self.mk = Marks(self.rng, self.xh, look.mess)
        self.pages: list[Page] = [Page()]
        self.line_angle = 0.0
        self.cursor_word = 0

    # -- helpers -------------------------------------------------------------------
    @property
    def page(self) -> Page:
        return self.pages[-1]

    def marks_out(self, strokes):
        for s in strokes:
            self.page.strokes.append((np.asarray(s), 1.0 + self.rng.normal(0, 0.06)))

    def progress(self) -> float:
        """0 at the top of the writing area, 1 at the bottom."""
        top, bot = getattr(self, "top", 0.0), getattr(self, "bottom", self.H)
        return float(np.clip((getattr(self, "y", top) - top) / max(1.0, bot - top), 0, 1))

    def fatigue_k(self) -> float:
        return 1 + self.look.fatigue * self.progress()

    def gap(self, k=1.0):
        m = self.look.mess * (1 + 5 * self.look.fatigue * self.progress())   # spacing loosens when tired
        g = self.xh * self.look.gap_xh * (1 + self.rng.normal(0, 0.13 + 0.29 * m)) * k
        return max(self.xh * 0.35, g)

    def slope_dy(self, w):
        return 0.0

    # -- token -> box ---------------------------------------------------------------
    def boxes(self, tokens, k=1.0):
        out = []
        for t in tokens:
            if isinstance(t, Word):
                out.append(WordBox(self, self.cursor_word, k))
                self.cursor_word += 1
            elif isinstance(t, Sym):
                if t.kind == "frac":
                    out.append(FracBox(self, HBox(self, self.boxes(t.args[0], k * 0.8), k * 0.8),
                                       HBox(self, self.boxes(t.args[1], k * 0.8), k * 0.8)))
                elif t.kind == "sqrt":
                    out.append(SqrtBox(self, HBox(self, self.boxes(t.args[0], k), k)))
                elif t.kind == "circle":
                    out.append(CircleBox(self, HBox(self, self.boxes(t.args[0], k), k)))
                elif t.kind == "sup":
                    out.append(SupBox(self, HBox(self, self.boxes(t.args[0], k * 0.65), k * 0.65)))
                else:
                    out.append(MarkBox(self, t.kind))
        return out

    # -- page flow --------------------------------------------------------------------
    def start(self):
        lm, tm, rm, bm = (v * MM for v in self.look.margin_mm)
        self.left, self.right, self.top, self.bottom = lm, self.W - rm, tm, self.H - bm
        self.y = self.top + self.xh * 1.6
        self.margin_walk = 0.0

    def new_line(self, extra=0.0):
        m = self.look.mess
        squash = 1 - self.look.tighten * self.progress() ** 3
        self.y += self.xh * (self.look.pitch * squash * (1 + self.rng.normal(0, 0.03 + 0.06 * m))) + extra
        if self.y > self.bottom:
            self.pages.append(Page())
            self.y = self.top + self.xh * 1.6

    def line_params(self):
        m = self.look.mess
        self.line_angle = -(self.look.slope_deg + self.slope_offset + self.rng.normal(0, 0.25 + 0.6 * m))  # degrees, ccw positive
        self.margin_walk = 0.85 * self.margin_walk + self.rng.normal(0, 3 + 10 * m)

    def flow(self, boxes, indent=0.0, hang=0.0, first_x=None):
        """Lay boxes left to right, wrapping at the right margin. Returns (x_end, y)."""
        self.line_params()
        x0 = self.left + indent + self.margin_walk
        x = x0 if first_x is None else first_x
        line_start = x0
        right = self.right - self.rng.uniform(0, self.xh * 2)
        for b in boxes:
            over = x + b.width - right
            if (over > 0 and isinstance(b, WordBox) and over < self.look.squeeze * b.width
                    and x + b.width < self.right + self.xh and x > line_start + 1):
                b.sx = max(0.82, (right - x) / b.width)      # squeezed in rather than wrapped
            elif x + b.width > right and x > line_start + 1:
                self.new_line()
                self.line_params()
                line_start = self.left + indent + hang + self.margin_walk
                x = line_start
                right = self.right - self.rng.uniform(0, self.xh * 2)
            t = math.tan(math.radians(self.line_angle))
            base = self.y - (x - line_start) * t
            b.place(self, x, base)
            extra = 0.0
            if isinstance(b, WordBox) and b.text[-1:] in ",.;:":
                extra = self.xh * self.look.punct_gap * self.rng.uniform(0, 1.6)
            x += b.width * getattr(b, "sx", 1.0) + self.gap() + extra
        return x, self.y

    def layout(self, blocks: list[Block]) -> list[Page]:
        self.start()
        first = True
        eq_x = None
        for b in blocks:
            if b.kind == "gap":
                self.y += self.xh * self.look.pitch * 0.45
                eq_x = None
                continue
            if not first:
                self.new_line()
            first = False
            if b.kind in ("heading", "subheading"):
                k = 1.45 if b.kind == "heading" else 1.2
                bx = self.boxes(b.tokens, k)
                if b.kind == "heading":
                    self.y += self.xh * 0.8
                x_end, y = self.flow([HBox(self, bx, k)], indent=self.xh * (0.5 if b.kind == "heading" else 0))
                start = self.left + self.margin_walk + (self.xh * 0.5 if b.kind == "heading" else 0)
                if b.kind == "heading":
                    self.marks_out(self.mk.underline(start, x_end - self.gap(), y + self.xh * 0.25))
                else:
                    self.marks_out(self.mk.underline(start, x_end - self.gap(), y))
                self.y += self.xh * 0.6
                eq_x = None
            elif b.kind == "rule":
                if self.rng.random() < self.look.session:
                    # written later: a slightly different slope and pressure from here on
                    self.slope_offset = self.rng.normal(0, 0.5)
                    self.session_ink = float(np.clip(self.rng.normal(1.0, 0.07), 0.85, 1.12))
                self.y += self.xh * 0.2
                self.marks_out(self.mk.rule(self.left + self.margin_walk, self.right - self.rng.uniform(0, self.xh * 3), self.y - self.xh * 0.8))
                self.y -= self.xh * 1.2
            elif b.kind in ("defn", "note"):
                lab = self.boxes(b.label)
                if b.kind == "defn":
                    lab = [HBox(self, lab)]
                    lab.append(MarkBox(self, "arrow"))
                body = self.boxes(b.tokens)
                # the term is underlined in definitions
                self.line_params()
                x = self.left + self.margin_walk
                base = self.y
                for bx in lab:
                    bx.place(self, x, base)
                    if b.kind == "defn" and isinstance(bx, HBox) and self.rng.random() < 0.7:
                        self.marks_out(self.mk.underline(x, x + bx.width, base))
                    x += bx.width + self.gap()
                self.flow(body, hang=self.xh * 2.2, first_x=x)
            elif b.kind in ("item", "bullet"):
                lab = self.boxes(b.label) if b.kind == "item" else [MarkBox(self, "bullet")]
                lw = sum(l.width for l in lab) + self.gap()
                self.line_params()
                x = self.left + self.margin_walk + self.xh * 0.6
                for l in lab:
                    l.place(self, x, self.y)
                    x += l.width
                x += self.gap()
                self.flow(self.boxes(b.tokens), indent=self.xh * 0.6 + lw, first_x=x)
            elif b.kind == "math":
                bx = self.boxes(b.tokens)
                indent = (eq_x - self.left) if eq_x is not None else self.xh * 2
                self.flow(bx, indent=indent - self.margin_walk * 0, first_x=None)
            elif b.kind == "table":
                self.table(b)
                eq_x = None
            else:  # para
                bx = self.boxes(b.tokens)
                # remember where the first "=" sits so following "= ..." lines align under it
                xpos = self.left + self.margin_walk
                for t, box in zip(b.tokens, bx):
                    if isinstance(t, Sym) and t.kind == "eq":
                        eq_x = xpos
                        break
                    xpos += box.width + self.xh * 0.75
                self.flow(bx)
        return self.pages

    def table(self, b: Block):
        rows = [[HBox(self, self.boxes(cell)) for cell in row] for row in b.cells]
        ncol = max(len(r) for r in rows)
        colw = [max((r[c].width for r in rows if c < len(r)), default=0) + self.xh * 1.6 for c in range(ncol)]
        x0 = self.left + self.margin_walk + self.xh * 0.5
        row_h = self.xh * self.look.pitch * 0.95
        y_top = self.y - self.xh * 1.9
        for ri, row in enumerate(rows):
            if ri:
                self.new_line(extra=-self.xh * self.look.pitch * 0.05)
            self.line_params()
            x = x0
            for c, cell in enumerate(row):
                cell.place(self, x + self.xh * 0.8, self.y)
                x += colw[c]
            if ri == 0:
                self.marks_out(self.mk.rule(x0 - self.xh * 0.3, x0 + sum(colw), self.y + self.xh * 0.75))
        y_bot = self.y + self.xh * 0.9
        xs = np.cumsum([x0] + colw)[1:-1]
        for xv in xs:
            self.marks_out(self.mk.vrule(xv, y_top, y_bot))

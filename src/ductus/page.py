"""End-to-end: text -> blocks -> words -> carriers -> diffusion -> OCR check -> layout -> page."""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import markup
from .layout import Layouter, Look, MM
from .markup import Word, Sym, Block

PRESETS = Path(__file__).with_name("presets")


def load_style(style: str) -> tuple[list[np.ndarray], dict]:
    """A style is a preset name, a line image, or a profile folder made by `ductus style`."""
    p = Path(style)
    if p.is_dir() and (p / "profile.json").exists():
        prof = json.loads((p / "profile.json").read_text())
        refs = [cv2.imread(str(p / "refs" / r), 0) for r in prof["refs"]]
        return [r for r in refs if r is not None], prof
    if p.is_file():
        return [cv2.imread(str(p), 0)], {}
    meta = json.loads((PRESETS / "presets.json").read_text())
    if style not in meta:
        raise ValueError(f"unknown style {style!r}; presets: {[k for k in meta if not k.startswith('_')]}")
    return [cv2.imread(str(PRESETS / meta[style]["file"]), 0)], {}


TYPO = "aeiounrst"


def typo(word: str, rng: random.Random) -> str:
    """A believable slip: swap two inner letters or change one."""
    if len(word) < 4:
        return word + rng.choice(TYPO)
    i = rng.randrange(1, len(word) - 2)
    if rng.random() < 0.5:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + rng.choice(TYPO) + word[i + 1:]


def add_corrections(blocks: list[Block], rate: float, rng: random.Random) -> None:
    if rate <= 0:
        return
    for b in blocks:
        if b.kind in ("table", "heading", "subheading"):
            continue
        new = []
        for t in b.tokens:
            if isinstance(t, Word) and not t.deco and len(t.text) >= 4 and t.text.isalpha() and rng.random() < rate:
                new.append(Word(typo(t.text, rng), frozenset({"oops"})))
            new.append(t)
        b.tokens = new


@dataclass
class Options:
    style: str = "quick"
    look: str = "natural"            # neat | natural | messy | profile
    mode: str = "plain"              # plain | notes | maths
    paper: str = "white"             # white | ruled | grid
    photo: bool = False
    ink: str | None = None
    seed: int = 0
    steps: int = 25
    verify: bool = True
    max_cer: float | None = None
    corrections: float | None = None
    pen: float | None = None
    size: float | None = None        # x-height mm
    log: object = print


@dataclass
class Result:
    pages: list = field(default_factory=list)       # RGB arrays
    layouts: list = field(default_factory=list)
    words: list = field(default_factory=list)
    look: Look | None = None
    stats: dict = field(default_factory=dict)


class Writer:
    """Keeps the models loaded between pages."""

    def __init__(self, steps: int = 25, verify: bool = True):
        from .engines.diffbrush import DiffBrushEngine
        self.engine = DiffBrushEngine(steps=steps)
        self.reader = None
        if verify:
            from .verify import Reader
            self.reader = Reader()

    def write(self, text: str, o: Options) -> Result:
        t0 = time.time()
        refs, prof = load_style(o.style)
        look = Look.preset(o.look if o.look != "profile" else "natural")
        if prof:
            pl = prof.get("look", {})
            for k in ("xh_mm", "pitch", "slope_deg", "mess", "ink"):
                if k in pl:
                    setattr(look, k, pl[k])
            if o.look == "neat":
                look.mess = min(look.mess, 0.2)
        if o.ink:
            look.ink = o.ink
        if o.corrections is not None:
            look.corrections = o.corrections
        if o.pen:
            look.pen = o.pen
        if o.size:
            look.xh_mm = o.size
        rng = random.Random(o.seed)
        blocks = markup.parse(text, o.mode)
        add_corrections(blocks, look.corrections, rng)
        word_objs = [w for b in blocks for w in markup.block_words(b)]
        texts = [w.text for w in word_objs]
        personal = bool(prof)
        # strict check for neat/preset writing; loose check that only catches wrong or
        # missing words for a personal hand, so it is never pushed towards neatness
        max_cer = o.max_cer if o.max_cer is not None else (prof.get("max_cer", 0.5) if personal else
                                                          (0.15 if o.look == "neat" else 0.25))
        from .synth import Synthesizer
        words = []
        # several reference lines from a personal profile -> rotate them for natural variety
        chunks = np.array_split(np.arange(len(texts)), max(1, min(len(refs), (len(texts) + 39) // 40)))
        stats = {"lines_generated": 0, "still_failing": 0}
        for ci, idx in enumerate(chunks):
            if len(idx) == 0:
                continue
            ref = refs[ci % len(refs)]
            syn = Synthesizer(self.engine, ref, self.reader if o.verify else None, max_cer=max_cer,
                              loose=personal, seed=o.seed * 1000 + ci * 37, log=o.log)
            words += syn.run([texts[i] for i in idx])
            for k in stats:
                stats[k] += syn.stats.get(k, 0)
        # oops words are struck through by the layout
        deco = [w.deco for w in word_objs]
        L = Layouter(words, deco, look, seed=o.seed)
        pages = L.layout(blocks)
        from .render import ink_coverage, compose
        from .paper import make_paper, phone_photo
        # pen width for drawn marks follows the measured stroke width of the writing
        from .segment import stroke_width
        sw = float(np.median([stroke_width(w.alpha) for w in words[:40]])) * L.s
        out = []
        nrng = np.random.default_rng(o.seed)
        for pi, pg in enumerate(pages):
            ink = ink_coverage(pg, words, look.mess, look.pen, max(2.0, sw), nrng)
            img = compose(ink, look.ink, look.pressure, nrng, make_paper(o.paper, seed=o.seed + pi))
            if o.photo:
                img = phone_photo(img, seed=o.seed + pi)
            out.append(img)
        stats.update(words=len(texts), seconds=round(time.time() - t0, 1), pages=len(pages),
                     ocr_mean_score=round(float(np.mean([w.score for w in words])), 3) if words else None,
                     pen_px=round(sw, 2), max_cer=max_cer)
        return Result(out, pages, words, look, stats)

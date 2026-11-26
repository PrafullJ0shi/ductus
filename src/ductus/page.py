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
    style: str | None = None        # None: "upright", the plainest print-like preset
    engine: str = "default"          # default (rnn if installed, else diffusion) | rnn | diffusion | auto
    rnn_style: int | None = 1        # built-in RNN writer style 0-7; 1 has plain print r, f and l
    rnn_bias: float = 1.5            # RNN sampling bias: higher is neater
    legibility: str = "strict"       # strict (default: line + word read-back) | loose (personal hands, old behaviour)
    plain: str = "abcdefghijklmnopqrstuvwxyz"          # letters that must use the plain form (no looped f/r/ascenders); "" = allow loops
    look: str = "natural"            # neat | natural | messy | profile
    mode: str = "plain"              # plain | notes | maths
    paper: str | None = None         # white | ruled | grid (default white; grid for a notebook photo)
    photo: str | bool | None = None  # None | "a4" (loose sheet on a desk) | "notebook" (spiral grid pad); True = a4
    ink: str | None = None
    seed: int = 0
    steps: int = 25
    verify: bool = True
    max_cer: float | None = None
    corrections: float | None = None
    pen: str | None = None           # pen preset (pens.PENS); None = blue ballpoint
    tip: float | None = None         # nominal tip size in mm (0.5, 0.7, 1.0)
    weight: float | None = None      # stroke weight multiplier on top of the pen
    flat_ink: bool = False           # old flat ink model (kept for comparison)
    cache: bool = True               # reuse words generated earlier for the same text and settings
    ink_model: str = "realism"       # realism (default) | pens | flat
    humanize: float = 0.5            # rate of the realism.human_errors pass on the laid-out page (0 = off)
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
        self.steps = steps
        self._diff = None
        self._rnn = {}
        self.reader = None
        if verify:
            from .verify import Reader
            self.reader = Reader()

    @property
    def engine(self):
        """The diffusion engine (style copying), loaded on first use."""
        if self._diff is None:
            from .engines.diffbrush import DiffBrushEngine
            self._diff = DiffBrushEngine(steps=self.steps)
        return self._diff

    def rnn(self, style=1, bias: float = 1.5):
        key = (style, bias)
        if key not in self._rnn:
            from .engines.rnn import RNNEngine
            self._rnn[key] = RNNEngine(style=style, bias=bias)
        return self._rnn[key]

    def engines(self, o: "Options"):
        """(primary, fallback, primary_rounds) for the engine mode.
        default/rnn: plain, readable letters from the Graves RNN; words it cannot write or that
        still fail the read-back go to diffusion. diffusion: letter shapes copied from the
        reference image. auto: diffusion first, the RNN for words that do not read."""
        from .engines import rnn as R
        mode = o.engine
        if mode == "rnn" or (mode == "default" and R.available()):
            if not R.available():
                raise RuntimeError("--engine rnn needs the RNN model; see src/ductus/engines/rnn/__init__.py")
            return self.rnn(o.rnn_style, o.rnn_bias), self.engine_lazy(), 4   # then diffusion for what still fails
        if mode == "auto" and R.available():
            return self.engine, self.rnn(o.rnn_style, o.rnn_bias), 2
        return self.engine, None, 99

    def engine_lazy(self):
        w = self

        class _Lazy:
            name = "diffusion"

            def generate(self, *a, **k):
                return w.engine.generate(*a, **k)
        return _Lazy()

    def write(self, text: str, o: Options) -> Result:
        t0 = time.time()
        refs, prof = load_style(o.style or "upright")
        look = Look.preset(o.look if o.look != "profile" else "natural")
        if prof:
            pl = sane_look(prof.get("look", {}))
            for k in ("xh_mm", "pitch", "slope_deg", "mess", "ink", "gap_xh"):
                if k in pl:
                    setattr(look, k, pl[k])
            if o.look == "neat":
                look.mess = min(look.mess, 0.2)
        _match_reference(look, refs, prof)
        if o.ink:
            look.ink = o.ink
        if o.corrections is not None:
            look.corrections = o.corrections
        if o.weight:
            look.pen = o.weight
        if o.size:
            look.xh_mm = o.size
        rng = random.Random(o.seed)
        if prof and o.look != "neat":
            from . import habits as H
            hb = H.load(o.style)
            H.apply_look(look, hb)
            text = H.apply_text(text, hb, o.seed)
        blocks = markup.parse(text, o.mode)
        add_corrections(blocks, look.corrections, rng)
        word_objs = [w for b in blocks for w in markup.block_words(b)]
        texts = [w.text for w in word_objs]
        personal = bool(prof)
        # readable first: every word must pass both the line and the word read-back (strict).
        # A personal hand is compared case- and punctuation-insensitively and keeps its own
        # proportions; messiness lives in spacing, baseline and shape, not garbled letters.
        # legibility="loose" restores the old check that only caught wrong or missing words.
        max_cer = _max_cer(o, prof)
        strict = o.legibility != "loose"
        from .synth import Synthesizer
        words = []
        ckey = _cache_key(texts, o, max_cer, o.steps)
        cached = _cache_get(ckey) if o.cache else None
        # several reference lines from a personal profile -> rotate them for natural variety
        chunks = np.array_split(np.arange(len(texts)), max(1, min(len(refs), (len(texts) + 39) // 40)))
        stats = {"lines_generated": 0, "still_failing": 0}
        for ci, idx in enumerate(chunks if cached is None else []):
            if len(idx) == 0:
                continue
            ref = refs[ci % len(refs)]
            prim, fb, pr = self.engines(o)
            syn = Synthesizer(prim, ref, self.reader if o.verify else None, fallback=fb, primary_rounds=pr, fallback_all=o.engine == "auto", max_cer=max_cer,
                              loose=personal, seed=o.seed * 1000 + ci * 37, log=o.log,
                              strict=strict, rounds=(pr + 2) if fb is not None and pr < 99 else (5 if o.look == "neat" else 4), plain=o.plain, letter_profile=_letters(o.style))
            words += syn.run([texts[i] for i in idx])
            for k in stats:
                stats[k] += syn.stats.get(k, 0)
        if cached is not None:
            words, stats = cached
        elif o.cache:
            _cache_put(ckey, (words, stats))
        # oops words are struck through by the layout
        deco = [w.deco for w in word_objs]
        L = Layouter(words, deco, look, seed=o.seed)
        pages = L.layout(blocks)
        from .render import ink_coverage, compose
        from .paper import make_paper, phone_photo
        # pen width for drawn marks follows the measured stroke width of the writing
        from .segment import stroke_width
        pen = _pen_for(o, look, prof)
        sw = _stroke_px(look, pen, o, prof)
        from .pens import nib_kernel
        from .render import compose_physical, SS
        nib = nib_kernel(sw * SS, pen)
        out = []
        nrng = np.random.default_rng(o.seed)
        for pi, pg in enumerate(pages):
            paper = make_paper(_paper(o), seed=o.seed + pi)
            if o.humanize and o.look != "neat":
                from .humanize_bridge import humanize_page
                humanize_page(pg, words, L, o.humanize, o.seed + pi)
            model = "flat" if o.flat_ink else o.ink_model
            if model == "flat":
                ink = ink_coverage(pg, words, look.mess, look.pen, sw, nrng)
                img = compose(ink, look.ink, look.pressure, nrng, paper)
            elif model == "realism":
                from .render import compose_realism
                from .render import hex_rgb
                shape, count = ink_coverage(pg, words, look.mess, look.pen, sw, nrng, nib=nib, with_count=True)
                rgb = tuple(int(v) for v in hex_rgb(o.ink or look.ink)) if (o.ink or (prof and not o.pen)) else None
                img = compose_realism(shape, count, pen, o.seed + pi, paper, rgb=rgb, smudges=look.smudges)
            else:
                shape, count = ink_coverage(pg, words, look.mess, look.pen, sw, nrng, nib=nib, with_count=True)
                img = compose_physical(shape, count, pen, look.pressure, nrng, paper, smudges=look.smudges)
            if o.photo:
                img = phone_photo(img, seed=o.seed + pi, surface=_surface(o.photo))
            out.append(img)
        stats.update(words=len(texts), seconds=round(time.time() - t0, 1), pages=len(pages),
                     ocr_mean_score=round(float(np.mean([w.score for w in words])), 3) if words else None,
                     pen_px=round(sw, 2), max_cer=max_cer)
        look.ink = pen.ink
        stats["pen"] = pen.name
        return Result(out, pages, words, look, stats)


def _match_reference(look: Look, refs, prof) -> None:
    """Take word spacing from the reference lines (profiles carry their own measurement)."""
    from .segment import ref_habits
    hs = [ref_habits(r) for r in refs[:8] if r is not None]
    gaps = [h["gap_xh"] for h in hs if "gap_xh" in h]
    if gaps and not (prof and "gap_xh" in prof.get("look", {})):
        look.gap_xh = float(np.median(gaps))
    sx = [h["stroke_xh"] for h in hs if "stroke_xh" in h]
    look._ref_stroke_xh = float(np.median(sx)) if sx else None
    if prof and prof.get("look", {}).get("stroke_xh"):
        look._ref_stroke_xh = float(prof["look"]["stroke_xh"])   # measured over all pages


def _pen_for(o: Options, look: Look, prof: dict):
    """The pen preset, with the ink colour of a personal profile (measured from the photos)
    when the user did not pick a pen or colour."""
    from .pens import get_pen
    ink = o.ink or (look.ink if (prof and not o.pen and prof.get("look", {}).get("ink")) else None)
    return get_pen(o.pen, o.tip, ink)


def _stroke_px(look: Look, pen=None, o: Options | None = None, prof: dict | None = None) -> float:
    """Line width in page pixels. A personal profile gives the measured width of the
    writer's own pen (stroke / x-height times the page x-height) unless a pen or tip is
    chosen; otherwise the pen preset's line width."""
    from .layout import MM
    ref = getattr(look, "_ref_stroke_xh", None)
    if prof and ref and not (o and (o.pen or o.tip)):
        mm = float(np.clip(ref * look.xh_mm, 0.25, 0.75))
    elif pen is not None:
        mm = pen.line_mm
    else:
        mm = look.pen_mm
    return mm * MM


def _pen_match(look: Look, words) -> float:
    """Stroke weight multiplier that brings generated strokes to the reference's weight."""
    from .segment import stroke_width
    ref = getattr(look, "_ref_stroke_xh", None)
    if not ref:
        return look.pen
    gen = float(np.median([stroke_width(w.alpha) / max(w.xheight, 4.0) for w in words[:60]]))
    return float(np.clip(ref / max(gen, 1e-3), 0.85, 1.6))


def write_lines(writer: "Writer", lines: list[str], o: Options) -> tuple[list[np.ndarray], dict]:
    """Write each string as one unwrapped line and return tight grayscale crops.
    Used by the evaluation scripts so every system is scored on the same line images."""
    from .synth import Synthesizer
    from .render import ink_coverage, compose
    from .segment import stroke_width, tight_bbox
    t0 = time.time()
    refs, prof = load_style(o.style or "upright")
    look = Look.preset(o.look if o.look != "profile" else "natural")
    if prof:
        for k, v in sane_look(prof.get("look", {})).items():
            if hasattr(look, k):
                setattr(look, k, v)
    look.margin_mm = (4, 4, 4, 4)
    _match_reference(look, refs, prof)
    blocks = [markup.parse(t, o.mode)[0] for t in lines]
    per_line = [[w.text for w in markup.block_words(b)] for b in blocks]
    flat = [w for ws in per_line for w in ws]
    max_cer = _max_cer(o, prof)
    prim, fb, pr = writer.engines(o)
    syn = Synthesizer(prim, refs[0], writer.reader if o.verify else None, fallback=fb, primary_rounds=pr, fallback_all=o.engine == "auto", max_cer=max_cer,
                      loose=bool(prof), seed=o.seed, log=o.log,
                      strict=o.legibility != "loose", rounds=(pr + 2) if fb is not None and pr < 99 else (5 if o.look == "neat" else 4), plain=o.plain, letter_profile=_letters(o.style))
    ckey = _cache_key(["lines"] + flat, o, max_cer, o.steps)
    cached = _cache_get(ckey) if o.cache else None
    if cached is not None:
        words, syn.stats = cached
    else:
        words = syn.run(flat)
        if o.cache:
            _cache_put(ckey, (words, dict(syn.stats)))
    out, k = [], 0
    rng = np.random.default_rng(o.seed)
    import cv2
    from .pens import nib_kernel
    from .render import compose_physical, SS
    pen = _pen_for(o, look, prof)
    sw = _stroke_px(look, pen, o, prof)
    nib = nib_kernel(sw * SS, pen)
    for b, ws in zip(blocks, per_line):
        sub = words[k:k + len(ws)]
        k += len(ws)
        L = Layouter(sub, [w.deco for w in markup.block_words(b)], look, seed=o.seed + k, page_size=(5600, 520))
        pg = L.layout([b])[0]
        if o.flat_ink:
            ink = ink_coverage(pg, sub, look.mess, look.pen, sw, rng, size=(5600, 520))
            img = compose(ink, "#000000", look.pressure, rng)[..., 0]
        elif o.ink_model == "pens":
            shape, count = ink_coverage(pg, sub, look.mess, look.pen, sw, rng, size=(5600, 520), nib=nib, with_count=True)
            img = cv2.cvtColor(compose_physical(shape, count, pen, look.pressure, rng), cv2.COLOR_RGB2GRAY)
        else:
            from .render import compose_realism
            shape, count = ink_coverage(pg, sub, look.mess, look.pen, sw, rng, size=(5600, 520), nib=nib, with_count=True)
            img = cv2.cvtColor(compose_realism(shape, count, pen, o.seed + k, None), cv2.COLOR_RGB2GRAY)
        bb = tight_bbox((255 - img.astype(np.float32)) / 255.0, 0.2)
        x0, y0, x1, y1 = bb
        out.append(img[max(0, y0 - 16):y1 + 16, max(0, x0 - 16):x1 + 16])
    stats = dict(syn.stats, seconds=round(time.time() - t0, 1))
    return out, stats


CACHE_VERSION = 8  # bump when word synthesis changes


def _cache_dir():
    import os
    from pathlib import Path
    return Path(os.environ.get("DUCTUS_CACHE", Path.home() / ".cache" / "ductus" / "words"))


def _cache_key(texts, o, max_cer, steps) -> str:
    import hashlib, json
    sig = o.style
    if o.style:
        from pathlib import Path
        sp = Path(o.style)
        f = sp / "profile.json" if sp.is_dir() else sp
        if f.exists():
            sig = [o.style, hashlib.sha1(f.read_bytes()).hexdigest()]
    k = json.dumps([CACHE_VERSION, o.engine, o.rnn_style, o.rnn_bias, texts, sig, o.look, o.seed, max_cer, o.verify, steps, o.mode, o.plain, o.legibility], default=str)
    return hashlib.sha1(k.encode()).hexdigest()


def _cache_get(key):
    import pickle
    f = _cache_dir() / f"{key}.pkl"
    if f.exists():
        try:
            return pickle.loads(f.read_bytes())
        except Exception:
            return None
    return None


def _cache_put(key, val) -> None:
    import pickle
    d = _cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{key}.pkl").write_bytes(pickle.dumps(val))


def _surface(photo) -> str:
    return "a4" if photo is True or photo in (None, "", "a4") else str(photo)


def _paper(o: "Options") -> str:
    if o.paper:
        return o.paper
    return "grid" if o.photo and _surface(o.photo) == "notebook" else "white"


def _max_cer(o: "Options", prof: dict) -> float:
    if o.max_cer is not None:
        return o.max_cer
    if o.look == "neat":
        return 0.15
    if prof and o.legibility == "loose":
        return prof.get("max_cer", 0.5)
    return 0.15   # one slip per 7 letters; 0.25 let "Coeffiuent" through


# plausible ranges for measured profile values; a value outside (a bad grid or scale estimate
# from a photo) is dropped and the preset value is kept. mess is capped so a measured messy
# hand stays readable: the mess goes into spacing and baseline, not letter shapes.
SANE = {"xh_mm": (1.5, 5.0), "pitch": (1.6, 4.0), "slope_deg": (-6.0, 6.0), "gap_xh": (0.5, 2.0),
        "mess": (0.0, 0.6)}


def sane_look(pl: dict) -> dict:
    out = {}
    for k, v in pl.items():
        if k in SANE and isinstance(v, (int, float)):
            lo, hi = SANE[k]
            if k == "mess":
                v = min(max(v, lo), hi)
            elif not lo <= v <= hi:
                continue
        out[k] = v
    return out


def _letters(style) -> dict | None:
    """Per-letter loop rates of a profile (letters.json from scripts/letter_glyphs.py)."""
    if not style:
        return None
    f = Path(style) / "letters.json"
    return json.loads(f.read_text()) if f.exists() else None

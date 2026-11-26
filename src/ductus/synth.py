"""Word synthesis: pack words into full-length carrier lines, generate them with the
diffusion engine, split them back into words, OCR-check, and retry the failures.

Why carriers: the line model is trained on 35-61 character lines and is weakest on
short inputs. Writing every word inside a normal-length line keeps it in its comfort
zone, and every occurrence of a word is a fresh sample, so no two copies match.
"""
from __future__ import annotations

import re
import sys
import random
from dataclasses import dataclass, field

import numpy as np

from .segment import split_by_length, clean_line, split_words, baseline_and_xheight, tight_bbox
from .verify import cer, cer_wild, normalise
from .letterforms import flourishes

FILLER = ("and the of with over into from about under while still both more when then "
          "light river window paper simple garden morning letter quiet number travel").split()
CARRIER_LEN = 46


def align_words(targets: list[str], hyp: list[str]) -> list[str]:
    """Align OCR words to target words (edit-distance DP over word sequences).
    Returns, for each target word, the OCR text matched to it ("" if deleted)."""
    n, m = len(targets), len(hyp)
    def sub(a, b):
        return min(1.0, cer(normalise(a) or a, normalise(b) or b))
    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + sub(targets[i - 1], hyp[j - 1]))
    out = [""] * n
    i, j = n, m
    while i > 0 and j > 0:
        if np.isclose(D[i, j], D[i - 1, j - 1] + sub(targets[i - 1], hyp[j - 1])):
            out[i - 1] = hyp[j - 1]; i -= 1; j -= 1
        elif np.isclose(D[i, j], D[i - 1, j] + 1):
            i -= 1
        else:
            j -= 1
    return out



def _drop_specks(sub: np.ndarray, xh: float) -> np.ndarray:
    """Remove tiny isolated marks that are not i/j dots (nothing below them within 1.2 xh)."""
    from scipy import ndimage
    m = sub > 0
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    if n < 2:
        return sub
    out = sub.copy()
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w_ = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if max(h, w_) > 0.3 * xh:
            continue
        below = m[sl[0].stop:min(m.shape[0], sl[0].stop + int(1.2 * xh)), max(0, sl[1].start - 2):sl[1].stop + 2]
        if below.any():
            continue                       # a dot over a stem
        out[lab == i] = 0
    return out

@dataclass
class WordImage:
    text: str
    alpha: np.ndarray        # ink coverage 0..1, cropped, at model scale (64 px lines)
    baseline: float          # row of the baseline inside alpha
    xheight: float
    score: float = 1.0       # 1 - CER of the OCR read-back (1.0 when unchecked)
    read: str = ""


_DEBUG = bool(__import__("os").environ.get("DUCTUS_DEBUG"))
SOLO = re.compile(r"^[(\[]?[0-9A-Za-z]{1,2}[).\]]$")


@dataclass
class Synthesizer:
    engine: object
    style_img: np.ndarray
    reader: object | None = None
    max_cer: float = 0.25          # neat: 0.15-0.25; personal style: calibrated, looser
    loose: bool = False            # True: compare case/punctuation-insensitively
    rounds: int = 3
    strict: bool = False           # neat: a word must pass both the line and the word read
    batch: int = 8
    seed: int = 0
    log: callable = print
    fallback: object | None = None # second engine, used from round primary_rounds on and for
    primary_rounds: int = 99       # words the primary cannot write (engine.supported)
    fallback_all: bool = False     # False: fallback rounds only take words the primary cannot write
    letter_profile: dict | None = None  # per-letter loop rates of the writer (letters.json)
    plain: str = "abcdefghijklmnopqrstuvwxyz"   # letters that must use the plain form (no closed loops); "" = off
    stats: dict = field(default_factory=dict)

    def _carriers(self, words: list[tuple[int, str]], rng: random.Random):
        """Greedy pack (id, text) pairs into carrier strings of exactly CARRIER_LEN chars."""
        out, cur, n = [], [], 0
        solo = []
        for wid, w in words:
            if SOLO.match(w):          # "4)", "(b)": the gap inside them looks like a word gap
                solo.append([(wid, w)])
                continue
            extra = len(w) + (1 if cur else 0)
            if cur and n + extra > CARRIER_LEN:
                out.append(cur)
                cur, n = [], 0
                extra = len(w)
            cur.append((wid, w))
            n += extra
        if cur:
            out.append(cur)
        carriers = [(g, g[0][1]) for g in solo]
        for group in out:
            text = " ".join(w for _, w in group)
            fill = []
            while len(text) + 1 + len(" ".join(fill)) < CARRIER_LEN:
                fill.append(rng.choice(FILLER))
            full = (text + (" " + " ".join(fill) if fill else ""))
            if len(full) > CARRIER_LEN and fill:
                full = full[:CARRIER_LEN].rstrip()
                if len(full.split()) <= len(group):     # trimmed every filler away
                    full = text
            carriers.append((group, full))
        return carriers

    def _accept(self, target: str, read: str) -> float:
        """Acceptance error. TrOCR often drops case and punctuation on its own, so both
        modes compare normalised text; the threshold (max_cer) is what differs."""
        t, r = normalise(target), normalise(read)
        if not t:                      # pure punctuation such as "-" or ":"
            if not read.strip():
                return 1.0
            # the crop must not hold a neighbouring word (the read has letters or digits)
            return 1.0 if normalise(read) else 0.0
        wild = "rf" if getattr(self, "_glyph_mode", "") == "cursive" else ""
        return cer_wild(t, r, wild)

    def run(self, texts: list[str]) -> list[WordImage]:
        """Return one WordImage per input word (same order)."""
        rng = random.Random(self.seed)
        self.stats = {}
        best: dict[int, WordImage] = {}
        pending = list(enumerate(texts))
        seed = self.seed
        gen_lines = 0
        for rnd in range(self.rounds):
            if not pending:
                break
            eng = self.engine if (rnd < self.primary_rounds or self.fallback is None) else self.fallback
            todo = pending
            if eng is self.engine and self.fallback is not None and hasattr(eng, "supported"):
                todo = [(wid, w) for wid, w in pending if eng.supported(w)]   # others wait for the fallback
                if not todo:
                    eng, todo = self.fallback, pending
            self._glyph_mode = getattr(eng, "glyphs", "")
            if hasattr(eng, "round"):
                eng.round = rnd                  # the RNN writes neater on later rounds
            carriers = self._carriers(todo, rng)
            self.log(f"round {rnd + 1} ({getattr(eng, 'name', 'diffusion')}): {len(todo)} words in {len(carriers)} lines")
            for i in range(0, len(carriers), self.batch):
                chunk = carriers[i:i + self.batch]
                seed += 1
                imgs = eng.generate([c[1] for c in chunk], self.style_img, seed=seed)
                gen_lines += len(chunk)
                for (group, full), img in zip(chunk, imgs):
                    a = clean_line(img)
                    n_full = len(full.split())
                    spans = split_words(a, n_full) or split_words(a, n_full, min_gap=1)
                    if spans is None:
                        spans = split_by_length(a, full.split())   # joined-up hand: cut by length
                    if spans is None:
                        continue
                    bl, xh = baseline_and_xheight(a)
                    real = spans[:len(group)]
                    read_words = None
                    if self.reader is not None:
                        x_end = min(img.shape[1], real[-1][1] + 6)
                        line_read = self.reader.read([img[:, :x_end]])[0]
                        read_words = align_words([w for _, w in group], line_read.split())
                    ink_w = real[-1][1] - real[0][0]
                    chars = sum(len(w) for _, w in group) + len(group) - 1
                    cw = ink_w / max(1, chars)             # mean character width on this line
                    items = []
                    for k, ((wid, w), (x0, x1)) in enumerate(zip(group, real)):
                        sub = a[:, max(0, x0 - 1):x1 + 1]
                        if getattr(eng, "glyphs", "") == "cursive" and w.isalpha():
                            sub = _drop_specks(sub, xh)   # leftovers of the r/f redraw
                        if tight_bbox(sub) is None:
                            continue
                        ratio = (x1 - x0) / max(1.0, cw * len(w))
                        rgb = img[:, max(0, x0 - 1):x1 + 1].copy()
                        if getattr(eng, "glyphs", "") == "cursive" and w.isalpha():
                            rgb[sub <= 0] = 255 if rgb.dtype == np.uint8 else 1.0
                        items.append((wid, w, sub, rgb, k, ratio))
                    # word-level read-back as a second opinion (catches split errors and
                    # misspellings the line-level language prior quietly corrects)
                    word_reads = None
                    if self.reader is not None and items:
                        word_reads = self.reader.read([it[3] for it in items])
                    for n, (wid, w, sub, _, k, ratio) in enumerate(items):
                        if read_words is not None:
                            e_line = self._accept(w, read_words[k])
                            e_word = self._accept(w, word_reads[n]) if (len(normalise(w)) >= 3 or any(c.isdigit() for c in w)) else e_line
                            err = max(e_line, e_word) if self.strict else 0.5 * (e_line + e_word)
                            lw = normalise(w)
                            if (self._glyph_mode == "cursive" and lw and not any(c.isdigit() for c in w)
                                    and sum(c in "rf" for c in lw) >= 0.4 * len(lw)
                                    and abs(len(re.sub(r"[^a-z]", "", normalise(word_reads[n]))) - len(lw)) <= 1):
                                err = e_line     # isolated word OCR cannot read the cursive r/f; the line read can
                        else:
                            err = 0.0
                        lo, hi = (0.55, 1.9) if self.strict and not self.loose else (0.4, 2.4)
                        if not (lo <= ratio <= hi) and len(w) >= 3:
                            err = max(err, 1.0)          # the split put this word in the wrong span
                        elif len(w) <= 2 and ratio > 2.6:
                            err = max(err, 1.0)          # a short token whose crop holds more than itself
                        if self.plain and any(c in self.plain for c in w):
                            fl = flourishes(sub, w, bl, float(np.clip(cw * getattr(eng, "xh_per_cw", 1.15), 0.5 * xh, 1.5 * xh)), set(self.plain),
                                            self.letter_profile)
                            if fl:
                                err = max(err, max(0.6, self.max_cer + 0.05))   # looped f/r etc: write it again
                                self.stats["flourish_rejects"] = self.stats.get("flourish_rejects", 0) + 1
                        if _DEBUG:
                            print(f"word {w!r} line={read_words[k] if read_words else None!r} "
                                  f"word={word_reads[n] if word_reads else None!r} ratio={ratio:.2f} err={err:.2f}", file=sys.stderr)
                        r = read_words[k] if read_words is not None else ""
                        # size by mean letter width: the row-profile x-height is fooled by ascenders
                        wxh = float(np.clip(cw * getattr(eng, "xh_per_cw", 1.15), 0.5 * xh, 1.5 * xh))
                        wbl = bl
                        if getattr(eng, "per_word_xh", False):
                            wbl, wxh = word_band(sub, wxh, bl)   # RNN lines drift: own baseline and size
                        cand = WordImage(w, sub, wbl, wxh, 1.0 - err, r)
                        if _DEBUG:
                            import os
                            from PIL import Image as _I
                            os.makedirs("/tmp/dbg_words", exist_ok=True)
                            _I.fromarray((255 - np.clip(sub, 0, 1) * 255).astype(np.uint8)).save(f"/tmp/dbg_words/r{rnd}_{wid}_{w}.png")
                        if wid not in best or cand.score > best[wid].score:
                            best[wid] = cand
            pending = [(wid, w) for wid, w in pending
                       if wid not in best or (1.0 - best[wid].score) > self.max_cer]
        self.stats = {"flourish_rejects": self.stats.get("flourish_rejects", 0), "lines_generated": gen_lines, "words": len(texts),
                      "still_failing": len(pending), "rounds_used": rnd + 1}
        missing = [wid for wid, _ in enumerate(texts) if wid not in best]
        if missing:
            raise RuntimeError(f"could not synthesise {len(missing)} words, e.g. {texts[missing[0]]!r}")
        return [best[i] for i in range(len(texts))]


def word_band(alpha: np.ndarray, line_xh: float, line_bl: float) -> tuple[float, float]:
    """(baseline, x-height) of one word from its own ink. Rows are scored by how many strokes
    cross them; the x-band is the contiguous run of rows around the busiest one that keeps at
    least half the peak crossings (ascenders and descenders carry one or two strokes). Falls back
    to the line values when the word is too small to measure."""
    m = alpha > 0.35
    runs = (np.diff(m.astype(np.int8), axis=1) == 1).sum(axis=1).astype(np.float32)
    if runs.max() < 2:
        return line_bl, line_xh
    runs = np.convolve(runs, np.ones(3) / 3, mode="same")
    pk = int(np.argmax(runs)); thr = 0.5 * runs[pk]
    t = pk
    while t > 0 and runs[t - 1] >= thr:
        t -= 1
    b = pk
    while b < len(runs) - 1 and runs[b + 1] >= thr:
        b += 1
    h = float(b - t + 1)
    if not (0.55 * line_xh <= h <= 1.6 * line_xh):
        return line_bl, float(np.clip(h, 0.65 * line_xh, 1.5 * line_xh))
    return float(b + 1), h


def word_xheight(alpha: np.ndarray, line_xh: float) -> float:
    return word_band(alpha, line_xh, 0.0)[1]
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
from .verify import cer, cer_wild, confused, normalise
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


def _lcs(a, b) -> int:
    """Length of the longest common subsequence."""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]

@dataclass
class WordImage:
    text: str
    alpha: np.ndarray        # ink coverage 0..1, cropped, at model scale (64 px lines)
    baseline: float          # row of the baseline inside alpha
    xheight: float
    score: float = 1.0       # 1 - CER of the OCR read-back (1.0 when unchecked)
    read: str = ""
    template: bool = False   # drawn from printglyphs (numbers, symbols, capitals), not a model


_DEBUG = bool(__import__("os").environ.get("DUCTUS_DEBUG"))
def _cap(text: str) -> int:
    """Token cap for reading a crop of `text`: every token is at least one character, so a
    read this long is already wrong; stopping there saves the decoder steps of garbage reads."""
    return len(text) + 10


SOLO = re.compile(r"^[(\[]?[0-9A-Za-z]{1,2}[).\]]$|^-?[0-9]{1,3}$|^[A-Za-z]$")   # tiny tokens: own line


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
    templates: bool = False        # numbers, symbols and capital-only tokens drawn from printglyphs
    checker: object | None = None  # larger OCR model that re-reads what `reader` (a fast one) rejects
    deadline: float | None = None  # time.time() after which no new rewrite round starts
    bank: object | None = None     # wordbank.Bank: verified words kept between runs
    mess: float = 0.3

    def _carriers(self, words: list[tuple[int, str]], rng: random.Random):
        """Greedy pack (id, text) pairs into carrier strings of exactly CARRIER_LEN chars."""
        out, cur, n = [], [], 0
        solo = []
        for wid, w in words:
            if getattr(self, "_solo_ok", True) and (SOLO.match(w) or getattr(self, "_solo_all", False)):
                # "4)", "(b)"; later RNN rounds: every word alone (diffusion repeats a lone word)
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
        if confused(t, r):
            return 1.0
        return cer_wild(t, r, wild)

    def run(self, texts: list[str]) -> list[WordImage]:
        """Return one WordImage per input word (same order)."""
        rng = random.Random(self.seed)
        self.stats = {}
        best: dict[int, WordImage] = {}
        pending = list(enumerate(texts))
        if self.templates:
            from . import printglyphs as PG
            nrng = np.random.default_rng(self.seed + 7)
            for wid, w in pending:
                if PG.handles(w):
                    a, bl, xh = PG.render(w, nrng, mess=self.mess)
                    best[wid] = WordImage(w, a, bl, xh, 1.0, w, template=True)
            pending = [(wid, w) for wid, w in pending if wid not in best]
            if best:
                self.log(f"templates: {len(best)} number, symbol and capital tokens drawn")
        banked = set()
        if self.bank is not None and pending:
            for wid, w in pending:
                v = self.bank.take(w)
                if v is not None:
                    best[wid] = v
                    banked.add(wid)
            pending = [(wid, w) for wid, w in pending if wid not in best]
            if banked:
                self.log(f"word bank: {len(banked)} words taken")
        seed = self.seed
        gen_lines = 0
        best_of = int(__import__("os").environ.get("DUCTUS_BEST_OF", "3"))   # candidates per stubborn word in the solo rounds
        skip_until = 0
        for rnd in range(self.rounds):
            if not pending:
                break
            if rnd < skip_until:
                continue                         # those rounds ran as parallel candidates
            if rnd and self.deadline is not None and __import__("time").time() > self.deadline:
                self.log(f"time budget reached: {len(pending)} words keep their best attempt")
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
            self._solo_ok = getattr(eng, "name", "") == "rnn"
            self._solo_all = rnd >= 2 and self._solo_ok   # a crop from a line of one word cannot belong to another word
            if best_of > 1 and self._solo_all and eng is self.engine:
                # several candidates of each stubborn word in one batch instead of one per round
                n_c = max(1, min(best_of, self.primary_rounds - rnd))
                todo = [x for x in todo for _ in range(n_c)]
                skip_until = rnd + n_c
            carriers = self._carriers(todo, rng)
            self.log(f"round {rnd + 1} ({getattr(eng, 'name', 'diffusion')}): {len(todo)} words in {len(carriers)} lines")
            for i in range(0, len(carriers), self.batch):
                chunk = carriers[i:i + self.batch]
                seed += 1
                imgs = eng.generate([c[1] for c in chunk], self.style_img, seed=seed)
                gen_lines += len(chunk)
                # split every line of the chunk first, then read all lines and all word
                # crops in two batched OCR calls (one call per crop was most of the time)
                prepped = []
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
                    prepped.append((group, img, bl, xh, real, cw, items))
                line_reads = word_reads_all = None
                if self.reader is not None and prepped:
                    line_reads = self.reader.read([img[:, :min(img.shape[1], real[-1][1] + 6)]
                                                   for group, img, bl, xh, real, cw, items in prepped],
                                                  max_tokens=[_cap(" ".join(w for _, w in p_[0])) for p_ in prepped])
                    # word-level read-back as a second opinion (catches split errors and
                    # misspellings the line-level language prior quietly corrects)
                    # crops whose word read cannot change the verdict are not read: words
                    # already rejected by a shape check, and short plain words the line read judges
                    flat = [(p_, it) for p_ in prepped for it in p_[6]]
                    need = [self._needs_word_read(eng, p_, n_) for p_ in prepped for n_ in range(len(p_[6]))]
                    got = iter(self.reader.read([it[3] for (p_, it), nd in zip(flat, need) if nd],
                                                max_tokens=[_cap(it[1]) for (p_, it), nd in zip(flat, need) if nd]))
                    word_reads_all = [next(got) if nd else "" for nd in need]
                    self.stats["word_reads_skipped"] = self.stats.get("word_reads_skipped", 0) + need.count(False)
                scored = []
                wpos = 0
                for li, (group, img, bl, xh, real, cw, items) in enumerate(prepped):
                    read_words = word_reads = None
                    if line_reads is not None:
                        read_words = align_words([w for _, w in group], line_reads[li].split())
                        if items:
                            word_reads = word_reads_all[wpos:wpos + len(items)]
                            wpos += len(items)
                    for n, (wid, w, sub, _, k, ratio) in enumerate(items):
                        err = self._score(eng, group, w, k, n, read_words, word_reads, sub, bl, xh, cw, ratio)
                        scored.append([li, n, err, read_words[k] if read_words is not None else ""])
                if self.checker is not None and line_reads is not None:
                    self._recheck(eng, prepped, scored)
                for li, n, err, r in scored:
                    group, img, bl, xh, real, cw, items = prepped[li]
                    wid, w, sub, _, k, ratio = items[n]
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
                      "template_words": sum(1 for v in best.values() if v.template),
                      "still_failing": len(pending), "rounds_used": rnd + 1,
                      "rechecked": self.stats.get("rechecked", 0),
                      "word_reads_skipped": self.stats.get("word_reads_skipped", 0)}
        if self.bank is not None:
            for wid, v in best.items():
                if wid not in banked and not v.template and (1.0 - v.score) <= self.max_cer:
                    self.bank.add(v)
            self.stats.update(self.bank.stats)
        missing = [wid for wid, _ in enumerate(texts) if wid not in best]
        if missing:
            raise RuntimeError(f"could not synthesise {len(missing)} words, e.g. {texts[missing[0]]!r}")
        return [best[i] for i in range(len(texts))]


    def _needs_word_read(self, eng, p_, n: int) -> bool:
        group, img, bl, xh, real, cw, items = p_
        wid, w, sub, _, k, ratio = items[n]
        if __import__("os").environ.get("DUCTUS_SKIP_READS", "1") == "0":
            return True
        lw = normalise(w)
        if (len(lw) < 3 and not any(c.isdigit() for c in w) and not any(c in "rfRM" for c in w)
                and not w[:1].isupper()):
            return False                 # e_word = e_line; no other rule looks at this read
        ideal = self._score(eng, group, w, k, n, [x for _, x in group], [x[1] for x in items],
                            sub, bl, xh, cw, ratio, count=False)
        return ideal <= self.max_cer     # a shape check fails it whatever the reads say

    def _recheck(self, eng, prepped, scored) -> None:
        """Second reader for the words the fast first reader rejected. The first pass reads
        with a small OCR model (several times faster); a word it reads right is legible. A word
        it rejects only because of what it read is read again, line and word, by the larger
        model, whose verdict then stands. Rejections that do not depend on the reading (split
        errors, looped letters) are not re-read."""
        redo = []
        for j, (li, n, err, _) in enumerate(scored):
            if err <= self.max_cer:
                continue
            group, img, bl, xh, real, cw, items = prepped[li]
            wid, w, sub, _, k, ratio = items[n]
            ideal = self._score(eng, group, w, k, n, [x for _, x in group], [it[1] for it in items],
                                sub, bl, xh, cw, ratio, count=False)
            if ideal <= self.max_cer:
                redo.append(j)
        if not redo:
            return
        lines = sorted({scored[j][0] for j in redo})
        lr = self.checker.read([prepped[li][1][:, :min(prepped[li][1].shape[1], prepped[li][4][-1][1] + 6)] for li in lines],
                               max_tokens=[_cap(" ".join(w for _, w in prepped[li][0])) for li in lines])
        line_read = {li: align_words([w for _, w in prepped[li][0]], t.split()) for li, t in zip(lines, lr)}
        wr = self.checker.read([prepped[scored[j][0]][6][scored[j][1]][3] for j in redo],
                               max_tokens=[_cap(prepped[scored[j][0]][6][scored[j][1]][1]) for j in redo])
        self.stats["rechecked"] = self.stats.get("rechecked", 0) + len(redo)
        for j, wread in zip(redo, wr):
            li, n, _, _ = scored[j]
            group, img, bl, xh, real, cw, items = prepped[li]
            wid, w, sub, _, k, ratio = items[n]
            word_reads = [it[1] for it in items]       # only item n is looked at
            word_reads[n] = wread
            err = self._score(eng, group, w, k, n, line_read[li], word_reads, sub, bl, xh, cw, ratio)
            scored[j][2], scored[j][3] = err, line_read[li][k]

    def _score(self, eng, group, w, k, n, read_words, word_reads, sub, bl, xh, cw, ratio, count=True) -> float:
        """Acceptance error of one word crop from its line and word reads and its shape."""
        def bump(key):
            if count:
                self.stats[key] = self.stats.get(key, 0) + 1
        if read_words is not None:
            nos = (lambda x: x.replace(" ", "")) if " " not in w else (lambda x: x)
            e_line = self._accept(w, nos(read_words[k]) if len(group) == 1 else read_words[k])
            e_word = self._accept(w, nos(word_reads[n])) if (len(normalise(w)) >= 3 or any(c.isdigit() for c in w)) else e_line
            err = max(e_line, e_word) if self.strict else 0.5 * (e_line + e_word)
            lw = normalise(w)
            if (self._glyph_mode == "cursive" and lw and not any(c.isdigit() for c in w)
                    and sum(c in "rf" for c in lw) >= 0.4 * len(lw)
                    and abs(len(re.sub(r"[^a-z]", "", normalise(word_reads[n]))) - len(lw)) <= 1):
                err = e_line     # isolated word OCR cannot read the cursive r/f; the line read can
        else:
            err = 0.0
        if (self._glyph_mode == "cursive" and word_reads and w.isalpha()
                and any(c in "rf" for c in w.lower())):
            # the redraw must not eat a neighbour ("of" -> "cf"): the other letters
            # have to be in the single-word read, in order
            core = [c for c in w.lower() if c not in "rf"]
            rd = re.sub(r"[^a-z]", "", word_reads[n].lower())
            miss = len(core) - _lcs(core, rd)
            if miss > (1 if len(core) > 4 else 0) or len(rd) > len(w) + 1:
                err = max(err, 0.5)
                bump("core_rejects")
        if (w[:1].isupper() and not w.isupper() and read_words is not None and word_reads
                and read_words[k][:1].islower() and word_reads[n].lstrip("#&*.,'\" ")[:1].islower()):
            err = max(err, 0.5)          # the capital came out as a small letter ("Note" -> "note")
            bump("cap_misread")
        for cap_, wrong_ in (("M", "H"), ("R", "P")):
            if cap_ in w and read_words is not None:
                # capital M that reads as H ("Hean"), R as P ("Pange"): write it again
                reads = [read_words[k], word_reads[n] if word_reads else ""]
                if any(x and wrong_ in x and x.count(cap_) < w.count(cap_) for x in reads):
                    err = max(err, 0.6)
                    bump("cap_misread")
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
                bump("flourish_rejects")
        if _DEBUG:
            print(f"word {w!r} line={read_words[k] if read_words else None!r} "
                  f"word={word_reads[n] if word_reads else None!r} ratio={ratio:.2f} err={err:.2f}", file=sys.stderr)
        return err

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
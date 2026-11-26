"""Word synthesis: pack words into full-length carrier lines, generate them with the
diffusion engine, split them back into words, OCR-check, and retry the failures.

Why carriers: the line model is trained on 35-61 character lines and is weakest on
short inputs. Writing every word inside a normal-length line keeps it in its comfort
zone, and every occurrence of a word is a fresh sample, so no two copies match.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .segment import split_by_length, clean_line, split_words, baseline_and_xheight, tight_bbox
from .verify import cer, normalise
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


@dataclass
class WordImage:
    text: str
    alpha: np.ndarray        # ink coverage 0..1, cropped, at model scale (64 px lines)
    baseline: float          # row of the baseline inside alpha
    xheight: float
    score: float = 1.0       # 1 - CER of the OCR read-back (1.0 when unchecked)
    read: str = ""


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
    plain: str = "fr" + "bdhklt"   # letters that must use the plain form (no closed loops); "" = off
    stats: dict = field(default_factory=dict)

    def _carriers(self, words: list[tuple[int, str]], rng: random.Random):
        """Greedy pack (id, text) pairs into carrier strings of exactly CARRIER_LEN chars."""
        out, cur, n = [], [], 0
        for wid, w in words:
            extra = len(w) + (1 if cur else 0)
            if cur and n + extra > CARRIER_LEN:
                out.append(cur)
                cur, n = [], 0
                extra = len(w)
            cur.append((wid, w))
            n += extra
        if cur:
            out.append(cur)
        carriers = []
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
        if not t:                      # pure punctuation such as "1)" or "-"
            return 0.0 if read.strip() else 1.0
        return cer(t, r)

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
                        if tight_bbox(sub) is None:
                            continue
                        ratio = (x1 - x0) / max(1.0, cw * len(w))
                        items.append((wid, w, sub, img[:, max(0, x0 - 1):x1 + 1], k, ratio))
                    # word-level read-back as a second opinion (catches split errors and
                    # misspellings the line-level language prior quietly corrects)
                    word_reads = None
                    if self.reader is not None and items:
                        word_reads = self.reader.read([it[3] for it in items])
                    for n, (wid, w, sub, _, k, ratio) in enumerate(items):
                        if read_words is not None:
                            e_line = self._accept(w, read_words[k])
                            e_word = self._accept(w, word_reads[n]) if len(normalise(w)) >= 3 else e_line
                            err = max(e_line, e_word) if self.strict else 0.5 * (e_line + e_word)
                        else:
                            err = 0.0
                        lo, hi = (0.55, 1.9) if self.strict and not self.loose else (0.4, 2.4)
                        if not (lo <= ratio <= hi) and len(w) >= 3:
                            err = max(err, 1.0)          # the split put this word in the wrong span
                        if self.plain and any(c in self.plain for c in w):
                            fl = flourishes(sub, w, bl, float(np.clip(cw * 1.15, 0.5 * xh, 1.5 * xh)), set(self.plain))
                            if fl:
                                err = max(err, max(0.6, self.max_cer + 0.05))   # looped f/r etc: write it again
                                self.stats["flourish_rejects"] = self.stats.get("flourish_rejects", 0) + 1
                        r = read_words[k] if read_words is not None else ""
                        # size by mean letter width: the row-profile x-height is fooled by ascenders
                        cand = WordImage(w, sub, bl, float(np.clip(cw * 1.15, 0.5 * xh, 1.5 * xh)), 1.0 - err, r)
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

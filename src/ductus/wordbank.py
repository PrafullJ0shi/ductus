"""Word bank: verified word images kept between runs, so common words skip the writer and the
OCR read-back.

Every word that passes the read-back is saved under its style and settings (at most
`per_word` variants per word). A later run takes a saved variant instead of writing the word
again, with rules that keep pages from looking copied:

  * one variant is used at most once per run (a document never shows the same image twice;
    a word that occurs more often than the bank has variants is written fresh);
  * every use is warped slightly (width, height, shear), and the ink, pressure and paper are
    rendered afresh on the page anyway;
  * a variant is dropped after `max_uses` runs, so the bank keeps turning over and no image
    follows the writer from document to document.

Folder: $DUCTUS_CACHE/bank (default ~/.cache/ductus/bank). Off with --no-bank.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np

BANK_VERSION = 1


def bank_root() -> Path:
    return Path(os.environ.get("DUCTUS_CACHE", Path.home() / ".cache" / "ductus")) / "bank"


def signature(o, max_cer: float, style_sig) -> str:
    """Settings that change how a word is written or judged (not the page look or the seed)."""
    k = [BANK_VERSION, o.engine, o.rnn_style, o.rnn_bias, o.glyphs, style_sig, max_cer, o.mode, o.plain,
         o.legibility, "small" if getattr(o, "fast", False) else getattr(o, "ocr", "cascade")]
    return hashlib.sha1(json.dumps(k, default=str).encode()).hexdigest()[:16]


def jitter(wi, rng: random.Random):
    """A slightly different copy of a WordImage: width +-5 %, height +-3 %, shear +-0.04."""
    import cv2
    from .synth import WordImage
    a = wi.alpha.astype(np.float32)
    h, w = a.shape
    sx, sy, sh = 1 + rng.uniform(-0.05, 0.05), 1 + rng.uniform(-0.03, 0.03), rng.uniform(-0.04, 0.04)
    pad = int(abs(sh) * h) + 2
    W = int(round(w * sx)) + 2 * pad
    H = max(1, int(round(h * sy)))
    bl = wi.baseline * sy
    M = np.float32([[sx, -sh, pad + sh * bl], [0, sy, 0]])      # shear about the baseline
    out = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    cols = np.where(out.max(0) > 0.02)[0]
    if len(cols):
        out = out[:, max(0, cols[0] - 1):cols[-1] + 2]
    return WordImage(wi.text, out, bl, wi.xheight * sy, wi.score, wi.read, wi.template)


class Bank:
    def __init__(self, sig: str, per_word: int = 6, max_uses: int = 3, seed: int = 0):
        self.dir = bank_root() / sig
        self.per_word, self.max_uses = per_word, max_uses
        self.rng = random.Random(seed * 7919 + 13)
        self.used: set[tuple[str, int]] = set()
        self.stats = {"bank_hits": 0, "bank_added": 0}

    def _file(self, word: str) -> Path:
        return self.dir / (hashlib.sha1(word.encode()).hexdigest()[:20] + ".pkl")

    def _load(self, word: str) -> list:
        f = self._file(word)
        if not f.exists():
            return []
        try:
            return pickle.loads(f.read_bytes())
        except Exception:
            return []

    def _save(self, word: str, items: list) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        f = self._file(word)
        tmp = f.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(pickle.dumps(items))
        os.replace(tmp, f)

    def take(self, word: str):
        """A jittered variant not used yet in this run, or None."""
        items = self._load(word)
        free = [i for i, it in enumerate(items) if (word, it["id"]) not in self.used]
        if not free:
            return None
        i = self.rng.choice(free)
        it = items[i]
        self.used.add((word, it["id"]))
        it["uses"] += 1
        if it["uses"] >= self.max_uses:
            items.pop(i)
        self._save(word, items)
        self.stats["bank_hits"] += 1
        return jitter(it["wi"], self.rng)

    def add(self, wi) -> None:
        items = self._load(wi.text)
        if len(items) >= self.per_word:
            return
        nid = max([it["id"] for it in items], default=-1) + 1
        items.append({"id": nid, "uses": 0, "wi": wi})
        self.used.add((wi.text, nid))             # written in this run: not taken again in it
        self._save(wi.text, items)
        self.stats["bank_added"] += 1

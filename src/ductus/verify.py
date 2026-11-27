"""OCR verifier: reads generated words/lines back with TrOCR and scores them.

Used two ways:
  * neat mode: strict, regenerate anything whose character error rate is above a threshold;
  * personal-style mode: loose, only catches missing or wrong words, so a messy hand stays messy.
"""
from __future__ import annotations

import numpy as np

OCR_MODEL = "microsoft/trocr-base-handwritten"   # MIT licence
OCR_SMALL = "microsoft/trocr-small-handwritten"  # MIT licence; about 4x faster on a CPU, first pass


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    ref = " ".join(ref.split())
    hyp = " ".join(hyp.split())
    return levenshtein(ref, hyp) / max(1, len(ref))


def cer_wild(ref: str, hyp: str, wild: str = "") -> float:
    """CER where substituting a reference letter in `wild` costs nothing. Used for letters the
    generator redraws in a hand the OCR model was not trained on (a school-cursive r or f):
    the OCR cannot judge them, the template fixes their shape, and a missing letter still counts."""
    ref = " ".join(ref.split()); hyp = " ".join(hyp.split())
    if not wild:
        return cer(ref, hyp)
    prev = list(range(len(hyp) + 1))
    for i, ca in enumerate(ref, 1):
        cur = [i]
        for j, cb in enumerate(hyp, 1):
            sub = 0 if (ca == cb or ca in wild) else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub))
        prev = cur
    return prev[-1] / max(1, len(ref))


def normalise(s: str) -> str:
    """Case and punctuation insensitive form used for scoring the generator, not the OCR."""
    import re
    return " ".join(re.sub(r"[^a-z0-9 ]", "", s.lower()).split())


class Reader:
    batch = 16    # crops per forward pass on a GPU; on a CPU see __init__

    def __init__(self, model: str = OCR_MODEL):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from . import speed
        speed.setup()
        self.torch = torch
        self.device = speed.device()
        self.proc = TrOCRProcessor.from_pretrained(model)
        self.model = VisionEncoderDecoderModel.from_pretrained(model).eval().to(self.device)
        if self.device.type == "cpu":
            # on a busy CPU one crop at a time was faster than a padded batch (44 s against
            # 62 s for 16 word crops with one thread); with free cores small batches help
            self.batch = 1 if torch.get_num_threads() <= 2 else 4

    def read(self, grays: list[np.ndarray], with_conf: bool = False, max_tokens=None):
        """OCR a batch of grayscale crops (dark ink on white). max_tokens (one value or one
        per crop) stops a read early: a crop that reads as more tokens than its target has
        characters fails anyway, and garbage reads otherwise run to 64 tokens."""
        if isinstance(max_tokens, (list, tuple)) and len(grays) <= self.batch:
            max_tokens = max(max_tokens) if max_tokens else None
        if len(grays) > self.batch:
            mt = max_tokens if isinstance(max_tokens, (list, tuple)) else [max_tokens] * len(grays)
            parts = [self.read(grays[i:i + self.batch], with_conf, mt[i:i + self.batch])
                     for i in range(0, len(grays), self.batch)]
            if not with_conf:
                return [t for p in parts for t in p]
            return [t for p in parts for t in p[0]], [c for p in parts for c in p[1]]
        if not grays:
            return ([], []) if with_conf else []
        from PIL import Image
        ims = []
        for g in grays:
            g = np.pad(g, 8, constant_values=255)
            ims.append(Image.fromarray(g.astype(np.uint8)).convert("RGB"))
        px = self.proc(images=ims, return_tensors="pt").pixel_values.to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(px, max_new_tokens=min(64, int(max_tokens or 64)), num_beams=1,
                                      output_scores=with_conf, return_dict_in_generate=True)
        texts = self.proc.batch_decode(out.sequences.cpu(), skip_special_tokens=True)
        if not with_conf:
            return texts
        # mean max-probability of emitted tokens = a simple legibility proxy
        probs = [self.torch.softmax(s, -1).max(-1).values for s in out.scores]
        P = self.torch.stack(probs, 1)  # [B, T]
        seq = out.sequences[:, 1:]
        mask = (seq != self.proc.tokenizer.pad_token_id).float()[:, : P.shape[1]]
        conf = ((P * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().tolist()
        return texts, conf

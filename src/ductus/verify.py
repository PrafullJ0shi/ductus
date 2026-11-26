"""OCR verifier: reads generated words/lines back with TrOCR and scores them.

Used two ways:
  * neat mode: strict, regenerate anything whose character error rate is above a threshold;
  * personal-style mode: loose, only catches missing or wrong words, so a messy hand stays messy.
"""
from __future__ import annotations

import numpy as np

OCR_MODEL = "microsoft/trocr-base-handwritten"   # MIT licence


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


def normalise(s: str) -> str:
    """Case and punctuation insensitive form used for scoring the generator, not the OCR."""
    import re
    return " ".join(re.sub(r"[^a-z0-9 ]", "", s.lower()).split())


class Reader:
    def __init__(self, model: str = OCR_MODEL):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.torch = torch
        self.proc = TrOCRProcessor.from_pretrained(model)
        self.model = VisionEncoderDecoderModel.from_pretrained(model).eval()

    def read(self, grays: list[np.ndarray], with_conf: bool = False):
        """OCR a batch of grayscale crops (dark ink on white)."""
        from PIL import Image
        ims = []
        for g in grays:
            g = np.pad(g, 8, constant_values=255)
            ims.append(Image.fromarray(g.astype(np.uint8)).convert("RGB"))
        px = self.proc(images=ims, return_tensors="pt").pixel_values
        with self.torch.no_grad():
            out = self.model.generate(px, max_new_tokens=64, num_beams=1,
                                      output_scores=with_conf, return_dict_in_generate=True)
        texts = self.proc.batch_decode(out.sequences, skip_special_tokens=True)
        if not with_conf:
            return texts
        # mean max-probability of emitted tokens = a simple legibility proxy
        probs = [self.torch.softmax(s, -1).max(-1).values for s in out.scores]
        P = self.torch.stack(probs, 1)  # [B, T]
        seq = out.sequences[:, 1:]
        mask = (seq != self.proc.tokenizer.pad_token_id).float()[:, : P.shape[1]]
        conf = ((P * mask).sum(1) / mask.sum(1).clamp(min=1)).tolist()
        return texts, conf

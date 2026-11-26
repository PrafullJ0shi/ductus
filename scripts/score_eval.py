"""Score generated evaluation lines with an OCR judge that ductus does NOT use internally.

  python scripts/score_eval.py scripts/data/eval_sentences.txt /workspace/eval rnn_b15,rnn_b25,font,...
Judge: microsoft/trocr-large-handwritten (ductus verifies with trocr-base, a different model).
Prints a markdown table and writes <eval>/scores.json.
"""
import json, sys
from pathlib import Path

import cv2
import numpy as np

from ductus.verify import Reader, cer, normalise

src, root = sys.argv[1], Path(sys.argv[2])
systems = sys.argv[3].split(",")
judge = sys.argv[4] if len(sys.argv) > 4 else "microsoft/trocr-large-handwritten"
refs = Path(src).read_text().splitlines()
R = Reader(judge)
res = {}
for s in systems:
    d = root / s
    if not d.exists():
        print("missing", s)
        continue
    imgs = [cv2.imread(str(d / f"{i:03d}.png"), 0) for i in range(len(refs))]
    hyps = []
    for i in range(0, len(imgs), 8):
        hyps += R.read(imgs[i:i + 8])
    raw = [cer(r, h) for r, h in zip(refs, hyps)]
    nor = [cer(normalise(r), normalise(h)) for r, h in zip(refs, hyps)]
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    res[s] = dict(cer=float(np.mean(raw)), cer_norm=float(np.mean(nor)),
                  cer_median=float(np.median(raw)), lines_ok=int(sum(c <= 0.1 for c in nor)),
                  n=len(refs), sec=meta.get("seconds_per_line"), reads=hyps)
    print(f"{s:18s} CER {res[s]['cer']:.3f}  normCER {res[s]['cer_norm']:.3f}  lines<=10% {res[s]['lines_ok']}/{len(refs)}  s/line {res[s]['sec']}", flush=True)
outp = root / f"scores-{judge.split('/')[-1]}.json"
old = json.loads(outp.read_text())["results"] if outp.exists() else {}
old.update(res)                      # keep systems scored in earlier runs
json.dump({"judge": judge, "results": old}, open(outp, "w"), indent=1)

"""OCR every system's line crops with one judge and compute character error rates.
Judge: microsoft/trocr-large-handwritten (ductus checks itself with trocr-base, a different
model). Same judge, same preprocessing (ductus.verify.Reader: 8 px white pad, greedy decode)
for every system.
  raw CER  : Levenshtein(ref, hyp) / len(ref), whitespace collapsed, case and punctuation count
  norm CER : same after lowercasing and keeping only [a-z0-9 ] (ductus.verify.normalise)
The reference is always the full intended text, so characters a tool cannot write count as errors.
  python ocr_score.py sys1 sys2 ...   -> out/ocr.json (merged with earlier runs)
"""
import json, sys
from pathlib import Path
import cv2
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from ductus.verify import Reader, cer, normalise

ROOT = Path(__file__).resolve().parents[1]
rows = [l.split("\t") for l in (ROOT / "texts/tests.tsv").read_text().splitlines()[1:]]
ids, refs = [r[0] for r in rows], [r[3] for r in rows]
R = Reader("microsoft/trocr-large-handwritten")
outp = ROOT / "out/ocr.json"
res = json.loads(outp.read_text()) if outp.exists() else {}
for s in sys.argv[1:]:
    d = ROOT / "out" / s / "lines"
    imgs = [cv2.imread(str(d / f"{i:03d}.png"), 0) for i in range(len(ids))]
    if any(im is None for im in imgs):
        print("missing lines for", s); continue
    hyps = R.read(imgs[:4]) + R.read(imgs[4:])
    res[s] = {i: dict(ref=r, hyp=h, cer=round(cer(r, h), 4), cer_norm=round(cer(normalise(r), normalise(h)), 4))
              for i, r, h in zip(ids, refs, hyps)}
    print(s, {i: v["cer_norm"] for i, v in res[s].items()}, flush=True)
    outp.write_text(json.dumps(res, indent=1))

"""Generate the evaluation lines for the font baseline, raw DiffBrush and ductus.

  python scripts/gen_eval_lines.py scripts/data/eval_sentences.txt /workspace/eval
Writes <out>/<system>/<NNN>.png and <out>/<system>/meta.json (seconds per line).
"""
import json, sys, time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

src, out = sys.argv[1], Path(sys.argv[2])
systems = sys.argv[3].split(",") if len(sys.argv) > 3 else ["font", "diffbrush_raw", "ductus_neat", "ductus_natural"]
lines = Path(src).read_text().splitlines()
FONT = Path("/workspace/research/fonts/Caveat%5Bwght%5D.ttf")


def save(sysname, imgs, secs, extra=None):
    d = out / sysname
    d.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(imgs):
        cv2.imwrite(str(d / f"{i:03d}.png"), im)
    json.dump({"seconds_per_line": secs, **(extra or {})}, open(d / "meta.json", "w"), indent=1)


if "font" in systems:
    f = ImageFont.truetype(str(FONT), 56)
    imgs = []
    t0 = time.time()
    for t in lines:
        w = int(f.getlength(t)) + 40
        im = Image.new("L", (w, 96), 255)
        ImageDraw.Draw(im).text((20, 12), t, font=f, fill=0)
        imgs.append(np.array(im))
    save("font", imgs, (time.time() - t0) / len(lines), {"font": "Caveat (Google Fonts, OFL)"})

if any(s.startswith("diffbrush") or s.startswith("ductus") for s in systems):
    from ductus.engines.diffbrush import DiffBrushEngine
    from ductus.page import load_style, Writer, Options, write_lines
    from ductus.markup import WRITABLE
    if "diffbrush_raw" in systems:
        e = DiffBrushEngine()
        ref = load_style("quick")[0][0]
        imgs = []
        t0 = time.time()
        for i in range(0, len(lines), 8):
            chunk = ["".join(c for c in t if c in WRITABLE) for t in lines[i:i + 8]]
            for t, g in zip(chunk, e.generate(chunk, ref, seed=100 + i)):
                a = (g < 200)
                cols = np.where(a.any(0))[0]
                imgs.append(g[:, max(0, cols.min() - 8):cols.max() + 8] if len(cols) else g)
        save("diffbrush_raw", imgs, (time.time() - t0) / len(lines), {"style": "quick", "note": "one sample per line, no carriers, no OCR check"})
    w = None
    for name in [s for s in systems if s.startswith("ductus")]:
        # ductus_<look>[_<preset>]; ductus_noverify = natural look without the OCR check
        parts = name.split("_")
        look = "natural" if parts[1] == "noverify" else parts[1]
        style = parts[2] if len(parts) > 2 else "quick"
        verify = name != "ductus_noverify"
        if w is None or (w.reader is None) == verify:
            w = Writer(verify=True)
        o = Options(style=style, look=look, seed=7, verify=verify, log=lambda *a: print(*a, file=sys.stderr))
        imgs, st = write_lines(w, lines, o)
        save(name, imgs, st["seconds"] / len(lines), st)

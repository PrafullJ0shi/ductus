"""Render the showcase pages and the samples gallery.

  python scripts/make_samples.py OUT_FULL [SAMPLES_DIR] [names...]
Full-resolution A4 PNG + PDF go to OUT_FULL; reduced PNGs for the repository go to SAMPLES_DIR.
"""
import json, sys, time
from pathlib import Path

import cv2

from ductus.page import Writer, Options
from ductus.render import save_pdf, save_svg

full = Path(sys.argv[1]); full.mkdir(parents=True, exist_ok=True)
small = Path(sys.argv[2]) if len(sys.argv) > 2 else None
only = set(sys.argv[3:])
EX = Path(__file__).resolve().parents[1] / "examples"

JOBS = [
    # name, text file, options
    ("intro", "intro.txt", dict(mode="plain", look="natural", style="slant", seed=11)),
    ("definitions", "definitions.txt", dict(mode="notes", look="natural", style="quick", seed=3)),
    ("maths", "maths.txt", dict(mode="maths", look="natural", style="upright", seed=5)),
    ("checklist", "checklist.txt", dict(mode="notes", look="natural", style="round", seed=9)),
    ("intro-neat", "intro.txt", dict(mode="plain", look="neat", style="upright", seed=2)),
    ("intro-messy", "intro.txt", dict(mode="plain", look="messy", style="quick", seed=4)),
    ("notes-grid-blue", "notes-statistics.txt", dict(mode="notes", look="natural", style="loose",
                                                    paper="grid", seed=6)),
    ("checklist-ruled-black", "checklist.txt", dict(mode="notes", look="natural", style="slant",
                                                   paper="ruled", pen="black-ballpoint", seed=8)),
    ("notes-photo", "notes-statistics.txt", dict(mode="notes", look="messy", style="quick",
                                                paper="grid", photo=True, seed=10)),
]

w = Writer()
log = {}
for name, src, kw in JOBS:
    if only and name not in only:
        continue
    t0 = time.time()
    r = w.write((EX / src).read_text(), Options(log=lambda *a: print(*a, file=sys.stderr), **kw))
    pngs = []
    for i, img in enumerate(r.pages):
        p = full / (f"{name}.png" if len(r.pages) == 1 else f"{name}-{i + 1}.png")
        cv2.imwrite(str(p), img[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 6])
        pngs.append(str(p))
        if small:
            small.mkdir(parents=True, exist_ok=True)
            s = cv2.resize(img[..., ::-1], (1240, 1754), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(small / Path(p).with_suffix(".jpg").name), s, [cv2.IMWRITE_JPEG_QUALITY, 88])
    save_pdf(pngs, str(full / f"{name}.pdf"))
    if name == "definitions":
        save_svg(r.layouts[0], r.words, str(full / f"{name}.svg"), r.look.ink, r.stats["pen_px"])
        save_svg(r.layouts[0], r.words, str(full / f"{name}.anim.svg"), r.look.ink, r.stats["pen_px"], animate=True)
    log[name] = dict(r.stats, options=kw, wall=round(time.time() - t0, 1))
    print(name, json.dumps(log[name]), flush=True)
json.dump(log, open(full / "samples.json", "w"), indent=1)

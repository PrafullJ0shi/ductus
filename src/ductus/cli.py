"""ductus command line.

  ductus write  notes.txt -o out/page [--style quick|<profile dir>|<line.png>] [--look neat|natural|messy]
                [--mode plain|notes|maths] [--paper white|ruled|grid] [--photo] [--svg] [--animate]
  ductus style  photo1.jpg photo2.jpg ... -o styles/my-hand      build a personal style profile
  ductus presets                                                list built-in styles
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_write(a):
    import cv2
    from .page import Options, Writer
    from .render import save_pdf, save_svg
    text = Path(a.input).read_text(encoding="utf-8") if a.input != "-" else sys.stdin.read()
    if a.text:
        text = a.text
    o = Options(style=a.style, look=a.look, mode=a.mode, paper=a.paper, photo=a.photo, ink=a.ink,
                seed=a.seed, steps=a.steps, verify=not a.no_verify, max_cer=a.max_cer,
                corrections=a.corrections, pen=a.pen, size=a.size,
                log=(lambda *x: None) if a.quiet else (lambda *x: print(*x, file=sys.stderr)))
    w = Writer(steps=a.steps, verify=not a.no_verify)
    r = w.write(text, o)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pngs = []
    for i, img in enumerate(r.pages):
        p = f"{out}.png" if len(r.pages) == 1 else f"{out}-{i + 1}.png"
        cv2.imwrite(p, img[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 6])
        pngs.append(p)
    if not a.no_pdf:
        save_pdf(pngs, f"{out}.pdf")
    if a.svg or a.animate:
        for i, pg in enumerate(r.layouts):
            sfx = "" if len(r.layouts) == 1 else f"-{i + 1}"
            pen = r.stats["pen_px"]
            if a.svg:
                save_svg(pg, r.words, f"{out}{sfx}.svg", r.look.ink, pen)
            if a.animate:
                save_svg(pg, r.words, f"{out}{sfx}.anim.svg", r.look.ink, pen, animate=True)
    print(json.dumps({"files": pngs, **r.stats}))


def cmd_style(a):
    from .profile import build
    reader = None
    if not a.no_ocr:
        from .verify import Reader
        reader = Reader()
    prof = build(a.photos, a.out, name=a.name, reader=reader)
    print(json.dumps(prof, indent=2))


def cmd_presets(a):
    from .page import PRESETS
    meta = json.loads((PRESETS / "presets.json").read_text())
    for k, v in meta.items():
        if not k.startswith("_"):
            print(f"{k:9s} {v['note']}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ductus", description="Human-looking handwriting on A4.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="write text onto A4 pages")
    w.add_argument("input", nargs="?", default="-", help="text file (markup allowed) or - for stdin")
    w.add_argument("--text", help="text given inline instead of a file")
    w.add_argument("-o", "--out", default="page", help="output path without extension")
    w.add_argument("--style", default="quick", help="preset name, a line image, or a profile folder")
    w.add_argument("--look", default="natural", choices=["neat", "natural", "messy", "profile"])
    w.add_argument("--mode", default="plain", choices=["plain", "notes", "maths"])
    w.add_argument("--paper", default="white", choices=["white", "ruled", "grid"])
    w.add_argument("--photo", action="store_true", help="make it look like a phone photo")
    w.add_argument("--ink", help="ink colour, e.g. '#1c2a6e'")
    w.add_argument("--pen", type=float, help="stroke weight multiplier (1.0 = as sampled)")
    w.add_argument("--size", type=float, help="x-height in mm (default 2.5)")
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--steps", type=int, default=25, help="diffusion steps (quality vs speed)")
    w.add_argument("--max-cer", type=float, help="OCR acceptance threshold")
    w.add_argument("--corrections", type=float, help="share of words written wrong then struck out")
    w.add_argument("--no-verify", action="store_true", help="skip the OCR read-back check")
    w.add_argument("--no-pdf", action="store_true")
    w.add_argument("--svg", action="store_true", help="also write a vector SVG")
    w.add_argument("--animate", action="store_true", help="also write an animated SVG")
    w.add_argument("-q", "--quiet", action="store_true")
    w.set_defaults(fn=cmd_write)
    s = sub.add_parser("style", help="build a personal style profile from photos of your writing")
    s.add_argument("photos", nargs="+")
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--name", default="my-hand")
    s.add_argument("--no-ocr", action="store_true", help="skip OCR-based upside-down detection")
    s.set_defaults(fn=cmd_style)
    p = sub.add_parser("presets", help="list built-in styles")
    p.set_defaults(fn=cmd_presets)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

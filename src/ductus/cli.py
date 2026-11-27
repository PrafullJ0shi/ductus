"""ductus command line.

  ductus write  notes.txt -o out/page [--style upright|<profile dir>|<line.png>] [--look neat|natural|messy]
                [--mode plain|notes|maths] [--paper white|ruled|grid] [--photo [a4|notebook]] [--svg] [--animate]
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
                seed=a.seed, steps=a.steps, verify=not a.no_verify, max_cer=a.max_cer, cache=not a.no_cache, plain="" if a.allow_loops else "abcdefghijklmnopqrstuvwxyz", legibility=a.legibility, engine=a.engine, rnn_style=a.rnn_style, rnn_bias=a.rnn_bias, glyphs=a.glyphs,
                corrections=a.corrections, pen=a.pen, tip=a.tip, weight=a.weight,
                flat_ink=a.flat_ink, ink_model=a.ink_model, humanize=a.humanize, size=a.size,
                border=a.border, border_inset=a.border_inset, border_header=a.border_header,
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
    w.add_argument("--style", help="preset name, a line image, or a profile folder (default: upright, a plain print-like hand)")
    w.add_argument("--look", default="natural", choices=["neat", "natural", "messy", "profile"])
    w.add_argument("--mode", default="plain", choices=["plain", "notes", "maths"])
    w.add_argument("--paper", default=None, choices=["white", "ruled", "grid"],
                   help="default white; grid with --photo notebook")
    w.add_argument("--photo", nargs="?", const="a4", default=None, choices=["a4", "notebook"],
                   help="phone photo of the page: a4 (default; loose sheet on a desk) or notebook (spiral grid pad)")
    w.add_argument("--ink", help="override the pen's ink colour, e.g. '#1c2a6e'")
    w.add_argument("--pen", choices=["blue-ballpoint", "black-ballpoint", "blue-gel", "black-gel", "fountain", "pencil"],
                   help="pen preset (default blue-ballpoint; a profile uses its measured ink and width)")
    w.add_argument("--tip", type=float, help="nominal tip size in mm: 0.5, 0.7 or 1.0")
    w.add_argument("--weight", type=float, help="stroke weight multiplier on top of the pen (1.0 = as is)")
    w.add_argument("--flat-ink", action="store_true", help="old flat ink model, for comparison")
    w.add_argument("--ink-model", default="realism", choices=["realism", "pens", "flat"],
                   help="realism (default, calibrated), pens (simpler model in pens.py) or flat")
    w.add_argument("--humanize", type=float, default=0.5,
                   help="rate of the human-error pass on the laid-out page (0 = off; ignored for --look neat)")
    w.add_argument("--size", type=float, help="x-height in mm (default 2.8)")
    w.add_argument("--border", default="none", choices=["none", "box", "box-margin", "margin"],
                   help="hand-ruled border box and/or left margin line, drawn in the pen's ink; the text stays inside")
    w.add_argument("--border-inset", type=float, default=12.0, help="border box distance from the paper edge in mm (default 12)")
    w.add_argument("--border-header", type=float, default=0.0,
                   help="also rule a line this many mm from the top edge, under the heading area (0 = none)")
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--steps", type=int, default=25, help="diffusion steps (quality vs speed)")
    w.add_argument("--max-cer", type=float, help="OCR acceptance threshold")
    w.add_argument("--corrections", type=float, help="share of words written wrong then struck out")
    w.add_argument("--no-verify", action="store_true", help="skip the OCR read-back check")
    w.add_argument("--engine", default="default", choices=["default", "rnn", "diffusion", "auto"],
                   help="default: RNN letters (plain, readable, fast) when installed, else diffusion; "
                        "diffusion: letter shapes copied from --style; auto: diffusion, RNN for words that do not read")
    w.add_argument("--rnn-style", type=int, default=1, help="built-in RNN writer style 0-7 (1: plain print letters)")
    w.add_argument("--glyphs", choices=["cursive", "print", "off"], default="cursive", help="r and f: cursive (school-cursive r, looped f, default), print (plain print), off (as the RNN writes them)")
    w.add_argument("--rnn-bias", type=float, default=1.5, help="RNN neatness (higher = neater)")
    w.add_argument("--legibility", default="strict", choices=["strict", "loose"],
                   help="strict (default): every word must pass the line and word OCR read-back; loose: only catch wrong or missing words")
    w.add_argument("--allow-loops", action="store_true",
                   help="keep looped f, r and ascenders (default re-writes them in the plain form)")
    w.add_argument("--no-cache", action="store_true", help="always generate words afresh (default reuses ~/.cache/ductus)")
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

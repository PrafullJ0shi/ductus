"""Render the pages behind docs/GUIDE.md and compose the comparison figures.

Every figure uses the same input text and the default (public) style unless the row says
otherwise; seed 7 for pages and seed 21 for the single line. The settings are written in the caption row of each figure.

  python scripts/make_guide_samples.py            # pages into out/guide, figures into docs/assets/guide
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "guide"
FIG = ROOT / "docs" / "assets" / "guide"
LINE = "The quick brown fox jumps over the lazy dog."
MONO = "/usr/share/fonts/truetype/sand-box/google/IBM Plex Mono/IBMPlexMono-Regular.ttf"
INK, MUTED, RULE = (29, 31, 36), (92, 97, 112), (217, 214, 206)


def jobs():
    guide = (ROOT / "examples" / "guide.txt").read_text()
    j = [(f"border-{b}", guide, dict(mode="notes", border=b)) for b in ("box", "margin", "box-margin")]
    j += [(f"style-{s}", LINE, dict(style=s)) for s in ("upright", "slant", "quick", "round", "loose")]
    j += [(f"pen-{p}", LINE, dict(pen=p)) for p in
          ("blue-ballpoint", "black-ballpoint", "blue-gel", "black-gel", "fountain", "pencil")]
    j += [(f"glyphs-{g}", LINE, dict(glyphs=g)) for g in ("cursive", "print", "off")]
    j += [(f"look-{x}", guide, dict(mode="notes", look=x)) for x in ("neat", "natural", "messy")]
    j += [(f"paper-{x}", guide, dict(mode="notes", paper=x)) for x in ("white", "ruled", "grid")]
    return j


def render():
    from ductus.page import Options, Writer
    OUT.mkdir(parents=True, exist_ok=True)
    w = Writer()
    for name, text, kw in jobs():
        if (OUT / f"{name}.png").exists():
            continue
        t = time.time()
        r = w.write(text, Options(seed=21 if text == LINE else 7, bank=True, log=lambda *x: None, **kw))
        cv2.imwrite(str(OUT / f"{name}.png"), r.pages[0][..., ::-1])
        print(json.dumps({"page": name, "s": round(time.time() - t, 1), "still_failing": r.stats.get("still_failing")}))


def font(size):
    try:
        return ImageFont.truetype(MONO, size)
    except OSError:
        return ImageFont.load_default()


def ink_box(img: Image.Image, pad=40):
    a = np.asarray(img.convert("L"), dtype=np.int16)
    bg = np.median(a)
    rows = np.where((bg - a > 60).sum(1) > 2)[0]
    cols = np.where((bg - a > 60).sum(0) > 2)[0]
    return (max(0, cols[0] - pad), max(0, rows[0] - pad), min(a.shape[1], cols[-1] + pad), min(a.shape[0], rows[-1] + pad))


def rows_figure(names, labels, out, width=1600):
    """One cropped line per row, label on the left."""
    crops = []
    for n in names:
        im = Image.open(OUT / f"{n}.png").convert("RGB")
        crops.append(im.crop(ink_box(im)))
    cw = max(c.size[0] for c in crops)
    lab_w, f = 360, font(26)
    scale = (width - lab_w - 40) / cw
    rh = [int(c.size[1] * scale) + 24 for c in crops]
    fig = Image.new("RGB", (width, sum(rh) + 40), "white")
    d = ImageDraw.Draw(fig)
    y = 20
    for c, lab, h in zip(crops, labels, rh):
        c = c.resize((int(c.size[0] * scale), int(c.size[1] * scale)), Image.LANCZOS)
        fig.paste(c, (lab_w, y + 12))
        d.text((32, y + h // 2), lab, fill=INK, font=f, anchor="lm")
        y += h
        d.line([(32, y), (width - 32, y)], fill=RULE, width=1)
    fig.save(FIG / out, quality=88)


def cols_figure(names, labels, out, crop_top=None, width=1800):
    """Pages (or their top part) side by side with a caption row."""
    ims = [Image.open(OUT / f"{n}.png").convert("RGB") for n in names]
    if crop_top:
        ims = [im.crop((0, 0, im.size[0], int(im.size[1] * crop_top))) for im in ims]
    gap, cap, f = 24, 64, font(24)
    cw = (width - gap * (len(ims) + 1)) // len(ims)
    ims = [im.resize((cw, int(im.size[1] * cw / im.size[0])), Image.LANCZOS) for im in ims]
    h = max(im.size[1] for im in ims)
    fig = Image.new("RGB", (width, h + cap + gap * 2), "white")
    d = ImageDraw.Draw(fig)
    for k, (im, lab) in enumerate(zip(ims, labels)):
        x = gap + k * (cw + gap)
        fig.paste(im, (x, gap))
        d.rectangle([x, gap, x + cw - 1, gap + im.size[1] - 1], outline=RULE)
        d.text((x + cw // 2, gap + h + cap // 2), lab, fill=MUTED, font=f, anchor="mm")
    fig.save(FIG / out, quality=88)


def figures():
    FIG.mkdir(parents=True, exist_ok=True)
    cols_figure([f"border-{b}" for b in ("box", "margin", "box-margin")],
                ["--border box", "--border margin", "--border box-margin"], "borders.jpg")
    rows_figure([f"style-{s}" for s in ("upright", "slant", "quick", "round", "loose")],
                [f"--style {s}" for s in ("upright", "slant", "quick", "round", "loose")], "styles.jpg")
    pens = ("blue-ballpoint", "black-ballpoint", "blue-gel", "black-gel", "fountain", "pencil")
    rows_figure([f"pen-{p}" for p in pens], [f"--pen {p}" for p in pens], "pens.jpg")
    rows_figure([f"glyphs-{g}" for g in ("cursive", "print", "off")],
                [f"--glyphs {g}" for g in ("cursive", "print", "off")], "glyphs.jpg")
    cols_figure([f"look-{x}" for x in ("neat", "natural", "messy")],
                ["--look neat", "--look natural", "--look messy"], "looks.jpg", crop_top=0.36)
    cols_figure([f"paper-{x}" for x in ("white", "ruled", "grid")],
                ["--paper white", "--paper ruled", "--paper grid"], "paper.jpg", crop_top=0.36)


if __name__ == "__main__":
    render()
    figures()

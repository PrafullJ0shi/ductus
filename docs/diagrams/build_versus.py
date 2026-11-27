"""Versus figure: the same three lines written by a handwriting font, two RNN baselines and ductus.

Reads the line crops of benchmark round 5 (one folder per system, `lines/000.png` ...,
line order as in benchmark/texts/tests.tsv) and draws one editorial grid in the DESIGN.md
tokens: paper background, graphite text, the ink blue only on the ductus rows.

  python docs/diagrams/build_versus.py BENCH_OUT docs/assets/versus.png

BENCH_OUT is the benchmark `out/` folder (default /workspace/ductus-parts/benchmark/out).
Every crop is scaled to the same height; nothing else is touched. The numbers in the
captions are read from benchmark/metrics.csv, not typed in.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/workspace/ductus-parts/benchmark/out")
DEST = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "docs/assets/versus.png"
DARK = "--dark" in sys.argv

S = 2  # supersampling
if DARK:
    PAPER, INK, MUTED, RULE, ACC = (21, 23, 28), (232, 230, 225), (160, 164, 176), (58, 61, 70), (140, 155, 224)
else:
    PAPER, INK, MUTED, RULE, ACC = (251, 250, 247), (29, 31, 36), (92, 97, 112), (217, 214, 206), (28, 42, 110)
FONTS = "/usr/share/fonts/truetype/sand-box/google/"
SERIF = FONTS + "Newsreader/Newsreader-VariableFont_opsz,wght.ttf"
MONO = FONTS + "IBM Plex Mono/IBMPlexMono-Regular.ttf"
MONOB = FONTS + "IBM Plex Mono/IBMPlexMono-SemiBold.ttf"


def font(path, size, weight=None):
    f = ImageFont.truetype(path, size * S)
    if weight is not None:
        try:
            f.set_variation_by_axes([weight, min(max(size, 6), 72)])
        except Exception:
            pass
    return f


TITLE = font(SERIF, 28, 550)
SUB = font(SERIF, 16, 450)
NAME = font(MONOB, 14)
SMALL = font(MONO, 12)
TAG = font(MONOB, 12)

TEXTS = [(0, "short", "Meet me at the library at 5 pm."),
         (3, "pangram", "The quick brown fox jumps over the lazy dog."),
         (2, "maths", "mean = 30/5 = 6 and x^2 = 36")]
SYSTEMS = [  # id in metrics.csv, folder, caption name, settings
    ("tth_caveat", "tth_caveat", "Font: Caveat", "typed by text-to-handwriting"),
    ("hws_b075_s9", "hws_b075_s9", "sjvasquez RNN", "Graves model, style 9, bias 0.75"),
    ("rnn_b1.5", "rnn_b1.5", "Grzego RNN", "Graves model, style 4, bias 1.5"),
    ("ductus_v5", "ductus_v5", "ductus", "default look, round 5"),
]
metrics = {r["system"]: r for r in csv.DictReader(open(REPO / "benchmark/metrics.csv"))}

CAP_W, IMG_W, ROW_H, INK_H, PAD = 330, 1000, 84, 56, 28
W = CAP_W + IMG_W + 2 * PAD
HEAD = 108
GROUP_GAP = 20
H = HEAD + len(TEXTS) * (len(SYSTEMS) * ROW_H + 34 + GROUP_GAP) + 28


def tile(p, h):
    g = Image.open(p).convert("L")
    s = h * S / g.height
    w = int(g.width * s)
    if w > IMG_W * S:
        s = IMG_W * S / g.width
        w = IMG_W * S
    g = g.resize((max(1, w), max(1, int(g.height * s))), Image.LANCZOS)
    a = np.asarray(g, np.float32) / 255.0
    return Image.fromarray(np.clip(np.array(PAPER) * a[..., None] + np.array(INK) * (1 - a[..., None]), 0, 255).astype(np.uint8))


def text_ink(d, x, y, s, f, fill):
    d.text((x * S, y * S), s, font=f, fill=fill)


im = Image.new("RGB", (W * S, H * S), PAPER)
d = ImageDraw.Draw(im)
text_ink(d, PAD, 24, "The same text, four writers", TITLE, INK)
text_ink(d, PAD, 62, "Each line is one unedited output, cut from benchmark round 5 and scaled to the same height.", SUB, MUTED)
text_ink(d, PAD, 84, "OCR CER is the main-line average from benchmark/metrics.csv; lower reads easier.", SUB, MUTED)

y = HEAD
for idx, tid, text in TEXTS:
    text_ink(d, PAD, y + 4, f"{tid.upper()}", TAG, MUTED)
    text_ink(d, PAD + 80, y + 4, text, SMALL, MUTED)
    y += 34
    for sid, folder, name, settings in SYSTEMS:
        m = metrics[sid]
        focal = sid == "ductus_v5"
        d.line([(PAD * S, y * S), ((W - PAD) * S, y * S)], fill=RULE, width=S)
        col = ACC if focal else INK
        if focal:
            d.rectangle([PAD * S, (y + 1) * S, (PAD + 3) * S, (y + ROW_H) * S], fill=ACC)
        text_ink(d, PAD + 14, y + 14, name, NAME, col)
        text_ink(d, PAD + 14, y + 36, settings, SMALL, MUTED)
        text_ink(d, PAD + 14, y + 54, f"norm CER {float(m['cer_norm_main']):.3f}  ·  {float(m['sec_per_line']):g} s/line", SMALL, MUTED)
        tl = tile(OUT / folder / "lines" / f"{idx:03d}.png", INK_H)
        im.paste(tl, ((CAP_W + PAD) * S, (y + (ROW_H - tl.height / S) / 2) * S if False else int((y + (ROW_H - tl.height / S) / 2) * S)))
        y += ROW_H
    d.line([(PAD * S, y * S), ((W - PAD) * S, y * S)], fill=RULE, width=S)
    y += GROUP_GAP

im.save(DEST, optimize=True)
print("wrote", DEST, im.size)

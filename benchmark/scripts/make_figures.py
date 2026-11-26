"""Side-by-side figures: same text, one row per system, grayscale, light paper background.
Each crop is scaled to the same height (row height minus padding) and capped in width;
nothing else is touched. Caption: system, settings, measured seconds per line.
  python make_figures.py  -> figures/*.png
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
PAPER, INK, MUTED, RULE = (251, 250, 247), (29, 31, 36), (92, 97, 112), (217, 214, 206)
MONO = "/usr/share/fonts/truetype/sand-box/google/IBM Plex Mono/IBMPlexMono-Regular.ttf"
MONOB = "/usr/share/fonts/truetype/sand-box/google/IBM Plex Mono/IBMPlexMono-Bold.ttf"
f1, f2 = ImageFont.truetype(MONOB, 15), ImageFont.truetype(MONO, 12)
SYSTEMS = json.loads((ROOT / "systems.json").read_text())
rows = [l.split("\t") for l in (ROOT / "texts/tests.tsv").read_text().splitlines()[1:]]
ids = [r[0] for r in rows]
CAP_W, IMG_W, ROW_H, INK_H = 300, 1100, 92, 64


def tile(p):
    g = Image.open(p).convert("L")
    s = INK_H / g.height
    w = int(g.width * s)
    if w > IMG_W:
        s = IMG_W / g.width; w = IMG_W
    g = g.resize((max(1, w), max(1, int(g.height * s))), Image.LANCZOS)
    a = np.asarray(g, np.float32) / 255.0
    rgb = np.array(PAPER) * a[..., None] + np.array(INK) * (1 - a[..., None])
    return Image.fromarray(rgb.astype(np.uint8))


def figure(name, test_ids, title):
    present = [s for s in SYSTEMS if all((ROOT / "out" / s["id"] / "lines" / f"{ids.index(t):03d}.png").exists() for t in test_ids)]
    H = 56 + len(present) * len(test_ids) * ROW_H + 12
    im = Image.new("RGB", (CAP_W + IMG_W + 40, H), PAPER); d = ImageDraw.Draw(im)
    d.text((20, 16), title, font=f1, fill=INK)
    y = 56
    for s in present:
        for k, t in enumerate(test_ids):
            d.line([(20, y), (CAP_W + IMG_W + 20, y)], fill=RULE)
            if k == 0:
                d.text((20, y + 14), s["label"], font=f1, fill=INK)
                d.text((20, y + 36), s["settings"], font=f2, fill=MUTED)
                d.text((20, y + 54), s.get("speed", ""), font=f2, fill=MUTED)
            tl = tile(ROOT / "out" / s["id"] / "lines" / f"{ids.index(t):03d}.png")
            im.paste(tl, (CAP_W + 20, y + (ROW_H - tl.height) // 2))
            y += ROW_H
    im.save(FIG / f"{name}.png")
    print("wrote", name, [s["id"] for s in present])


texts = {r[0]: r[1] for r in rows}
for t in ["short", "definition", "maths", "pangram"]:
    figure(f"compare-{t}", [t], f'Same input: "{texts[t]}"')
figure("compare-charset", ["caps", "digits", "symbols"], "Capitals, digits, symbols (same three lines for every system)")

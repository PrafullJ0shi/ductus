"""Side-by-side comparison figure with a caption row (IBM Plex Mono), per DESIGN.md.

  python scripts/figure_grid.py out.jpg IMG1 "caption 1" IMG2 "caption 2" [...] [--crop y0,y1] [--height 1400]
--crop keeps a vertical band of every image (fractions of its height), so pages compare
at a readable size."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/sand-box/google/IBM Plex Mono/IBMPlexMono-Regular.ttf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("items", nargs="+")
    ap.add_argument("--crop", default="0,1"); ap.add_argument("--height", type=int, default=1400)
    a = ap.parse_args()
    y0f, y1f = map(float, a.crop.split(","))
    pairs = list(zip(a.items[::2], a.items[1::2]))
    tiles = []
    for path, cap in pairs:
        im = Image.open(path).convert("RGB")
        im = im.crop((0, int(y0f * im.height), im.width, int(y1f * im.height)))
        im = im.resize((int(im.width * a.height / im.height), a.height), Image.LANCZOS)
        tiles.append((im, cap))
    try:
        font = ImageFont.truetype(FONT, 30)
    except OSError:
        font = ImageFont.load_default()
    gap, cap_h = 24, 64
    W = sum(t.width for t, _ in tiles) + gap * (len(tiles) - 1)
    out = Image.new("RGB", (W, a.height + cap_h), "#FBFAF7")
    d = ImageDraw.Draw(out)
    x = 0
    for t, cap in tiles:
        out.paste(t, (x, cap_h))
        d.text((x + 8, 16), cap, fill="#5C6170", font=font)
        d.rectangle((x, cap_h, x + t.width - 1, cap_h + a.height - 1), outline="#D9D6CE")
        x += t.width + gap
    out.save(a.out, quality=85)
    print(a.out, out.size)


if __name__ == "__main__":
    main()

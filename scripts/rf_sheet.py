"""Zoomed sheet: RNN output (top) against the r and f redrawn from the glyph templates (bottom).
  python scripts/rf_sheet.py out.png [mode] [seed]"""
import sys
import cv2, numpy as np
from ductus.engines.rnn import RNNEngine, render_line
out = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else "cursive"; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 5
L = ["range river from", "first after for", "fifty rare often"]
e = RNNEngine(style=1, bias=1.5)
from PIL import Image, ImageDraw, ImageFont
F = ImageFont.truetype("/usr/share/fonts/truetype/sand-box/google/IBM Plex Mono/IBMPlexMono-Regular.ttf", 22)
blocks = []
for m, cap in (("off", "RNN letters"), (mode, f"r and f redrawn ({mode})")):
    e.glyphs = m
    rows = []
    for i, t in enumerate(L):
        s, _ = e.strokes(t, seed=seed + i)
        im = render_line(s, len(t), pen_px=2.0)
        ys, xs = np.where(im < 160)
        im = im[max(0, ys.min() - 10):ys.max() + 10, max(0, xs.min() - 10):xs.max() + 10]
        im = cv2.resize(im, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        rows.append(cv2.copyMakeBorder(im, 0, 0, 0, max(0, 1100 - im.shape[1]), cv2.BORDER_CONSTANT, value=255)[:, :1100])
    head = Image.new("L", (1100, 44), 255); ImageDraw.Draw(head).text((8, 10), cap, fill=60, font=F)
    blocks.append(np.vstack([np.array(head)] + rows))
e.close()
cv2.imwrite(out, np.vstack([blocks[0], np.full((20, 1100), 255, np.uint8), blocks[1]]))

"""Three generated-only example lines from the default-hand blind test (proxy_v3).

  python docs/diagrams/build_blind_examples.py GENERATED_DIR docs/assets/blind-generated-examples.png

No real handwriting is shown: these are ductus outputs only.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

src, dest = Path(sys.argv[1]), Path(sys.argv[2])
PAPER, INK, MUTED, RULE = (251, 250, 247), (29, 31, 36), (92, 97, 112), (217, 214, 206)
MONO = "/usr/share/fonts/truetype/sand-box/google/IBM Plex Mono/IBMPlexMono-Regular.ttf"
S = 2
f = ImageFont.truetype(MONO, 12 * S)
files = ["173_000.png", "204_001.png", "209_002.png"]
W, ROW = 1100, 96
im = Image.new("RGB", (W * S, (len(files) * ROW + 40) * S), PAPER)
d = ImageDraw.Draw(im)
d.text((28 * S, 12 * S), "Generated lines used in the default-hand blind test (ductus output only, unedited)", font=f, fill=MUTED)
y = 40
for n in files:
    g = Image.open(src / n).convert("L")
    h = 56 * S
    s = h / g.height
    w = min(int(g.width * s), (W - 56) * S)
    g = g.resize((w, int(g.height * w / int(g.width * s) * 0 + g.height * s * (w / (g.width * s)))), Image.LANCZOS)
    a = np.asarray(g, np.float32) / 255
    t = Image.fromarray(np.clip(np.array(PAPER) * a[..., None] + np.array(INK) * (1 - a[..., None]), 0, 255).astype(np.uint8))
    d.line([(28 * S, y * S), ((W - 28) * S, y * S)], fill=RULE, width=S)
    im.paste(t, (28 * S, y * S + (ROW * S - t.height) // 2))
    y += ROW
d.line([(28 * S, y * S), ((W - 28) * S, y * S)], fill=RULE, width=S)
im.save(dest, optimize=True)
print("wrote", dest, im.size)

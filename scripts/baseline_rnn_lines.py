"""Write each evaluation sentence with the 2016-era Graves RNN (Grzego/handwriting-generation).

Runs inside the hand-style environment that has TensorFlow and the pretrained checkpoint:
  HAND=/path/to/handwriting-generation  python baseline_rnn_lines.py sentences.txt out_dir [bias] [style]
One sample per sentence, no seed picking (that is the honest baseline).
"""
import os, sys, time, json
import numpy as np
from PIL import Image, ImageDraw

HAND = os.environ.get("HAND", "/home/box/agent-data/hand-style")
sys.path.insert(0, HAND)
from write_page import Model, to_strokes, clean  # noqa: E402

src, out = sys.argv[1], sys.argv[2]
bias = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
style = int(sys.argv[4]) if len(sys.argv) > 4 else 4
os.makedirs(out, exist_ok=True)
m = Model(os.path.join(HAND, "pretrained", "model-29"))
times = []
for i, line in enumerate(open(src).read().splitlines()):
    t0 = time.time()
    text = clean(line, m.charset)
    coords, ok = m.line(text, bias, None if style < 0 else style, np.random.RandomState(1000 + i))
    times.append(time.time() - t0)
    st = to_strokes(coords)
    allp = np.concatenate(st)
    x0, x1 = allp[:, 0].min(), allp[:, 0].max()
    y0, y1 = np.percentile(allp[:, 1], 0.5), np.percentile(allp[:, 1], 99.5)
    S, H = 4, 96
    scale = (H * S * 0.6) / max(1e-6, y1 - y0)
    W = int((x1 - x0) * scale + 24 * S)
    im = Image.new("L", (W, H * S), 255)
    d = ImageDraw.Draw(im)
    for s in st:
        pts = [((x - x0) * scale + 12 * S, (y - (y0 + y1) / 2) * scale + H * S / 2) for x, y in s]
        if len(pts) > 1:
            d.line(pts, fill=0, width=int(2.6 * S), joint="curve")
    im.resize((W // S, H), Image.LANCZOS).save(f"{out}/{i:03d}.png")
json.dump({"seconds_per_line": float(np.mean(times)), "bias": bias, "style": style}, open(f"{out}/meta.json", "w"))

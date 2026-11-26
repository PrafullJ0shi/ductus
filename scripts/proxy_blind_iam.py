"""Stand-in blind test on writers the model never saw in training (IAM test writers that
ship with the DiffBrush repository). For each writer: one real line is the style reference,
the writer's other lines are the "real" set, and ductus writes the same number of new lines
in that style. Images are not redistributed; only the scores are reported.

  python scripts/proxy_blind_iam.py /path/to/DiffBrush/test_data /path/to/DiffBrush/data/wikitext103.te out_dir
"""
import random, sys
from pathlib import Path

import cv2

from ductus.page import Writer, Options, write_lines

test_data, corpus, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
writers = sys.argv[4].split(",") if len(sys.argv) > 4 else ["173", "204", "205", "206", "209", "285"]
look = sys.argv[5] if len(sys.argv) > 5 else "natural"
texts = [t.strip() for t in corpus.read_text().splitlines() if 30 <= len(t.strip()) <= 50]
rng = random.Random(0)
w = Writer(verify=True)
for wid in writers:
    files = sorted((test_data / wid).glob("*.png"))
    imgs = [(f, cv2.imread(str(f), 0)) for f in files]
    imgs.sort(key=lambda t: -t[1].shape[1])
    ref_f, ref = imgs[0]
    (out / "real" ).mkdir(parents=True, exist_ok=True)
    (out / "fake").mkdir(parents=True, exist_ok=True)
    (out / "refs").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "refs" / f"{wid}.png"), ref)
    for f, im in imgs[1:]:
        cv2.imwrite(str(out / "real" / f"{wid}_{f.stem}.png"), im)
    n = len(imgs) - 1
    lines = rng.sample(texts, n)
    lines = ["".join(c for c in t if c.isalnum() or c in " .,;:'!?-()") for t in lines]
    o = Options(style=str(out / "refs" / f"{wid}.png"), look=look, seed=int(wid), max_cer=0.5)
    gen, st = write_lines(w, lines, o)
    for i, g in enumerate(gen):
        cv2.imwrite(str(out / "fake" / f"{wid}_{i:03d}.png"), g)
    print(wid, n, st, flush=True)

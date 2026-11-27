"""Photo-against-photo blind test for a personal profile.

Both sides go through the same pipeline: the writer's real page photos, and pages ductus
writes in that profile and puts through the calibrated phone capture (--photo notebook), are
each read by `ductus style` (profile.build), and the line crops it extracts are compared with
blind_test.py. So scale, ink extraction and cropping are identical on both sides; what is left
is the writing and the capture. The writer's photos and crops are private: keep OUT outside
the repository.

  python scripts/blind_photo.py STYLE_DIR WIKITEXT OUT photo1.jpg photo2.jpg ... [--pages N] [--words N]
"""
import argparse, random, shutil
from pathlib import Path

import cv2

from ductus.page import Writer, Options
from ductus import profile

ap = argparse.ArgumentParser()
ap.add_argument("style"); ap.add_argument("corpus"); ap.add_argument("out"); ap.add_argument("photos", nargs="+")
ap.add_argument("--pages", type=int, default=2); ap.add_argument("--words", type=int, default=130)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--surface", default="notebook")
ap.add_argument("--skip-real", action="store_true")
a = ap.parse_args()
out = Path(a.out)


def crops(photos, sub):
    prof_dir = out / f"{sub}_prof"
    shutil.rmtree(prof_dir, ignore_errors=True)
    profile.build(photos, str(prof_dir), reader=w.reader, max_refs=400)
    d = out / sub
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    for i, f in enumerate(sorted((prof_dir / "refs").glob("*.png"))):
        shutil.copy(f, d / f"p_{i:03d}.png")
    return len(list(d.glob("*.png")))


w = Writer(verify=True)
if not a.skip_real:
    print("real crops:", crops(a.photos, "real"), flush=True)
texts = [t.strip() for t in Path(a.corpus).read_text().splitlines() if 60 <= len(t.strip()) <= 400]
rng = random.Random(a.seed)
pages = []
(out / "fake_pages").mkdir(parents=True, exist_ok=True)
for p in range(a.pages):
    words = []
    while len(words) < a.words:
        words += "".join(c for c in rng.choice(texts) if c.isalnum() or c in " .,;:'!?-()").split()
    txt = " ".join(words[:a.words])
    r = w.write(txt, Options(style=a.style, look="profile", photo=a.surface, seed=a.seed * 100 + p, log=lambda *x: None))
    for k, img in enumerate(r.pages):
        f = out / "fake_pages" / f"page{p}_{k}.png"
        cv2.imwrite(str(f), img[..., ::-1])
        pages.append(str(f))
    print("page", p, r.stats, flush=True)
print("fake crops:", crops(pages, "fake"), flush=True)

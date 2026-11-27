"""One step of the realism loop: write fake pages in a profile, put them through the phone
capture, extract line crops with the same pipeline as the real photos, and score both
classifiers (normal and binarized) against a fixed folder of real crops.

  python scripts/blind_iter.py STYLE CORPUS REAL_CROPS OUT [--pages 3] [--tag name] [--knob k=v ...]

Knobs are set on ductus.realism.knobs.K before writing, so a candidate change can be scored
without changing the defaults. Results are appended to OUT/results.jsonl.
"""
import argparse, json, random, shutil, subprocess, sys, time
from pathlib import Path

import cv2

from ductus.page import Writer, Options
from ductus import profile
from ductus.realism import knobs

ap = argparse.ArgumentParser()
ap.add_argument("style"); ap.add_argument("corpus"); ap.add_argument("real"); ap.add_argument("out")
ap.add_argument("--pages", type=int, default=3); ap.add_argument("--words", type=int, default=130)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--surface", default="notebook")
ap.add_argument("--tag", default="run"); ap.add_argument("--knob", action="append", default=[])
ap.add_argument("--verify", action="store_true"); ap.add_argument("--strong", action="store_true")
a = ap.parse_args()
for kv in a.knob:
    k, v = kv.split("=", 1)
    knobs.K[k] = json.loads(v)
out = Path(a.out) / a.tag
shutil.rmtree(out, ignore_errors=True)
(out / "pages").mkdir(parents=True)
t0 = time.time()
w = Writer(verify=a.verify)
texts = [t.strip() for t in Path(a.corpus).read_text().splitlines() if 60 <= len(t.strip()) <= 400]
rng = random.Random(a.seed)
pages = []
for p in range(a.pages):
    words = []
    while len(words) < a.words:
        words += "".join(c for c in rng.choice(texts) if c.isalnum() or c in " .,;:'!?-()").split()
    r = w.write(" ".join(words[:a.words]), Options(style=a.style, look="profile", photo=a.surface,
                                                   seed=a.seed * 100 + p, log=lambda *x: None))
    for k, img in enumerate(r.pages):
        f = out / "pages" / f"page{p}_{k}.png"
        cv2.imwrite(str(f), img[..., ::-1])
        pages.append(str(f))
from ductus.verify import Reader
profile.build(pages, str(out / "prof"), reader=Reader(), max_refs=400, log=lambda *x: None)
fake = out / "fake"; fake.mkdir()
for i, f in enumerate(sorted((out / "prof" / "refs").glob("*.png"))):
    shutil.copy(f, fake / f"p_{i:03d}.png")
res = {"tag": a.tag, "knobs": dict(knobs.K), "fake_crops": len(list(fake.glob("*.png")))}
here = Path(__file__).parent
for mode in ("", "--binarize"):
    cmd = [sys.executable, str(here / "blind_test.py"), "--real", a.real, "--fake", str(fake),
           "--out", str(out / ("bt" + mode.replace("-", "_")))] + ([mode] if mode else []) + (["--strong"] if a.strong else [])
    txt = subprocess.run(cmd, capture_output=True, text=True).stdout
    j = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
    key = "bin" if mode else "raw"
    res[key] = {"lr": round(j["hog_lbp_logreg_acc"][0], 3), "lr_sd": round(j["hog_lbp_logreg_acc"][1], 3),
                "cnn": round(j["small_cnn_acc"][0], 3), "cnn_sd": round(j["small_cnn_acc"][1], 3),
                "n": [j["windows_real"], j["windows_generated"]]}
    if "resnet18_ft_acc" in j:
        res[key]["resnet"] = round(j["resnet18_ft_acc"][0], 3); res[key]["resnet_sd"] = round(j["resnet18_ft_acc"][1], 3)
res["secs"] = round(time.time() - t0)
print(json.dumps(res))
with open(Path(a.out) / "results.jsonl", "a") as fh:
    fh.write(json.dumps(res) + "\n")

"""Before/after images for the two phone capture surfaces.

usage: python scripts/photo_compare.py clean_page.png out_dir [seed]
The clean page (white A4, 300 dpi) is shown next to `--photo a4`; the same ink laid on grid
paper is shown next to `--photo notebook`."""
import sys
from pathlib import Path

import cv2
import numpy as np

from ductus.paper import make_paper, phone_photo


def side(a, b, h=1920):
    r = [cv2.resize(x, (int(x.shape[1] * h / x.shape[0]), h), interpolation=cv2.INTER_AREA) for x in (a, b)]
    gap = np.full((h, 24, 3), 255, np.uint8)
    return np.hstack([r[0], gap, r[1]])


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    page = cv2.cvtColor(cv2.imread(str(src)), cv2.COLOR_BGR2RGB)
    a4 = phone_photo(page, seed=seed, surface="a4")
    # transmittance of the ink over white, laid on grid paper
    grid = make_paper("grid", seed=seed, size=(page.shape[1], page.shape[0])).astype(np.float32)
    white = np.percentile(page.reshape(-1, 3), 90, axis=0)
    T = np.clip(page.astype(np.float32) / white, 0, 1)
    gpage = (grid * T).clip(0, 255).astype(np.uint8)
    nb = phone_photo(gpage, seed=seed, surface="notebook")
    out.mkdir(parents=True, exist_ok=True)
    for name, a, b in (("photo_a4_compare.jpg", page, a4), ("photo_notebook_compare.jpg", gpage, nb)):
        cv2.imwrite(str(out / name), cv2.cvtColor(side(a, b), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(out / name)


if __name__ == "__main__":
    main()

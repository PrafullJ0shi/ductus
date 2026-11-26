"""Blind realism test: can a classifier (or a person) tell real handwriting from ductus output?

  python scripts/blind_test.py --real real_lines/ --fake fake_lines/ --out blind/

Both sets get the same treatment: ink extracted from the background, written as dark ink on
white, scaled to 64 px line height, then cut into 64x256 windows with ink in them.

(a) Classifier: HOG + local binary pattern histograms -> logistic regression, and a small CNN,
    each with 5-fold cross-validation grouped by source line (windows of one line never sit
    in both train and test). Accuracy near 50% means the classifier cannot separate them.
(b) Human / reviewer sheet: a shuffled grid of 24 windows, numbered, with the answer key in
    a separate JSON file. Give the sheet to people (or fresh vision-model reviewers) who have
    not seen the key, ask "real or generated?" for each number, then score with --score.
"""
import argparse, json, random
from pathlib import Path

import cv2
import numpy as np


def load_dir(d):
    return [p for p in sorted(Path(d).glob("*.png")) + sorted(Path(d).glob("*.jpg"))]


def normalise(gray):
    from ductus.segment import ink_alpha
    g = gray.astype(np.float32)
    bg = cv2.medianBlur(g.astype(np.uint8), 31).astype(np.float32)
    rel = np.clip(255 - (bg - g) * 255 / np.maximum(bg, 1), 0, 255)
    a = ink_alpha(rel)
    ys = np.where(a.max(1) > 0.3)[0]
    if len(ys) < 4:
        return None
    a = a[max(0, ys.min() - 4):ys.max() + 5]
    h = 64
    w = max(8, int(a.shape[1] * h / a.shape[0]))
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    return (255 * (1 - a)).astype(np.uint8)


def windows(img, w=256, step=160):
    out = []
    for x in range(0, max(1, img.shape[1] - w + 1), step):
        win = img[:, x:x + w]
        if win.shape[1] < w:
            continue
        if (win < 128).mean() > 0.02:
            out.append(win)
    return out


def feats(win):
    from skimage.feature import hog, local_binary_pattern
    h = hog(win, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    lbp = local_binary_pattern(win, 8, 1, "uniform")
    lh, _ = np.histogram(lbp[win < 200], bins=10, range=(0, 10), density=True)
    ink = (255 - win.astype(np.float32)) / 255
    return np.r_[h, lh, ink.mean(), ink.std(), (ink > 0.5).mean()]


def cnn_cv(X, y, groups, folds=5, epochs=20):
    import torch, torch.nn as nn
    from sklearn.model_selection import GroupKFold
    accs = []
    for tr, te in GroupKFold(folds).split(X, y, groups):
        torch.manual_seed(0)
        net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(), nn.Linear(64, 2))
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        Xt = torch.tensor(X[tr], dtype=torch.float32)[:, None] / 255
        yt = torch.tensor(y[tr])
        for _ in range(epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 32):
                b = perm[i:i + 32]
                loss = nn.functional.cross_entropy(net(Xt[b]), yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            p = net(torch.tensor(X[te], dtype=torch.float32)[:, None] / 255).argmax(1).numpy()
        accs.append(float((p == y[te]).mean()))
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real"); ap.add_argument("--fake"); ap.add_argument("--out", default="blind")
    ap.add_argument("--sheet", type=int, default=24)
    ap.add_argument("--binarize", action="store_true",
                    help="threshold both sets and soften edges the same way, so only letter shapes differ")
    ap.add_argument("--score", help="answers JSON {number: 'real'|'generated'} to score against the key")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.score:
        key = json.loads((out / "answer_key.json").read_text())
        ans = json.loads(Path(a.score).read_text())
        right = sum(1 for k, v in ans.items() if key.get(str(k)) == v)
        print(json.dumps({"correct": right, "total": len(ans), "accuracy": right / max(1, len(ans))}))
        return
    X, y, g, wins = [], [], [], []
    fake_files = load_dir(a.fake)
    prefixes = {p.name.split("_")[0] for p in fake_files}
    # only writers that appear in both sets
    real_files = [p for p in load_dir(a.real) if p.name.split("_")[0] in prefixes]
    for label, files in ((1, real_files), (0, fake_files)):
        for li, p in enumerate(files):
            im = cv2.imread(str(p), 0)
            n = normalise(im) if im is not None else None
            if n is not None and a.binarize:
                n = cv2.GaussianBlur(np.where(n < 128, 0, 255).astype(np.uint8), (3, 3), 0.8)
            if n is None:
                continue
            for w_ in windows(n):
                wins.append((label, w_))
                X.append(w_); y.append(label); g.append(f"{label}-{li}")
    X, y = np.array(X), np.array(y)
    g = np.array(g)
    # balance the classes so chance is 50%
    rs = np.random.RandomState(0)
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    k = min(len(i1), len(i0))
    keep = np.sort(np.r_[rs.choice(i1, k, replace=False), rs.choice(i0, k, replace=False)])
    X, y, g = X[keep], y[keep], list(g[keep])
    wins = [wins[i] for i in keep]
    print(f"windows: real {int(y.sum())}, generated {int((1 - y).sum())}")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    F = np.array([feats(w_) for w_ in X])
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    acc_lr = cross_val_score(lr, F, y, groups=g, cv=GroupKFold(5))
    acc_cnn = cnn_cv(X, y, np.array(g))
    res = {"windows_real": int(y.sum()), "windows_generated": int((1 - y).sum()),
           "hog_lbp_logreg_acc": [float(np.mean(acc_lr)), float(np.std(acc_lr))],
           "small_cnn_acc": [float(np.mean(acc_cnn)), float(np.std(acc_cnn))],
           "chance": float(max(y.mean(), 1 - y.mean()))}
    print(json.dumps(res, indent=1))
    (out / "classifier.json").write_text(json.dumps(res, indent=1))
    # reviewer sheet
    rng = random.Random(7)
    reals = [w_ for l, w_ in wins if l == 1]
    fakes = [w_ for l, w_ in wins if l == 0]
    k = min(a.sheet // 2, len(reals), len(fakes))
    picks = [(1, w_) for w_ in rng.sample(reals, k)] + [(0, w_) for w_ in rng.sample(fakes, k)]
    rng.shuffle(picks)
    tiles, key = [], {}
    for i, (l, w_) in enumerate(picks, 1):
        t = np.full((64 + 22, 256 + 12), 255, np.uint8)
        t[20:84, 6:262] = w_
        cv2.putText(t, str(i), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 90, 1)
        tiles.append(t)
        key[str(i)] = "real" if l else "generated"
    cols = 3
    rows = [np.hstack(tiles[r:r + cols] + [np.full_like(tiles[0], 255)] * (cols - len(tiles[r:r + cols])))
            for r in range(0, len(tiles), cols)]
    cv2.imwrite(str(out / "review_sheet.png"), np.vstack(rows))
    (out / "answer_key.json").write_text(json.dumps(key, indent=1))
    print("sheet:", out / "review_sheet.png", "key:", out / "answer_key.json")


if __name__ == "__main__":
    main()

"""Build the README demo GIFs: a definitions page written line by line.

Uses the same input and settings as the definitions sample in scripts/make_samples.py
(examples/definitions.txt, notes mode, natural look, seed 3, default upright hand).
Uses the default upright hand, not a personal style profile.

  python docs/diagrams/build_demo.py [OUT_DIR]

Writes demo-light.gif and demo-dark.gif under docs/assets/ by default.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "assets"

# Design tokens (DESIGN.md)
PAPER = np.array([0xFB, 0xFA, 0xF7], dtype=np.float32)
INK = np.array([0x1C, 0x2A, 0x6E], dtype=np.float32)
DARK_PAPER = np.array([0x15, 0x17, 0x1C], dtype=np.float32)
DARK_INK = np.array([0x8C, 0x9B, 0xE0], dtype=np.float32)
WIDTH = 720
HOLD = 12          # final-frame repeats (at 10 fps this is 1.2 s)
FPS = 10
LINE_STEPS = 1     # one frame per written line


def _write_page():
    from ductus.page import Writer, Options
    text = (ROOT / "examples" / "definitions.txt").read_text(encoding="utf-8")
    r = Writer().write(
        text,
        Options(mode="notes", look="natural", seed=3,
                log=lambda *a: print(*a, file=sys.stderr)),
    )
    return r


def _ink_amount(rgb: np.ndarray) -> np.ndarray:
    """0 = paper, 1 = full ink, from a rendered RGB page."""
    paper = np.median(rgb.reshape(-1, 3).astype(np.float32), axis=0)
    # ink is darker than paper; use the strongest channel drop
    drop = (paper - rgb.astype(np.float32)).max(axis=2)
    amt = np.clip(drop / max(8.0, float(np.percentile(drop, 99.5))), 0, 1)
    # ignore faint paper texture
    amt[amt < 0.08] = 0
    return amt


def _line_groups(page, words, h, w):
    """Group word placements and pen strokes into writing-order lines."""
    events = []
    for k, pl in enumerate(page.words):
        events.append((pl.base, pl.x, "w", k))
    for k, (pts, _wf) in enumerate(page.strokes):
        events.append((float(np.median(pts[:, 1])) + 1e-3, float(pts[0, 0]), "s", k))
    events.sort(key=lambda e: (round(e[0] / 100), e[1]))

    groups, cur_key, cur = [], None, []
    for y, x, kind, k in events:
        key = round(y / 100)
        if cur_key is None:
            cur_key = key
        if key != cur_key:
            groups.append(cur)
            cur, cur_key = [], key
        cur.append((y, x, kind, k))
    if cur:
        groups.append(cur)

    lines = []
    for evs in groups:
        mask = np.zeros((h, w), dtype=bool)
        xs = []
        for _y, x, kind, k in evs:
            if kind == "w":
                pl = page.words[k]
                wi = words[pl.idx]
                hh, ww = wi.alpha.shape
                # axis-aligned box of the placed crop (good enough to wipe a line)
                x0 = int(np.floor(pl.x))
                y0 = int(np.floor(pl.base - wi.baseline * pl.scale))
                x1 = int(np.ceil(pl.x + ww * pl.scale * getattr(pl, "sx", 1.0)))
                y1 = int(np.ceil(y0 + hh * pl.scale))
                x0, y0 = max(0, x0 - 4), max(0, y0 - 4)
                x1, y1 = min(w, x1 + 4), min(h, y1 + 4)
                mask[y0:y1, x0:x1] = True
                xs.append(x0)
                xs.append(x1)
            else:
                pts, _wf = page.strokes[k]
                x0 = max(0, int(pts[:, 0].min()) - 8)
                x1 = min(w, int(pts[:, 0].max()) + 8)
                y0 = max(0, int(pts[:, 1].min()) - 8)
                y1 = min(h, int(pts[:, 1].max()) + 8)
                mask[y0:y1, x0:x1] = True
                xs.append(x0)
                xs.append(x1)
        if not mask.any():
            continue
        lines.append((mask, (min(xs) if xs else 0, max(xs) if xs else w)))
    return lines


def _compose(amt: np.ndarray, paper: np.ndarray, ink: np.ndarray) -> np.ndarray:
    a = amt[..., None]
    return np.clip(paper * (1 - a) + ink * a, 0, 255).astype(np.uint8)


def _crop_box(amt: np.ndarray, pad: int = 56):
    ys, xs = np.where(amt > 0.08)
    y0, y1 = max(0, ys.min() - pad), min(amt.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(amt.shape[1], xs.max() + pad)
    return y0, y1, x0, x1


def _resize(img: np.ndarray, width: int) -> np.ndarray:
    im = Image.fromarray(img)
    h = int(round(im.height * (width / im.width)))
    return np.array(im.resize((width, h), Image.Resampling.LANCZOS))


def _frames(amt, lines, y0, y1, x0, x1, paper, ink):
    h, w = amt.shape
    revealed = np.zeros((h, w), dtype=np.float32)
    frames = []
    # blank sheet first
    full = _compose(amt, paper, ink)
    blank = np.broadcast_to(paper.astype(np.uint8), full.shape).copy()
    frames.append(_resize(blank[y0:y1, x0:x1], WIDTH))
    for mask, (xmin, xmax) in lines:
        span = max(1, xmax - xmin)
        for s in range(1, LINE_STEPS + 1):
            cut = xmin + int(round(span * (s / LINE_STEPS)))
            step = mask.copy()
            step[:, cut:] = False
            revealed[step] = amt[step]
            shot = _compose(revealed, paper, ink)
            frames.append(_resize(shot[y0:y1, x0:x1], WIDTH))
    # guarantee the last frame is the finished crop
    frames.append(_resize(full[y0:y1, x0:x1], WIDTH))
    return frames


def _save_gif(frames, path: Path, hold: int = HOLD):
    ims = [Image.fromarray(f).convert("P", palette=Image.Palette.ADAPTIVE, colors=24) for f in frames]
    # hold the finished page
    durations = [int(1000 / FPS)] * len(ims)
    durations[-1] = hold * int(1000 / FPS)
    path.parent.mkdir(parents=True, exist_ok=True)
    ims[0].save(
        path,
        save_all=True,
        append_images=ims[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )


def _optimize(path: Path, ceiling_kb: int = 1800) -> None:
    import shutil, subprocess
    if not shutil.which("gifsicle"):
        return
    tmp = path.with_suffix(".opt.gif")
    for lossy in (30, 50, 80, 120):
        subprocess.check_call(
            ["gifsicle", "-O3", f"--lossy={lossy}", "--colors", "16",
             "-o", str(tmp), str(path)],
        )
        if tmp.stat().st_size <= ceiling_kb * 1024:
            break
    tmp.replace(path)


def main():
    print("writing definitions page (word cache used when present)", flush=True)
    r = _write_page()
    rgb = r.pages[0]
    page = r.layouts[0]
    amt = _ink_amount(rgb)
    y0, y1, x0, x1 = _crop_box(amt)
    lines = _line_groups(page, r.words, *amt.shape)
    print(f"lines={len(lines)} crop={x1-x0}x{y1-y0} frames_est={1 + len(lines)*LINE_STEPS + 1}", flush=True)
    light = _frames(amt, lines, y0, y1, x0, x1, PAPER, INK)
    dark = _frames(amt, lines, y0, y1, x0, x1, DARK_PAPER, DARK_INK)
    lp, dp = OUT / "demo-light.gif", OUT / "demo-dark.gif"
    _save_gif(light, lp)
    _save_gif(dark, dp)
    _optimize(lp)
    _optimize(dp)
    print(f"wrote {lp} ({lp.stat().st_size/1024:.0f} KB), {dp} ({dp.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()

# Pens and tip sizes

This page lists what each `--pen` and `--tip` value does. Every number below is copied from
the code, and each table names the file it comes from. If the code and this page disagree,
the code is right; please open an issue.

How the ink is drawn once the pen is chosen is in [INK.md](INK.md).

## What a pen choice changes

A pen preset sets three things, in two different places in the code:

| What | Where it is set | Used by |
|---|---|---|
| Line width on paper, and the nib shape of the fountain pen | `src/ductus/pens.py` (`PENS`, `TIPS`, `get_pen`, `nib_kernel`) | every ink model; every word is re-inked at this width in `render.word_layer` |
| Ink colour and texture, default model | `src/ductus/realism/ink.py` (`PENS`); `render.REALISM_PENS` maps each `--pen` name to an entry there | `--ink-model realism` (the default) |
| Ink colour and texture, simpler model | `src/ductus/pens.py` (`PENS`, `concentration`, `to_rgb`) | `--ink-model pens` |

With `--ink-model flat` the pen only sets the line width. The colour is `--ink` or `#1c2a6e`.

## The six presets

`--pen` accepts these names. The default is `blue-ballpoint`.

Line width without `--tip`, from `pens.py`:

| `--pen` | Kind | Line on paper | Nib |
|---|---|---|---|
| `blue-ballpoint` (default) | ballpoint | 0.38 mm | round |
| `black-ballpoint` | ballpoint | 0.38 mm | round |
| `blue-gel` | gel | 0.48 mm | round |
| `black-gel` | gel | 0.48 mm | round |
| `fountain` | fountain | 0.42 mm | thin ellipse, ratio 0.38, angle 40 degrees |
| `pencil` | pencil | 0.42 mm | round |

Colour of one average stroke on white paper. The default model reads the first column; the
`pens` model reads the second:

| `--pen` | Default model (`realism/ink.py`) | `--ink-model pens` (`pens.py`) |
|---|---|---|
| `blue-ballpoint` | `#344296` | `#2c3a86` |
| `black-ballpoint` | `#303038` | `#2a2a30` |
| `blue-gel` | `#1e3ec4` | `#1f3fc4` |
| `black-gel` | `#16161c` | `#141418` |
| `fountain` | `#243276` | `#2a3a7e` |
| `pencil` | `#58585e` | `#55565c` |

Texture of each pen in the default model (`realism/ink.py`). Higher means more of the effect:

| `--pen` | Density | Striations | Skips | Blobs | Bleed (px at 600 dpi) | Other |
|---|---|---|---|---|---|---|
| `blue-ballpoint` | 1.0 | 0.22 | 0.25 | 0.55 | 0.5 | |
| `black-ballpoint` | 1.05 | 0.22 | 0.25 | 0.55 | 0.5 | |
| `blue-gel` | 1.3 | 0 | 0.02 | 0.15 | 1.0 | darker rim `edge_dark` 0.25 |
| `black-gel` | 1.35 | 0 | 0.02 | 0.15 | 1.0 | darker rim `edge_dark` 0.25 |
| `fountain` | 1.1 | 0 | 0 | 0.8 | 1.4 | darker rim 0.35, nib ratio 0.35 |
| `pencil` | 0.8 | 0 | 0 | 0.05 | 0.2 | paper grain 0.6 |

In words: a ballpoint is lighter on fast straight strokes, with faint light tracks along the
line, small gaps and small blobs at stroke ends. A gel pen is dense and even with almost no
gaps and a little more bleed. A fountain pen pools ink at stroke ends and dries darker at the
edges. A pencil is lighter and shows the grain of the paper.

Some fields are only used on some paths. `edge_dark`, `taper`, `width_scale` and the nib
angle matter when strokes are given as pen paths (`render_strokes`, `line_width_px`), which the
page writer does not use for words. For words, which arrive as images, `ink_from_mask` uses
density, flow variation, speed fade, striations, skips, blobs, bleed and grain.

## Tip sizes

`--tip` takes the nominal tip size in millimetres. The tip printed on a pen is not the line it
leaves, so `pens.py` maps it through `TIPS`:

| `--tip` | Line on paper: ballpoint, pencil | Line on paper: gel, fountain |
|---|---|---|
| `0.5` | 0.30 mm | 0.36 mm |
| `0.7` | 0.38 mm | 0.456 mm |
| `1.0` | 0.50 mm | 0.60 mm |

Rules, from `get_pen` in `pens.py`:

- The three values above are looked up in `TIPS`. Any other number `t` uses `0.42 * t + 0.08` mm.
  The command line help names 0.5, 0.7 and 1.0, but other values are accepted.
- Gel and fountain pens are multiplied by 1.2, because their ink spreads.
- Giving `--tip` replaces the preset's own width. So `--pen blue-gel` alone is 0.48 mm, while
  `--pen blue-gel --tip 0.7` is 0.456 mm.

## Other width and colour options

| Option | Effect | Code |
|---|---|---|
| `--weight W` | Multiplies the stroke width of words and pen marks. `1.0` leaves it as is. | `Options.weight` sets `look.pen` in `page.py`; `render.ink_coverage` applies it |
| `--ink '#rrggbb'` | Overrides the colour of the chosen pen. | `get_pen(..., ink)` in `pens.py`; `rgb` passed to `compose_realism` |
| `--size MM` | x-height of the writing in mm. Not a pen setting, but it changes how thick a line looks next to the letters. | `Options.size` |

## Personal profiles and pens

With `--style` pointing at a profile folder made by `ductus style`, and no `--pen` and no
`--tip`:

- the ink colour is the one measured from the photos;
- the line width is the measured stroke width, in x-heights, times the page x-height, clipped
  to 0.25 to 0.75 mm (`_stroke_px` in `page.py`).

Giving `--pen` or `--tip` switches back to the preset. See [HABITS.md](HABITS.md) and
[STYLE_GUIDE.md](STYLE_GUIDE.md).

## Try them

`scripts/ink_lab.py` writes one passage with every pen into a folder, reusing cached words so
only the ink changes:

```bash
python scripts/ink_lab.py out/pens --mode notes
```

It writes `pen-<name>.png` for each pen and a stacked `pens.png`. These images are not stored
in the repository.

# CLI and API reference

## Install

```
pip install -e ".[model,pdf]"     # from a clone; Python 3.10+
```

First use downloads the DiffBrush checkpoint (1.2 GB, Google Drive via gdown), the Stable
Diffusion 1.5 VAE (335 MB, Hugging Face) and TrOCR base (1.3 GB, Hugging Face) into
`~/.cache/ductus` and the Hugging Face cache. No login is needed.

The RNN writer, which is the default, needs more: a checkout of
[Grzego/handwriting-generation](https://github.com/Grzego/handwriting-generation) and a Python
with TensorFlow 2. See the environment variables below and the README.

## `ductus write`

```
ductus write INPUT -o OUT [options]
```

The table follows `src/ductus/cli.py`. If you change one, change the other.

**Input and output**

| Option | Default | Meaning |
|---|---|---|
| `INPUT` | `-` | text file with optional markup, or `-` for stdin |
| `--text TEXT` | | inline text instead of a file (wins over `INPUT`) |
| `-o`, `--out PATH` | `page` | output path without extension; writes `PATH.png` (or `PATH-1.png`, ... for several pages) and `PATH.pdf` |
| `--no-pdf` | off | skip the PDF |
| `--svg` | off | also write a vector SVG |
| `--animate` | off | also write an SVG that draws itself (`PATH.anim.svg`) |
| `-q`, `--quiet` | off | no progress messages on stderr |

**Content and look**

| Option | Default | Meaning |
|---|---|---|
| `--mode` | `plain` | `plain`, `notes` (also reads `Term: meaning`), or `maths` (`= ...` lines align) |
| `--look` | `natural` | `neat`, `natural`, `messy`, or `profile` (the natural look plus the measurements and habits of the `--style` profile) |
| `--style` | `upright` | a preset (`upright`, `slant`, `quick`, `round`, `loose`; list them with `ductus presets`), a line image, or a profile folder from `ductus style` |
| `--size MM` | `2.8` | x-height of the writing in mm |
| `--paper` | `white` | `white`, `ruled`, or `grid` (5 mm). With `--photo notebook` the default is `grid` |
| `--photo [a4\|notebook]` | off | phone photo of the page. `a4` (used when the flag has no value): one loose white sheet on a desk, slight corner lift, contact shadow. `notebook`: spiral-bound grid pad, pages below, curl at the binding. Both use the same calibrated camera model |
| `--corrections RATE` | by look | share of words written wrong, struck out and rewritten (`messy` 0.03, others 0) |
| `--humanize RATE` | `0.5` | rate of the human-error pass on the laid-out page (drift of size, baseline and slant, retouched letters); `0` turns it off; ignored for `--look neat` |
| `--seed N` | `0` | random seed; same input, options and seed give the same page |

**Pen and ink** (details in [PENS.md](PENS.md) and [INK.md](INK.md))

| Option | Default | Meaning |
|---|---|---|
| `--pen` | `blue-ballpoint` | `blue-ballpoint`, `black-ballpoint`, `blue-gel`, `black-gel`, `fountain`, `pencil`. With a profile and no `--pen`, the measured ink colour and width are used |
| `--tip MM` | pen's own | nominal tip size: `0.5`, `0.7` or `1.0` (other numbers are accepted) |
| `--weight W` | `1.0` | stroke weight multiplier on top of the pen |
| `--ink '#rrggbb'` | pen's colour | override the ink colour |
| `--ink-model` | `realism` | `realism` (striations, skips, blobs, overlap darkening), `pens` (simpler model), or `flat` |
| `--flat-ink` | off | same as `--ink-model flat` |

**Writer and read-back** (details in [ARCHITECTURE.md](ARCHITECTURE.md))

| Option | Default | Meaning |
|---|---|---|
| `--engine` | `default` | `default`: RNN letters when installed, else diffusion, with diffusion as fallback; `rnn`: same, but stops if the RNN is missing; `diffusion`: letter shapes copied from `--style`; `auto`: diffusion first, RNN for words that do not read |
| `--rnn-style N` | `1` | built-in RNN writer style 0 to 7 (style 1 writes plain print letters) |
| `--rnn-bias B` | `1.5` | RNN neatness; higher is neater |
| `--legibility` | `strict` | `strict`: every word must pass both the line and the word OCR read-back, or it is written again; `loose`: only catch wrong or missing words |
| `--allow-loops` | off | keep looped `f`, stylish `r` and looped ascenders; by default such words are written again until the plain form appears (`letterforms.py`) |
| `--max-cer X` | `0.15` | OCR acceptance threshold (character error rate); with `--legibility loose` and a profile, the profile's value (0.5 by default) |
| `--no-verify` | off | skip the OCR read-back (faster, more mistakes) |
| `--steps N` | `25` | diffusion steps; more is slower |
| `--no-cache` | off | generate words afresh instead of reusing earlier ones |

The command prints a JSON line with the files and statistics: words, carrier lines generated,
words still failing after the last round, seconds, the pen and the OCR mean score.

## Environment variables

| Variable | Meaning |
|---|---|
| `DUCTUS_RNN_DIR` | folder of the Grzego/handwriting-generation checkout (default `~/.cache/ductus/handwriting-generation`) |
| `DUCTUS_RNN_PYTHON` | Python with TensorFlow 2 that runs the RNN worker |
| `DUCTUS_RNN_THREADS` | CPU threads for the RNN worker (default 2) |
| `DUCTUS_CACHE` | folder for downloaded weights (default `~/.cache/ductus`). The word cache uses the same variable as its own folder (default `~/.cache/ductus/words`) |

## Markup

Block level, one per line:

| Write | Result |
|---|---|
| `# Heading` / `## Sub heading` | larger writing with a hand-drawn rule under it |
| `---` | separator across the page |
| `Term :: meaning` | definition: underlined term, arrow, meaning with a hanging indent |
| `Note: text` | `Note ->` then the text |
| `1) item`, `2. item`, `- item` | numbered or bulleted list |
| `\| a \| b \|` | table rows; ruled columns, rule under the first row |
| `= expr` | maths continuation, aligned under the first `=` of the line above |

Inline:

| Write | Result |
|---|---|
| `->`, `<-`, `=>` | drawn arrows |
| `[v]`, `[x]` | tick, cross |
| `~~words~~` | struck through |
| `__words__` | underlined |
| `((words))` | circled |
| `{a/b}` | stacked fraction |
| `sqrt(x)` | radical over x |
| `{sum}` | capital sigma |
| `x^2` | superscript |
| `{oops:word}` | word written, then crossed out |
| `=`, `<`, `>`, `%`, `~` | drawn marks (the model has no glyphs for them) |

Characters the model cannot write and that have no mark fall back to a close ASCII form
(`@` becomes `at`, curly quotes become straight quotes).

## `ductus style`

```
ductus style photo1.jpg photo2.jpg ... -o styles/my-hand [--name my-hand]
```

| Option | Default | Meaning |
|---|---|---|
| `photos` | required | one or more page photos |
| `-o`, `--out` | required | profile folder |
| `--name` | `my-hand` | name stored in `profile.json` |
| `--no-ocr` | off | skip the OCR check that finds upside-down pages |

Writes `styles/my-hand/profile.json`, reference line crops in `refs/` and cleaned page ink maps
in `debug/`. Use the folder as `--style styles/my-hand --look profile`. See
[STYLE_GUIDE.md](STYLE_GUIDE.md) and [HABITS.md](HABITS.md).

## `ductus presets`

Lists the built-in styles and a note on each.

## Python

```python
from ductus.page import Writer, Options
from ductus.render import save_pdf, save_svg
import cv2

w = Writer()                                   # loads the models once
r = w.write(open("notes.txt").read(), Options(mode="notes", look="natural", seed=3))
cv2.imwrite("page.png", r.pages[0][..., ::-1])
save_svg(r.layouts[0], r.words, "page.svg", r.look.ink, r.stats["pen_px"], animate=True)
print(r.stats)
```

`write_lines(writer, lines, options)` returns a list with one tight grayscale crop per input
line, and a dict of statistics; the evaluation scripts use it.

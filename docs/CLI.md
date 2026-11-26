# CLI and API reference

## Install

```
pip install -e ".[model,pdf]"     # from a clone; Python 3.10+
```

First use downloads the DiffBrush checkpoint (1.2 GB, Google Drive via gdown), the Stable
Diffusion 1.5 VAE (335 MB, Hugging Face) and TrOCR base (1.3 GB, Hugging Face) into
`~/.cache/ductus` and the Hugging Face cache. No login is needed.

## `ductus write`

```
ductus write INPUT -o OUT [options]
```

| Option | Default | Meaning |
|---|---|---|
| `INPUT` | `-` | text file with optional markup, or `-` for stdin |
| `--text` | | inline text instead of a file |
| `-o, --out` | `page` | output path without extension; writes `OUT.png` (or `OUT-1.png`, ...) and `OUT.pdf` |
| `--style` | `quick` | a preset (`upright`, `slant`, `quick`, `round`, `loose`), a line image, or a profile folder from `ductus style` |
| `--look` | `natural` | `neat`, `natural`, `messy`, or `profile` (use the measured habits of a profile) |
| `--mode` | `plain` | `plain`, `notes` (auto-detects `Term: meaning`), `maths` (`= ...` lines align) |
| `--paper` | `white` | `white` A4, `ruled`, or `grid` (5 mm) |
| `--photo` | off | phone-photo look: perspective, light falloff, noise, JPEG |
| `--ink` | `#1c2a6e` | ink colour |
| `--pen` | `1.0` | stroke weight multiplier |
| `--size` | `2.5` | x-height in mm |
| `--seed` | `0` | random seed (same seed, same page) |
| `--steps` | `25` | diffusion steps; 50 is slightly cleaner and twice as slow |
| `--max-cer` | by look | OCR acceptance threshold (neat 0.15, natural 0.25, profile 0.5) |
| `--corrections` | by look | share of words written wrong, struck out and rewritten |
| `--no-verify` | off | skip OCR read-back (faster, more spelling slips) |
| `--svg` | off | also write a vector SVG |
| `--animate` | off | also write an SVG that draws itself |
| `--no-pdf` | off | skip the PDF |

The command prints a JSON line with the files and statistics (words, carrier lines generated,
words still failing after the retries, seconds).

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

Writes `styles/my-hand/profile.json`, reference line crops in `refs/` and cleaned page ink maps
in `debug/`. Use the folder as `--style styles/my-hand --look profile`. See
[STYLE_GUIDE.md](STYLE_GUIDE.md).

## `ductus presets`

Lists the built-in styles.

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

`write_lines(writer, lines, options)` returns one tight grayscale crop per input line; the
evaluation scripts use it.

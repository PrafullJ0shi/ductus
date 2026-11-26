# Architecture

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="diagrams/architecture-dark.svg">
  <img alt="ductus architecture: text and markup go to an RNN writer, every word is read back by OCR and rewritten if it fails, words that still fail go to a diffusion writer, then words are laid out, inked on paper, optionally photographed and saved as PNG, PDF and SVG" src="diagrams/architecture.svg">
</picture>

The diagram has nine boxes. Dashed boxes are optional and dashed arrows are returns or
optional links. The source is [`diagrams/build_architecture.py`](diagrams/build_architecture.py);
it writes light and dark SVG files and an HTML copy for the diagram kit's checks.

What the diagram leaves out, so that it stays readable:

- The style profile also sets the layout look (size, spacing, slant, ink colour, habits). See
  [HABITS.md](HABITS.md).
- Symbols such as arrows, ticks and fraction bars do not go through a writer. They are drawn
  as pen marks inside the layout step.
- Without `--photo` the ink step goes straight to the outputs.

## Two writers

| Writer | When it is used | Strength | Weakness |
|---|---|---|---|
| RNN (`engines/rnn/`) | Default, when the model is installed | Plain print-like letters, fast on a CPU | Cannot copy a style from a photo; fails on capitals, digits and symbols it was not trained on |
| Diffusion (`engines/diffbrush/`) | Fallback by default; the only writer with `--engine diffusion` | Copies a style from a reference line; wider character set | Slow on a CPU, and less readable |

`--engine` picks the order (`Writer.engines` in `page.py`):

| `--engine` | Primary writer | Fallback | Rounds |
|---|---|---|---|
| `default` (RNN installed) or `rnn` | RNN, 4 rounds | diffusion, 2 more rounds | 6 |
| `default` (RNN not installed) or `diffusion` | diffusion | none | 4 (5 for `--look neat`) |
| `auto` | diffusion, 2 rounds | RNN, 2 more rounds | 4 |

Words that the RNN cannot write at all (`supported()` in `engines/rnn/`) wait for the
fallback instead of being tried first. `--engine rnn` stops with an error if the RNN model is
missing; `--engine default` quietly uses diffusion.

## Modules

| Module | Role |
|---|---|
| `markup.py` | Parses text into blocks (heading, definition, note, list item, table, maths line, rule) and inline tokens (words with decorations, symbols, fractions, roots, circles, superscripts). |
| `engines/rnn/` | The Graves RNN as a line writer. It runs as a separate process with its own Python (`DUCTUS_RNN_PYTHON`) and draws strokes into 64 px line images. |
| `engines/diffbrush/` | The vendored DiffBrush network (MIT) and a CPU wrapper: weights download, style and content tensors, DDIM sampling, VAE decode. |
| `synth.py` | Packs words into carrier lines, generates them in batches of 8, splits lines back into words, reads them back, rewrites failures. |
| `verify.py` | TrOCR reader, character error rate, text normalisation. |
| `letterforms.py` | Letter-shape check: rejects words with a looped `f`, a stylish `r` or looped ascenders. |
| `segment.py` | Ink maps, word splitting, baseline and x-height, stroke width, skew, line finding. |
| `pen.py` | Pen marks as strokes: arrows, ticks, crosses, `=`, `<`, `>`, `%`, sigma, radicals, fraction bars, underlines, strike-throughs, circles, rules. |
| `layout.py` | Box model and page flow with human variation; notebook structures; multi-page. |
| `humanize_bridge.py`, `realism/human_errors.py` | The `--humanize` pass: slow drift of size, baseline and slant, and letters written over a second time. |
| `render.py` | Ink coverage and overlap count, the three ink models, PNG, PDF, SVG and animated SVG. |
| `pens.py`, `realism/ink.py` | Pen presets and the ink models. See [PENS.md](PENS.md) and [INK.md](INK.md). |
| `paper.py` | White, ruled and grid paper; entry to the phone photo. |
| `realism/phone_capture.py`, `realism/calibration.json` | The phone camera model and its measured parameters. `photo.py` is the older model, kept for tests. |
| `profile.py`, `habits.py` | Personal style profile from photos, and writer habits. See [HABITS.md](HABITS.md). |
| `page.py` | Orchestration: `Writer.write` and `write_lines`. |
| `cli.py` | `ductus write`, `ductus style`, `ductus presets`. |

## Data flow in one page

1. `markup.parse` returns blocks. Corrections (`{oops:word}` or the `--corrections` rate) are
   inserted as extra struck-out words. With a profile, `habits.apply_text` first rewrites
   words the writer spells or abbreviates in a personal way.
2. All words are listed in layout order. `Synthesizer` packs them into carriers of 46
   characters, adds filler words to reach the length, and generates 8 carriers per batch.
   Short list markers such as `4)` or `(b)` get a carrier of their own, so the word split
   cannot cut them.
3. Each carrier image is cleaned and split into words at the widest gaps. It is read back
   twice: the whole line, aligned to the target words with an edit-distance search, and each
   word crop on its own. With `--legibility strict` (default) the worse of the two errors
   counts; with `loose` the mean counts. A width check catches a word cut in the wrong place,
   and `letterforms.py` rejects looped letters.
4. Words above the error threshold go to the next round in new carriers with a new seed. The
   best attempt for each word is kept. `stats` in the printed JSON reports words that still
   fail after the last round.
5. `Layouter` turns blocks into boxes. Words keep the baseline the writer gave them; the
   layout adds line drift, margin wander, gap and size jitter, small rotations, fatigue and
   squeezed words at line ends. Pen marks are drawn at the stroke width of the pen.
6. The humanize pass changes drift and adds retraced letters.
7. `render` builds the page at 600 dpi, inks it with the chosen model, multiplies it onto
   the paper and reduces it to 300 dpi.
8. With `--photo`, `paper.phone_photo` applies the camera model.

OCR threshold, `max_cer`: 0.15 by default; `--legibility loose` with a profile uses the
profile's value, 0.5 by default (`_max_cer` in `page.py`). `--max-cer` overrides both.

## Why carriers and not one call per line of the page

A page line in notes mode is interrupted by arrows, fractions, ticks and table rules. The
writers cannot draw those, and splitting a line into short runs moves a line model out of its
training range. Carriers decouple generation from layout: the writer always produces
ordinary sentences, and the layout is free to put words anywhere. A side effect is that every
occurrence of a word is a new sample.

## Speed

Speed depends on the writer, the machine and the cache. Measured numbers, with the machine
and the text they were taken on, are in [BENCHMARK.md](BENCHMARK.md) (per line) and
[EVALUATION.md](EVALUATION.md). Generated words are cached under `~/.cache/ductus`
(`--no-cache` turns this off), so changing only the pen, paper or photo re-renders without
writing the words again.

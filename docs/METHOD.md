# Method

This page explains what ductus does and why, step by step. For numbers see
[EVALUATION.md](EVALUATION.md); for the survey that led to these choices see
[RESEARCH.md](RESEARCH.md).

## 1. Generator: one-shot line diffusion (DiffBrush, ICCV 2025)

The 2016 approach (Graves 2013, as in greydanus/scribe) predicts pen offsets one step at a
time with an LSTM and a mixture density output. It draws plausible strokes, but it drifts on
long text, it was trained on IAM-OnDB pen data only, and the style "prime" must itself be a
pen trace. It cannot copy a style from a photo.

DiffBrush writes a whole text line as an image in the latent space of the Stable Diffusion 1.5
VAE. The style comes from one image of a written line, encoded by a CNN plus transformer
style encoder trained with separate vertical and horizontal style objectives. Content is a
glyph image sequence (Unifont) attended by the UNet. Because both inputs are images, a photo
of any handwriting can be the reference.

We run the released checkpoint on CPU: 25 DDIM steps, 64 x 1024 output, 8 lines per batch.

## 2. Carrier lines

The model was trained on full IAM lines. Given a short word, or a line padded with spaces, it
produces stretched or broken writing. Notebook pages are full of short runs (`Term ->`, a
fraction numerator, one table cell). So ductus never asks the model for a layout line. It
packs all the page's words in order into sentences of about 46 characters, tops them up with
common filler words, and generates those. Each result is cut back into words at the widest
column gaps. The layout then places the words wherever the page needs them.

Side effects that help realism:

- Every occurrence of a word is generated in a different context, so repeated words are
  never identical copies.
- Words keep the baseline and x-height the model drew, measured over the whole carrier, so
  the descenders and the slight up and down of real writing survive.

## 3. Read-back and rewrite

Diffusion samples occasionally drop or swap a letter. ductus reads every carrier back with
TrOCR (base, handwritten) and aligns the reading to the target words with an edit-distance
DP over words. Each word crop is also read on its own. A word is accepted when the mean of
the two character error rates is below the threshold and its width is plausible for its
length. Failed words are regenerated in new carriers with new seeds, for up to 3 rounds; the
best attempt is kept.

| Look | Threshold | Intent |
|---|---|---|
| `neat` | 0.15 | legible; catches most slips |
| `natural` | 0.25 | default; catches wrong and missing words |
| `profile` | 0.50 | personal hand; only wrong or missing words, never neatened |

The thresholds are on normalised text (case and punctuation folded). The evaluation uses a
different and larger OCR model as the judge so the check is not graded by itself.

## 4. Pen marks

Arrows, ticks, crosses, `=`, `<`, `>`, `%`, sigma, radical signs, fraction bars, underlines,
rules, strike-throughs and circles are drawn as strokes: a few control points, a quadratic or
polyline path, then low-frequency wobble, pressure taper at both ends and the measured stroke
width of the generated writing. Each mark is drawn fresh, so no two are the same.

## 5. Human layout

A page is laid out by a box model (word, mark, horizontal group, fraction, root, circle,
superscript). The human part is in the parameters, all scaled by one `mess` value:

- line pitch varies a little; each line has its own slope, drifting with a random walk;
- the left margin wanders and the right margin is ragged;
- word gaps, sizes and angles vary;
- headings are written larger and get a hand-drawn rule;
- definitions get an arrow and a hanging indent; maths continuation lines align under `=`;
- a small share of words can be written wrong, struck out and rewritten (`corrections`).

## 6. Ink and paper

Words are rendered as ink coverage at 600 dpi. A smooth random pressure field and per-word
pressure make some strokes darker; a slight blur gives bleed. The ink colour is multiplied
onto the paper (white, ruled or 5 mm grid) and the result downsampled to 300 dpi A4. The
optional phone-photo look adds perspective, lighting falloff, noise and JPEG compression.

## 7. Personal style profiles

`ductus style` turns photos of handwritten pages into a style: lighting normalisation and an
ink map, grid line removal by morphology, rotation to upright (projection profile plus an
OCR confidence check for upside-down), deskew, line segmentation, then the best line crops
become reference images. It also measures habits from the photos (x-height in mm via the grid
period, line pitch, baseline slope and its spread, word gaps, baseline residual, stroke
width) and stores them in `profile.json` for `--look profile`. See
[STYLE_GUIDE.md](STYLE_GUIDE.md).

## Limits of the method

- The engine writes the 80 characters in its alphabet. Other characters are drawn as marks
  or replaced (see [CLI.md](CLI.md)).
- Style copying is one-shot and works best when the reference is a clean, long line. It
  captures slant, width, weight and letter shapes but not every personal glyph.
- CPU speed is about a minute per 100 words.

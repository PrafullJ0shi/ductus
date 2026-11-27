# Why Ductus: 10 differences

What sets this project apart from a handwriting font and from the 2018 Graves RNNs
([sjvasquez](https://github.com/sjvasquez/handwriting-synthesis),
[Grzego](https://github.com/Grzego/handwriting-generation)), with a number or a file for
each claim. The side-by-side figure and the full table are in [VERSUS.md](VERSUS.md). This
is not a claim that the writing cannot be detected; a classifier still separates it, and
those results are in [BLIND_TEST.md](BLIND_TEST.md).

## 1. Best readability among the handwriting models in the public test

On the four main lines of the 7-line benchmark, ductus round 5 has a normalised OCR
character error rate of **0.038**. The sjvasquez RNN scores 0.084 and the Grzego RNN
(bias 1.5) scores 0.124. A Caveat handwriting font still leads, at 0.000. Lower is easier
to read. The judge, the texts and the caveats (n = 7, one sample each) are in
[BENCHMARK.md](BENCHMARK.md) and [`benchmark/metrics.csv`](../benchmark/metrics.csv).

## 2. No repeated letter shapes, unlike a font

In the same run, the share of ink shapes that have a near-identical twin is **0.000** for
ductus and for both RNNs, and **0.462** for Caveat (about 46% of shapes). A font reuses one
image per letter; each ductus character is drawn with its own jitter. See
[VERSUS.md](VERSUS.md#numbers) and `benchmark/scripts/glyph_repetition.py`.

## 3. A style profile from a few phone photos

`ductus style` builds a profile from one or more page photos (3 to 8 is the usual range).
The profile records x-height, line pitch, slope, spacing, stroke width and ink colour, plus
optional habits in `habits.json`. With the default engine those measurements are applied;
letter shapes are copied only with `--engine diffusion` or `auto`, which is slower and often
less readable. See [STYLE_GUIDE.md](STYLE_GUIDE.md) and [HABITS.md](HABITS.md).

## 4. Every word is read back and written again if it fails

Each word is read by TrOCR and rewritten when its character error rate is above the
threshold (**0.15** by default, several rounds). `--legibility loose` only catches wrong or
missing words; `--no-verify` skips the check and is faster, with more mistakes. The method
is in [METHOD.md](METHOD.md#3-read-back-and-rewrite) and the flags are in
[CLI.md](CLI.md#ductus-write).

## 5. Capitals, digits and list markers come from templates

Tokens made only of digits, maths symbols or capitals (`BTC`, `3.14`, `4)`) are drawn from
plain print templates in `src/ductus/printglyphs.py`, each with its own jitter, because the
RNN used to garble them. In round 5, capitals score **0.000** and digits **0.037**, level
with the fonts on those lines. Symbols are still weak (raw CER **0.435**). See
[BENCHMARK.md](BENCHMARK.md#round-5-the-pangram-wrap-fixed-maths-read-better).

## 6. A pen and ink model, not a flat colour

Six presets (`blue-ballpoint`, `black-ballpoint`, `blue-gel`, `black-gel`, `fountain`,
`pencil`) and tip sizes 0.5, 0.7 and 1.0 mm. The default ink model (`--ink-model realism`)
adds striations, skips, end blobs, bleed and darker overlaps, composited in linear light at
600 dpi. A flat-colour mode is kept for comparison. Details and the limits of the model are
in [PENS.md](PENS.md) and [INK.md](INK.md).

## 7. A phone-photo look calibrated on a real photo

`--photo a4` places a loose sheet on a desk; `--photo notebook` is a spiral-bound grid pad.
Both use the same camera model (geometry, light, defocus, noise, JPEG). The parameters in
`src/ductus/realism/calibration.json` were measured from one real phone photo of handwritten
notes (`python -m ductus.realism.calibrate`; more photos would tighten the estimates). See
[REALISM.md](REALISM.md#g-the-realism-package-srcductusrealism) and
[CLI.md](CLI.md#ductus-write).

## 8. Human habits are optional; a neat look turns them down

Strike-throughs (`--corrections`, or `{oops:word}` in the text) are off in the default
`natural` look and in `neat` (the `messy` look sets 0.03). Baseline drift, size changes and
retouches are a separate pass (`--humanize`, default 0.5) which is ignored for `--look
neat`. `neat` also lowers slope, fatigue and smudges (`mess` 0.12, smudges 0). The checklist
is in [REALISM.md](REALISM.md); the flags are in [CLI.md](CLI.md#ductus-write).

## 9. Text flows across as many A4 pages as it needs

The layout is a 300 dpi A4 page (2480 × 3508 px) with margins, and it starts a new page when
the text no longer fits. `ductus write` writes `out.png` or `out-1.png`, `out-2.png`, … and
a single PDF. A 243-word run filled 3 pages at the default size; one page is about 80 to 110
words. See [CLI.md](CLI.md#ductus-write) and [EVALUATION.md](EVALUATION.md#3-speed).

## 10. Open source, runs on CPU, with a public benchmark and an honest blind test

The code is MIT ([LICENSE](../LICENSE)). It runs offline on CPU; no account is required.
The benchmark scripts, the raw scores and the side-by-side figure are in the repository
([BENCHMARK.md](BENCHMARK.md), [VERSUS.md](VERSUS.md)). The blind test, including the runs
the classifier wins, is published in [BLIND_TEST.md](BLIND_TEST.md). The pretrained writer
weights were trained on IAM, which is licensed for non-commercial research use.

## What it doesn't do yet

- **It is not indistinguishable from real handwriting.** A classifier still separates ductus
  lines from real ones. The published scores are overall accuracy on balanced windows, not a
  separate count of generated windows labelled real. The closest figure, **100% minus
  accuracy**, is 1.5% on the default-hand run (`proxy_v3`, both classifiers at 98.5%) and
  0% on the style-matched logistic test (accuracy 100%). That figure equals the share of
  generated windows labelled real only if errors are split evenly between the two classes.
  50% would mean the classifier cannot tell the sets apart. These figures are updated after
  every test. See [BLIND_TEST.md](BLIND_TEST.md).
- **Maths and symbols are still the weak lines.** Maths normalised CER is 0.150 (the judge
  often misreads a drawn `=`); symbols raw CER is 0.435 (`%`, `+`, `;` and `:` are read as
  `#` or dropped). [VERSUS.md](VERSUS.md#where-each-one-wins-and-loses).
- **It is slower than the fonts and the RNNs.** 12.09 seconds a line in the warm benchmark,
  against 0.7 s for Caveat, 2.73 s for sjvasquez and 5.2 s for Grzego, all on a shared CPU.
  A cold 243-word run took about 10 minutes. [VERSUS.md](VERSUS.md#numbers).

# Blind test: can a classifier tell real handwriting from ductus?

**Short answer: yes, every time so far.** The strongest result: a fine-tuned ResNet-18 tells the writer's real photographed lines from lines generated in his own profile 98.5% of the time. On every run below a simple classifier separates
real lines from ductus lines far better than chance. ductus output is **not** indistinguishable
from real handwriting. These are results, not a pass.

**Fooled rate** (updated after every test). The published runs report overall accuracy on
balanced windows (half real, half generated), not a separate count of generated windows
labelled as real. The figure used here is **100% minus accuracy**. It equals the share of
generated windows labelled real only if errors are split evenly between the two classes.
50% would mean the classifier cannot tell the two sets apart; 0% means it labelled none of
the generated windows as real, if that even split holds.

On the published accuracies, that figure is **1.5%** on the default-hand run (`proxy_v3`,
both classifiers at 98.5%), **6.4% / 9.4%** on the earlier style-copy run (`proxy_v1`,
logistic 93.6% / CNN 90.6%), and on the photo-matched run in the writer's own profile
**8.7%** (logistic), **30.1%** (small CNN) and **1.5%** (fine-tuned ResNet-18). The older
style-matched rows are void (see the photo-matched caveats). These are not a claim
that a person, or a stronger detector, would miss generated pages at this rate.

All numbers are in [`benchmark/blind_test.csv`](../benchmark/blind_test.csv). The test is
described in [EVALUATION.md](EVALUATION.md#4-blind-test); this page is the published record of
the runs.

## What 50% means

Windows are balanced half real, half generated, so a classifier that guesses scores 50%. A
score of 50% would mean it cannot find any difference between the two sets; 100% means it
finds one in every window. Lower is better for ductus. A score near 50% only counts if the
classifier is strong enough to find a difference when one exists (see the caveats).

## Method

- **Real lines.** For the IAM runs: lines by writers 173, 204 and 209 from the IAM test set that
  ships with the DiffBrush repository, writers the model was not trained on. 20 lines per writer
  are "real". IAM images are not redistributed here; only scores are.
- **Generated lines.** ductus writes 20 new lines per writer (WikiText sentences, 30 to 50
  characters), 60 lines in all. Run `proxy_v1` copies a style from one reference line per writer;
  `proxy_v3` is the default hand with no style image.
- **Treatment.** Both sets get the same treatment ([`scripts/blind_test.py`](../scripts/blind_test.py)):
  ink extracted from the background, written as dark on white, scaled to 64 px line height, cut
  into 64 x 256 windows that contain ink. The `binarized` variant thresholds both sets so that only
  letter shapes remain and ink texture is gone.
- **Classifiers.** HOG + local-binary-pattern features with logistic regression, and a small CNN.
  5-fold cross-validation grouped by source line, so windows of one line never sit in both train
  and test. Accuracy is the mean over folds; sd is the standard deviation over folds.
- **Scripts.** [`proxy_blind_iam.py`](../scripts/proxy_blind_iam.py),
  [`blind_default.py`](../scripts/blind_default.py), [`blind_test.py`](../scripts/blind_test.py).

## Results

| Run | What it tests | Windows (real / generated) | HOG+LBP logistic | Small CNN |
|---|---|---|---|---|
| `proxy_v1` | 3 IAM writers, flat-ink renderer, style copied from one line | 235 / 235 | 93.6% (sd 3.1) | 90.6% (sd 6.0) |
| `proxy_v1_bin` | same run, binarized | 235 / 235 | 75.5% (sd 3.2) | 62.1% (sd 4.0) |
| `proxy_v3` | 3 IAM writers, default hand (no style image), round 3 engine and ink | 258 / 258 | 98.5% (sd 1.9) | 98.5% (sd 1.6) |
| `proxy_v3_bin` | same run, binarized (letter shape only) | 258 / 258 | 94.6% (sd 1.4) | 91.1% (sd 5.0) |
| `style_matched_r4` (void) | the writer's own profile, 12 real line crops against 12 generated, round 4 engine | 53 / 53 | 100% (sd 0) | 53% (sd 26) |
| `style_matched_r4` binarized (void) | same, binarized | 53 / 53 | 100% (sd 0) | 51% (sd 17) |

`proxy_v1_bin` was scored in the same session as `proxy_v1` and sits in the evaluation
folder, but it was not listed in EVALUATION.md before this page. The style-matched numbers are
the rounded values recorded in EVALUATION.md.

![Three generated lines from the default-hand run](assets/blind-generated-examples.png)

What the rows say:

- **`proxy_v1` to `proxy_v3`.** The earlier run, which copies a style, scores 93.6% and 90.6%.
  The later default-hand run scores 98.5% on both classifiers. That rise is expected: the
  default hand is not trying to look like those writers, so the gap should be larger. It does
  not show the engine got worse.
- **Binarized.** Removing ink texture lowers the score (94.6% and 91.1% on `proxy_v3`), so ink
  is part of the giveaway, but the classifier still wins by a wide margin on letter shapes and
  spacing alone.
- **Style-matched (void).** These crops came from a page reader that cut slices across lines
  on the writer's photos; see the [photo-matched run](#photo-matched-run) for the valid
  measurement. As recorded: the logistic classifier separated the two sets perfectly. The CNN, near 50%, is not evidence of realism: with 53 windows per class it has
  too little data to learn anything. See the caveats.

## Caveats

- **Small.** 3 writers and 20 lines each, or 12 lines in the style-matched run. The standard
  deviations are over folds, not over independent samples, and are optimistic.
- **A near-50% score is not a pass.** The style-matched CNN result is a training-size effect,
  not a measure of how human the writing looks.
- **Capture mismatch.** In the style-matched run the real crops come from a phone photo of a
  notebook page and the generated lines from a renderer. A classifier may be using the capture
  difference (paper, lighting, noise) and not the writing. The photo-matched run below is meant
  to remove this.
- **Default hand versus real writers.** `proxy_v3` compares a generic hand with specific
  writers, so a high score there is expected and says little about a profile that is trained
  on the writer.
- **Proxy classifiers, not people.** HOG+LBP and a small CNN are not forensic examiners. A
  person, or a stronger detector, may do better or worse. No tool can promise that a reader
  cannot tell, and handwriting generated to be handed in as one's own work can break the rules
  of a course or workplace whether or not anyone can tell.
- **Reviewer sheets.** [`scripts/blind_test.py`](../scripts/blind_test.py) can also write a shuffled
  24-tile sheet for human or vision-model reviewers. No reviewer results are published yet.
- **Private data.** The writer's photos and crops are not in the repository. Only scores and
  generated images are published.

## Photo-matched run

This run puts the generated pages through the same phone-photo capture as the writer's real
photos (`--photo notebook`, calibrated on a real photo) and extracts line crops from both sides
with the same pipeline (`ductus style`: page mask, orientation, deskew, lines found by chaining
marks), so scale, ink extraction and capture are the same and the writing is what differs.
Script: [`scripts/blind_iter.py`](../scripts/blind_iter.py) (pages, crops, both classifiers);
cues: [`scripts/blind_cues.py`](../scripts/blind_cues.py).

- **Real side.** 69 line crops from 8 phone photos of one writer's pages (7 notebook pages and
  one A4 sheet of known sentences). The photos and crops are private and not published.
- **Generated side.** 3 pages of WikiText prose (130 words each, 6 page images) written in that
  writer's profile (`--look profile`, letter shapes copied by the diffusion writer from his
  reference lines, his own traced digits, symbols and capitals), then `--photo notebook`.
  96 to 105 crops depending on the run. Written through the Python API, which does not use the
  word bank (the CLI does unless `--no-bank`), so no generated word is reused.
- **Windows.** 98 real and 98 generated 64 x 256 windows (the real side is the limit),
  5-fold cross-validation grouped by source line.
- **Detectors.** HOG+LBP logistic regression, the small CNN, and a stronger detector: an
  ImageNet ResNet-18 fine-tuned on the windows (`blind_test.py --strong`).

| Run | What changed | HOG+LBP logistic | Small CNN | ResNet-18 fine-tuned | Binarized, logistic / CNN |
|---|---|---|---|---|---|
| `photo_matched_base` | current defaults | 91.3% (sd 4.5) | 69.9% (sd 17.7) | 98.5% (sd 3.1) | 89.3% / 44.9% |
| `photo_matched_pressure` | thinner pen (0.75), pressure field (width and darkness), downstroke pressure, end taper | 90.3% (sd 4.1) | 71.9% (sd 7.2) | 96.4% (sd 2.6) | 91.3% / 55.0% |

Other candidates on the same pages (logistic / small CNN, not binarized): pen 0.65 with the
same pressure 89.3% / 74.0%; stronger pressure 86.2% / 82.1%; more skips and blobs 88.8% /
77.0%. None lowered all detectors, so **none was made a default**; the switches stay in
[`realism/knobs.py`](../src/ductus/realism/knobs.py) for the next round.

**Fooled rate** (100% minus accuracy, with the even-split caveat at the top of this page) for
the current defaults: 8.7% against the logistic regression, 30.1% against the small CNN and
**1.5% against the fine-tuned ResNet-18**. The stronger detector finds the generated lines
almost every time. **The blind test is not passed.**

What the cues say (Cohen's d, real against generated, current defaults on one page): the
largest gaps were stroke-width spread (real strokes vary much more, d -4.9), ink darkness spread
(d -2.1), share of faint ink (d -1.7) and stroke width (generated strokes thicker, d 1.4).
Pressure and a thinner pen closed most of those gaps (stroke-width spread d -1.2, ink spread
d -0.4), but the detectors barely moved, so they are using something the simple cues do not
measure, most likely letter shapes and joins.

Caveats for this run:

- **Small.** 3 generated pages and 8 real photos of one writer. Fold standard deviations are
  3 to 18 points; differences of a few points between rows are noise.
- **Earlier style-matched rows are void.** The `style_matched_r4` rows above (100% logistic,
  ~50% CNN) and an earlier photo run used line crops from a page reader that, on these photos,
  cut slices across lines and picked up the spiral binding. That reader was fixed (page mask,
  blob removal, sideways and upside-down detection, line chaining) before this run. Those rows
  are kept for the record but measure nothing.
- **Same pipeline, different paper.** The real photos are of a spiral notebook and one A4
  sheet; the generated pages use the calibrated notebook capture. Capture differences may still
  help the detectors.

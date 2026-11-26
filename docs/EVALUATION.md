# Evaluation

Everything here can be re-run with the scripts in [`scripts/`](../scripts). Machine: 8 CPU
cores, no GPU.

## 1. Legibility (OCR character error rate)

**Data.** 40 sentences ([`scripts/data/eval_sentences.txt`](../scripts/data/eval_sentences.txt)):
20 of ordinary lowercase prose and 20 with names, capitals, digits and symbols.

**Systems.** See [COMPARISON.md](COMPARISON.md#systems). Every system writes each sentence
once; nothing is cherry-picked.

**Judge.** `microsoft/trocr-large-handwritten`. ductus checks its own words with the smaller
`trocr-base-handwritten`, so the judge is a different model. *Normalised* CER ignores case,
punctuation and spacing (the judge often inserts a space before a full stop even on the font).

```
python scripts/baseline_rnn_lines.py scripts/data/eval_sentences.txt eval/rnn_b15 1.5 4   # TensorFlow env
python scripts/gen_eval_lines.py scripts/data/eval_sentences.txt eval font,diffbrush_raw,ductus_natural,ductus_noverify,ductus_neat_upright
python scripts/score_eval.py scripts/data/eval_sentences.txt eval rnn_b15,font,diffbrush_raw,ductus_natural,ductus_noverify,ductus_neat_upright
python scripts/repetition_probe.py scripts/data/eval_sentences.txt eval font,rnn_b15,diffbrush_raw,ductus_natural
```

| System | CER | CER, normalised | Lines with CER at most 10% | Identical copies of a repeated word | Seconds per line (CPU) |
|---|---|---|---|---|---|
| Handwriting font (Caveat) | 3.2% | 0.4% | 40/40 | 42.5% (median similarity 0.92) | 0.002 |
| 2016 RNN, bias 1.5 | 6.3% | 3.5% | 37/40 | 0% | 3.1 |
| 2016 RNN, bias 2.5 | 6.8% | 4.0% | 33/40 | 0% | 4.5 |
| DiffBrush raw (2025) | 11.8% | 8.5% | 30/40 | 0% | 4.7 |
| ductus, no read-back | 19.1% | 15.7% | 18/40 | 0% | 4.6 |
| ductus natural (`quick` style) | 10.5% | 7.2% | 29/40 | 0% | 10.2 |
| **ductus neat (`upright` style)** | **6.7%** | **3.2%** | **37/40** | **0%** | 10.1 |


Where each system fails (normalised CER per group of 20 lines, from the same runs):

| System | Lines with capitals or digits | Plain lowercase lines |
|---|---|---|
| 2016 RNN, bias 1.5 | higher: "BTC" read as "BIT", "9:30" as "9 so", "450 rupees" lost | 0 to 2% on most lines |
| ductus neat | "BTC", "9:30", "12.5", "ID" written correctly | slips on some long words ("Wednesday", "library") |

Notes:

- The ductus rows were produced before the latest layout and ink changes (re-inked strokes,
  letter-width normalisation, word-level check for the natural look). Those changes target
  stroke consistency and wrong words; the table will be refreshed with the next run.
- The neat look's strict rule (a word must pass both the line read and its own read, 5
  rounds) was measured at 3.2% normalised CER; the earlier averaging rule gave 3.0%. The
  difference is within noise for 40 lines.

## 2. Repetition

`repetition_probe.py` compares every pair of occurrences of the same word. A font repeats
glyphs exactly (42.5% of pairs above 0.95 similarity, median 0.92); every generative system,
ductus included, has 0% identical pairs. Pairs whose widths differ by more than 15% are
skipped as likely split errors.

## 3. Speed

| System | Seconds per line |
|---|---|
| Font | 0.002 |
| 2016 RNN | 3.1 to 4.5 |
| DiffBrush raw | 4.7 |
| ductus with read-back | 10.1 to 12.6 |
| ductus without read-back | 4.6 |

A notes page of 150 to 250 words takes about 2 to 5 minutes on CPU.

## 4. Blind test

**Question.** Can a classifier or a reviewer tell real handwriting from ductus output when
both get the same treatment?

**Real data.** The user's own pages are private and are not used on the development machine.
As a stand-in, IAM test-set writers that ship with the DiffBrush repository (writers the
model was not trained on) were used: for each writer one line is the style reference, 20
other lines are "real", and ductus writes 20 new lines (WikiText sentences) in that style
([`scripts/proxy_blind_iam.py`](../scripts/proxy_blind_iam.py)). IAM images are not
redistributed; only scores are reported.

**Treatment.** Background removed, ink map written as dark on white, height 64 px, cut into
64 x 256 windows. Optional `--binarize` thresholds both sets the same way so only letter
shapes differ. Classes are balanced (chance = 50%).

**(a) Classifiers.** HOG + LBP features with logistic regression, and a small CNN; 5-fold
cross-validation grouped by source line.

| Run | Windows (real / generated) | HOG+LBP logistic | Small CNN |
|---|---|---|---|
| 3 IAM writers, flat-ink renderer | 235 / 235 | 93.6% (sd 3.1) | 90.6% (sd 6.0) |

This is far from chance. **ductus output is not yet indistinguishable from real writing in
this test.** Looking at the windows, the main cues are (1) stroke weight and ink texture
(the flat renderer's even strokes; the new ink model and re-inked line width target this),
(2) style mismatch: one reference line does not carry every habit of a writer, and (3) word
spacing. The test will be re-run with the new ink model and with the user's pages once his
profile is built.

**(b) Reviewers.** `blind_test.py` writes a shuffled 24-tile sheet and a separate answer key.
Reviewers (people or fresh vision-model reviewers who have not seen the key) mark each tile
real or generated; `--score` grades the answers. Sheet results will be added here when run.

```
python scripts/blind_test.py --real real_lines/ --fake fake_lines/ --out blind/
python scripts/blind_test.py --binarize --real real_lines/ --fake fake_lines/ --out blind_bin/
python scripts/blind_test.py --out blind/ --score answers.json
```

## 5. With the user's pages

Once `ductus style` has been run on the 8 photos (on the user's machine):

1. Real line crops: `styles/<name>/refs/` and the line crops in `debug/`.
2. Generated: `python scripts/proxy_blind_iam.py` adapted to the profile, or
   `ductus write ... --style styles/<name> --look profile` and crop lines.
3. Realism metric: OCR CER of his real lines (needs a transcript) against OCR CER of
   generated lines in his style. In his style the check is loose (threshold 0.5) and only
   catches wrong or missing words, so the generated CER should sit close to his real CER,
   not below it.

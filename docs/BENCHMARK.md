# Benchmark round 1: ductus against other tools

Every number on this page comes from the scripts in [`benchmark/scripts/`](../benchmark/scripts) run on 26 Nov 2026 on one
shared 8-core CPU machine with no GPU. Where another tool does better, this page says so.
This is a small test: 7 lines, one sample per system, no seed picking. Read the numbers as
indications, not rankings with error bars.

## Readability comes first

The goal is pages that read easily: plain letters, no looped or decorative `f` and `r`.
So the main measure here is **OCR character error rate (CER)**. A machine reader that has
never seen any of these systems reads each line, and we count how many characters it gets
wrong. Lower is easier to read. It stands in for a human reader and is imperfect: it misses
things like letter shapes that read fine but look odd.

## Test texts

Every system got the same 7 lines ([`benchmark/texts/tests.tsv`](../benchmark/texts/tests.tsv)):

| id | text |
|---|---|
| short | `Meet me at the library at 5 pm.` |
| definition | `Variance -> mean of squared deviations` |
| maths | `mean = 30/5 = 6 and x^2 = 36` (ductus input: `mean = {30/5} = 6 and x^2 = 36`, its stacked-fraction markup) |
| pangram | `The quick brown fox jumps over the lazy dog.` |
| caps | `BTC NASA HTML GPU ICCV` |
| digits | `0123456789 and 3.14 and 2026` |
| symbols | `% = + ( ) @ & ? ! # : ;` |

The first four are the main test. The last three only probe the character set.

## Systems

| System | What it is | How it was run |
|---|---|---|
| **ductus natural** | this project, default look | `ductus write texts/ductus_page.txt --look natural --seed 0 --no-pdf` (all 7 lines on one A4 page, then cut into lines by `scripts/split_page.py`) |
| Grzego RNN (2018, MIT) | Graves 2013 LSTM + mixture density network, the same model as greydanus/scribe | pretrained `model-29`, style 4, bias 1.5 and 0.75 (`scripts/baseline_rnn_lines.py`) |
| sjvasquez/handwriting-synthesis (2018) | Graves 2013 model, the most-starred implementation | upstream checkpoint 17900, style 9, bias 0.75. Ported to TensorFlow 2 `compat.v1` for this test: API renames only, with sampling rewritten using the same distributions (`tools/hws-tf2/`) |
| text-to-handwriting (saurabhdaware, MIT, ~5k stars) | the popular free "assignment" generator; types a handwriting font onto paper | the real web app, cloned and served locally, driven in headless Chrome (`scripts/run_text_to_handwriting.py`). Fonts: Homemade Apple (site default), Caveat, Liu Jian Mao Cao. Effect "no effect", paper lines and margin off, resolution "Very High" |

## Method

- **OCR CER**: judge `microsoft/trocr-large-handwritten`, the same for every system
  (`scripts/ocr_score.py`). ductus checks its own output with `trocr-base`, a different
  model. *Raw CER* counts case and punctuation. *Normalised CER* lowercases and keeps only
  `a-z 0-9` and spaces. The reference is always the full intended text, so a character a tool
  silently drops counts as an error. For the symbols line, normalised CER is undefined (the
  normalised reference is empty), so only raw CER is reported.
- **Glyph repetition** (`scripts/glyph_repetition.py`): every connected ink shape in all 7
  lines is compared with every other shape of the same size (±5%). Two shapes count as
  twins when their normalised cross-correlation is ≥ 0.95. The rate is the share of shapes
  that have a twin. A font reuses the same glyph image, so it shows up here.
- **Character coverage** (`scripts/charset_coverage.py`): read from each tool's own
  vocabulary or the font's character map, not guessed. A character only counts if the tool
  writes it as that character. Dropping it or swapping in a substitute (ductus writes `@`
  as "at") does not count.
- **Speed**: seconds per line as logged by each tool, on a **heavily shared machine** (load
  average 20 to 32 on 8 cores during the ductus run, from other jobs). The figures show the
  order of magnitude only.

## Results

![Short sentence](assets/benchmark/compare-short.png)
![Definition with arrow](assets/benchmark/compare-definition.png)
![Maths line](assets/benchmark/compare-maths.png)
![Pangram](assets/benchmark/compare-pangram.png)
![Capitals, digits, symbols](assets/benchmark/compare-charset.png)

### Readability (OCR CER, lower is better)

| System | norm CER, 4 main lines | raw CER, 4 main lines | short | definition | maths | pangram | caps (norm) | digits (norm) | symbols (raw) |
|---|---|---|---|---|---|---|---|---|---|
| ductus natural | 0.148 | 0.193 | 0.133 | 0.057 | 0.400 | **0.000** | **0.000** | 0.333 | 0.652 |
| Grzego RNN, bias 1.5 | 0.124 | 0.201 | **0.000** | 0.171 | 0.300 | 0.023 | 0.591 | 0.111 | 0.826 |
| Grzego RNN, bias 0.75 | 0.166 | 0.199 | 0.067 | 0.086 | 0.350 | 0.163 | 0.591 | 0.296 | 0.696 |
| sjvasquez, bias 0.75 | 0.084 | 0.191 | **0.000** | 0.114 | 0.200 | 0.023 | 1.500 | 0.482 | 0.783 |
| text-to-handwriting, Homemade Apple | 0.030 | 0.119 | **0.000** | **0.000** | 0.050 | 0.070 | 0.318 | 0.037 | **0.348** |
| text-to-handwriting, Caveat | **0.000** | **0.088** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | 0.037 | 0.478 |
| text-to-handwriting, Liu Jian Mao Cao | 0.031 | 0.109 | **0.000** | **0.000** | 0.100 | 0.023 | 0.091 | 0.074 | 0.435 |

What the judge read for ductus (from the OCR output of the run):

| line | ductus wrote, as read by OCR |
|---|---|
| short | `Most me at the library at 5 in` (on the page: "Meet" with a malformed e, and "pm" written as "m") |
| definition | `Naiance # mean of squared deviations .` (the drawn arrow is read as `#`) |
| maths | `mean # # # and and ? # 36` ("x^2" came out as "and and ²"; the stacked fraction is not read) |
| pangram | `The quick brown fox jumps over the lazy dog` |
| caps | `BTC NASA HTML GPU ICCV.` |
| digits | `07780 and 3/4 and 2016 .` |

ductus logged `still_failing: 5` of 50 words on this page: words that still failed its own
check after 4 rewrite rounds and were written anyway.

### Repetition, character coverage, speed

| System | glyph twin rate | ink shapes | capitals | symbols `%=+@&#/*<>^$` | maths Unicode (√ Σ ² ½ → × ÷ ≠ ≤ π) | stacked fraction / root / sigma | s per line |
|---|---|---|---|---|---|---|---|
| ductus natural | **0.000** | 99 | 26/26 | 9/12 (drawn `= < > %`; `@`→"at", `$`→"USD", no `^`) | 0/10 as characters | **yes** (markup) | 390 (load 20 to 32) |
| Grzego RNN | **0.000** | 111 / 102 | 26/26 | 5/12 | 0/10 | no | 5.2 / 4.9 |
| sjvasquez | **0.000** | 98 | 23/26 (no Q X Z) | 1/12 | 0/10 | no | 2.7 |
| text-to-handwriting, Homemade Apple | 0.266 | 79 | 26/26 | 12/12 | 4/10 | no | 0.8 (browser render) |
| text-to-handwriting, Caveat | 0.462 | 158 | 26/26 | 12/12 | 6/10 | no | 0.7 |
| text-to-handwriting, Liu Jian Mao Cao | 0.357 | 129 | 26/26 | 12/12 | 2/10 | no | 0.8 |

### Features (from each tool's docs and code, not measured)

| Feature | ductus | Grzego / sjvasquez RNN | text-to-handwriting |
|---|---|---|---|
| A4 page layout | yes | no (single lines) | yes |
| Notes marks (arrows, ticks, underline, strike-through, circles) | yes, drawn | no | no (only what the font has as characters) |
| Maths layout (stacked fractions, roots, sigma, aligned `=`) | yes | no | no |
| Style imitation from a photo | yes, one line image (`ductus style`) | no; only built-in pen-trace styles | no; user can upload a .ttf font |
| Pen choices | yes (`--pen`, `--ink`) | stroke width/colour in SVG | 3 ink colours |
| Phone-photo look | yes (`--photo`) | no | "shadows" and "scanner" effects |
| Vector output | SVG | SVG | no (JPEG/PDF) |
| Runs offline on CPU | yes, slowly | yes | yes (static site) |
| Licence | MIT code; DiffBrush weights trained on IAM (non-commercial research data) | Grzego MIT; sjvasquez has no licence file; weights trained on IAM-OnDB | MIT; fonts Apache-2.0 (Homemade Apple) and OFL |

## Reading the results honestly

Where competitors win:

- **Readability: the font-based site wins clearly.** Caveat scored 0.000 normalised CER on all
  four main lines, and every font beat ductus on the main lines (0.000 to 0.031 against 0.148).
  Caveat is also the plainest-looking hand, with no decorative loops. If readability is the
  goal, a plain font is very hard to beat on OCR.
- **The 2018 RNNs beat ductus on the four main lines in this sample**: sjvasquez 0.084 and Grzego
  bias 1.5 0.124 against ductus 0.148. Both wrote the short sentence perfectly, where ductus
  failed on "Meet" and "pm".
- **Maths line**: ductus was the worst at 0.400. The stacked fraction and drawn `=` marks are
  not read by the OCR judge (this partly penalises the layout itself), and `x^2` was miswritten.
- **Digits**: ductus scored 0.333 against 0.037 for the fonts and 0.111 for Grzego bias 1.5.
- **Speed**: ductus took about 390 s per line under heavy load. A single-line CLI call earlier
  took 1,399 s for 8 words. The RNNs take 3 to 5 s per line and the fonts under 1 s.
  ductus's rewrite loop (up to 4 or 5 OCR-checked rounds) is the main cost.
- **Symbols**: fonts cover every symbol tested. ductus covers 9/12, the RNNs 5/12 and 1/12.

Where ductus does better in this round:

- **Capitals**: 0.000 CER on `BTC NASA HTML GPU ICCV`. The RNNs scored 0.59 and 1.50 (one RNN
  looped and wrote "N.N.N..." without stopping).
- **No repeated glyphs**: 0.000 twin rate, the same as the RNNs, against 27 to 46% of shapes
  having an identical twin for the fonts. This is the measurable "a font is a font" signal.
- **Page features**: maths and notes layout, style from a photo, and phone-photo mode. No
  other tested tool has these.
- **Pangram**: 0.000 normalised CER.

## Not tested, and why

| System | Reason |
|---|---|
| ductus `neat` look | Started on the 4 main lines, still in its 3rd rewrite round after about 22 minutes on the loaded machine, and stopped when the time budget ran out. An earlier single-line neat run (`out/ductus_neat_single_short/`) finished in 1,399 s with 2 of 8 words still failing ("Meet me at the liirary at 5 m"). It was not scored, so it is not in the tables. |
| One-DM (ECCV 2024, MIT) | Code and weights were downloaded and run on CPU (`tools/One-DM/bench_lines.py`, word level, 50 DDIM steps, style from ductus' `quick` preset; its Laplacian style input had to be approximated). It took 2 to 21 s per denoising step on this machine, about 2 to 8 minutes per word. Only the first line finished (`out/onedm/000.png`, image only, not scored). |
| greydanus/scribe (2016) | Needs TensorFlow 0.x / Python 2 and a separately hosted checkpoint. It implements the same Graves model as the two RNNs tested. |
| DiffusionPen (ECCV 2024), SDT (CVPR 2023), Emuru (CVPR 2025), DiffBrush raw | Not attempted in this round for lack of time. SDT is character-level and Chinese-first. Emuru fails to load with current `transformers` (see RESEARCH.md). Raw DiffBrush is already covered by the existing eval in docs/COMPARISON.md. |
| calligrapher.ai and other hosted web tools | No public API or static output. Driving the live site was out of scope for this round. |
| Tesseract as a second OCR judge | Not installed on the box. Only one judge was used. |

## Round 2: after the readability changes

Same 7 lines, same judge (`trocr-large-handwritten`), same scripts
(`benchmark/scripts/run_ductus_page.sh natural`, `split_page.py`, `ocr_score.py`). Changes since
round 1: the Graves RNN (Grzego weights) writes the letters by default and diffusion is only the
fallback for words the RNN cannot write or that still fail the read-back; every word must pass
the line and the word read-back; a letter-shape check re-writes looped letters.

| System | norm CER, 4 main lines | raw CER, 4 main lines | short | definition | maths | pangram | caps (norm) | digits (norm) | symbols (raw) | s per line |
|---|---|---|---|---|---|---|---|---|---|---|
| ductus natural, round 1 | 0.148 | 0.193 | 0.133 | 0.057 | 0.400 | 0.000 | 0.000 | 0.333 | 0.652 | 390 (load 20 to 32) |
| **ductus natural, round 2** | **0.113** | **0.153** | 0.000 | 0.000 | 0.450 | 0.000 | 0.227 | 0.185 | 0.696 | 23 (load 4 to 12) |
| sjvasquez, bias 0.75 | 0.084 | 0.191 | 0.000 | 0.114 | 0.200 | 0.023 | 1.500 | 0.482 | 0.783 | 2.7 |
| Grzego RNN, bias 1.5 | 0.124 | 0.201 | 0.000 | 0.171 | 0.300 | 0.023 | 0.591 | 0.111 | 0.826 | 5.2 |
| text-to-handwriting, Caveat | 0.000 | 0.088 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.037 | 0.478 | 0.7 |

What changed, honestly:

- Three of the four main lines now read perfectly (short, definition, pangram). All of the
  remaining main-line error is the maths line: the judge cannot read the drawn stacked fraction
  and `=` marks, and `x^2` came out as "X ?". On plain prose ductus is now level with the fonts.
- ductus is still behind the sjvasquez RNN on the main-line average (0.113 against 0.084) and
  far behind the fonts.
- Capitals got worse (0.000 to 0.227: "BTC" read as "Bf", "HTML" as "HTHL"); digits got better
  (0.333 to 0.185). The RNN writes capitals less clearly than diffusion did.
- Speed: 158 s for the page (23 s per line), against about 390 s per line in round 1. The
  machine was less loaded this time, so part of the gain is not ours; the RNN itself takes a few
  seconds per line, and most of the remaining time is model loading and the OCR read-back.
- 4 of 50 words still failed the read-back after 6 rounds and were written anyway.

## Round 3: cursive r and f, later solo-line rewrites

Same 7 lines, same judge, same scripts. Changes since round 2: the default hand uses traced
cursive `r` and `f` and template capitals (M, R, H, E, N); words that still fail in later
RNN rounds are rewritten on their own line; read-back stays at max CER 0.10. Eight RNN
rounds run before the diffusion fallback.

| System | norm CER, 4 main lines | raw CER, 4 main lines | short | definition | maths | pangram | caps (norm) | digits (norm) | symbols (raw) | s per line |
|---|---|---|---|---|---|---|---|---|---|---|
| ductus natural, round 1 | 0.148 | 0.193 | 0.133 | 0.057 | 0.400 | 0.000 | 0.000 | 0.333 | 0.652 | 390 (load 20 to 32) |
| ductus natural, round 2 | 0.113 | 0.153 | 0.000 | 0.000 | 0.450 | 0.000 | 0.227 | 0.185 | 0.696 | 23 (load 4 to 12) |
| **ductus natural, round 3** | **0.093** | **0.150** | 0.000 | 0.000 | 0.350 | 0.023 | 0.136 | 0.111 | 0.609 | 38 (load 3 to 5) |
| sjvasquez, bias 0.75 | 0.084 | 0.191 | 0.000 | 0.114 | 0.200 | 0.023 | 1.500 | 0.482 | 0.783 | 2.7 |
| Grzego RNN, bias 1.5 | 0.124 | 0.201 | 0.000 | 0.171 | 0.300 | 0.023 | 0.591 | 0.111 | 0.826 | 5.2 |
| text-to-handwriting, Caveat | 0.000 | 0.088 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.037 | 0.478 | 0.7 |

What the judge read for ductus in this round:

| line | ductus wrote, as read by OCR |
|---|---|
| short | `meet me at the library at 5 p.m.` (normalised CER 0.000) |
| definition | `Variance - mean of squared deviations` (the drawn arrow is read as `-`) |
| maths | `mean # # and X # 36` (stacked fraction and drawn `=` marks are not read; `x^2` came out messy) |
| pangram | `The quick brown for jumps over the lazy dog .` ("fox" read as "for") |
| caps | `BC NASA HTM. GPI ICCV.` ("BTC" as "BC", "HTML" as "HTM", "GPU" as "GPI") |
| digits | `01234589 and.314 and 2026` (the 6 and 7 were unclear; "3.14" lost the leading 3) |
| symbols | `# # C ) ARC. " ) . "` |

What changed, honestly:

- Main-line normalised CER improved from 0.113 to 0.093. Short and definition still read
  perfectly. The pangram is no longer perfect (0.000 to 0.023): the judge read "fox" as
  "for", most likely the cursive `f`.
- Maths improved a little (0.450 to 0.350) but is still most of the main-line error, and
  still the weakest of the four. The judge cannot read the stacked fraction and drawn `=`
  marks, and `x^2` was miswritten.
- ductus is closer to, but still behind, the sjvasquez RNN (0.093 against 0.084) and far
  behind the fonts.
- Capitals improved on round 2 but remain worse than round 1 (0.000 to 0.227 to 0.136).
  Digits improved again (0.333 to 0.185 to 0.111). Symbols raw CER is 0.609, a little
  better than round 2 (0.696) and still well behind the fonts.
- Speed: 268 s for the 7-line page (38 s per line), slower than round 2 (158 s, 23 s per
  line). Load was similar or slightly lower this time, so the extra time is mostly the
  added rewrite rounds, not the machine. 5 of 50 words still failed the read-back after
  10 rounds and were written anyway. Glyph twin rate is still 0.000.
- A separate cold-cache run (`--no-cache`, look natural, mode plain, seed 0) of 243 words
  took 610 s of engine time (615 s wall on this machine). The default letter size spread
  that text over 3 A4 pages, so one full page at this size is closer to 80 to 110 words
  than to 250. 8 of 243 words still failed the read-back.

## Raw data

Per-system numbers: [`benchmark/metrics.csv`](../benchmark/metrics.csv). Systems and settings:
[`benchmark/systems.json`](../benchmark/systems.json). Generated images and third-party tool
checkouts are not in the repository.

## Caveats

- n = 7 lines, one sample each. A single bad word moves a line's CER by 10 to 15 points.
- The OCR judge rewards printed-looking text and cannot read drawn marks (arrows, stacked
  fractions), which works against ductus's notes and maths layout.
- Speed was measured under heavy load from other jobs and should be re-measured on an idle
  machine.
- The sjvasquez result uses our TF2 port, not the original TF 1.6 environment.

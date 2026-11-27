# Changelog

All notable changes are listed here. Versions follow semantic versioning.

## 1.0.0 - 2026-11-27

First stable release. Benchmark: normalised CER 0.038 (raw 0.108) on the 4 main test lines,
61 s per benchmark page on a CPU. The blind test is not passed yet (see [BLIND_TEST.md](docs/BLIND_TEST.md)).

- Default writer is the Graves RNN (plain, print-like letters), with diffusion as the fallback
  for words it cannot write or that still fail the read-back (`--engine`).
- Strict read-back by default (`--legibility strict`): every word must pass the line and the
  word check; a letter-shape check rewrites looped `f`, `r` and ascenders (`--allow-loops` to
  keep them).
- Docs: new [PENS.md](docs/PENS.md), rewritten [INK.md](docs/INK.md), [HABITS.md](docs/HABITS.md),
  [FAQ.md](docs/FAQ.md), [ARCHITECTURE.md](docs/ARCHITECTURE.md) with a new diagram, an option
  table in the README and CLI reference that matches `cli.py`, and `scripts/check_links.py`.
- Calibrated ink model (`--ink-model realism`, default): striations, skips, flow variation,
  blobs and overlap darkening, composited in linear light at 600 dpi.
- Phone photo surfaces: `--photo a4` (loose white sheet on a desk, default) and
  `--photo notebook` (spiral grid pad), sharing one camera pipeline calibrated on a real photo.
- Human-error pass on the laid-out page (`--humanize`): size and baseline drift, retouched letters.
- Word cache: re-rendering the same text with other pens, paper or photo settings reuses the
  generated words (`--no-cache` to disable).
- Speed: `--fast` draft mode, the OCR image encoder in ONNX Runtime when installed, skipped
  word reads and best-of-N candidates for stubborn words.
- Word bank: words that passed the read-back are reused across runs, shared by `--fast` and
  full runs (on in the CLI, off in the API; `--no-bank` to disable).
- Hand-ruled page borders and margin lines (`--border`, `--border-inset`, `--border-header`).
- `ductus charset`: trace your own capitals, digits and symbols from a photo of a character
  sheet; they replace the template shapes.
- Read-back rejects `l`/`t` swaps (word cache version 32).
- Style profiles from photos: paper mask, blob removal, rotation by OCR confidence, spectral
  grid period, line chaining with dots attached, and steadier pitch, slope and margin measures.
- Realism switches for pressure, skips, blobs, pitch and stroke width (off by default).
- Blind test: photo-matched run in the writer's own profile with logistic, CNN and fine-tuned
  ResNet-18 detectors (98.5% on 3 real pages), `scripts/blind_iter.py` and
  `blind_test.py --strong`.
- Docs: [ROADMAP_V2.md](docs/ROADMAP_V2.md).

## 0.1.0 - 2026-11-26

First public version.

- One-shot line diffusion engine (DiffBrush, ICCV 2025) wrapped for CPU inference, 25 DDIM steps.
- Carrier-line synthesis: every word is written inside a normal-length line, then cut out,
  so short inputs, capitals and digits stay in the model's training regime.
- OCR read-back with TrOCR and word alignment; strict for neat output, loose for personal styles.
- Notebook layouts: headings with rules, `Term :: meaning` definitions with arrows, notes,
  numbered and bulleted lists, ticks and crosses, strike-throughs and corrections,
  fractions, square roots, sigma, superscripts, tables with ruled columns, circled answers.
- Human layout model: baseline drift, wandering margins, gap and size jitter, pressure field.
- Paper: white A4 (default), ruled, 5 mm grid; optional phone-photo look.
- Output: 300 dpi PNG, A4 PDF, vector SVG and animated SVG.
- `ductus style`: personal style profile from phone photos (rotation, deskew, grid removal,
  line segmentation, layout statistics).
- Five built-in styles rendered from the 2016-era RNN as one-shot references.

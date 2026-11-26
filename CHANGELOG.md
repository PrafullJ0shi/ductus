# Changelog

All notable changes are listed here. Versions follow semantic versioning.

## Unreleased

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

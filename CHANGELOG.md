# Changelog

All notable changes are listed here. Versions follow semantic versioning.

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

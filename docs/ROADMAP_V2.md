# Roadmap for version 2

Version 1.0 reads well (normalised CER 0.038 on the benchmark lines) but does not pass a blind
test: a fine-tuned ResNet-18 tells real lines from generated ones 98.5% of the time
([BLIND_TEST.md](BLIND_TEST.md#photo-matched-run)). Version 2 is mostly about closing that gap
without losing readability, and about speed.

## 1. Blind-test realism

Each item is tested on its own against the photo-matched blind test and kept only if it lowers
detector accuracy without raising CER. Switches already in the code are in
`src/ductus/realism/knobs.py` and are off by default.

| # | Item | State at 1.0 |
|---|---|---|
| 1 | Pen pressure along strokes, words and lines | switch exists (`press`, `press_w`, `press_l`, `taper`); closes the stroke-width cue but did not lower every detector |
| 2 | Ballpoint skips and ink blobs | switch exists (`skips_x`, `blob_rate_x`, `blob_x`); not adopted |
| 3 | Joined and separate letters as the writer joins them | not started |
| 4 | Per-letter variation (no two `e` alike in the same way the writer varies) | not started |
| 5 | Uneven spacing and line drift | switch exists (`pitch_x`); profile pitch, slope and margin spread measured |
| 6 | Fatigue down the page | measured in the profile (capped at 0.12); effect not tested alone |
| 7 | Paper grain and show-through | not started |
| 8 | More of the writer's own letter shapes | capitals, digits and symbols traced by `ductus charset`; lowercase inside words still from the RNN |
| 9 | Stronger detector | fine-tuned ResNet-18 added (`blind_test.py --strong`) |

Also needed:

- **More real pages.** The test has 3 real pages (98 windows). Aim for 10 or more, written on
  different days and with different pens, so detector numbers are not about one sheet.
- **A stronger detector still**, trained on several writers, as the bar to beat.

## 2. GPU timings and half precision

CUDA and MPS are used when present but have not been timed. To do: time a page on a common GPU,
add half precision for the OCR and diffusion models, batch the DDIM sampler, and publish the
numbers next to the CPU ones (84 s default and 36 s `--fast` for a new 84-word page; 22 s
when the page repeats words already in the word bank).

## 3. Web service with a fast and a quality mode

A small service on a VPS: `POST /write` with text and options, returning PNG, PDF or SVG.

- **Fast mode** maps to `--fast` with the word bank on: draft quality in seconds for common words.
- **Quality mode** is the full read-back with the diffusion fallback, queued, with a
  progress callback.
- Personal profiles stay with the user; nothing is kept after the job unless asked.

## 4. Loose ends from 1.0

- A stray mark before `fox` on a benchmark page written with the word bank on.
- `l` and `t` confusions: swaps are now rejected at read-back, but look-alikes the OCR still
  reads correctly can pass.
- A stray dot left of `0123456789` on the round-6 benchmark page.
- The symbol `x` from a character sheet overwrites the lowercase `x` key in `glyphs.json`.

## 5. Carried over from the earlier plan

1. Fine-tune on your own pages (LoRA on the style encoder and UNet from `ductus style` crops).
2. Full character set: `@`, `<`, `>`, accented Latin, currency signs written by the model.
3. Stroke output for pen plotters, and an animated SVG in real stroke order.
4. Editor plug-ins (Figma, Canva, Notion, Obsidian) and a LaTeX/Markdown maths importer.
5. Provenance by default: an invisible watermark, a C2PA manifest and a public detector.
6. Hosted tiers, a plotter card service and a consented style marketplace, to pay for compute.
   Signatures are never offered.

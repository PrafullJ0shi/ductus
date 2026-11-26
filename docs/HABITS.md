# Writer habit profile

A style is more than letter shapes. ductus keeps a writer's *habits* next to the style
references and samples from them when it writes a page.

## Two parts

**Measured** by `ductus style` from the photos and stored under `habits` in
`profile.json`:

| Key | Meaning | How it is measured |
|---|---|---|
| `fatigue` | how much the writing grows from the top to the bottom of a page | slope of body-line x-height against position on the page, median over pages |
| `tighten` | how much the line pitch shrinks in the lower half | 1 - (median pitch, lower half) / (median pitch, upper half) |
| `margin_sd_xh` | how much the left margin wanders | spread of line starts, in x-heights |
| `heading_share`, `heading_scale` | how often lines are written larger, and how much larger | lines with x-height above 1.3x the median |
| `section_gap_xh` | the space left between sections | line gaps of 4.5 to 12 x-heights |

The `look` section of the same file holds x-height in mm (from the 5 mm grid), line pitch,
baseline slope and its spread, word gap, baseline wobble, stroke width and ink colour.

**Personal**, written by hand in `habits.json` beside `profile.json` (it is easier for the
writer to list these than for a program to guess them):

```json
{
  "spellings": {"software": "softwear"},
  "abbreviations": {"explains": "->", "because": "bcz", "therefore": "=>"},
  "apply_rate": 0.85,
  "corrections": 0.02,
  "hesitation": 0.02,
  "retrace": 0.015,
  "smudges": 0.8
}
```

- `spellings` and `abbreviations` rewrite a word the way the writer writes it. Each
  occurrence is rewritten with probability `apply_rate`, so a page is not perfectly
  consistent, which matches real notes.
- `->` and `=>` become drawn arrows, as in the notes markup.
- Any key from the human-error layer (`fatigue`, `tighten`, `squeeze`, `hesitation`,
  `retrace`, `punct_gap`, `session`, `smudges`, `corrections`) overrides the measured or
  default value. See [REALISM.md](REALISM.md) for what each one does.

Habits are applied with `--look profile` (or any look except `neat`) when `--style` points
at a profile folder.

## Habits that need more data

Letter and digit variants (his `1`, `7`, `4`), how often each form appears, i-dot and t-cross
placement, and arrow, tick and circle shapes copied from his pages need a fine-tune on his
own line crops or stroke separation. They are listed as deferred in
[REALISM.md](REALISM.md) and as roadmap item 1 in the README.

## What to collect

1. All 8 pages through `ductus style` (done on the writer's machine; the photos stay private).
2. A typed transcript of two pages, to fill `spellings` and `abbreviations` and to calibrate
   the loose OCR threshold against his real writing.
3. One sample page with the alphabet, digits and symbols (see [STYLE_GUIDE.md](STYLE_GUIDE.md)).

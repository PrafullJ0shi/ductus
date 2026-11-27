# Writer profile and habits

A style is more than letter shapes. ductus keeps two kinds of information about a writer next
to the reference lines: **measurements** taken from photos of real pages, and **habits** that
the writer fills in by hand. Both live in a profile folder made by `ductus style`.

Everything here applies only when `--style` points at a profile folder. The built-in presets
have no profile.

## What a profile folder holds

```
styles/my-hand/
  profile.json    measurements, written by ductus style
  refs/           clean one-line crops used as style references
  debug/          cleaned page ink maps, for checking
  habits.json     optional, written by you
  letters.json    optional, from scripts/letter_glyphs.py
```

Photos of your own pages and everything derived from them are private. Do not commit a
profile folder.

## Building it

```bash
ductus style page1.jpg page2.jpg page3.jpg -o styles/my-hand --name my-hand
```

For each photo `src/ductus/profile.py` does the following:

1. Lighting is removed and ink is found: dark and, for blue or black pens, more saturated than
   the printed grid.
2. Long thin straight lines (printed grid or ruling) are removed.
3. The page is rotated to upright (0, 90, 180 or 270 degrees; the last step uses an OCR
   confidence check, which `--no-ocr` skips) and deskewed.
4. The page is split into text lines. Clean, wide lines of average size are kept in `refs/`
   (at most 12) as one-shot style references.
5. The grid period gives the scale, because notebook squares are 5 mm.

Then it measures, over all pages, and writes `profile.json`:

| Key under `look` | Meaning |
|---|---|
| `xh_mm` | x-height in mm (2.6 if no grid was found) |
| `pitch` | line pitch in x-heights |
| `slope_deg`, `slope_sd` | mean baseline drift and its spread |
| `gap_xh`, `gap_sd_xh` | word gap in x-heights and its spread |
| `baseline_resid_xh` | wobble of the baseline around a straight fit |
| `stroke_xh` | stroke width in x-heights |
| `ink` | ink colour, from the strongest ink pixels |
| `mess` | a 0 to 1 estimate from wobble, drift spread and gap spread |

| Key under `habits` | Meaning | How it is measured |
|---|---|---|
| `fatigue` | how much the writing grows from the top to the bottom of a page | slope of body-line x-height against position on the page, median over pages, clipped to 0 to 0.3 |
| `tighten` | how much the line pitch shrinks in the lower half | 1 minus (median pitch, lower half) over (median pitch, upper half), clipped to 0 to 0.4 |
| `margin_sd_xh` | how much the left margin wanders | spread of line starts, in x-heights |
| `heading_share`, `heading_scale` | how often lines are written larger, and how much larger | lines with x-height above 1.3 times the median |
| `section_gap_xh` | the space left between sections | line gaps of 4.5 to 12 x-heights |

Pages with fewer than six lines are skipped for the habit measurements.

## What is used when you write

Status of each value in the current code, so you know what changes the output:

| Value | Used? | Where |
|---|---|---|
| `look.xh_mm`, `pitch`, `slope_deg`, `mess`, `ink`, `gap_xh` | yes, for any `--look` | `Writer.write` in `page.py` copies them onto the page look. Values outside a plausible range are dropped (`SANE`), and `mess` is capped at 0.6, so a hurried hand stays readable. |
| `look.stroke_xh` | yes | sets the pen width unless `--pen` or `--tip` is given (see [PENS.md](PENS.md)) |
| `habits.fatigue`, `habits.tighten` | yes, unless `--look neat` | `habits.apply_look`; the layout grows the writing and tightens the lines toward the page end |
| `habits.margin_sd_xh`, `heading_share`, `heading_scale`, `section_gap_xh` | recorded only | measured and stored, but the layout does not read them yet |
| `look.slope_sd`, `gap_sd_xh`, `baseline_resid_xh` | recorded only for now | used inside the `mess` estimate |

`--look neat` caps `mess` at 0.2 and skips habits and text rewrites. `--look profile` is the
`natural` look with the profile values on top; with a profile, `natural` and `messy` also use
the profile values.

## Personal habits: `habits.json`

Some habits are easier to list than to measure. Put them in `habits.json` next to
`profile.json`:

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

- `spellings` and `abbreviations` rewrite a word the way the writer writes it. They are matched
  without regard to case. Each occurrence is rewritten with probability `apply_rate` (default
  0.85), so a page is not perfectly consistent, which matches real notes. A capital first letter
  is kept.
- `->` and `=>` in a replacement become drawn arrows, as in the notes markup.
- These keys, if present, override the measured or preset value of the same name:
  `fatigue`, `tighten`, `squeeze`, `hesitation`, `retrace`, `punct_gap`, `session`, `smudges`,
  `corrections` (`habits.LOOK_KEYS`). What each does is in [REALISM.md](REALISM.md).
- `habits.json` is read after `profile.json`, so it wins when a key appears in both.

The docstring at the top of `src/ductus/habits.py` also mentions `correction`,
`corrections_rate` and `heading`. The code does not read them; use `corrections` as above.

## Letter shapes: `letters.json`

`scripts/letter_glyphs.py` cuts single letters from your reference lines, reads each with OCR
and records how often each letter has a closed loop above the x-height, inside it, or below
the baseline. With `letters.json` in the profile folder, the letter-shape check
(`src/ductus/letterforms.py`) lets a generated letter keep a loop if you use that loop in at
least 20% of at least 5 samples. It never forbids a loop on that basis, and `f`, `l` and `r`
are always checked against the plain form. The output holds crops of your own writing, so keep
it private.

Single letters cut from joined-up writing are often broken, so treat these rates as rough.

## Habits that need more data

Letter and digit variants (your `1`, `7`, `4`), how often each form appears, i-dot and t-cross
placement, and arrow, tick and circle shapes copied from your pages need a fine-tune on your
own line crops or stroke separation. They are marked deferred in [REALISM.md](REALISM.md) and
are roadmap item 1 in the [README](ROADMAP_V2.md).

## What to collect

1. Several pages through `ductus style` (on your own machine; the photos stay private).
2. A typed transcript of two pages, to fill `spellings` and `abbreviations`.
3. One sample page with the alphabet, digits and symbols (see [STYLE_GUIDE.md](STYLE_GUIDE.md)).

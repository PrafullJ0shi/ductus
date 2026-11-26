# Writing in your own hand

## What to photograph

- 3 to 8 pages of your normal writing, in the pen you usually use. More pages give more
  reference lines and steadier habit measurements.
- Grid, ruled or plain paper all work. Grid and ruled lines are removed.
- Daylight or even room light, phone held roughly parallel to the page, whole page in frame.
  Shadows and slight rotation are fine; the tool rotates and deskews.

Best single extra page: one page written specifically as a sample, with

1. the alphabet in lower case and in capitals,
2. digits 0 to 9 and the symbols you use (`= + - x / % ( ) -> < >`, ticks, crosses),
3. two pangrams, for example "The quick brown fox jumps over the lazy dog" and
   "Pack my box with five dozen liquor jugs",
4. a few lines of ordinary prose at your usual speed.

A typed transcript of one or two of your existing pages lets the evaluation measure how
readable your real writing is to OCR, so generated pages can be matched to it.

## Build the profile

```
ductus style page1.jpg page2.jpg page3.jpg ... -o styles/my-hand --name my-hand
```

Output:

| Path | Content |
|---|---|
| `styles/my-hand/profile.json` | measured habits: x-height (mm), line pitch, slope, slope spread, gaps, baseline wobble, stroke width, ink colour |
| `styles/my-hand/refs/` | the cleaned line crops used as style references |
| `styles/my-hand/debug/` | each page after rotation, deskew and grid removal, with detected lines |

Check `debug/`: every page should be upright and the grid gone. Delete reference crops that
are mostly drawings, tables or crossed-out text; those make weaker references.

`styles/` is in `.gitignore`. Your handwriting is personal data; keep the profile private.

## Write with it

```
ductus write notes.txt -o out/notes --style styles/my-hand --look profile --mode notes
```

`--look profile` is the `natural` look with the measurements and habits of your profile on
top (see [HABITS.md](HABITS.md) for which values are used). Words must still read back:
`--legibility strict` (the default) uses the OCR threshold 0.15 and keeps your own
proportions, and messiness stays in spacing, baseline and shape rather than in garbled
letters. Add `--legibility loose` to only regenerate words that came out wrong or missing,
with the profile's threshold (0.5 unless `profile.json` sets `max_cer`). Each block of about
40 words uses a different reference line, so the page shows the variety of your writing
rather than one line repeated.

The letter shapes follow the reference lines only with `--engine diffusion` or `--engine
auto`. With the default engine the letters come from the plain RNN hand, and your profile sets
the size, spacing, slant, ink colour and habits.

Useful overrides: `--paper grid` to match grid notebooks, `--ink "#1b3fa0"` for a blue
ballpoint (or pick a pen with `--pen`, see [PENS.md](PENS.md)), `--size` to set the x-height, `--corrections 0.02` for occasional crossed-out
words.

## Responsible use

Write your own hand, or a hand you have permission to use. Do not use ductus to imitate
someone's signature or to pass generated pages off as another person's work.

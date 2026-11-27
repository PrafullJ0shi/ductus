# Page borders and margin lines

Students rule a box round every page of a project file, often with a margin line down the
left. `--border` draws that by hand-ruled lines, in the same pen as the writing.

```
ductus write notes.txt -o out/notes --mode notes --border box-margin
ductus write notes.txt -o out/notes --border box --border-inset 15 --photo a4
ductus write notes.txt -o out/notes --border margin --border-header 40
```

| Value | Draws |
|---|---|
| `none` (default) | nothing; the layout is unchanged |
| `box` | four lines round the page, `--border-inset` mm in from the paper edge (default 12) |
| `margin` | a vertical margin line about 28 mm from the left paper edge |
| `box-margin` | both; the margin line sits at least 14 mm inside the box |

`--border-header MM` adds a horizontal line that many mm from the top edge, for a heading
area. Code: [`src/ductus/border.py`](../src/ductus/border.py).

## How the lines look

- **Same ink, not a vector overlay.** Each line is added to the page's stroke layer next to the
  words, so it goes through the ink model ([INK.md](INK.md)): striations, skips, bleed, and
  darker overlaps. With `--photo` the border is drawn before the capture and gets the photo look.
  The SVG output (`--svg`) does not include the border.
- **One pull of the pen per line**, with a slow wobble and a slight bow (more with a messier
  `--look`), pressure that changes the width and density along the line, and a lifted pen at the
  end.
- **Loose corners.** Each end overshoots the corner or stops short of it by 0 to 2 mm, and every
  corner is a slightly different point, so the box is not exactly square.
- **Different on every page.** Each side moves up to 2 mm from its nominal position, per page and
  per seed. About one line in three is gone over a second time, a little to one side.

## Where the text goes

The writing area shrinks by the worst case of those random offsets plus a gap of 4.5 mm, so
text never touches a line, and it does not hang into the margin: a squeezed last word, the
last words of a line that run past the right margin, and the sideways wander of the left edge
are all capped. With a box the text area is 19 to 22 mm in from each paper edge at the default
inset (inset 12 mm + 2 mm offset + 0.35 mm line + 4.5 mm gap, plus allowances for
ascenders, descenders and the last line's slope); with a margin line the text starts about 9 mm to the right of it. A longer text
therefore takes more pages than without a border. The border does not change which words are
generated, so the word cache is reused.

Tests in [`tests/test_border.py`](../tests/test_border.py) check that the text area shrinks,
that rendered text ink never overlaps the border ink, and that there is a gap of at least 1.5 mm
everywhere.

## Limits

- The margin line is a plain vertical line. Printed notebook margin lines and double margins
  are not drawn.
- Borders are not decorated (no corner flourishes, no double lines).
- The 0 to 2 mm corner and inset ranges are fixed values chosen by eye; they are not measured
  from real project files.

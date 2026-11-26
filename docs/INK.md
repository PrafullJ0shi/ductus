# The ink model

Generated handwriting usually gives itself away in the ink before the letters: one flat
colour, perfect edges, the same density everywhere. ductus first works out where the pen
went, then lets a pen model put ink along that path.

Which pen does what, with every width and colour, is in [PENS.md](PENS.md). This page
explains the steps.

![Old flat ink and new ballpoint ink, 100% crop](assets/ink-before-after.png)

## Three models

`--ink-model` picks one. All three start from the same page layout.

| Value | What it is | Code |
|---|---|---|
| `realism` (default) | Re-inks the page at 600 dpi with striations, skips, blobs, flow variation, bleed and overlap darkening. Colours are mixed in linear light. | `render.compose_realism`, `realism/ink.py` |
| `pens` | A simpler model: ink concentration from speed, blobs, skips, rough edges and bleed, then the Beer-Lambert law. | `pens.concentration`, `pens.to_rgb`, `render.compose_physical` |
| `flat` | One colour, one density with a gentle pressure field. Kept for comparison. `--flat-ink` does the same. | `render.compose` |

## Steps in the default model

1. **Stroke shape.** Every word image is skeletonised (reduced to its centre line) and drawn
   again at the pen's line width: a disc for ballpoint, gel and pencil, a thin rotated ellipse
   (the nib) for the fountain pen. Every word on a page therefore has the same stroke weight,
   whichever carrier line it came from. Pen marks such as arrows, ticks and rules are drawn at
   the same width. (`render.word_layer`, `render.ink_coverage`.)
2. **Overlap count.** While the page is assembled, the renderer counts how many strokes pass
   over each point. A strike-through or a retraced letter is counted twice.
3. **Smudges.** A few short directional smears are added to the stroke shape, where a hand
   would have dragged wet ink. The rate comes from the look (`smudges`, default 0.6 per page;
   0 for `neat`). (`render.smudge`.)
4. **Ink texture.** The word image is turned into ink by `ink_from_mask` in `realism/ink.py`:
   - stroke direction is estimated with a structure tensor;
   - light tracks (striations) are drawn along that direction for ballpoints;
   - straight, fast stretches get less ink, and short gaps (skips) for ballpoints;
   - slow variation of density along the line;
   - extra ink at stroke ends, as a ballpoint leaves a blob when the ball stops;
   - bleed, a small wicking blur whose size depends on the pen;
   - pencil grain, from paper-tooth noise.
5. **Overlaps.** Where two strokes cross, the second is inked again on top, so crossings and
   retraces are darker.
6. **Colour.** The pen's single-stroke colour is turned into a transmittance with the
   Beer-Lambert law and multiplied onto the paper in linear light. Ink is a filter, not paint,
   so it also works on ruled and grid paper.
7. **Resolution.** The page is built at 600 dpi (supersampling factor 2) and reduced to the
   300 dpi A4 output. The SVG output keeps vector outlines; the raster output is the realistic
   one.

The simpler `pens` model replaces steps 4 to 6 with `pens.concentration`: speed is estimated
from a coherence map (straight stretches are fast and lighter), blobs sit on centre-line ends
and junctions, ballpoint skips appear on fast stretches, a fountain pen gets a darker rim,
and the result goes through Beer-Lambert in `to_rgb`.

## What is real and what is a model

The effects are chosen to match what forensic and pen-physics sources describe for
ballpoint ink: striations, ink depots, skips, flow variation, overlap darkening. The
parameters of the default model are set by hand; `src/ductus/realism/calibrate.py` measures
values such as the blue ballpoint colour from photos. This repository does not report a
pixel-level comparison with scans of real pen lines. Sources and status of each detail are
in [REALISM.md](REALISM.md) (section D, and the package checklist in section G).

## Compare with your own ink

`scripts/ink_compare.py` puts crops of your own photographed writing next to the generated
ink. Photos of real pages are never part of the repository.

## Sources

- Ballpoint line features (striations, gooping, skipping):
  [Norwitch Document Laboratory](https://www.questioneddocuments.com/individual-characteristics/),
  [Northwestern JCLC](https://scholarlycommons.law.northwestern.edu/cgi/viewcontent.cgi?article=4544&context=jclc).
- Speed and curvature:
  [Lacquaniti, Terzuolo and Viviani, 1983](https://doi.org/10.1016/0001-6918(83)90027-6).
- Ink deposition: Franke and Rose, 2004, and Franke, 2005; full references in
  [REALISM.md](REALISM.md#further-sources-realism-package).

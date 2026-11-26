# Realism checklist

Small details decide whether a page reads as written or as generated. This list records
each detail ductus considers: what it is, why it matters, the evidence, how it is done, and
its status.

Status: **done** (implemented and on by default where it fits the look), **partial**,
**deferred** (not yet implemented; reason given). The column *Checked* says whether the
detail was compared against the user's own photographed pages. Those photos are private and
are processed only on the user's machine by `ductus style`; checks marked *pending* need
that run (see [STYLE_GUIDE.md](STYLE_GUIDE.md)).

## Sources

- Natural variation: no writer repeats a form exactly; examiners judge a range of variation
  ([FBI review, 2009](https://archives.fbi.gov/archives/about-us/lab/forensic-science-communications/fsc/oct2009/review/2009_10_review02.htm);
  [NIST IR 8282, 2020](https://nvlpubs.nist.gov/nistpubs/ir/2020/NIST.IR.8282.pdf)).
- Fatigue: letters grow taller in over 90% of fatigued writers and wider in 77%; quality
  becomes more scrawled; word-length misjudged at line ends
  ([Garnacho-Castaño et al., Applied Sciences 2020](https://www.mdpi.com/2076-3417/10/21/7630));
  legibility falls and speed rises after 10 minutes of copying
  ([Schwellnus et al., 2012](https://onlinelibrary.wiley.com/doi/10.1111/j.1440-1630.2012.01014.x)).
- Speed and curvature: pen speed falls in tight curves (two-thirds power law,
  [Lacquaniti, Terzuolo and Viviani, 1983](https://doi.org/10.1016/0001-6918(83)90027-6));
  strokes are lognormal velocity pulses ([Plamondon et al., 2013](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2013.00945/full)).
- Ballpoint line features: striations, gooping (blobs after direction changes), skipping
  ([Norwitch Document Laboratory](https://www.questioneddocuments.com/individual-characteristics/);
  [Northwestern JCLC on ballpoint pens](https://scholarlycommons.law.northwestern.edu/cgi/viewcontent.cgi?article=4544&context=jclc)).
- Revisions: crossing out is the most common revision in student handwriting; spelling
  revisions are 10 to 26% of revisions
  ([Sumner and Connelly, UCL](https://discovery.ucl.ac.uk/id/eprint/10088070/1/Sumner%20&%20Connelly%20in%20press_writing%20and%20revisions%20JoLD.pdf));
  learner corpora transcribe strike-throughs, insertions and smudges
  ([Gold et al., CAWL 2023](https://aclanthology.org/2023.cawl-1.3.pdf)).
- Phone capture: reflectance times shading, smooth warps, real shadow maps, JPEG, noise and
  blur ([DocAligner](https://arxiv.org/html/2306.05749v2), [Doc3D](https://github.com/cvlab-stonybrook/doc3D-dataset/),
  [SyntheticDoc](https://arxiv.org/html/2609.15503), [Augraphy](https://augraphy.readthedocs.io/en/latest/)).

## A. Letter forms and words

| # | Detail | Why / evidence | Implementation | Status | Checked |
|---|---|---|---|---|---|
| 1 | Every occurrence of a word is drawn anew | natural variation (FBI 2009) | each word comes from a different carrier line and seed | done | repetition probe: 0% identical pairs |
| 2 | Letter forms vary inside one page | range of variation (NIST 8282) | diffusion sampling; profile pages rotate reference lines every 40 words | done | pending |
| 3 | Inconsistent capital forms | writers keep 1 to 3 allographs per letter (NIST 8282, fig. 1.3) | follows from 1 and 2; no explicit allograph control | partial | pending |
| 4 | His digit styles (1, 7, 4) | personal digit forms are strong identifiers | needs the alphabet sample page and a fine-tune (roadmap item 1) | deferred | pending |
| 5 | i-dots and t-crosses placed late and offset | dots and crosses are added after the word | inside the generator's output; a dot-shift pass needs stroke separation | deferred | pending |
| 6 | Dot written as a short dash | fast writers flick dots | as 5 | deferred | pending |
| 7 | Ligatures and joins vary | joins depend on speed | generator output | done (model) | pending |
| 8 | Pen lifts mid-word | common in print-cursive mixes | generator output | done (model) | pending |
| 9 | Double-traced letters | writer goes over a letter that came out thin | `retrace`: a second, offset pass over the word, lighter | done | pending |
| 10 | Retouched letters darker | retraced ink stacks | overlap darkening in the ink model (11 in section D) | done | pending |
| 11 | Spelling slips struck out and rewritten | crossing out is the most common revision (Sumner and Connelly) | `{oops:word}` and `--corrections` rate | done | pending |
| 12 | Cross-outs only where wanted | random strikes look wrong | strikes come only from markup or the corrections rate (0 by default for natural) | done | |
| 13 | Caret insertion of a forgotten word | learner corpora mark insertions | needs a two-level line layout | deferred | pending |
| 14 | Overwrite (letter corrected in place) | common quick fix | needs letter-level control | deferred | pending |
| 15 | Hesitation (pen-rest) dots | pen rests before starting | `hesitation` rate: a small dot before the word | done | pending |
| 16 | Consistent stroke weight across words | real pens have one line width | words re-inked along their centre line at the pen's width | done | |
| 17 | Consistent word size | generator lines come out at different sizes | each word normalised to the page x-height, then a slow size drift | done | |

## B. Lines and spacing

| # | Detail | Why / evidence | Implementation | Status | Checked |
|---|---|---|---|---|---|
| 18 | Baseline drift and per-line slope | lines are not ruled | per-line slope with random walk; profile measures his slope | done | pending |
| 19 | Upward drift to the right | his habit (reported by the user) | `slope_deg` negative; measured in a profile | done | pending |
| 20 | Words bob on the baseline | motor variability | per-word vertical jitter | done | |
| 21 | Word spacing varies | inter-word spacing is a habit with variation (NIST 8282 table 1.1) | gap from the reference line, with noise | done | |
| 22 | Spacing after punctuation inconsistent | habit | `punct_gap`: extra random gap after , . ; : | done | pending |
| 23 | Squeezed words at the line end | writers misjudge word length at line ends (fatigue study) | `squeeze`: a word up to 14% too long is compressed instead of wrapped | done | pending |
| 24 | No hyphenation | handwriting rarely hyphenates | wrap at word boundaries only | done | |
| 25 | Writing grows and loosens toward the page end | fatigue: taller and wider letters | `fatigue`: size and spacing grow with page progress | done | pending |
| 26 | Line spacing tightens at the page bottom | running out of page | `tighten`: pitch shrinks with progress cubed | done | pending |
| 27 | Last line cramped | as 26 | follows from 26 and 23 | partial | pending |
| 28 | Uneven left margin | margins wander | margin random walk | done | pending |
| 29 | Ragged right margin | no justification | random right limit per line | done | |
| 30 | Indentation habits | lists and definitions hang | hanging indents in notes mode | done | pending |
| 31 | Slope and pressure change between sessions | pages written on different days | `session`: after a section rule, new slope offset and pressure | done | pending |
| 32 | Mixed heading sizes | his headings are larger, some much larger | headings 1.45x, sub-headings 1.2x, plus size jitter | done | pending |
| 33 | Notebook margin line ignored | writers cross the red line | not drawn: margin lines are not part of the paper types yet | deferred | |
| 34 | Letters touching grid lines | writing is not aligned to the grid | grid is independent of the writing | done | |

## C. Marks

| # | Detail | Why / evidence | Implementation | Status | Checked |
|---|---|---|---|---|---|
| 35 | Arrows drawn by hand, open heads, slightly curved | personal arrow habit | quadratic shaft with wobble, two open head strokes | done | pending |
| 36 | Underlines of uneven length | drawn freehand | random overshoot and undershoot | done | |
| 37 | Ticks of varying size | drawn fresh each time | random size and angle | done | |
| 38 | Circles not closed or overlapping | one quick loop | ellipse with 1.12 turns and random start | done | pending |
| 39 | Numbers poorly aligned in table columns | freehand tables | cell content keeps word jitter; column rules drawn freehand | done | pending |
| 40 | Section rules wobble and stop short | freehand | wobbling rule with random end | done | |

## D. Ink

| # | Detail | Why / evidence | Implementation | Status | Checked |
|---|---|---|---|---|---|
| 41 | Lighter on fast straight strokes, darker in curves | speed falls with curvature (two-thirds power law) | structure-tensor coherence lowers concentration | done | pending |
| 42 | Blobs at stroke starts, ends and turns | ballpoint gooping | blob field from centre-line ends and junctions | done | pending |
| 43 | Faint skips on fast strokes | ballpoint skipping | streak noise on fast stretches | done | pending |
| 44 | Pressure varies along a line | pressure is not constant | low-frequency concentration field | done | pending |
| 45 | Heavier downstrokes | more force pulling toward the body | needs stroke direction; the coherence term is direction-free | deferred | pending |
| 46 | Ink starts faint after a pause | ball dries while resting | needs stroke order | deferred | pending |
| 47 | Tiny hooks at stroke starts | pen lands moving | generator output | partial | pending |
| 48 | Ragged edges, no vector-perfect outline | paper fibres | edge noise on the stroke boundary at 600 dpi | done | |
| 49 | Bleed into the paper | ink wicks | per-pen bleed blur | done | |
| 50 | Overlaps darker | two layers of ink | Beer-Lambert: concentration adds, transmittance multiplies | done | |
| 51 | Hand smudges | side of the hand drags fresh ink | `smudges` per page: short directional smear | done | pending |
| 52 | His ink colour | blue ballpoint differs by brand | profile stores the measured ink colour; default `#2c3a86` | partial | pending |

## E. Phone photo (`--photo`)

| # | Detail | Why / evidence | Implementation | Status | Checked |
|---|---|---|---|---|---|
| 53 | Keystone tilt | phone not parallel to page | random corner perspective | done | pending |
| 54 | Small rotation | hand-held | random in-plane rotation | done | pending |
| 55 | Page curl near the binding | spiral notebooks lift | displacement and shading band near the bound edge | done | pending |
| 56 | Spiral holes | notebook pages | dark holes along the bound edge | done | pending |
| 57 | Uneven light | room light | directional gradient plus low-frequency field | done | pending |
| 58 | Hand or phone shadow | photographer over the page | soft blurred shadow from one side | done | pending |
| 59 | White-balance tint | auto white balance misses | random per-channel gain, usually warm | done | pending |
| 60 | Vignetting | lens falloff | radial darkening | done | |
| 61 | Edge defocus | shallow depth of field at an angle | radial blend with a blurred copy | done | |
| 62 | Sensor noise, stronger in shadows | small sensors | signal-dependent luma and chroma noise | done | |
| 63 | Phone sharpening halos | phone image pipelines sharpen | unsharp mask | done | |
| 64 | JPEG compression | phones save JPEG or HEIC | JPEG quality 84 | done | |
| 65 | Desk visible around the page | photos are not cropped scans | textured desk background | done | pending |
| 66 | Show-through from the back page | thin notebook paper | mirrored, blurred ghost of writing | done | pending |
| 67 | Finger at the photo edge | holding the page | optional skin-tone blur at an edge | done (off by default) | |
| 68 | Moire with the grid | grid frequency against the sensor | arises from resampling a fine grid; not modelled | deferred | |
| 69 | Ink sheen | ballpoint paste is glossy at angles | not modelled | deferred | |

## F. Page furniture

| # | Detail | Why / evidence | Implementation | Status | Checked |
|---|---|---|---|---|---|
| 70 | Page number or date in a corner | students date their notes | needs a corner text slot in markup | deferred | pending |
| 71 | Margin notes | side remarks | needs a margin column in layout | deferred | pending |
| 72 | Writing around page edges | running out of space | squeeze and tighten cover part of it | partial | pending |

Implemented in `layout.py` (B, C), `render.py` and `pens.py` (D), `photo.py` (E). The
rates live in `Look` and in a profile's `look` section, so a personal profile can set them
from measurements.

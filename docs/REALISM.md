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
| 35 | Arrows drawn by hand, open heads, slightly curved | personal arrow habit | wobbling shaft with slight tilt; open head drawn in one or two strokes | done | pending |
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


## G. The realism package (`src/ductus/realism/`)

The default ink model, the phone capture and the human-error pass come from
`ductus.realism`. Its capture parameters and ink colour are measured from a real phone photo
of the user's notes by `python -m ductus.realism.calibrate` (`calibration.json`, status
`measured`, 1 photo so far; more photos tighten the estimates). How it is wired in:

- Ink: `--ink-model realism` (default) re-inks the 600 dpi stroke shape with
  `realism.ink.ink_from_mask` and composites it in linear light (`render.compose_realism`);
  strokes drawn twice are inked twice. `--ink-model pens` and `--flat-ink` keep the older models.
- Behaviour: `humanize_bridge.humanize_page` runs `drift_size_slant` and `add_overwrites` on
  the laid-out page (`--humanize RATE`, off for `--look neat`). Fatigue, tightening, margins and
  squeezed line ends stay in `layout.py`, which already does them.
- Photo: `--photo` uses `realism.phone_capture.capture` with one of two surfaces
  (`CaptureParams.surface`): `a4` (default; a loose sheet on a desk: no spiral, no pages below,
  one or two corners lifted with a soft cast shadow, a gentle bow, contact shadow along the
  edges, no show-through) and `notebook` (spiral binding, pages below, curl at the binding,
  grid paper by default). The older model stays in `photo.py`.

Before/after images: [loose A4 photo](../examples/photo_a4_compare.jpg),
[notebook photo](../examples/photo_notebook_compare.jpg). (The package's own demo images used
strokes from the 2016 RNN, whose looped f and stylish r are not what ductus aims for, so they
are not shipped.)

Evidence numbers in the tables below refer to the "Further sources" list in this section.

### Further sources (realism package)

Handwriting, motor control, forensics

1. Plamondon, R. (1995). A kinematic theory of rapid human movements, Part I. *Biological Cybernetics* 72(4), 295–307.
2. Plamondon, R., & Srihari, S. N. (2000). On-line and off-line handwriting recognition: a comprehensive survey. *IEEE TPAMI* 22(1), 63–84.
3. Lacquaniti, F., Terzuolo, C., & Viviani, P. (1983). The law relating the kinematic and figural aspects of drawing movements. *Acta Psychologica* 54, 115–130. (two-thirds power law)
4. Van Galen, G. P. (1991). Handwriting: issues for a psychomotor theory. *Human Movement Science* 10, 165–191.
5. Wann, J., & Nimmo-Smith, I. (1991). The control of pen pressure in handwriting: a subtle point. *Human Movement Science* 10, 223–246.
6. Huber, R. A., & Headrick, A. M. (1999). *Handwriting Identification: Facts and Fundamentals*. CRC Press. (natural variation, alignment, spacing, i-dot and t-cross habits)
7. Franke, K., & Rose, S. (2004). Ink-deposition model: the relation of writing and ink deposition processes. *IWFHR-9*, 173–178. doi:10.1109/IWFHR.2004.59
8. Franke, K. (2005). *The influence of physical and biomechanical processes on the ink trace*. PhD thesis, University of Groningen. (ballpoint striations, ink depots, gaps, feathering; force vs intensity)
9. Garnacho-Castaño, M. V., Faundez-Zanuy, M., & Lopez-Xarbau, J. (2020). On the handwriting tasks' analysis to detect fatigue. *Applied Sciences* 10(21), 7630. (letter height grew in >90 % of writers, width in 77 %)
10. Kushki, A., Schwellnus, H., Ilyas, F., & Chau, T. (2011). Changes in kinetics and kinematics of handwriting during a prolonged writing task. *Research in Developmental Disabilities* 32(3), 1058–1064.
11. Ellis, A. W. (1979). Slips of the pen. *Visible Language* 13(3), 265–282.
12. Wing, A. M., & Baddeley, A. D. (1980). Spelling errors in handwriting: a corpus and a distributional analysis. In U. Frith (Ed.), *Cognitive Processes in Spelling* (pp. 251–285). Academic Press. (slips cluster in the middle of words)
13. Bilgin Taşdemir, E. F., & Yanikoglu, B. A comparative study of delayed stroke handling approaches in online handwriting. *IJDAR* (2019). http://myweb.sabanciuniv.edu/berrin/files/2018/12/delayed-2018.pdf (i-dots and t-crosses written after the word)
14. Graves, A. (2013). Generating sequences with recurrent neural networks. arXiv:1308.0850. (source of the demo strokes)

Document images, cameras

15. Journet, N., Visani, M., Mansencal, B., Van-Cuong, K., & Billy, A. (2017). DocCreator: a new software for creating synthetic ground-truthed document images. *Journal of Imaging* 3(4), 62.
16. Groleau, A., Chee, K. W., Larson, S., Maini, S., & Boarman, J. (2023). Augraphy: a data augmentation library for document images. *ICDAR 2023*, 384–401. arXiv:2208.14558.
17. Kieu, V. C., Visani, M., Journet, N., Domenger, J.-P., & Mullot, R. (2012). A character degradation model for grayscale ancient document images. *ICPR 2012*.
18. Baird, H. S. (1992). Document image defect models. In *Structured Document Image Analysis*, Springer, 546–556.
19. Das, S., Ma, K., Shu, Z., Samaras, D., & Shilkrot, R. (2019). DewarpNet: single-image document unwarping with stacked 3D and 2D regression networks. *ICCV 2019*. (Doc3D)
20. Cao, H., Ding, X., & Liu, C. (2003). Rectifying the bound document image captured by the camera: a model based approach. *ICDAR 2003*. (curl near the binding)
21. Burie, J.-C., et al. (2015). ICDAR2015 competition on smartphone document capture and OCR (SmartDoc). *ICDAR 2015*. (perspective, light and defocus ranges in phone captures)
22. Foi, A., Trimeche, M., Katkovnik, V., & Egiazarian, K. (2008). Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data. *IEEE TIP* 17(10), 1737–1754.
23. Brooks, T., Mildenhall, B., Xue, T., Chen, J., Sharlet, D., & Barron, J. T. (2019). Unprocessing images for learned raw denoising. *CVPR 2019*. (camera pipeline order: noise in linear raw, then WB, demosaic, tone curve)
24. Goldman, D. B. (2010). Vignette and exposure calibration and compensation. *IEEE TPAMI* 32(12), 2276–2288.
25. McCamy, C. S. (1992). Correlated color temperature as an explicit function of chromaticity coordinates. *Color Research & Application* 17(2), 142–144.
26. Sharma, G. (2001). Show-through cancellation in scans of duplex printed documents. *IEEE TIP* 10(5), 736–754.
27. Postl, W. (1986). Detection of linear oblique structures and skew scan in digitized documents. *ICPR 1986*, 687–689. (projection-profile skew)
28. Pertuz, S., Puig, D., & Garcia, M. A. (2013). Analysis of focus measure operators for shape-from-focus. *Pattern Recognition* 46(5), 1415–1432. (variance of Laplacian)
29. Wallace, G. K. (1992). The JPEG still picture compression standard. *IEEE Trans. Consumer Electronics* 38(1). (IJG quantisation tables used to estimate quality)
30. Kubelka, P., & Munk, F. (1931). Ein Beitrag zur Optik der Farbanstriche. *Z. Tech. Phys.* 12, 593–601. (ink as a subtractive layer; this module uses the simpler Beer–Lambert form)

### Package checklist

Legend: **I** = implemented (function), **D** = deferred (why). Evidence numbers refer to the list above.

#### Ink trace (`realism/ink.py`)

| # | Detail | Status | Evidence |
|---|---|---|---|
| 1 | Line width set by the tip (0.5/0.7/1.0 mm tips leave ~0.30/0.38/0.50 mm lines) | I `line_width_px`, `BALLPOINT_LINE_MM` | manufacturer line widths; 8 |
| 2 | Width grows with pen pressure | I `stroke_concentration` (half-width 0.78–1.18×) | 5, 7 |
| 3 | Ink deposit grows with force and falls with speed | I (`pressure_gain`, `speed_fade`) | 7, 8 |
| 4 | Speed from real timing; if none, from curvature (two-thirds power law) | I `stroke_speed` | 3 |
| 5 | Downstrokes heavier than upstrokes | I `down_gain` | 5, 6 |
| 6 | Ballpoint striations (light tracks along the stroke) | I `striation` | 8 |
| 7 | Ink gaps (skips) on fast stretches, mostly mid-stroke | I `skips` | 8 |
| 8 | Ink depots (blobs) at a slow start, sharp turns and the end | I `blob`, `blob_rate` | 8 |
| 9 | Flying start and finish: fast strokes taper in and out | I `taper` | 6 |
| 10 | Slow along-stroke opacity variation | I `flow_var` | 7 |
| 11 | Stroke-to-stroke ink-flow variation | I `stroke_var` | 8 |
| 12 | Ragged edges from paper fibres, both sides independent | I `rough` | 8 |
| 13 | Fibre bleed (wicking) with oriented fibre texture | I `_finish`, `fibre_field` | 8, 16 |
| 14 | Overlaps darker (ink adds), with saturation | I (`_paste_add` + tanh) | 7 |
| 15 | Colour by Beer–Lambert in linear light: thin = paler and bluer, dense = navy | I `transmittance` | 30 |
| 16 | Ink multiplies the paper and grid lines (not painted on) | I `composite` | 30 |
| 17 | Gel pen: even, saturated, darker dried rim, almost no skips | I `gel-*` presets | 8 |
| 18 | Fountain pen: width by nib angle, pooling at ends, feathering | I `fountain` preset (`nib_angle`, `edge_dark`, bleed) | 8 |
| 19 | Pencil: graphite only on the paper tooth, grey without sheen | I `pencil` preset (`grain`) | 8 |
| 20 | Re-ink raster word images (diffusion output) with the same texture | I `ink_from_mask` | — |
| 21 | Ballpoint sheen (glossy paste reflecting light at an angle) | D: needs light direction per pixel; small at phone distance | 8 |
| 22 | Indentation (embossing) visible in raking light | D: needs a height map; slight value in close-ups only | 6 |
| 23 | Ink build-up on the ball released as a blob after a long pause between words | partly I via start blobs; per-word pause timing D | 8 |

#### Writing behaviour (`realism/human_errors.py`)

| # | Detail | Status | Evidence |
|---|---|---|---|
| 24 | Slow drift of size and slant across words (random walk, not white noise) | I `drift_size_slant` | 6 |
| 25 | Baseline wobble and small word rotations | I `drift_size_slant` | 6 |
| 26 | Uneven word gaps | I (`gap_after`) | 6 |
| 27 | Writing grows larger and wider down the page (fatigue) | I `apply_fatigue` | 9, 10 |
| 28 | Slant and unsteadiness increase late on the page | I `apply_fatigue` | 10 |
| 29 | Lines rise towards the right | I `apply_fatigue` (`rise_deg`) | 6 |
| 30 | Words re-break when they grow (no text spilling off the page) | I `rewrap` | — |
| 31 | Line spacing tightens and writing shrinks near the bottom | I `tighten_bottom` | 6 |
| 32 | Left margin creeps and jitters; ragged right edge | I `uneven_margins` | 6 |
| 33 | Last word squeezed or run into the margin instead of wrapping | I `squeeze_line_ends` | 6 |
| 34 | Slips of the pen (omission, doubling, transposition, anticipation, stopped part-way) | I `make_slip` | 11, 12 |
| 35 | Slips sit mostly in the middle of words | I (beta(2.2, 2.2) position) | 12 |
| 36 | Strike-through: single, double or scribble, then the right word after it | I `add_strike_corrections` | 11 |
| 37 | Caret insertion: forgotten word added smaller above the line | I `add_caret_insertions` | 6 |
| 38 | Retouched letter written over itself (darker, doubled) | I `add_overwrites` (+ demo renderer) | 6 |
| 39 | i-dots placed late: right of the stem, high, sometimes a dash, sometimes missing | I `delayed_strokes` | 13, 6 |
| 40 | t-crosses rise to the right and can reach the next letter | I `delayed_strokes` | 13, 6 |
| 41 | Marks drawn as fast, even-speed strokes with the same pen | I `marks_to_strokes` | 7 |
| 42 | Hyphenation or squashed half-words at line ends | D: rare in student notes; `squeeze_line_ends` covers it | 6 |
| 43 | Personal letter variants (two forms of the same letter) learnt from his pages | D: belongs to the style profile (`profile.py`) | 6 |
| 44 | Writing speed (and so ink) rises when copying familiar words | D: needs word-frequency timing in the glyph engine | 4 |

#### Phone capture (`realism/phone_capture.py`)

| # | Detail | Status | Evidence |
|---|---|---|---|
| 45 | Grid, ruled, dotted or plain paper with uneven printed lines | I `make_paper` | — |
| 46 | Paper cloudiness and fibres | I `make_paper` | 16, 17 |
| 47 | Show-through of the back page, mirrored and blurred | I `add_show_through` | 26 |
| 48 | Camera tilt (pitch, yaw) from a real pinhole projection, not random corners | I `capture` (`_rot`, `focal`) | 21 |
| 49 | In-plane rotation | I (`roll_deg`) | 21 |
| 50 | Curl near the spiral with a shading band | I (`curl`, `curl_width`) | 19, 20 |
| 51 | Spiral holes and wire coils over the binding edge | I `_spiral` | — |
| 52 | Page edge, pages underneath, thin thickness shadow on the desk | I (`page_edge`) | — |
| 53 | Desk background | I `_desk` | 16 |
| 54 | Uneven light: gradient across the page | I (`light_gradient`) | 21 |
| 55 | Soft hand or phone shadow with a wide penumbra | I (`shadow`, `shadow_softness`) | 16 |
| 56 | Light multiplied in linear RGB | I | 23 |
| 57 | Auto exposure puts the paper at a fixed level | I (`exposure`) | 23 |
| 58 | Residual white-balance tint (warm indoor light) | I (`wb_gains`, `wb_jitter`) | 23, 25 |
| 59 | Vignetting | I (`vignette`) | 24 |
| 60 | Defocus that follows depth (tilted page, lifted curl) | I (`defocus`, depth map) | 21, 28 |
| 61 | Lens and demosaic softness everywhere | I (`base_blur`) | 23 |
| 62 | Lateral chromatic aberration towards the corners | I (`chroma_aberration`) | 23 |
| 63 | Poisson–Gaussian sensor noise, stronger in shadows | I (`noise_shot`, `noise_read`) | 22 |
| 64 | ISP chroma denoising (noise is mostly in luma) | I (`chroma_denoise`) | 23 |
| 65 | Tone curve and sharpening halos | I (`tone_contrast`, `sharpen`) | 23 |
| 66 | JPEG with 4:2:0 chroma subsampling at a measured quality | I (`jpeg_quality`) | 29 |
| 67 | Measurement of all the above from real photos | I `calibrate.py` | 22, 25, 27, 28, 29 |
| 68 | Motion blur from a hand-held shot | D: phones use short exposures in good light; add if his photos show it | 21 |
| 69 | Finger or thumb at the page edge | D: easy to add to `_shadow_polygon`; not seen as common | — |
| 70 | Moiré between the grid and the sensor | D: arises from resampling; not modelled explicitly | — |
| 71 | HDR / multi-frame local tone mapping (flat-looking shadows) | D: partly covered by the shadow strength | 23 |
| 72 | Specular glare on the page | D: paper is matte; would need a light position | — |


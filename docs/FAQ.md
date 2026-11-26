# FAQ

## Setup

**Does it need a GPU?**
No. Everything runs on a CPU. GPU and Apple Silicon paths are on the roadmap in the
[README](../README.md#v2-roadmap); there are no GPU timings in this repository.

**Does it need an account or a login?**
No. The DiffBrush checkpoint comes from a public Google Drive link, and the Stable Diffusion
VAE and TrOCR from Hugging Face. The RNN writer needs a checkout of
[Grzego/handwriting-generation](https://github.com/Grzego/handwriting-generation) and a Python
with TensorFlow 2 (see [CLI.md](CLI.md#environment-variables)).

**Does it work offline?**
After the first run has downloaded the models, yes. ductus does not upload your text or your
photos.

**It says the RNN model is not found.**
Check that `~/.cache/ductus/handwriting-generation/pretrained/model-29.meta` exists, or set
`DUCTUS_RNN_DIR` to your checkout, and set `DUCTUS_RNN_PYTHON` to a Python with TensorFlow 2.
With `--engine default`, ductus then falls back to the diffusion writer for every word, which
works but is slower and less readable. With `--engine rnn` it stops with an error instead.

## Speed

**How long does a page take?**
It depends on the machine and the writer. In [round 2 of the benchmark](BENCHMARK.md#round-2-after-the-readability-changes),
a 50-word test page took 158 s (23 s per line) on an 8-core CPU with the default writer,
against about 390 s per line for the diffusion-only version under heavier load. The OCR read-back
and model loading take most of the time. Treat these as examples, not promises.

**How do I make a draft faster?**
Use `--no-verify` to skip the OCR read-back (more mistakes). Words are cached under
`~/.cache/ductus/words`, so changing only the pen, paper, photo or ink model re-renders
without writing the words again. `--steps` only affects the diffusion writer.

## Quality

**Why does a word sometimes still look wrong?**
A word is accepted when OCR reads it closely enough. If it still fails after the last round
(6 rounds by default: 4 with the RNN and 2 with diffusion), the best attempt is used and counted
in `still_failing` in the printed statistics. Try another `--seed`, `--look neat`, or a simpler
word. The OCR judge also misreads drawn marks, so a correct page can score badly.

**Why are `f`, `r` and `l` so plain?**
That is the default on purpose. The target is the plain print-like hand that most people
write, and a letter-shape check rewrites looped or calligraphic forms. `--allow-loops` turns
the check off. See [README](../README.md#plain-letters).

**Why is the maths line the weakest in the benchmark?**
The OCR judge cannot read stacked fractions and drawn `=` marks, and `x^2` came out wrong in
the test. Both are in [BENCHMARK.md](BENCHMARK.md) with the raw OCR output.

**Can it write characters outside ASCII?**
Not yet. The diffusion writer's alphabet is 80 ASCII characters. `=`, `<`, `>`, `%` and a few
others are drawn as pen marks; other characters are replaced by close ASCII forms (`@`
becomes "at"). Accented letters and currency signs are not written.

**Is the output a font?**
No. Every word is generated in context, so a repeated word is not an identical copy. In the
benchmark the glyph twin rate is 0.000 for ductus and 0.27 to 0.46 for fonts
([BENCHMARK.md](BENCHMARK.md)).

**Can it fool a person or a detector?**
Not reliably. A classifier told real and generated lines apart far above chance in the
earlier blind test ([EVALUATION.md](EVALUATION.md#4-blind-test)). That test used the older
flat-ink version; it has not been re-run.

## Your own hand

**How do I use my own handwriting?**
Photograph some of your pages and run `ductus style`; see [STYLE_GUIDE.md](STYLE_GUIDE.md).
What the profile changes is in [HABITS.md](HABITS.md).

**Do the letter shapes copy mine?**
Only with `--engine diffusion` or `--engine auto`. With the default engine the letters are the
plain RNN hand, and your profile sets size, spacing, slant, ink colour and habits.

**Where do my photos go?**
They are read on your machine and not uploaded by ductus, and the profile folder (`styles/`) is ignored by git.
Do not commit it.

## Output

**Can I get vectors?**
`--svg` writes traced outlines for words and strokes for marks. `--animate` writes an SVG that
writes itself word by word.

**How do I change the pen?**
`--pen`, `--tip`, `--weight` and `--ink`. See [PENS.md](PENS.md).

## Licence and use

**Can I use it commercially?**
The ductus code and the vendored DiffBrush code are MIT. The DiffBrush weights were trained on
the IAM database, whose licence is for non-commercial research, and the RNN weights were trained
on IAM-OnDB. Check both before commercial use. See [RESEARCH.md](RESEARCH.md#6-licences-that-matter-for-users).

**Is it meant to fake documents?**
No. It is for notes, mock-ups, design and accessibility. Do not use it to forge signatures or
to hand in generated work as someone else's.

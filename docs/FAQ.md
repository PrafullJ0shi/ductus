# FAQ

## Setup

**Does it need a GPU?**
No. Everything runs on a CPU. If PyTorch can see a CUDA or Apple (MPS) GPU, the OCR read-back,
the RNN sampler and the diffusion fallback use it automatically (`DUCTUS_DEVICE=cpu` turns that
off). That path is written but has not been timed: this repository has no GPU numbers. The
read-back and the RNN are small models run step by step, so expect a GPU to help most with the
diffusion fallback and the OCR, and less than for a big model.

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
On a shared 8-core CPU, an 84-word A4 page took 167 s by default and 37 s with `--fast`, against
2,881 s before the speed work ([round 6](BENCHMARK.md#round-6-speed): the old figure was
measured under a heavier load, so the gain is smaller than 17 times for an idle machine; the
closest same-load comparison is 510 s to 167 s). Pages after the first in one run reuse the
loaded models. Treat these as examples, not promises.

**Why is it slow, and what makes it faster?**
Almost all the time is the OCR read-back that checks every word and rewrites the ones that
fail. The default reads with a small model first and asks the larger one only about words the
small one rejects. Fewer rewrite rounds, `--fast` and `--budget` trade a few rough words for
time. On a busy machine, fewer threads can be faster: set `DUCTUS_THREADS`.

**How do I make a draft faster?**
Use `--fast` (small OCR only, 4 rewrite rounds, no diffusion fallback: 4 to 5 times faster, a
few more rough words; 2 to 3 of 84 words failed in the test), `--budget SECONDS` to stop
starting new rewrite rounds, or `--no-verify` to skip the read-back entirely (more mistakes).
Words are cached under
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
blind tests in [EVALUATION.md](EVALUATION.md#4-blind-test), including a later default-hand
run with the current engine and ink (about 98% accuracy, or 91 to 95% when only letter
shape is left).

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

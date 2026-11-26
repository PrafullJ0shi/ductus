# Research notes

What changed in handwriting synthesis between 2016 and 2026, what can actually run on a
laptop CPU today, and why ductus is built the way it is. Every project below was checked on
GitHub or Hugging Face in November 2026; licences are as stated by each project.

## 1. The 2016 starting point

[scribe](https://github.com/greydanus/scribe) (Sam Greydanus, 2016,
[blog post](https://greydanus.github.io/2016/08/21/handwriting/)) is a clean TensorFlow 1
implementation of Alex Graves' handwriting model from
[Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850) (2013):

- a 3-layer LSTM predicts the next pen offset as a mixture of bivariate Gaussians plus an
  end-of-stroke bit (a mixture density network);
- a soft Gaussian "window" over the characters tells the network which letter it is writing;
- a bias term sharpens sampling (neater, less varied writing);
- style is imitated by "priming" the network with strokes from a known writer.

The same method lives on in [hardmaru/write-rnn-tensorflow](https://github.com/hardmaru/write-rnn-tensorflow)
(2015), [sjvasquez/handwriting-synthesis](https://github.com/sjvasquez/handwriting-synthesis)
(2018, the most popular one) and [Grzego/handwriting-generation](https://github.com/Grzego/handwriting-generation)
(2018, MIT, pretrained). We use Grzego's checkpoint as the reproducible "2016 method" baseline
because it is MIT-licensed and runs on CPU; scribe itself needs TensorFlow 1.x and a
checkpoint hosted on Google Drive, and implements the same model.

Known weaknesses, confirmed on our own baseline runs before starting:

| Weakness | Cause |
|---|---|
| Capital runs (`BTC`, `AI`) and mid-word capitals (`RupeeClarity`) come out wrong | IAM-OnDB has few of them; the attention window drifts |
| Limited character set (no `=`, `@`, `$`, `%` in some checkpoints) | training vocabulary |
| Lines over about 50 characters fall apart | attention and LSTM state drift |
| Needs manual seed picking | no internal check that the output says what was asked |
| Style only from strokes recorded on a tablet | priming needs online data, not a photo |

## 2. What changed, 2020 to 2026

Two families now dominate.

**Offline (image) generators** draw a picture of the text in the style of a reference
image. That matters here: a phone photo of a notebook is enough as a style sample.

| Work | Year | Idea | Code / licence |
|---|---|---|---|
| [Decoupled Style Descriptors](https://github.com/brownvc/decoupled-style-descriptors) (Kotani et al., ECCV) | 2020 | per-character style descriptors; BRUSH dataset (170 writers) | non-commercial licence (Brown University) |
| [Handwriting Transformers, HWT](https://github.com/ankanbhunia/Handwriting-Transformers) (Bhunia et al., ICCV) | 2021 | transformer encoder-decoder GAN, few-shot | MIT |
| [VATr](https://github.com/aimagelab/VATr) (Pippi et al., CVPR) | 2023 | visual archetypes for rare characters | MIT |
| [WordStylist](https://github.com/koninik/WordStylist) (Nikolaidou et al., ICDAR) | 2023 | latent diffusion, word level | MIT |
| [One-DM](https://github.com/dailenson/One-DM) (Dai et al., ECCV) | 2024 | one-shot diffusion with high-frequency style cues | MIT |
| [DiffusionPen](https://github.com/koninik/DiffusionPen) (Nikolaidou et al., ECCV) | 2024 | few-shot latent diffusion with a metric-learned style encoder | MIT |
| [Emuru](https://github.com/aimagelab/emuru-autoregressive-text-img) (Pippi et al., CVPR) | 2025 | autoregressive T5 over VAE latents, trained only on synthetic font renders | MIT, [weights](https://huggingface.co/blowing-up-groundhogs/emuru) MIT |
| [DiffBrush](https://github.com/dailenson/DiffBrush) (Dai et al., ICCV, [arXiv 2508.03256](https://arxiv.org/abs/2508.03256)) | 2025 | one-shot **text-line** diffusion with content-decoupled style learning and line + word discriminators | MIT, weights trained on IAM |
| Eruku (Zaccagnino et al., [WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Zaccagnino_Autoregressive_Styled_Text_Image_Generation_but_Make_it_Reliable_WACV_2026_paper.pdf)) | 2026 | Emuru with classifier-free guidance and special tokens for content fidelity | code announced, not checked |
| InkSpire ([ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/963ccebfefe8e349cad24e3ed4a92907-Paper-Conference.pdf)) | 2026 | single diffusion transformer, multi-line infilling, Chinese and English | paper only at time of writing |

**Online (stroke) generators** output pen trajectories, which give true vector output and
drawing animation, but need online training data (tablet recordings).

| Work | Year | Idea | Code / licence |
|---|---|---|---|
| [SDT](https://github.com/dailenson/SDT) (Dai et al., CVPR) | 2023 | style-disentangled transformer, character level | MIT; weights on Google Drive |
| [Cursive Transformer](https://github.com/greydanus/cursivetransformer) (Greydanus and Wimpee, [arXiv 2504.00051](http://arxiv.org/abs/2504.00051)) | 2025 | GPT-style decoder over polar stroke tokens, cross-attention to text; the scribe author's own update | no licence file |
| [ai-handwriting-generator](https://github.com/OmarMusayev/ai-handwriting-generator) (Musayev) | 2025 | cross-attention GPT with a polar tokenizer, trained on IAM-OnDB | MIT, weights included |
| [DiffInk](https://github.com/awei669/DiffInk) (ICLR 2026, [arXiv 2509.23624](https://arxiv.org/abs/2509.23624)) | 2026 | latent diffusion transformer for full lines of online ink | MIT code; released weights are Chinese (CASIA-OLHWDB) |
| [HandwritingAgent](https://arxiv.org/html/2606.18788) | 2026 | language-driven synthesis in vector space | paper |

**Photo to strokes.** [InkSight](https://github.com/google-research/inksight) (Google, TMLR 2025,
[arXiv 2402.05804](https://arxiv.org/abs/2402.05804)) converts photos of handwriting into digital ink
with a ViT + mT5 model ([Small-p weights](https://huggingface.co/Derendering/InkSight-Small-p), Apache-2.0).
Its model card explicitly warns against using it to forge handwriting.

A maintained index of the field: [awesome-handwritten-text-generation](https://github.com/koninik/awesome-handwritten-text-generation).

## 3. Reported quality

DiffBrush reports, on IAM text lines with one reference line (Table 1 of the paper):

| Method | Shot | HWD (lower is better) | OCR CER difference (lower is better) | FID |
|---|---|---|---|---|
| DiffusionPen | few | 1.72 | 54.75 | 10.24 |
| One-DM | one | 1.80 | 20.91 | 10.60 |
| DiffBrush | one | 1.41 | 8.59 | 8.69 |

These are the authors' numbers on their protocol, not ours. Our own measurements are in
[EVALUATION.md](EVALUATION.md).

## 4. What we tried on this machine

Machine: 8 CPU cores, 15 GB RAM, no GPU.

| Candidate | Result |
|---|---|
| Grzego RNN (2016 method) | runs, about 2 s per line; used as the baseline |
| DiffBrush, 25 DDIM steps | runs, **about 3.5 s per 64 x 1024 line** in batches of 8; strong one-shot style transfer; capitals, digits and mid-word capitals work |
| Musayev cross-attention GPT | runs, but generation has no key-value cache; three short samples did not finish in 10 minutes |
| Emuru (Hugging Face weights) | fails to load with transformers 5.x (`all_tied_weights_keys` missing in its remote code); would need a pinned older stack |
| DiffInk | English weights not released |
| SDT | English model is character level; whole lines need a separate layout stage |

## 5. Decision

ductus uses **DiffBrush** as its writing engine and builds the rest of the system around
its weak points:

1. **Carrier lines.** DiffBrush is trained on 35 to 61 character lines; short inputs come out
   badly (measured: single words often lose letters). ductus never asks it for a short
   input. Words are packed into 46-character carrier lines (topped up with filler words),
   written, and cut back into words at the widest gaps. Every occurrence of a word is a new
   sample, so there are no repeated glyph copies.
2. **OCR read-back.** Each carrier is read back by TrOCR (line level, then word level as a
   second opinion) and aligned to the intended words. Words that do not read back are
   rewritten in a new carrier. This replaces manual seed picking. For a personal hand the
   threshold is loose and compares case- and punctuation-insensitive text, so the check
   catches wrong or missing words without pushing the writing towards neatness.
3. **Pen marks are drawn, not written.** Arrows, ticks, crosses, `=`, `<`, `>`, `%`, fraction
   bars, radicals, sigma, underlines, strike-throughs, circles, rules and table lines are
   generated as strokes from nominal shapes plus random variation, at the measured pen width.
4. **Layout is where "human" lives.** Line drift, margins that wander, uneven gaps, per-word
   size and angle jitter, ink pressure, corrections, notebook structures. These are
   parameters measured from real pages when a style profile is built.

Why not train our own model: on CPU, a model that beats DiffBrush is not reachable in hours,
and the open English datasets (IAM, IAM-OnDB) require registration and allow research use
only. The sensible use of the time was a system around a strong open model.

## 6. Licences that matter for users

- DiffBrush weights are trained on the IAM Handwriting Database (registration,
  non-commercial research use). Treat generated output accordingly.
- The SD 1.5 VAE is under CreativeML OpenRAIL-M.
- ductus code is MIT. No weights or datasets are redistributed in this repository.

## 7. What would unlock better results

| Login or resource | Unlocks |
|---|---|
| IAM / IAM-OnDB registration (FKI, University of Bern) | fine-tuning on more writers; fair CER numbers on the standard test split |
| A GPU (any 8 GB card) | 50-step sampling, best-of-8 per line, fine-tuning DiffBrush on one person's pages |
| Hugging Face token | not needed today; every model used is public |
| A tablet recording of the writer | stroke-level imitation with an online model (true pen order in animations) |

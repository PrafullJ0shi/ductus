# Contributing

Thanks for taking the time to help.

## Setup

```bash
git clone https://github.com/PrafullJ0shi/ductus && cd ductus
python -m venv .venv && . .venv/bin/activate
pip install -e ".[model,pdf,dev]"
pytest -q            # unit tests; run them before every pull request
```

The first `ductus write` downloads about 3 GB of weights (the DiffBrush checkpoint, the Stable
Diffusion VAE and TrOCR) into `~/.cache/ductus` and the Hugging Face cache; set `DUCTUS_CACHE`
to move the first. The default RNN writer also needs the
[Grzego/handwriting-generation](https://github.com/Grzego/handwriting-generation) checkout and
a Python with TensorFlow 2; see the [README](README.md#quick-start) and
[docs/CLI.md](docs/CLI.md#environment-variables). Without it, ductus uses the diffusion writer.

## What helps most

- Bug reports with the exact command, the input text and the output image.
- New pen marks (`src/ductus/pen.py`): each mark must be built from a nominal shape plus
  random variation, so no two marks on a page are identical. Add a test.
- Layout habits (`src/ductus/layout.py`) backed by measurements from real pages.
- Pen and ink changes (`src/ductus/pens.py`, `src/ductus/realism/ink.py`), with the effect
  shown at 100% zoom (`scripts/ink_lab.py`).
- Evaluation scripts that make a claim checkable.

## Rules

- Keep numbers honest. Any number in the docs must come from a script in `scripts/` or
  `benchmark/scripts/`, and the page should say how it was made. Do not round in your favour.
- Say what the system cannot do in the same place you say what it can.
- Do not add model weights, datasets or other people's handwriting to the repository.
- Do not commit your own handwriting photos or a `styles/` profile; they are personal data.
- Do not add features meant to imitate a specific person's handwriting without their
  consent (see the README section on responsible use).
- Run `pytest -q` before opening a pull request.

## Documentation

Docs follow [DESIGN.md](DESIGN.md): short sentences, plain words, no marketing adjectives, no
emoji. Diagrams follow the diagram rules in that file (orthogonal connectors, at most nine
nodes, one accent colour, no shadows or glows).

- When you change a command line option in `src/ductus/cli.py`, change the tables in
  [README.md](README.md) and [docs/CLI.md](docs/CLI.md) in the same pull request.
- When you change a pen preset in `src/ductus/pens.py` or `src/ductus/realism/ink.py`, change
  [docs/PENS.md](docs/PENS.md).
- Check internal links before you push:

  ```bash
  python scripts/check_links.py
  ```

  It checks relative links, images and `#anchors` in every Markdown file, and lists how many
  external links it did not fetch.
- Diagrams are built by the scripts in `docs/diagrams/`; edit the script, not the SVG.

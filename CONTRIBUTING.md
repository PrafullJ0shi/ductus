# Contributing

Thanks for taking the time to help.

## Setup

```
git clone https://github.com/PrafullJ0shi/ductus && cd ductus
python -m venv .venv && . .venv/bin/activate
pip install -e ".[model,pdf,dev]"
pytest -q            # unit tests, no model download needed
```

The first `ductus write` downloads about 1.5 GB of weights into `~/.cache/ductus`
(set `DUCTUS_CACHE` to move it).

## What helps most

- Bug reports with the exact command, the input text and the output image.
- New pen marks (`src/ductus/pen.py`): each mark must be built from a nominal shape plus
  random variation, so no two marks on a page are identical. Add a test.
- Layout habits (`src/ductus/layout.py`) backed by measurements from real pages.
- Evaluation scripts that make a claim checkable.

## Rules

- Keep numbers honest: any number in the docs must come from a script in `scripts/`.
- Do not add model weights, datasets or other people's handwriting to the repository.
- Do not add features meant to imitate a specific person's handwriting without their
  consent (see the README section on responsible use).
- Run `pytest -q` before opening a pull request.

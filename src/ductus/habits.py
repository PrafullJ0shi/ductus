"""Writer habit profile: measured page habits plus hand-filled personal habits.

Measured by `ductus style` from the photos (see profile.measure_habits):
  fatigue, tighten, margin_sd_xh, heading_scale, heading_share, section_gap_xh,
  slope drift down the page.
Filled in by the writer or from a transcript (habits.json next to profile.json):
  spellings      {"software": "softwear"}           written the way he writes them
  abbreviations  {"explains": "->", "because": "bcz"} applied when the word appears
  correction     {"strike": 0.7, "scribble": 0.3}   how slips are crossed out
  corrections_rate                                   share of words with a slip
  heading        {"underline": 0.8}                  how often headings get a rule
The generator samples from these: replacements apply per occurrence with probability
`apply_rate`, so not every instance changes, as in real notes.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

LOOK_KEYS = ("fatigue", "tighten", "squeeze", "hesitation", "retrace", "punct_gap", "session",
             "smudges", "corrections")


def load(profile_dir: str | Path) -> dict:
    p = Path(profile_dir)
    out = {}
    prof = p / "profile.json"
    if prof.exists():
        out.update(json.loads(prof.read_text()).get("habits", {}))
    extra = p / "habits.json"
    if extra.exists():
        out.update(json.loads(extra.read_text()))
    return out


def apply_text(text: str, habits: dict, seed: int = 0) -> str:
    """Rewrite words the way the writer habitually writes them."""
    rng = random.Random(seed)
    rate = float(habits.get("apply_rate", 0.85))
    table = {**{k.lower(): v for k, v in habits.get("spellings", {}).items()},
             **{k.lower(): v for k, v in habits.get("abbreviations", {}).items()}}
    if not table:
        return text

    def sub(m):
        w = m.group(0)
        rep = table.get(w.lower())
        if rep is None or rng.random() > rate:
            return w
        return rep[:1].upper() + rep[1:] if w[:1].isupper() and rep[:1].isalpha() else rep

    return re.sub(r"[A-Za-z]+", sub, text)


def apply_look(look, habits: dict) -> None:
    for k in LOOK_KEYS:
        if k in habits:
            setattr(look, k, float(habits[k]))

"""Declared character coverage: which characters each system can put on paper as asked.

Read from each tool's own code or font file, not guessed:
  Grzego RNN       model.charset of pretrained/model-29 (write_page.Model)
  sjvasquez        drawing.alphabet
  One-DM           data_loader.loader.letters
  ductus           ductus.markup.WRITABLE (model glyphs) + marks drawn by ductus.pen (= < > % ~ arrows)
  font tools       the font's cmap (fontTools)
A character counts as covered only if the tool writes it as that character; ductus' ASCII
fallbacks (@ -> 'at', $ -> 'USD') and tools that silently drop a character count as not covered.
  -> out/charset.json
"""
import json, string, sys
from pathlib import Path
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
CATS = {
    "lowercase": string.ascii_lowercase,
    "capitals": string.ascii_uppercase,
    "digits": string.digits,
    "punctuation": ".,;:!?'\"()-",
    "symbols": "%=+@&#/*<>^$",
    "maths_unicode": "\u221a\u03a3\u00b2\u00bd\u2192\u00d7\u00f7\u2260\u2264\u03c0",  # sqrt Sigma ^2 1/2 arrow x / != <= pi
}
GRZEGO = set(' !"#%&\'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
SJV = set('\x00 !"#\'(),-.0123456789:;?ABCDEFGHIJKLMNOPRSTUVWYabcdefghijklmnopqrstuvwxyz')
ONEDM = set('_Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%')
DUCTUS_WRITABLE = set(" _!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
DUCTUS_MARKS = set("=<>%~")  # drawn as pen strokes; arrows via '->'; sqrt/sigma/fraction/superscript via markup
FONTS = {
    "tth_homemade": "/usr/share/fonts/truetype/sand-box/google/Homemade Apple/HomemadeApple-Regular.ttf",
    "tth_caveat": "/usr/share/fonts/truetype/sand-box/google/Caveat/Caveat-VariableFont_wght.ttf",
    "tth_liujian": "/usr/share/fonts/truetype/sand-box/google/Liu Jian Mao Cao/LiuJianMaoCao-Regular.ttf",
}
sets = {"rnn_grzego": GRZEGO, "hws_sjvasquez": SJV, "onedm": ONEDM, "ductus": DUCTUS_WRITABLE | DUCTUS_MARKS}
for k, f in FONTS.items():
    sets[k] = {chr(c) for c in TTFont(f).getBestCmap()}
res = {}
for name, s in sets.items():
    r = {}
    for cat, chars in CATS.items():
        have = [c for c in chars if c in s]
        r[cat] = dict(covered=len(have), of=len(chars), missing="".join(c for c in chars if c not in s))
    res[name] = r
# maths constructs a page needs, beyond single characters (from each tool's documented input)
res["_maths_constructs"] = {
    "ductus": "stacked fraction {a/b}, radical sqrt(x), sigma {sum}, superscript x^2, circled answer, aligned '=' lines (docs/CLI.md)",
    "fonts": "only what the font has as Unicode characters (see maths_unicode); no stacked fractions or radicals over expressions",
    "rnn_grzego / hws_sjvasquez / onedm": "none; plain text line only",
}
(ROOT / "out/charset.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
for n, r in res.items():
    if not n.startswith("_"):
        print(f"{n:15s}", "  ".join(f"{c}:{v['covered']}/{v['of']}" for c, v in r.items()))

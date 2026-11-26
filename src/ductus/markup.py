"""Plain text with light markup -> a list of laid-out-able blocks.

Block syntax (one per line)
  # Heading            heading, drawn larger with a hand rule under it
  ## Sub heading       smaller heading, short underline
  ---                  hand-drawn separator across the page
  Term :: meaning      definition (notes mode also accepts `Term: meaning`)
  Note: text           written as "Note ->" followed by the text
  1) item / 2. item    numbered list item (hanging indent)
  - item               bullet item
  | a | b | c |        table row; consecutive rows form one table with ruled columns
  = expr               maths continuation line, aligned under the previous "="
  (blank line)         paragraph break

Inline syntax
  ->  <-  =>           arrows (drawn, not written)
  ~~words~~            struck through
  __words__            underlined
  ((words))            circled (final answers)
  [v]  [x]             tick / cross mark
  {a/b}                stacked fraction
  sqrt(x)              radical sign over x
  {sum}                capital sigma
  ^2                   superscript (one token)
  {oops:word}          write word, cross it out, keep writing (a correction)
  =  <  >  %  ~        drawn as pen marks (the model has no glyphs for these)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# characters the diffusion engine can write directly
WRITABLE = set(" _!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

FALLBACK = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
    "\u2026": "...", "\t": " ", "`": "'", "@": " at ", "$": "USD ", "[": "(", "]": ")",
    "|": "/", "\\": "/", "\u00d7": "x", "\u2212": "-",
}


@dataclass
class Word:
    text: str
    deco: frozenset = frozenset()      # {"strike", "underline", "circle", "oops"}


@dataclass
class Sym:
    kind: str                          # arrow, larrow, darrow, tick, cross, eq, lt, gt, pct, tilde, sigma
    args: tuple = ()                   # frac -> (num, den), sqrt -> (inner,), sup -> (text,)


@dataclass
class Block:
    kind: str                          # para, heading, subheading, rule, defn, note, item, bullet, table, math, gap
    tokens: list = field(default_factory=list)   # list[Word|Sym]
    label: list = field(default_factory=list)    # defn term / list number tokens
    cells: list = field(default_factory=list)    # table: list of rows, each list of token lists


def clean_text(s: str) -> str:
    out = []
    for ch in s:
        ch = FALLBACK.get(ch, ch)
        out.append(ch)
    return "".join(out)


_INLINE = re.compile(
    r"(\{oops:[^}]+\}|~~.+?~~|__.+?__|\(\(.+?\)\)|\[v\]|\[x\]|\{[^{}/]+/[^{}]+\}|sqrt\([^)]*\)|\{sum\}"
    r"|\^\S+|->|<-|=>|=|<|>|%|~)"
)

_SYM = {"->": "arrow", "<-": "larrow", "=>": "darrow", "[v]": "tick", "[x]": "cross", "=": "eq",
        "<": "lt", ">": "gt", "%": "pct", "~": "tilde", "{sum}": "sigma"}


def _words(text: str, deco=frozenset()) -> list:
    text = clean_text(text)
    out = []
    for w in text.split():
        w = "".join(c for c in w if c in WRITABLE)
        if w:
            out.append(Word(w, deco))
    return out


def parse_inline(text: str) -> list:
    """Split one line of text into Word and Sym tokens."""
    toks: list = []
    pos = 0
    for m in _INLINE.finditer(text):
        toks += _words(text[pos:m.start()])
        s = m.group(0)
        pos = m.end()
        if s in _SYM:
            toks.append(Sym(_SYM[s]))
        elif s.startswith("{oops:"):
            toks += _words(s[6:-1], frozenset({"oops"}))
        elif s.startswith("~~"):
            toks += _words(s[2:-2], frozenset({"strike"}))
        elif s.startswith("__"):
            toks += _words(s[2:-2], frozenset({"underline"}))
        elif s.startswith("(("):
            inner = parse_inline(s[2:-2])
            toks.append(Sym("circle", (inner,)))
        elif s.startswith("sqrt("):
            toks.append(Sym("sqrt", (parse_inline(s[5:-1]),)))
        elif s.startswith("^"):
            toks.append(Sym("sup", (parse_inline(s[1:]),)))
        elif s.startswith("{") and "/" in s:
            num, den = s[1:-1].split("/", 1)
            toks.append(Sym("frac", (parse_inline(num), parse_inline(den))))
    toks += _words(text[pos:])
    return toks


_ITEM = re.compile(r"^\s*(\d{1,2}|[a-z])([).])\s+(.*)$")
_DEFN = re.compile(r"^\s*([^:|]{1,40}?)\s*::\s*(.+)$")
_DEFN_LOOSE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 '()-]{0,38}?)\s*:\s+(.+)$")
_NOTE = re.compile(r"^\s*(note|nb|n\.b\.)\s*[:\-]+>?\s*(.+)$", re.I)


def parse(text: str, mode: str = "plain") -> list[Block]:
    """Parse a document. mode: plain | notes | maths."""
    blocks: list[Block] = []
    lines = text.replace("\r", "").split("\n")
    table: list = []

    def flush_table():
        if table:
            blocks.append(Block("table", cells=list(table)))
            table.clear()

    for raw in lines:
        line = raw.rstrip()
        s = line.strip()
        if s.startswith("|") and s.endswith("|") and len(s) > 2:
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue  # markdown separator row
            table.append([parse_inline(c) for c in cells])
            continue
        flush_table()
        if not s:
            blocks.append(Block("gap"))
        elif s == "---" or s == "***":
            blocks.append(Block("rule"))
        elif s.startswith("## "):
            blocks.append(Block("subheading", parse_inline(s[3:])))
        elif s.startswith("# "):
            blocks.append(Block("heading", parse_inline(s[2:])))
        elif (m := _NOTE.match(s)) and mode != "plain" or (m := _NOTE.match(s)) and s.lower().startswith("note"):
            blocks.append(Block("note", parse_inline(m.group(2)), label=_words("Note") + [Sym("arrow")]))
        elif m := _DEFN.match(s):
            blocks.append(Block("defn", parse_inline(m.group(2)), label=parse_inline(m.group(1))))
        elif mode == "notes" and (m := _DEFN_LOOSE.match(s)):
            blocks.append(Block("defn", parse_inline(m.group(2)), label=parse_inline(m.group(1))))
        elif m := _ITEM.match(s):
            blocks.append(Block("item", parse_inline(m.group(3)), label=_words(m.group(1) + ")")))
        elif s.startswith("- ") or s.startswith("* "):
            blocks.append(Block("bullet", parse_inline(s[2:])))
        elif mode == "maths" and s.startswith("="):
            blocks.append(Block("math", [Sym("eq")] + parse_inline(s[1:])))
        else:
            blocks.append(Block("para", parse_inline(s)))
    flush_table()
    # collapse runs of gaps
    out: list[Block] = []
    for b in blocks:
        if b.kind == "gap" and (not out or out[-1].kind == "gap"):
            continue
        out.append(b)
    while out and out[-1].kind == "gap":
        out.pop()
    return out


def iter_words(tokens):
    """All Word objects inside a token list, recursing into fractions, roots, circles."""
    for t in tokens:
        if isinstance(t, Word):
            yield t
        elif isinstance(t, Sym):
            for a in t.args:
                yield from iter_words(a)


def block_words(b: Block):
    yield from iter_words(b.label)
    yield from iter_words(b.tokens)
    for row in b.cells:
        for cell in row:
            yield from iter_words(cell)

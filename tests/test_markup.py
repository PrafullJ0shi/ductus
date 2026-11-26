from ductus.markup import parse, parse_inline, Word, Sym, block_words


def kinds(toks):
    return [t.kind if isinstance(t, Sym) else t.text for t in toks]


def test_inline_arrows_and_marks():
    assert kinds(parse_inline("Mean -> average [v]")) == ["Mean", "arrow", "average", "tick"]
    assert kinds(parse_inline("x = 2 [x]")) == ["x", "eq", "2", "cross"]


def test_inline_decorations():
    t = parse_inline("~~wrong word~~ __key__")
    assert [w.deco for w in t] == [frozenset({"strike"})] * 2 + [frozenset({"underline"})]


def test_fraction_sqrt_sup_circle():
    t = parse_inline("{a+b/2} sqrt(16) x^2 ((42))")
    assert [s.kind for s in t if isinstance(s, Sym)] == ["frac", "sqrt", "sup", "circle"]
    frac = t[0]
    assert [w.text for w in frac.args[0]] == ["a+b"]
    assert [w.text for w in frac.args[1]] == ["2"]


def test_oops_marks_correction():
    t = parse_inline("the {oops:mena} mean")
    assert t[1].deco == frozenset({"oops"}) and t[2].text == "mean"


def test_unsupported_characters_fall_back():
    t = parse_inline("email me @ home for $5")
    assert "at" in [w.text for w in t if isinstance(w, Word)]
    assert all(c not in "@$" for w in t if isinstance(w, Word) for c in w.text)


def test_blocks_notes_mode():
    doc = "# Title\nRange :: max minus min\nVariance: spread\nNote: be careful\n1) first\n- dash\n---\n| a | b |\n| 1 | 2 |"
    b = parse(doc, mode="notes")
    assert [x.kind for x in b] == ["heading", "defn", "defn", "note", "item", "bullet", "rule", "table"]
    assert len(b[-1].cells) == 2


def test_plain_mode_keeps_colon_text():
    b = parse("Time: 10 am", mode="plain")
    assert b[0].kind == "para"


def test_maths_continuation():
    b = parse("x = 3 + 4\n= 7", mode="maths")
    assert b[1].kind == "math" and b[1].tokens[0].kind == "eq"


def test_word_order_matches_layout_order():
    b = parse("1) alpha {beta/gamma} delta", mode="notes")[0]
    assert [w.text for w in block_words(b)] == ["1)", "alpha", "beta", "gamma", "delta"]


def test_gaps_collapse():
    b = parse("a\n\n\n\nb")
    assert [x.kind for x in b] == ["para", "gap", "para"]

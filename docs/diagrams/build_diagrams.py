"""Build the pipeline diagram (light and dark) as self-contained HTML with inline SVG.
Follows the diagram-design rules: orthogonal connectors, labels on masks, 4 px grid,
at most 9 nodes and 2 accent elements."""
from pathlib import Path

HERE = Path(__file__).parent
SKINS = {
    "light": dict(paper="#FBFAF7", ink="#1D1F24", muted="#5C6170", soft="#8A8F9C", rule="#D9D6CE", accent="#1C2A6E", fill="#FFFFFF", tint="rgba(28,42,110,0.06)"),
    "dark": dict(paper="#15171C", ink="#E8E6E1", muted="#A3A7B3", soft="#7D8290", rule="#3A3D45", accent="#8C9BE0", fill="#1C1F26", tint="rgba(140,155,224,0.10)"),
}
W, H = 176, 64
NODES = {  # key: (x, y, name, sub, tag, accent)
    "style": (496, 40, "Style reference", "preset or profile", "IN", False),
    "text": (40, 176, "Text and markup", "plain, notes, maths", "IN", False),
    "carr": (268, 176, "Carrier lines", "46 chars, filler words", "STEP", False),
    "diff": (496, 176, "DiffBrush", "one-shot line diffusion", "MODEL", True),
    "read": (724, 176, "Cut and read back", "TrOCR, word alignment", "CHECK", True),
    "pen": (40, 432, "Pen marks", "arrows, ticks, roots", "STEP", False),
    "lay": (268, 432, "Human layout", "drift, jitter, margins", "STEP", False),
    "ren": (496, 432, "Ink and paper", "pressure, bleed, paper", "STEP", False),
    "out": (724, 432, "Outputs", "PNG, PDF, SVG, animated", "OUT", False),
}


def node(k, c):
    x, y, name, sub, tag, acc = NODES[k]
    stroke = c["accent"] if acc else c["muted"]
    fill = c["tint"] if acc else c["fill"]
    tw = 8 + 6 * len(tag)
    return f'''<g id="n-{k}">
<rect x="{x}" y="{y}" width="{W}" height="{H}" rx="6" fill="{c['paper']}"/>
<rect x="{x}" y="{y}" width="{W}" height="{H}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="{1.4 if acc else 1}"/>
<rect x="{x + 8}" y="{y + 8}" width="{tw}" height="12" rx="2" fill="none" stroke="{stroke}" stroke-opacity="0.45" stroke-width="0.8"/>
<text x="{x + 8 + tw / 2}" y="{y + 17}" fill="{stroke}" font-size="7" font-family="'IBM Plex Mono', monospace" text-anchor="middle" letter-spacing="0.08em">{tag}</text>
<text x="{x + W / 2}" y="{y + 38}" fill="{c['ink']}" font-size="15" font-weight="600" font-family="Newsreader, Georgia, serif" text-anchor="middle" style="font-variant-ligatures:none">{name}</text>
<text x="{x + W / 2}" y="{y + 54}" fill="{c['muted']}" font-size="9.5" font-family="'IBM Plex Mono', monospace" text-anchor="middle">{sub}</text>
</g>'''


def label(x, y, text, c, w=None):
    w = w or 10 + 7 * len(text)
    return (f'<rect x="{x - w / 2}" y="{y}" width="{w}" height="12" rx="2" fill="{c["paper"]}"/>'
            f'<text x="{x}" y="{y + 9}" fill="{c["soft"]}" font-size="8" font-family="\'IBM Plex Mono\', monospace" text-anchor="middle" letter-spacing="0.06em">{text}</text>')


def svg(slug, c):
    m, ma = 'marker-end="url(#a)"', 'marker-end="url(#aa)"'
    arrows = [
        f'<path d="M584 104 V174" fill="none" stroke="{c["muted"]}" stroke-width="1.2" {m}/>',
        f'<path d="M216 208 H266" fill="none" stroke="{c["muted"]}" stroke-width="1.2" {m}/>',
        f'<path d="M444 208 H494" fill="none" stroke="{c["muted"]}" stroke-width="1.2" {m}/>',
        f'<path d="M672 208 H722" fill="none" stroke="{c["accent"]}" stroke-width="1.4" {ma}/>',
        # retry loop: read-back bottom -> carrier bottom (dashed)
        f'<path d="M784 240 V312 Q784 320 776 320 H340 Q332 320 332 312 V242" fill="none" stroke="{c["accent"]}" stroke-width="1.2" stroke-dasharray="5,4" {ma}/>',
        # pass: read-back bottom -> layout top
        f'<path d="M840 240 V360 Q840 368 832 368 H388 Q380 368 380 376 V430" fill="none" stroke="{c["muted"]}" stroke-width="1.2" {m}/>',
        f'<path d="M128 240 V430" fill="none" stroke="{c["muted"]}" stroke-width="1.2" {m}/>',
        f'<path d="M216 464 H266" fill="none" stroke="{c["muted"]}" stroke-width="1.2" {m}/>',
        f'<path d="M444 464 H494" fill="none" stroke="{c["muted"]}" stroke-width="1.2" {m}/>',
        f'<path d="M672 464 H722" fill="none" stroke="{c["muted"]}" stroke-width="1.2" {m}/>',
    ]
    labels = [label(560, 300, "REWRITE FAILED WORDS", c), label(608, 348, "WORDS THAT READ BACK", c),
              f'<rect x="136" y="324" width="64" height="12" rx="2" fill="{c["paper"]}"/>'
              f'<text x="140" y="333" fill="{c["soft"]}" font-size="8" font-family="\'IBM Plex Mono\', monospace" letter-spacing="0.06em">SYMBOLS</text>']
    legend = (f'<line x1="40" y1="556" x2="72" y2="556" stroke="{c["accent"]}" stroke-width="1.4"/>'
              f'<text x="80" y="559" fill="{c["muted"]}" font-size="10" font-family="\'IBM Plex Mono\', monospace">accent: model step and the check that replaces seed picking</text>'
              f'<line x1="520" y1="556" x2="552" y2="556" stroke="{c["accent"]}" stroke-width="1.2" stroke-dasharray="5,4"/>'
              f'<text x="560" y="559" fill="{c["muted"]}" font-size="10" font-family="\'IBM Plex Mono\', monospace">dashed: retry loop, up to 3 rounds</text>')
    nodes = "\n".join(node(k, c) for k in NODES)
    return f'''<svg viewBox="0 0 960 600" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="{slug}-title {slug}-desc">
<title id="{slug}-title">ductus pipeline: text to handwritten page</title>
<desc id="{slug}-desc">Text is packed into carrier lines, written by a one-shot diffusion model in the reference style, read back by OCR with failed words rewritten, then laid out with drawn pen marks and rendered as ink on paper.</desc>
<defs>
<marker id="a" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{c['muted']}"/></marker>
<marker id="aa" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{c['accent']}"/></marker>
</defs>
<rect width="100%" height="100%" fill="{c['paper']}"/>
{chr(10).join(arrows)}
{chr(10).join(labels)}
{nodes}
{legend}
</svg>'''


def page(slug, c, title):
    return f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:Newsreader,Georgia,serif;background:{c['paper']};color:{c['ink']};min-height:100vh;display:flex;align-items:center;justify-content:center;padding:3rem 2rem}}
.frame{{max-width:1200px;width:100%}}
.diagram-container{{width:100%;overflow-x:auto}}
.eyebrow{{font-family:'IBM Plex Mono',monospace;font-size:.66rem;letter-spacing:.18em;text-transform:uppercase;color:{c['muted']};margin-bottom:.5rem}}
h1{{font-weight:400;font-size:2rem;letter-spacing:-.01em;margin-bottom:1.5rem}}
svg{{width:100%;min-width:960px;display:block}}
@media print{{.diagram-container{{overflow-x:visible}}svg{{min-width:0}}}}
</style></head><body><div class="frame">
<p class="eyebrow">Flowchart · ductus</p><h1>{title}</h1>
<div class="diagram-container">
{svg(slug, c)}
</div></div></body></html>'''


for skin, c in SKINS.items():
    slug = "pipeline" if skin == "light" else "pipeline-dark"
    (HERE / f"{slug}.html").write_text(page(slug, c, "From text to a handwritten page"))
print("ok")

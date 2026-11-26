"""Build the architecture diagram (light and dark) as standalone SVG and as HTML around it.

Follows the diagram-design rules and DESIGN.md: orthogonal connectors only, at most nine
nodes, one accent colour on at most two elements, no shadows or glows, labels on opaque
masks, 4 px grid. The SVG files are what the docs embed; the HTML files are the same SVG in
a page, for the diagram kit's checks.

  python docs/diagrams/build_architecture.py
"""
from pathlib import Path

HERE = Path(__file__).parent
SKINS = {
    "light": dict(paper="#FBFAF7", ink="#1D1F24", muted="#5C6170", soft="#6F7482", rule="#D9D6CE", accent="#1C2A6E", fill="#FFFFFF", tint="rgba(28,42,110,0.06)"),
    "dark": dict(paper="#15171C", ink="#E8E6E1", muted="#A3A7B3", soft="#8E93A1", rule="#3A3D45", accent="#8C9BE0", fill="#1C1F26", tint="rgba(140,155,224,0.10)"),
}
W, H = 176, 64
NODES = {  # key: (x, y, name, sub, tag, accent, dashed)
    "text": (40, 112, "Text and markup", "plain, notes, maths", "IN", False, False),
    "rnn": (268, 112, "RNN writer", "plain print letters", "MODEL", False, False),
    "read": (496, 112, "Read-back check", "OCR and letter shapes", "CHECK", True, False),
    "lay": (724, 112, "Layout and marks", "drift, margins, pen marks", "STEP", False, False),
    "prof": (268, 288, "Style profile", "optional, from photos", "IN", False, True),
    "diff": (496, 288, "Diffusion writer", "fallback, copies a style", "MODEL", False, False),
    "ink": (724, 288, "Ink on paper", "pen model at 600 dpi", "STEP", False, False),
    "out": (496, 464, "Outputs", "PNG, PDF, SVG", "OUT", False, False),
    "photo": (724, 464, "Phone photo", "skipped without --photo", "STEP", False, True),
}


def node(k, c):
    x, y, name, sub, tag, acc, dashed = NODES[k]
    stroke = c["accent"] if acc else c["muted"]
    fill = c["tint"] if acc else c["fill"]
    tw = 8 + 6 * len(tag)
    dash = ' stroke-dasharray="4,3"' if dashed else ""
    return f'''<g id="n-{k}">
<rect x="{x}" y="{y}" width="{W}" height="{H}" rx="6" fill="{c['paper']}"/>
<rect x="{x}" y="{y}" width="{W}" height="{H}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="{1.4 if acc else 1}"{dash}/>
<rect x="{x + 8}" y="{y + 8}" width="{tw}" height="12" rx="2" fill="none" stroke="{stroke}" stroke-opacity="0.45" stroke-width="0.8"/>
<text x="{x + 8 + tw / 2}" y="{y + 17}" fill="{stroke}" font-size="7" font-family="'IBM Plex Mono', monospace" text-anchor="middle" letter-spacing="0.08em">{tag}</text>
<text x="{x + W / 2}" y="{y + 38}" fill="{c['ink']}" font-size="15" font-weight="600" font-family="Newsreader, Georgia, serif" text-anchor="middle" style="font-variant-ligatures:none">{name}</text>
<text x="{x + W / 2}" y="{y + 54}" fill="{c['muted']}" font-size="9.5" font-family="'IBM Plex Mono', monospace" text-anchor="middle">{sub}</text>
</g>'''


def label(cx, y, text, c, anchor="middle"):
    w = 10 + 7 * len(text)
    x0 = cx - w / 2 if anchor == "middle" else (cx if anchor == "start" else cx - w)
    tx = x0 + w / 2
    return (f'<rect x="{x0}" y="{y}" width="{w}" height="12" rx="2" fill="{c["paper"]}"/>'
            f'<text x="{tx}" y="{y + 9}" fill="{c["soft"]}" font-size="8" font-family="\'IBM Plex Mono\', monospace" text-anchor="middle" letter-spacing="0.06em">{text}</text>')


def svg(slug, c):
    m, ma = 'marker-end="url(#a)"', 'marker-end="url(#aa)"'
    mu = c["muted"]
    ac = c["accent"]
    arrows = [
        # text -> RNN writer
        f'<path d="M216 144 H266" fill="none" stroke="{mu}" stroke-width="1.2" {m}/>',
        # RNN writer -> read-back
        f'<path d="M444 144 H494" fill="none" stroke="{mu}" stroke-width="1.2" {m}/>',
        # retry loop: read-back top -> RNN writer top (dashed, return)
        f'<path d="M584 110 V88 Q584 80 576 80 H364 Q356 80 356 88 V110" fill="none" stroke="{ac}" stroke-width="1.2" stroke-dasharray="5,4" {ma}/>',
        # read-back -> layout
        f'<path d="M672 144 H722" fill="none" stroke="{ac}" stroke-width="1.4" {ma}/>',
        # words that still fail go down to the diffusion writer
        f'<path d="M556 176 V286" fill="none" stroke="{mu}" stroke-width="1.2" {m}/>',
        # diffusion words come back to be read (return, dashed)
        f'<path d="M612 286 V178" fill="none" stroke="{mu}" stroke-width="1.2" stroke-dasharray="5,4" {m}/>',
        # style profile -> diffusion writer
        f'<path d="M444 320 H494" fill="none" stroke="{mu}" stroke-width="1.2" stroke-dasharray="5,4" {m}/>',
        # layout -> ink
        f'<path d="M812 176 V286" fill="none" stroke="{mu}" stroke-width="1.2" {m}/>',
        # ink -> phone photo
        f'<path d="M812 352 V462" fill="none" stroke="{mu}" stroke-width="1.2" {m}/>',
        # phone photo -> outputs
        f'<path d="M722 496 H674" fill="none" stroke="{mu}" stroke-width="1.2" {m}/>',
    ]
    labels = [
        label(470, 58, "REWRITE", c),
        label(548, 224, "STILL FAILING", c, anchor="end"),
        label(620, 224, "REWRITTEN", c, anchor="start"),
    ]
    legend = (f'<line x1="40" y1="584" x2="72" y2="584" stroke="{ac}" stroke-width="1.4"/>'
              f'<text x="80" y="587" fill="{mu}" font-size="10" font-family="\'IBM Plex Mono\', monospace">accent: the check every word must pass</text>'
              f'<line x1="400" y1="584" x2="432" y2="584" stroke="{mu}" stroke-width="1.2" stroke-dasharray="5,4"/>'
              f'<text x="440" y="587" fill="{mu}" font-size="10" font-family="\'IBM Plex Mono\', monospace">dashed: optional or returning</text>')
    nodes = "\n".join(node(k, c) for k in NODES)
    return f'''<svg viewBox="0 0 960 624" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="{slug}-title {slug}-desc">
<title id="{slug}-title">ductus architecture</title>
<desc id="{slug}-desc">Text is written word by word by a plain-letter RNN, every word is read back by OCR and rewritten if it fails, words that still fail go to a diffusion writer, and the words are then laid out, inked, optionally photographed and saved.</desc>
<defs>
<marker id="a" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{mu}"/></marker>
<marker id="aa" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{ac}"/></marker>
</defs>
<rect width="960" height="624" fill="{c['paper']}"/>
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
<p class="eyebrow">Architecture · ductus</p><h1>{title}</h1>
<div class="diagram-container">
{svg(slug, c)}
</div></div></body></html>'''


for skin, c in SKINS.items():
    slug = "architecture" if skin == "light" else "architecture-dark"
    (HERE / f"{slug}.svg").write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + svg(slug, c) + "\n")
    (HERE / f"{slug}.html").write_text(page(slug, c, "How ductus writes a page"))
print("ok")

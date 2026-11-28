"""Build the read-back cascade diagram (light and dark) as standalone SVG and as HTML around it.

Follows the diagram-design rules and DESIGN.md: orthogonal connectors only, at most nine
nodes, one accent colour on at most two elements, no shadows or glows, labels on opaque
masks, 4 px grid. The SVG files are what the docs embed; the HTML files are the same SVG in
a page, for the diagram kit's checks.

  python docs/diagrams/build_readback.py
"""
from pathlib import Path

HERE = Path(__file__).parent
SKINS = {
    "light": dict(paper="#FBFAF7", ink="#1D1F24", muted="#5C6170", soft="#6F7482", rule="#D9D6CE", accent="#1C2A6E", fill="#FFFFFF", tint="rgba(28,42,110,0.06)"),
    "dark": dict(paper="#15171C", ink="#E8E6E1", muted="#A3A7B3", soft="#8E93A1", rule="#3A3D45", accent="#8C9BE0", fill="#1C1F26", tint="rgba(140,155,224,0.10)"),
}
W, H = 176, 64
NODES = {  # key: (x, y, name, sub, tag, accent, dashed)
    "bank": (40, 112, "Word bank", "verified in earlier runs", "IN", False, True),
    "ok": (744, 112, "Accepted word", "goes on the page", "OUT", False, False),
    "word": (40, 288, "New word", "cut from an RNN line", "IN", False, False),
    "small": (248, 288, "trocr-small", "reads every word", "CHECK", False, False),
    "base": (520, 288, "trocr-base", "re-reads the rejects", "CHECK", True, False),
    "redo": (520, 464, "Rewrite round", "new carrier, best of 3", "MODEL", False, False),
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
    mu, ac = c["muted"], c["accent"]
    arrows = [
        f'<path d="M216 320 H246" fill="none" stroke="{mu}" stroke-width="1.2" {m}/>',
        f'<path d="M424 320 H518" fill="none" stroke="{ac}" stroke-width="1.4" {ma}/>',
        f'<path d="M336 286 V164 Q336 156 344 156 H742" fill="none" stroke="{mu}" stroke-width="1.2" {m}/>',
        f'<path d="M216 132 H742" fill="none" stroke="{mu}" stroke-width="1.2" stroke-dasharray="5,4" {m}/>',
        f'<path d="M696 320 H824 Q832 320 832 312 V178" fill="none" stroke="{ac}" stroke-width="1.4" {ma}/>',
        f'<path d="M608 352 V462" fill="none" stroke="{mu}" stroke-width="1.2" {m}/>',
        f'<path d="M518 496 H136 Q128 496 128 488 V354" fill="none" stroke="{mu}" stroke-width="1.2" stroke-dasharray="5,4" {m}/>',
        f'<path d="M832 110 V88 Q832 80 824 80 H136 Q128 80 128 88 V110" fill="none" stroke="{mu}" stroke-width="1.2" stroke-dasharray="5,4" {m}/>',
    ]
    labels = [
        label(470, 74, "SAVED", c),
        label(470, 126, "BANKED", c),
        label(560, 150, "READS", c),
        label(472, 296, "REJECTED", c),
        label(764, 296, "READS", c),
        label(616, 400, "REJECTED", c, anchor="start"),
        label(310, 490, "NEW CARRIER", c),
    ]
    legend = (f'<line x1="40" y1="584" x2="72" y2="584" stroke="{ac}" stroke-width="1.4"/>'
              f'<text x="80" y="587" fill="{mu}" font-size="10" font-family="\'IBM Plex Mono\', monospace">accent: the second read, only for rejected words</text>'
              f'<line x1="440" y1="584" x2="472" y2="584" stroke="{mu}" stroke-width="1.2" stroke-dasharray="5,4"/>'
              f'<text x="480" y="587" fill="{mu}" font-size="10" font-family="\'IBM Plex Mono\', monospace">dashed: word bank or returning</text>')
    nodes = "\n".join(node(k, c) for k in NODES)
    return f'''<svg viewBox="0 0 960 624" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="{slug}-title {slug}-desc">
<title id="{slug}-title">ductus read-back cascade</title>
<desc id="{slug}-desc">Each new word is read by trocr-small; words it rejects are re-read by trocr-base; words both reject are written again in a new carrier line, and accepted words go on the page and into the word bank.</desc>
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
<p class="eyebrow">Read-back · ductus</p><h1>{title}</h1>
<div class="diagram-container">
{svg(slug, c)}
</div></div></body></html>'''


for skin, c in SKINS.items():
    slug = "readback" if skin == "light" else "readback-dark"
    (HERE / f"{slug}.svg").write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + svg(slug, c) + "\n")
    (HERE / f"{slug}.html").write_text(page(slug, c, "How one word is checked"))
print("ok")

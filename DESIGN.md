# Design

Ductus is a technical project and its pages should read like a short research report:
plain, exact and calm. The handwriting is the only expressive element; everything around
it stays out of the way.

## Principles

1. Evidence before decoration. Every figure is a real output, a measurement, or a diagram
   that explains a step. No stock imagery, no illustrations, no mock-ups.
2. One accent colour, used for one thing: the ink. Charts and diagrams are drawn in
   graphite tones and the accent marks only the element a reader should look at first.
3. Same text, same settings. Comparisons always use identical input text and say how
   each image was made.
4. Numbers are measured, never estimated. Every number in the docs links to the script
   that produced it.

## Tokens

| Role | Value | Use |
|---|---|---|
| paper | `#FBFAF7` | page and diagram background |
| paper-2 | `#F2F0EA` | table stripes, containers |
| ink | `#1D1F24` | body text, strokes |
| muted | `#5C6170` | secondary text, default connectors |
| rule | `#D9D6CE` | hairlines |
| accent | `#1C2A6E` | ballpoint blue: focal element only, at most two per figure |
| dark paper | `#15171C` | dark-mode diagram background |
| dark ink | `#E8E6E1` | dark-mode text |
| dark accent | `#8C9BE0` | accent on dark backgrounds |

The default ink colour of generated pages (`#1C2A6E`) is the same blue, so the accent in
diagrams always means "this is where the ink goes".

## Type

Two families only.

- **Newsreader** (serif, OFL): titles, body text on the demo page, diagram titles and node
  names. Optical sizes make it work at both 11 px and 40 px.
- **IBM Plex Mono** (OFL): commands, file names, numbers in tables, diagram sublabels.

Not used: Inter, Roboto, Space Grotesk, any display or "handwriting" font. A handwriting
font next to real model output would undermine the point of the project.

Scale (px): 12 / 14 / 16 / 20 / 28 / 40. Line length on the demo page: 68 characters max.

## Figures

- Diagrams follow the diagram-design kit: orthogonal connectors only, at most nine nodes,
  no shadows, no gradients, no glows, radius 6 px or less, labels on opaque masks,
  4 px grid. Each diagram ships as HTML (source of truth) and as light and dark PNGs.
- README images use `<picture>` with `prefers-color-scheme` so diagrams read in both
  GitHub themes. Handwriting samples are on white paper and need no dark variant.
- Comparison figures are grids with a caption row in Plex Mono: system, settings, time.

## Writing

- Short sentences. No marketing adjectives. No emoji.
- Say what the system cannot do in the same place you say what it can.
- Credit prior work by name, with links, next to the place it is used.

## Repository layout

```
src/ductus/          package (engine, synthesis, layout, pen marks, render, profile)
docs/                method, architecture, evaluation, comparison, guides, diagrams
examples/            input texts with markup
samples/             generated pages (PNG/PDF/SVG) referenced by the README
scripts/             evaluation and showcase scripts that produce every number in docs
tests/               unit tests for everything that does not need the model
```

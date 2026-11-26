"""Check internal links in Markdown files.

Usage: python scripts/check_links.py [root]

Checks relative links and images, and #anchors (GitHub-style heading slugs).
External http(s) links are listed, not fetched.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

LINK = re.compile(r"(?<!\\)!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HTML_REF = re.compile(r"""(?:src|srcset|href)=["']([^"']+)["']""")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
SKIP_DIRS = {".git", ".venv", "node_modules", ".pytest_cache", "src/ductus.egg-info"}


def slug(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    out: set[str] = set()
    fence = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        m = HEADING.match(line)
        if m:
            s = slug(m.group(2))
            n = seen.get(s, 0)
            seen[s] = n + 1
            out.add(s if n == 0 else f"{s}-{n}")
    return out


def md_files(root: Path):
    for p in sorted(root.rglob("*.md")):
        rel = p.relative_to(root).as_posix()
        if any(rel == d or rel.startswith(d + "/") or f"/{d}/" in f"/{rel}" for d in SKIP_DIRS):
            continue
        yield p


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    cache: dict[Path, set[str]] = {}
    bad = 0
    ext = 0
    for md in md_files(root):
        text = md.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        fence = False
        for n, line in enumerate(lines, 1):
            if line.lstrip().startswith("```"):
                fence = not fence
                continue
            if fence:
                continue
            line_nocode = re.sub(r"`[^`]*`", "", line)
            refs = LINK.findall(line_nocode) + HTML_REF.findall(line_nocode)
            for ref in refs:
                if ref.startswith(("http://", "https://", "mailto:")):
                    ext += 1
                    continue
                target, _, frag = ref.partition("#")
                if target == "":
                    dest = md
                else:
                    dest = (md.parent / target).resolve()
                if not dest.exists():
                    print(f"{md.relative_to(root)}:{n}: missing file {ref}")
                    bad += 1
                    continue
                if frag and dest.suffix == ".md":
                    if dest not in cache:
                        cache[dest] = anchors(dest)
                    if frag.lower() not in cache[dest]:
                        print(f"{md.relative_to(root)}:{n}: missing anchor {ref}")
                        bad += 1
    print(f"{bad} broken internal link(s); {ext} external link(s) not fetched")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

"""Drive the real text-to-handwriting web app (saurabhdaware/text-to-handwriting, MIT),
served locally from a clone, in headless Chrome. One generated image per test line.
Settings: site defaults except paper lines/margin switched off (an option the site offers)
and effect 'no-effect', so OCR sees ink on plain paper like the other systems.
  python run_text_to_handwriting.py FONT_LABEL FONT_VALUE OUTDIR
"""
import base64, json, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

label, value, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
lines = Path(__file__).resolve().parents[1].joinpath("texts/plain.txt").read_text().splitlines()
times = []
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1300, "height": 1800})
    pg.goto("http://localhost:8799/index.html", wait_until="networkidle")
    pg.select_option("#handwriting-font", value)
    pg.select_option("#page-effects", "no-effect")
    pg.select_option("#resolution", "4")
    for tog in ["#paper-margin-toggle", "#paper-line-toggle"]:
        if pg.is_checked(tog):
            pg.locator(tog).evaluate("e => e.click()")
    pg.wait_for_timeout(1500)
    for i, t in enumerate(lines):
        pg.evaluate("() => { document.querySelectorAll('#output img').forEach(()=>0); }")
        pg.evaluate("t => { document.querySelector('#note').innerText = t; }", t)
        t0 = time.time()
        n0 = pg.evaluate("() => document.querySelectorAll('#output img').length")
        pg.click("text=Generate Image")
        pg.wait_for_function(f"() => document.querySelectorAll('#output img').length > {n0}")
        times.append(time.time() - t0)
        src = pg.evaluate("() => { const a=document.querySelectorAll('#output img'); return a[a.length-1].src; }")
        (out / f"{i:03d}_page.png").write_bytes(base64.b64decode(src.split(",", 1)[1]))
    b.close()
json.dump({"seconds_per_line": sum(times) / len(times), "tool": "text-to-handwriting (saurabhdaware, MIT, commit at clone)",
           "font": label, "settings": "no-effect, lines off, margin off, resolution Very High (4)"}, open(out / "meta.json", "w"), indent=1)
print("ok", label)

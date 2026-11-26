"""Collect ocr.json, repetition.json, charset.json and per-system timing into metrics.csv."""
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "out"
ocr = json.loads((O / "ocr.json").read_text())
rep = json.loads((O / "repetition.json").read_text())
cs = json.loads((O / "charset.json").read_text())
CSKEY = {"rnn": "rnn_grzego", "hws": "hws_sjvasquez", "one": "onedm", "duc": "ductus"}
MAIN = ["short", "definition", "maths", "pangram"]


def speed(sid):
    d = O / sid
    if sid.startswith("ductus"):
        st = json.loads([l for l in (d / "page.json").read_text().splitlines() if l.startswith("{")][-1])
        w = json.loads((d / "wall.json").read_text())
        return st["seconds"] / 7, w.get("wall_s"), st
    m = json.loads((d / "meta.json").read_text())
    return m["seconds_per_line"], None, m


rows = []
for s in json.loads((ROOT / "systems.json").read_text()):
    sid = s["id"]
    if sid not in ocr:
        continue
    o = ocr[sid]
    c = cs[CSKEY.get(sid[:3], sid)]
    sec, wall, meta = speed(sid)
    r = dict(system=sid, label=s["label"], settings=s["settings"],
             cer_raw_main=round(np.mean([o[t]["cer"] for t in MAIN]), 3),
             cer_norm_main=round(np.mean([o[t]["cer_norm"] for t in MAIN]), 3),
             **{f"cer_norm_{t}": o[t]["cer_norm"] for t in MAIN},
             cer_norm_caps=o["caps"]["cer_norm"], cer_norm_digits=o["digits"]["cer_norm"], cer_raw_symbols=o["symbols"]["cer"],
             twin_rate=rep.get(sid, {}).get("twin_rate"), components=rep.get(sid, {}).get("components"),
             capitals=f'{c["capitals"]["covered"]}/26', symbols=f'{c["symbols"]["covered"]}/12', maths_unicode=f'{c["maths_unicode"]["covered"]}/10',
             sec_per_line=round(sec, 2), wall_s_total=round(wall, 1) if wall else "",
             license=s["license"])
    if sid.startswith("ductus"):
        r["still_failing_words"] = meta.get("still_failing")
    rows.append(r)
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(ROOT / "metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)

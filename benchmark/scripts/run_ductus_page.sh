#!/bin/bash
# One ductus CLI call per look: all 7 test lines on one A4 page (blank line between them),
# so the models load once. Lines are cut out of the page afterwards (split_page.py).
cd "$(dirname "$0")/.."
D=${DUCTUS:-ductus}
for look in "$@"; do
  mkdir -p out/ductus_$look
  s=$(date +%s.%N)
  $D write texts/ductus_page.txt -o out/ductus_$look/page --look $look --seed 0 --no-pdf > out/ductus_$look/page.json 2> out/ductus_$look/page.err
  echo "{\"look\":\"$look\",\"rc\":$?,\"wall_s\":$(python3 -c "import time;print(round(time.time()-$s,1))"),\"load\":\"$(cut -d' ' -f1-3 /proc/loadavg)\"}" > out/ductus_$look/wall.json
done
echo DONE

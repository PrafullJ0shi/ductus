#!/bin/bash
# Runs the ductus CLI (as documented in docs/CLI.md) once per test text and look.
cd "$(dirname "$0")/.."
D=${DUCTUS:-ductus}
for look in neat natural; do
  mkdir -p out/ductus_$look
  tail -n +2 texts/tests.tsv | while IFS=$'\t' read id plain markup ref; do
    s=$(date +%s.%N)
    $D write --text "$markup" -o out/ductus_$look/$id --look $look --seed 0 --no-pdf > out/ductus_$look/$id.json 2> out/ductus_$look/$id.err
    rc=$?
    e=$(date +%s.%N)
    echo "{\"id\":\"$id\",\"rc\":$rc,\"wall_s\":$(echo "$e - $s" | bc)}" >> out/ductus_$look/wall.jsonl
  done
done
echo DONE

#!/usr/bin/env bash
# Zero-shot evaluation of every finished run on data/external (one run at a time, GPU is shared with training).
cd "$(dirname "$0")/.."
for d in "$@"; do
  [ -f "$d/external.json" ] && continue
  [ -f "$d/results.json" ] || continue
  PYTHONPATH=src .venv/bin/python -m hybrid_forensics.external_eval "$d" 2>&1 | tail -1
done
echo "external eval finished"

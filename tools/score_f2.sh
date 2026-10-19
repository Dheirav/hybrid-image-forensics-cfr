#!/usr/bin/env bash
# Score every finished runs_f2 run on the forge2 test split (per-tool metrics, Dice, saved probabilities). Skips runs already scored.
cd "$(dirname "$0")/.." || exit 1
for r in runs_f2/*; do
  [ -f "$r/results.json" ] && [ ! -f "$r/f2_test.json" ] && PYTHONPATH=src .venv/bin/python -m hybrid_forensics.eval_manifest score data/forge2/manifest.csv test f2 "$r" 2>&1 | grep -v -i warn
done
echo scored

#!/usr/bin/env bash
# Score every finished run on the forge_hard test split (tool-disjoint, content-matched). Skips runs already scored.
# usage: tools/score_hard_all.sh [split]    progress: tools/count_progress.py logs/hard.stamp 'runs*/*/hard_test.json' 42
cd "$(dirname "$0")/.." || exit 1
split=${1:-test}
for r in runs/* runs_extra/*; do
  [ -f "$r/best.pt" ] && [ ! -f "$r/hard_${split}.json" ] && PYTHONPATH=src .venv/bin/python -m hybrid_forensics.eval_manifest score data/forge_hard/manifest.csv "$split" hard "$r" 2>&1 | grep -v -i warn
done
echo finished

#!/usr/bin/env bash
# Run training jobs one after another, detached-friendly. A job is VARIANT:SEED[:extra args].
# usage: tools/run_queue.sh "indep:0" "ml:0" ...       progress: tools/run_progress.py --watch
# Each run writes runs/<variant>_s<seed>/{progress.json,results.json,best.pt,probs.npz}; finished runs are skipped.
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
for job in "$@"; do
  IFS=: read -r variant seed extra <<<"$job"
  out="${OUT:-runs}"
  tag=""; [[ "$extra" =~ --tag[[:space:]]+([^[:space:]]+) ]] && tag="${BASH_REMATCH[1]}"   # keeps tagged runs' dirs and logs apart from the headline runs
  [ -f "$out/${variant}${tag}_s${seed}/results.json" ] && { echo "skip $job (done)"; continue; }
  echo "$(date +%T) start $job"
  PYTHONPATH=src .venv/bin/python -m hybrid_forensics.train --variant "$variant" --seed "$seed" --out "$out" $extra \
    > "logs/train_${variant}${tag}_s${seed}.log" 2>&1 || echo "FAILED $job (see logs/train_${variant}${tag}_s${seed}.log)"
done
echo "$(date +%T) queue finished"

#!/usr/bin/env bash
# Stage 2 of the next phase: the headline comparison on forge2 (tool-disjoint, headroom). Outputs go to runs_f2/.
# Seed-major, so a partial queue always has complete seeds for the decision-critical variants. Seeds 0-2 run all 8 variants, seeds 3-4 only the 6 that the
# decision rule needs (single_b and indep are diagnostic controls). progress: tools/queue_f2_progress.py --watch
cd "$(dirname "$0")/.."
export OUT=runs_f2
M="--manifest data/forge2/manifest.csv"
jobs=()
for s in 0 1 2 3 4; do
  jobs+=("single_a:$s:$M" "ml:$s:$M --lam 0.25 --tag _lam0.25" "ml:$s:$M" "ml_hem:$s:$M" "cfr_adaptive:$s:$M" "joint_concat:$s:$M")
  [ "$s" -le 2 ] && jobs+=("single_b:$s:$M" "indep:$s:$M")
done
tools/run_queue.sh "${jobs[@]}"

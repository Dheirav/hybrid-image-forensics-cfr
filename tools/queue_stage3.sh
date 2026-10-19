#!/usr/bin/env bash
# Stage 3: runs the first pass left out although the brief asks for them. All go to runs_extra/ so the headline set is untouched.
#   joint_concat / joint_gate : learned fusion (brief phase 4 items 3 and 5)
#   cfr_scratch --lam 0       : competitive training with NO alignment (phase 7, aligned vs non-aligned)
#   ml --lam 0.25 / 4         : alignment-strength sweep (phase 5); the headline ml uses lam 1
# Seed-major so a partial queue still has complete seeds for the earliest variants.
cd "$(dirname "$0")/.."
export OUT=runs_extra
jobs=()
for s in 0 1 2; do
  jobs+=("joint_concat:$s" "joint_gate:$s" "cfr_scratch:$s:--lam 0 --tag _noalign" "ml:$s:--lam 0.25 --tag _lam0.25" "ml:$s:--lam 4 --tag _lam4")
done
tools/run_queue.sh "${jobs[@]}"

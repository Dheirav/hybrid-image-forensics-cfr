#!/usr/bin/env bash
# Stage 2: headline comparison, seed-major so each seed finishes for every headline variant before the next
# starts, then the transition controls, then the remaining baseline seeds. lr, lam, T, beta, floor stay at defaults.
cd "$(dirname "$0")/.."
jobs=()
for s in 0 1 2; do jobs+=("ml:$s" "cfr_adaptive:$s" "ml_hem:$s"); done
for s in 1 2; do jobs+=("indep:$s"); done
for s in 0 1 2; do jobs+=("cfr_sudden:$s" "cfr_fixed:$s" "cfr_scratch:$s"); done
for s in 1 2; do jobs+=("single_a:$s" "single_b:$s"); done
tools/run_queue.sh "${jobs[@]}"

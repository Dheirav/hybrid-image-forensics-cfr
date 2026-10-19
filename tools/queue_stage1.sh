#!/usr/bin/env bash
# Stage 1: single-branch baselines and the independent pair at the default lr, then two lr checks
# for the pair (the lr is chosen on validation AUROC, then frozen for every later variant).
cd "$(dirname "$0")/.."
tools/run_queue.sh single_a:0 single_b:0 indep:0
OUT=runs_lr1e-3 tools/run_queue.sh "indep:0:--lr 1e-3"
OUT=runs_lr2e-4 tools/run_queue.sh "indep:0:--lr 2e-4"

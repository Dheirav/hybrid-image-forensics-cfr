#!/usr/bin/env bash
# Runs the CFR-weight and external-failure diagnoses one after the other (GPU is shared). Each writes reports/evidence/next_phase/<name>.md.
cd "$(dirname "$0")/.." || exit 1
for m in diag_cfr diag_external; do
  echo "$(date +%T) start $m"
  PYTHONPATH=src .venv/bin/python -m hybrid_forensics.$m > reports/evidence/next_phase/$m.md 2> logs/$m.err && echo "$(date +%T) done $m" || echo "FAILED $m (logs/$m.err)"
done

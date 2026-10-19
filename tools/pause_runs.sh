#!/usr/bin/env bash
# pause_runs.sh stop|cont [HH:MM UTC to wait for]
# Freezes (SIGSTOP) or thaws (SIGCONT) the stage 3 training queue and the external-eval job, whole process groups, so the data-loader
# workers stop with their parent. State and GPU memory are kept; the runs continue where they were. The elapsed-time counter in
# queue3_progress.py will include the paused time, so its ETA is pessimistic after a resume until the next run finishes.
action=$1; at=$2
if [ -n "$at" ]; then
  target=$(date -u -d "today $at" +%s)
  until [ "$(date +%s)" -ge "$target" ]; do sleep 20; done
fi
sig=STOP; [ "$action" = cont ] && sig=CONT
for pat in '[q]ueue_stage3.sh' '[e]xternal_all.sh'; do
  for g in $(ps -eo pgid,args | awk -v p="$pat" '$0 ~ p {print $1}' | sort -u); do
    kill -$sig -- -"$g" && echo "$(date +%T) $sig process group $g ($pat)"
  done
done

#!/usr/bin/env bash
# Progress for `uv pip install`, read from its log. Sizes come from the "Downloading X (N MiB)"
# lines, completion from the "Downloaded X" lines. Rate = completed MiB / elapsed since the log
# was created, so the ETA is only shown once something has completed, and it is lumpy while one
# large wheel (torch is 874 MiB) is still in flight.
# usage: tools/install-progress.sh [--once] [logfile]
ONCE=0; [ "$1" = "--once" ] && { ONCE=1; shift; }
LOG="${1:-logs/install.log}"
while :; do
  start=$(stat -c %W "$LOG" 2>/dev/null); [ "${start:-0}" -gt 0 ] || start=$(stat -c %Y "$LOG")
  el=$(( $(date +%s) - start ))
  read -r total done_mb n_total n_done < <(awk '
    /^Downloading /{match($0,/\(([0-9.]+)MiB\)/,m); sz[$2]=m[1]; tot+=m[1]; nt++}
    /^ *Downloaded /{n=$2; if(n in sz){dn+=sz[n]; nd++}}
    END{printf "%d %d %d %d\n", tot, dn, nt, nd}' "$LOG")
  state=running; pgrep -f "uv pip install" >/dev/null || state=finished
  msg="$state  ${done_mb}/${total} MiB  ${n_done:-0}/${n_total:-0} packages  elapsed ${el}s"
  if [ "${done_mb:-0}" -gt 0 ] && [ "$state" = running ]; then
    msg="$msg  rate $(( done_mb / (el>0?el:1) )) MiB/s  ETA ~$(( (total-done_mb) * el / done_mb ))s"
  else msg="$msg  (no ETA yet: nothing completed)"; fi
  printf '\r%s   ' "$msg"
  if [ "$state" = finished ]; then echo; tail -2 "$LOG" | cut -c1-160; break; fi
  [ "$ONCE" = 1 ] && { echo; break; }
  sleep 3
done

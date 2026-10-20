#!/usr/bin/env bash
# Run a command at low priority and kill it (and its process group) if MemAvailable falls below a threshold.
#   scripts/mem_guard.sh <min_available_gib> <log_file> -- <command...>
# Exit code: the command's, or 99 if the guard killed it. The peak of (MemTotal - MemAvailable) is appended to the log.
set -u
min_gib=$1; log=$2; shift 3
setsid nice -n 10 "$@" &
pid=$!
peak=0
while kill -0 "$pid" 2>/dev/null; do
  avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  total_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
  used=$(( (total_kb - avail_kb) / 1024 ))
  [ "$used" -gt "$peak" ] && peak=$used
  if [ "$avail_kb" -lt $(( min_gib * 1024 * 1024 )) ]; then
    echo "$(date +%H:%M:%S) mem_guard: MemAvailable $((avail_kb / 1024)) MiB < ${min_gib} GiB, killing $pid" >> "$log"
    kill -TERM -- "-$pid" 2>/dev/null; sleep 5; kill -KILL -- "-$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    echo "mem_guard peak_used_mib=$peak killed=1" >> "$log"
    exit 99
  fi
  sleep 2
done
wait "$pid"; rc=$?
echo "mem_guard peak_used_mib=$peak killed=0 rc=$rc" >> "$log"
exit $rc

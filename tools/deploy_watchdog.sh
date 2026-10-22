#!/usr/bin/env bash
# Process watchdog for deployment jobs on the Spark (host side, uses no GPU slot).
#
# Every INTERVAL seconds, for every running python process whose command line matches PATTERN (default: the
# deploy tools), append a heartbeat line to WATCHDIR/<pid>.heartbeat: time, pid, elapsed, %CPU, RSS, the log file's
# size / mtime / last [deploy]|[t50] line, GPU util / memory, host free memory. When a watched pid disappears, write
# WATCHDIR/<pid>.death with the last heartbeat, the container exit code (from `docker events`, captured here),
# dmesg tail (OOM / Xid lines), nvidia-smi and free -m. Added 2026-10-22 after an adapting T50 run died silently
# at minute 4 (no traceback, no OOM record).
#
# usage:  nohup tools/deploy_watchdog.sh [WATCHDIR] [INTERVAL] [PATTERN] > /dev/null 2>&1 &
set -u
watchdir=${1:-$HOME/src/isaaclab-learning/logs/deploy/watchdog}
interval=${2:-60}
pattern=${3:-"python3 /workspace/learning/hexapod/tools/deploy_"}
mkdir -p "$watchdir"
# container deaths with exit codes (containers run with --rm, so this is the only record)
docker events --filter event=die --format '{{.Time}} die container={{.Actor.ID}} exitCode={{index .Actor.Attributes "exitCode"}} image={{.Actor.Attributes.image}}' >> "$watchdir/docker_die_events.log" 2>&1 &
events_pid=$!
trap 'kill $events_pid 2>/dev/null' EXIT
declare -A watched
while true; do
  now=$(date -Is)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr -d '\n')
  mem=$(free -m | awk '/Mem:/ {print "used " $3 " MiB, avail " $7 " MiB"}')
  for pid in $(pgrep -f "$pattern"); do
    args=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null) || continue
    case "$args" in *python3*) ;; *) continue ;; esac
    # the log is the spark_run.sh LOGFILE of the ancestor wrapper with the same tool name
    tool=$(echo "$args" | grep -o 'tools/deploy_[a-z_0-9]*\.py' | head -1)
    log=$(pgrep -fa "spark_run.sh .* -- .*${tool}" | grep -o 'logs/deploy/[^ ]*\.log' | head -1)
    [ -n "$log" ] && log="$HOME/src/isaaclab-learning/$log"
    if [ -z "${watched[$pid]:-}" ]; then
      watched[$pid]="$log"
      echo "$now start pid=$pid log=$log args=$args" >> "$watchdir/$pid.heartbeat"
    fi
    ps_line=$(ps -o etime=,pcpu=,rss= -p "$pid" 2>/dev/null | xargs)
    if [ -n "$log" ] && [ -f "$log" ]; then
      lsz=$(stat -c %s "$log"); lmt=$(stat -c %y "$log" | cut -c1-19)
      last=$(grep -E '\[deploy\]|\[t50\]|\[residual\]|\[ensemble\]' "$log" | tail -1 | cut -c1-160)
    else
      lsz=-; lmt=-; last=-
    fi
    echo "$now pid=$pid etime/cpu/rss=[$ps_line] log_size=$lsz log_mtime=$lmt gpu=[$gpu] host=[$mem] last=[$last]" >> "$watchdir/$pid.heartbeat"
  done
  for pid in "${!watched[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      {
        echo "== death of pid $pid noticed at $now (within ${interval}s of the last heartbeat)"
        echo "log: ${watched[$pid]}"
        echo "-- last heartbeat"; tail -1 "$watchdir/$pid.heartbeat"
        echo "-- log tail"; [ -f "${watched[$pid]}" ] && tail -c 1500 "${watched[$pid]}"
        echo; echo "-- docker die events (last 5)"; tail -5 "$watchdir/docker_die_events.log"
        echo "-- dmesg tail (OOM / Xid / killed)"; dmesg 2>&1 | grep -iE 'oom|xid|killed process|nvrm' | tail -10
        echo "-- nvidia-smi"; nvidia-smi 2>&1 | head -25
        echo "-- free -m"; free -m
        echo "-- running isaac containers"; docker ps --format '{{.ID}} {{.Names}} {{.Status}}' | head
      } > "$watchdir/$pid.death" 2>&1
      unset "watched[$pid]"
    fi
  done
  sleep "$interval"
done

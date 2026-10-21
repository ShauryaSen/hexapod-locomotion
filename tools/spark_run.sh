#!/usr/bin/env bash
# Run one Isaac Lab command on the Spark through ~/bin/isaac-shell, retrying the flaky Kit startup.
#
# Measured 2026-10-21: of 13 launches, 3 died in a Kit crash ~10 s in, 1 hung in Kit startup with nothing logged, ("[crash] A crash has occurred", then the
# process hangs) and 2 exited silently right after "AppLauncher initialization complete" ("There was an error
# running python", no traceback). Both happen before any env code runs; a relaunch works.
#
# usage:  tools/spark_run.sh LOGFILE -- <command run inside isaac-shell>
#   e.g.  tools/spark_run.sh ~/src/isaaclab-learning/logs/x.log -- env PYTHONPATH=/workspace/learning ./isaaclab.sh -p ...
# Exit code = the command's exit code of the last attempt. Failed attempts' logs are kept as LOGFILE.attemptN.
# Only containers started by this script (tagged with a unique SPARK_RUN_TAG env var) are ever stopped.
set -u
log=$1; shift; [ "${1:-}" = "--" ] && shift
max_attempts=${SPARK_RUN_ATTEMPTS:-4}
startup_timeout=${SPARK_RUN_STARTUP_S:-240}

stop_tagged() {
  for c in $(docker ps -q --filter ancestor=isaac-lab-base); do
    if docker inspect --format '{{.Args}}' "$c" | grep -q "SPARK_RUN_TAG=$1"; then docker stop "$c" >/dev/null; fi
  done
}

for attempt in $(seq 1 "$max_attempts"); do
  tag="$(date +%s)_$$_$attempt"
  # PYTHONUNBUFFERED: without it python block-buffers stdout under nohup and the log lags by minutes (or until exit)
  ~/bin/isaac-shell env SPARK_RUN_TAG="$tag" PYTHONUNBUFFERED=1 "$@" > "$log" 2>&1 &
  pid=$!
  started=0
  for _ in $(seq 1 $((startup_timeout / 2))); do
    sleep 2
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    if grep -q "\[crash\] A crash has occurred" "$log"; then
      echo "[spark_run] attempt $attempt: Kit crash at startup, stopping container $tag" >> "$log"
      stop_tagged "$tag"; sleep 5; kill "$pid" 2>/dev/null; break
    fi
    if grep -q "AppLauncher initialization complete" "$log"; then started=1; break; fi
  done
  if [ "$started" = 1 ]; then
    # past the crash window; the silent exit (if any) comes within seconds, otherwise the command runs to the end.
    # Stall guard: if the log has not grown for SPARK_RUN_STALL_S (default 1800 s; 0 disables) the instance is
    # treated as hung in physics start (seen under load) and retried.
    stall=${SPARK_RUN_STALL_S:-1800}
    stalled=0
    while kill -0 "$pid" 2>/dev/null; do
      sleep 20
      if [ "$stall" -gt 0 ] && [ $(( $(date +%s) - $(stat -c %Y "$log") )) -gt "$stall" ]; then
        echo "[spark_run] attempt $attempt: no log output for ${stall}s, stopping container $tag" >> "$log"
        stop_tagged "$tag"; sleep 5; kill "$pid" 2>/dev/null; stalled=1; break
      fi
    done
    wait "$pid"; rc=$?
    if [ "$stalled" = 1 ]; then mv "$log" "$log.attempt$attempt"; sleep 30; continue; fi
    if [ "$rc" != 0 ] && grep -q "There was an error running python" "$log" && ! grep -q "Traceback (most recent call last)" "$log"; then
      echo "[spark_run] attempt $attempt: silent python exit after AppLauncher init" >> "$log"
    else
      exit "$rc"
    fi
  else
    if kill -0 "$pid" 2>/dev/null; then
      # Kit hung in startup (seen: nothing logged after the protobuf warnings for minutes)
      echo "[spark_run] attempt $attempt: startup timeout after ${startup_timeout}s, stopping container $tag" >> "$log"
      stop_tagged "$tag"; sleep 5; kill "$pid" 2>/dev/null
    fi
    wait "$pid" 2>/dev/null; rc=$?
    if grep -q "Traceback (most recent call last)" "$log"; then exit "$rc"; fi
    echo "[spark_run] attempt $attempt: did not reach AppLauncher init (rc=$rc)" >> "$log"
  fi
  mv "$log" "$log.attempt$attempt"
  sleep 30
done
echo "[spark_run] giving up after $max_attempts attempts" > "$log"
exit 1

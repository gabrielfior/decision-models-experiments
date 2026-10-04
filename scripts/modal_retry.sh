#!/usr/bin/env bash
# Run a Modal entrypoint, retrying up to N times on failure (transient Modal/infra errors).
# usage: scripts/modal_retry.sh <log> <tries> <modal run args...>
LOG=$1; TRIES=$2; shift 2
for i in $(seq 1 "$TRIES"); do
  echo "=== attempt $i $(date -u +%FT%TZ)" >> "$LOG"
  uv run modal run "$@" >> "$LOG" 2>&1 && { echo "exit 0" >> "$LOG"; exit 0; }
  echo "attempt $i failed" >> "$LOG"; sleep 60
done
echo "exit 1 (all $TRIES attempts failed)" >> "$LOG"; exit 1

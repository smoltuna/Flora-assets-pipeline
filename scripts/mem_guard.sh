#!/usr/bin/env bash
# Memory-pressure guardrail for the pipeline.
#
# Usage:
#   scripts/mem_guard.sh <PID> <LOG_FILE>
#
# Polls memory_pressure(1), vm_stat, and swap every INTERVAL seconds. If any
# threshold is breached, sends SIGTERM (then SIGKILL after grace) to <PID>
# so the pipeline exits cleanly before the kernel watchdog panics.
#
# Thresholds tuned for a 16 GB M4 running Ollama qwen2.5:7b + docker
# observability stack. Adjust via env vars.

set -u

PID="${1:?usage: mem_guard.sh PID LOG}"
LOG="${2:?usage: mem_guard.sh PID LOG}"

INTERVAL="${MEM_GUARD_INTERVAL:-10}"
# Tuned after 2026-08-12 dry run: swap grew to 4.8 GB purely from the OS's
# normal compressor→swap relief valve (compressor dropped 2 GB → swap grew 3 GB
# in one 10s window). That is NOT a pre-panic signal, so the swap threshold is
# now generous. The actual panic-signature was compressor ≈ 13 GB + swap
# EXHAUSTED (couldn't extend), so we still watch compressor + free%.
MIN_FREE_PCT="${MEM_GUARD_MIN_FREE_PCT:-10}"     # abort if free < 10%
MAX_COMP_MB="${MEM_GUARD_MAX_COMP_MB:-10240}"    # abort if compressor > 10 GB
MAX_SWAP_MB="${MEM_GUARD_MAX_SWAP_MB:-10240}"    # abort if swap used > 10 GB
GRACE_SEC="${MEM_GUARD_GRACE_SEC:-25}"

trap 'echo "$(date -Iseconds) guard.exit reason=signal" >>"$LOG"; exit 0' TERM INT

PAGE_SIZE=$(vm_stat | awk '/page size of/ {print $8}')

echo "$(date -Iseconds) guard.start pid=$PID interval=${INTERVAL}s min_free=${MIN_FREE_PCT}% max_comp=${MAX_COMP_MB}M max_swap=${MAX_SWAP_MB}M" >>"$LOG"

while kill -0 "$PID" 2>/dev/null; do
  ts=$(date -Iseconds)

  free_pct=$(memory_pressure -Q 2>/dev/null | awk -F'[ %]' '/free percentage/ {print $(NF-1)}')
  [ -z "$free_pct" ] && free_pct=100

  comp_pages=$(vm_stat | awk '/Pages occupied by compressor/ {gsub(/\./,"",$5); print $5}')
  comp_mb=$(( comp_pages * PAGE_SIZE / 1024 / 1024 ))

  swap_used=$(sysctl -n vm.swapusage | awk '{gsub(/M/,"",$6); print int($6)}')

  echo "$ts guard.sample free_pct=$free_pct comp_mb=$comp_mb swap_mb=$swap_used" >>"$LOG"

  reason=""
  if [ "$free_pct" -lt "$MIN_FREE_PCT" ]; then
    reason="free_pct=${free_pct}% < ${MIN_FREE_PCT}%"
  elif [ "$comp_mb" -gt "$MAX_COMP_MB" ]; then
    reason="comp_mb=${comp_mb} > ${MAX_COMP_MB}"
  elif [ "$swap_used" -gt "$MAX_SWAP_MB" ]; then
    reason="swap_mb=${swap_used} > ${MAX_SWAP_MB}"
  fi

  if [ -n "$reason" ]; then
    echo "$ts guard.critical $reason - SIGTERM pid=$PID" >>"$LOG"
    kill -TERM "$PID" 2>/dev/null || true
    # give the pipeline a grace period to close db/session, then hard-kill
    for _ in $(seq 1 "$GRACE_SEC"); do
      kill -0 "$PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$PID" 2>/dev/null; then
      echo "$ts guard.escalate SIGKILL pid=$PID" >>"$LOG"
      kill -KILL "$PID" 2>/dev/null || true
    fi
    echo "$ts guard.stop pid=$PID" >>"$LOG"
    exit 2
  fi

  sleep "$INTERVAL"
done

echo "$(date -Iseconds) guard.pid_exited pid=$PID - guard shutting down" >>"$LOG"

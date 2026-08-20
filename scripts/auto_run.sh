#!/usr/bin/env bash
# Auto-restart wrapper around run_all.py + mem_guard.sh.
#
# Loops until every flower in data/batch4.txt has a complete asset triplet
# (home.png + info.jpg + lock.png) in output/FlowerAssets.xcassets/. On each
# iteration, regenerates data/batch4_remaining.txt from the current asset
# state, waits for memory to settle, and launches pipeline + guardrail. When
# the pipeline exits (naturally or via guard trip), records the outcome and
# restarts after a cooldown.
#
# Usage:
#   scripts/auto_run.sh
#
# Logs:
#   logs/auto_run_<TS>.log            — wrapper's own log
#   logs/batch4_remaining_<TS>.log    — pipeline stdout per iteration
#   logs/mem_guard_<TS>.log           — guard samples per iteration

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BATCH_SOURCE=data/batch4.txt
REMAINING=data/batch4_remaining.txt
ASSETS=output/FlowerAssets.xcassets

WRAPPER_TS=$(date +%Y%m%d_%H%M%S)
WRAPPER_LOG="logs/auto_run_${WRAPPER_TS}.log"

MAX_ITERS="${AUTO_RUN_MAX_ITERS:-25}"        # hard cap so we don't loop forever
COOLDOWN_SEC="${AUTO_RUN_COOLDOWN_SEC:-60}"  # wait between iterations
MEM_SETTLE_PCT="${AUTO_RUN_MEM_SETTLE_PCT:-40}"  # wait for free % >= this before restart

wlog() {
  echo "$(date -Iseconds) auto_run.$1 ${*:2}" | tee -a "$WRAPPER_LOG"
}

trap 'wlog exit reason=signal; kill_children; exit 130' INT TERM

kill_children() {
  [ -n "${PIPE_PID:-}" ] && kill -TERM "$PIPE_PID" 2>/dev/null || true
  [ -n "${GUARD_PID:-}" ] && kill -TERM "$GUARD_PID" 2>/dev/null || true
}

regenerate_remaining() {
  # Regenerate REMAINING from batch4.txt, filtering out flowers with a full triplet on disk.
  local n=0
  : > "$REMAINING"
  while IFS= read -r name; do
    [ -z "$name" ] && continue
    local slug
    slug=$(echo "$name" | tr '[:upper:] ' '[:lower:]-')
    if [ -f "$ASSETS/$slug.imageset/home.png" ] \
       && [ -f "$ASSETS/$slug-info.imageset/info.jpg" ] \
       && [ -f "$ASSETS/$slug-lock.imageset/lock.png" ]; then
      continue
    fi
    echo "$name" >> "$REMAINING"
    n=$((n+1))
  done < "$BATCH_SOURCE"
  echo "$n"
}

wait_for_memory_settle() {
  # Block until free% >= MEM_SETTLE_PCT (or 90s max).
  local i=0
  while [ $i -lt 18 ]; do
    local free_pct
    free_pct=$(memory_pressure -Q 2>/dev/null | awk -F'[ %]' '/free percentage/ {print $(NF-1)}')
    [ -z "$free_pct" ] && free_pct=100
    if [ "$free_pct" -ge "$MEM_SETTLE_PCT" ]; then
      wlog mem.settled free_pct="$free_pct" after_s=$((i*5))
      return
    fi
    sleep 5
    i=$((i+1))
  done
  wlog mem.settle_timeout free_pct="$free_pct" waited_s=90
}

wlog start wrapper_log="$WRAPPER_LOG" max_iters="$MAX_ITERS" cooldown_s="$COOLDOWN_SEC"

for iter in $(seq 1 "$MAX_ITERS"); do
  n_remaining=$(regenerate_remaining)
  wlog iter.begin iter="$iter" remaining="$n_remaining"

  if [ "$n_remaining" -eq 0 ]; then
    wlog done reason=all_flowers_complete iter="$iter"
    exit 0
  fi

  wait_for_memory_settle

  TS=$(date +%Y%m%d_%H%M%S)
  PIPE_LOG="logs/batch4_remaining_${TS}.log"
  GUARD_LOG="logs/mem_guard_${TS}.log"

  wlog pipeline.start iter="$iter" remaining="$n_remaining" log="$PIPE_LOG"
  uv run python scripts/run_all.py --file "$REMAINING" > "$PIPE_LOG" 2>&1 &
  PIPE_PID=$!

  scripts/mem_guard.sh "$PIPE_PID" "$GUARD_LOG" > /dev/null 2>&1 &
  GUARD_PID=$!

  # Wait for pipeline to exit
  wait "$PIPE_PID"
  pipe_exit=$?

  # Ensure guard shuts down
  kill -TERM "$GUARD_PID" 2>/dev/null || true
  wait "$GUARD_PID" 2>/dev/null || true

  # Post-iter accounting
  done_this_iter=$(grep -c "run_all.data_done" "$PIPE_LOG" 2>/dev/null || echo 0)
  guard_tripped=$(grep -c "guard.critical" "$GUARD_LOG" 2>/dev/null || echo 0)
  wlog pipeline.end iter="$iter" pipe_exit="$pipe_exit" flowers_completed="$done_this_iter" guard_tripped="$guard_tripped"

  # Cooldown for memory to drain before next iter
  sleep "$COOLDOWN_SEC"
done

wlog stop reason=hit_max_iters iter="$MAX_ITERS"
exit 1

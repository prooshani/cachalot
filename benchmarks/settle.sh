#!/bin/bash
# Wait until the machine has really given back the previous benchmark's memory.
#
# Why: a finished runtime's pages land on the inactive list, and the boot
# volume's file cache holds the shards it streamed. guarded_run.sh's preflight
# counts both as reclaimable and will happily start, but macOS has not yet
# reused them, so the next arm competes with the previous one's corpse. Two
# arms started in immediate succession on 2026-09-17 produced a 1.32 tok/s
# outlier and two preflight refusals. Every A/B in docs/HANDOFF-2026-09-17.md
# ran through this gate between arms.
#
# It waits for four conditions to hold together, then for a quiet period:
#   - no runtime process alive
#   - kern.memorystatus_vm_pressure_level back to 1 (normal)
#   - swap not growing
#   - free + speculative + purgeable + reclaimable file cache above the wired
#     set the next arm plans to hold
#
# Usage:
#   benchmarks/settle.sh [--budget-gib N] [--timeout S] [--quiet-seconds S]
set -uo pipefail
cd "$(dirname "$0")/.."

BUDGET_GIB=36
TIMEOUT=600
QUIET_SECONDS=20

while [ $# -gt 0 ]; do
  case "$1" in
    --budget-gib) BUDGET_GIB=$2; shift 2 ;;
    --timeout) TIMEOUT=$2; shift 2 ;;
    --quiet-seconds) QUIET_SECONDS=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

TRUNK_GIB=12
TRANSIENT_GIB=3
ACTIVATION_GIB=8
OS_FLOOR_GIB=6
NEED_GIB=$((TRUNK_GIB + BUDGET_GIB + TRANSIENT_GIB + ACTIVATION_GIB + OS_FLOOR_GIB))

page_size() { vm_stat | awk '/page size of/ {print $8}'; }
vm_pages() {
  vm_stat | awk -F'[: .]+' '
    /^Pages free/ {f=$3} /^Pages speculative/ {s=$3} /^Pages purgeable/ {p=$3}
    /^Pages inactive/ {i=$3} /^File-backed pages/ {fb=$3}
    END {printf "%d %d %d %d %d\n", f, s, p, i, fb}'
}
swap_used_mb() {
  sysctl -n vm.swapusage | awk '{
    for (n = 1; n <= NF; n++) if ($n == "used") { v = $(n+2); break }
    unit = substr(v, length(v)); num = substr(v, 1, length(v)-1)
    if (unit == "G") num *= 1024; else if (unit == "K") num /= 1024
    printf "%d\n", num }'
}
gib() { awk -v pages="$1" -v ps="$2" 'BEGIN {printf "%.1f", pages * ps / 1073741824}'; }

PS=$(page_size)
START=$(date +%s)
quiet_since=0
last_swap=$(swap_used_mb)

echo "settle: waiting for ${NEED_GIB} GiB available (next arm budget ${BUDGET_GIB} GiB), pressure 1, ${QUIET_SECONDS}s quiet"

while true; do
  elapsed=$(( $(date +%s) - START ))
  if [ "$elapsed" -ge "$TIMEOUT" ]; then
    echo "settle: TIMEOUT after ${elapsed}s; do not start the next arm, investigate first"
    exit 1
  fi

  ok=1
  runtime_alive=0
  # -f matches the whole command line, so any process that merely *mentions*
  # the interpreter counts -- including the shell of a script that runs an A/B
  # and calls this gate between arms, if that script was passed as text rather
  # than as a file. Such a script waits for itself and never starts its next
  # arm (2026-09-21). Put the arms in a file and run `bash the-file`.
  pgrep -f "deepseek-v41/bin/python|cachalot" >/dev/null && { ok=0; runtime_alive=1; }
  lvl=$(sysctl -n kern.memorystatus_vm_pressure_level)
  [ "$lvl" -eq 1 ] || ok=0
  swap=$(swap_used_mb)
  [ "$swap" -le "$last_swap" ] || ok=0
  last_swap=$swap
  read -r pf psp ppu pin pfb <<<"$(vm_pages)"
  reclaim=$(( pfb < pin ? pfb : pin ))
  avail=$(gib $((pf + psp + ppu + reclaim)) "$PS")
  awk -v a="$avail" -v n="$NEED_GIB" 'BEGIN {exit !(a >= n)}' || ok=0

  if [ "$ok" -eq 1 ]; then
    [ "$quiet_since" -eq 0 ] && quiet_since=$(date +%s)
    held=$(( $(date +%s) - quiet_since ))
    if [ "$held" -ge "$QUIET_SECONDS" ]; then
      echo "settle: ready after ${elapsed}s (available ${avail} GiB, pressure ${lvl}, swap ${swap} MB)"
      exit 0
    fi
  else
    quiet_since=0
    held=0
  fi

  echo "settle: ${elapsed}s available ${avail}/${NEED_GIB} GiB, pressure ${lvl}, swap ${swap} MB, runtime_alive ${runtime_alive}, quiet ${held}/${QUIET_SECONDS}s"
  sleep 5
done

#!/bin/bash
# Run a benchmark under a memory guardian so that memory pressure kills the
# benchmark instead of panicking the machine.
#
# Why: on 2026-09-15 two decode benchmarks ran with the auto expert budget
# (50 GiB experts, 66 GiB wired). Other applications plus the runtime's
# non-wired memory pushed macOS into swap; the boot volume had ~13 GiB free,
# swap grew to 29 swapfiles, the compressor hit "100% of segments limit" and
# kernel_task spun until the hardware watchdog panicked
# (/Library/Logs/DiagnosticReports/panic-full-2026-09-15-180051.0002.panic).
# Wired memory can neither be compressed nor swapped, so the only safe
# reaction to pressure is to kill the process that holds it. This script:
#
#   1. preflight: refuses to start when another runtime process is alive,
#      swap is already in use, memory pressure is not normal, the boot volume
#      is nearly full, or free+reclaimable memory cannot hold the planned
#      wired set;
#   2. runs the command with CACHALOT_EXPERT_CACHE_BUDGET_GIB set to an
#      explicit, conservative budget (never the auto formula) under nice;
#   3. samples vm_stat / swap / pressure / process RSS every second into a
#      CSV and SIGKILLs the command on critical pressure, sustained warning
#      pressure, swap growth, low disk, or a wall-clock timeout;
#   4. prints a footprint summary (peak wired, peak swap, min free, peak RSS).
#
# Usage:
#   benchmarks/guarded_run.sh [--budget-gib N] [--max-seconds S] [--tag NAME] -- <command...>
#
# Example:
#   benchmarks/guarded_run.sh --budget-gib 32 --max-seconds 900 --tag base -- \
#       env CACHALOT_PREDICT_TOPK=0 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python \
#       benchmarks/decode_throughput.py --prompt-tokens 512 --decode-tokens 64
set -uo pipefail
cd "$(dirname "$0")/.."

BUDGET_GIB=28
MAX_SECONDS=900
TAG=run
FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --budget-gib) BUDGET_GIB=$2; shift 2 ;;
    --max-seconds) MAX_SECONDS=$2; shift 2 ;;
    --tag) TAG=$2; shift 2 ;;
    --force) FORCE=1; shift ;;
    --) shift; break ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: $0 [--budget-gib N] [--max-seconds S] [--tag NAME] -- <command...>" >&2; exit 2; }

# ---- thresholds (GiB / MB / samples) ---------------------------------------
TRUNK_GIB=12            # resident non-expert weights
TRANSIENT_GIB=3         # 128 transient slots (2.4 GiB) rounded up
ACTIVATION_GIB=8        # MLX cache limit (2) + prefill temporaries + Python heap
OS_FLOOR_GIB=6          # free memory the OS must keep for itself and other apps
MIN_DISK_GIB=10         # boot volume space swap could still grow into
KILL_DISK_GIB=6
KILL_SWAP_MB=512        # swap growth during the run means the wired set does not fit
KILL_CRITICAL_SAMPLES=2 # kern.memorystatus_vm_pressure_level 4 = critical
KILL_WARN_SAMPLES=10    # level 2 = warning, sustained (preceded swapping by ~18 s on 2026-09-16)
PREFLIGHT_SWAP_MB=512

RESULTS=benchmarks/results/guarded
mkdir -p "$RESULTS"
STAMP=$(date +%Y%m%d-%H%M%S)
CSV="$RESULTS/${TAG}_${STAMP}.csv"
OUT="$RESULTS/${TAG}_${STAMP}.out"

# ---- helpers ---------------------------------------------------------------
page_size() { vm_stat | awk '/page size of/ {print $8}'; }
# prints: free speculative purgeable inactive wired filebacked anonymous compressor (pages)
vm_pages() {
  vm_stat | awk -F'[: .]+' '
    /^Pages free/ {f=$3} /^Pages speculative/ {s=$3} /^Pages purgeable/ {p=$3}
    /^Pages inactive/ {i=$3} /^Pages wired down/ {w=$4} /^File-backed pages/ {fb=$3}
    /^Anonymous pages/ {an=$3} /^Pages occupied by compressor/ {c=$5}
    END {printf "%d %d %d %d %d %d %d %d\n", f, s, p, i, w, fb, an, c}'
}
swap_used_mb() {
  sysctl -n vm.swapusage | awk '{
    for (n = 1; n <= NF; n++) if ($n == "used") { v = $(n+2); break }
    unit = substr(v, length(v)); num = substr(v, 1, length(v)-1)
    if (unit == "G") num *= 1024; else if (unit == "K") num /= 1024
    printf "%d\n", num }'
}
pressure_level() { sysctl -n kern.memorystatus_vm_pressure_level; }
disk_avail_gib() { df -g / | awk 'NR==2 {print $4}'; }
gib() { awk -v pages="$1" -v ps="$2" 'BEGIN {printf "%.1f", pages * ps / 1073741824}'; }

PS=$(page_size)

# ---- preflight -------------------------------------------------------------
fail=0
note() { echo "preflight: $*"; }
if pgrep -f "deepseek-v41/bin/python|cachalot" >/dev/null; then
  note "FAIL another runtime process is running:"; pgrep -fl "deepseek-v41/bin/python|cachalot"; fail=1
fi
lvl=$(pressure_level)
[ "$lvl" -eq 1 ] || { note "FAIL memory pressure level is $lvl (1 = normal)"; fail=1; }
sw=$(swap_used_mb)
# stale swap from an earlier process is harmless while pressure is normal and memory is free;
# the guardian watches swap GROWTH from this baseline during the run
SWAP_START=$sw
[ "$sw" -lt "$PREFLIGHT_SWAP_MB" ] || note "WARN swap already in use: ${sw} MB (stale; guardian kills on +${KILL_SWAP_MB} MB growth)"
disk=$(disk_avail_gib)
[ "$disk" -ge "$MIN_DISK_GIB" ] || { note "FAIL boot volume has ${disk} GiB free (< ${MIN_DISK_GIB})"; fail=1; }
read -r pf psp ppu pin pw pfb pan pc <<<"$(vm_pages)"
reclaim=$(( pfb < pin ? pfb : pin ))
avail_gib=$(gib $((pf + psp + ppu + reclaim)) "$PS")
need_gib=$((TRUNK_GIB + BUDGET_GIB + TRANSIENT_GIB + ACTIVATION_GIB + OS_FLOOR_GIB))
planned_wired=$((TRUNK_GIB + BUDGET_GIB + TRANSIENT_GIB))
echo "preflight: budget ${BUDGET_GIB} GiB experts -> planned wired ~${planned_wired} GiB, need ${need_gib} GiB; " \
     "free $(gib "$pf" "$PS") + speculative $(gib "$psp" "$PS") + purgeable $(gib "$ppu" "$PS") + reclaimable file cache $(gib "$reclaim" "$PS") = ${avail_gib} GiB; " \
     "wired now $(gib "$pw" "$PS") GiB; swap ${sw} MB; pressure ${lvl}; disk ${disk} GiB free"
awk -v a="$avail_gib" -v n="$need_gib" 'BEGIN {exit !(a >= n)}' || { note "FAIL only ${avail_gib} GiB available for ${need_gib} GiB needed; lower --budget-gib or close applications"; fail=1; }
if [ "$fail" -ne 0 ]; then
  [ "$FORCE" -eq 1 ] && echo "preflight: --force given, continuing" || { echo "preflight: aborting"; exit 3; }
fi

# ---- launch ----------------------------------------------------------------
echo "run: tag=${TAG} budget=${BUDGET_GIB} GiB max=${MAX_SECONDS}s"
echo "run: output -> ${OUT}"
echo "run: samples -> ${CSV}"
export CACHALOT_EXPERT_CACHE_BUDGET_GIB="$BUDGET_GIB"
nice -n 5 "$@" >"$OUT" 2>&1 &
PID=$!
START=$(date +%s)
trap 'kill -9 $PID 2>/dev/null; echo "run: interrupted, killed pid $PID"; exit 130' INT TERM

echo "elapsed_s,free_gib,wired_gib,swap_used_mb,pressure,rss_gib,disk_avail_gib,anon_gib,file_gib,compressor_gib,other_top_rss_gib,other_top_proc" >"$CSV"
crit=0; warn=0; reason=""
while kill -0 "$PID" 2>/dev/null; do
  now=$(( $(date +%s) - START ))
  read -r pf psp ppu pin pw pfb pan pc <<<"$(vm_pages)"
  sw=$(swap_used_mb); lvl=$(pressure_level); disk=$(disk_avail_gib)
  rss_kb=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' '); rss_kb=${rss_kb:-0}
  rss_gib=$(awk -v k="$rss_kb" 'BEGIN {printf "%.1f", k / 1048576}')
  # largest other process by RSS, to attribute interference (anonymous memory spikes) to its owner
  other=$(ps -axo rss=,pid=,comm= | awk -v me="$PID" '$2 != me' | sort -rn | head -1 | awk '{printf "%.1f,%s", $1 / 1048576, $3}')
  echo "${now},$(gib "$pf" "$PS"),$(gib "$pw" "$PS"),${sw},${lvl},${rss_gib},${disk},$(gib "$pan" "$PS"),$(gib "$pfb" "$PS"),$(gib "$pc" "$PS"),${other}" >>"$CSV"
  if [ "$lvl" -ge 4 ]; then crit=$((crit + 1)); else crit=0; fi
  if [ "$lvl" -ge 2 ]; then warn=$((warn + 1)); else warn=0; fi
  # the runtime must honour the budget: benchmarks and the server print
  # "(expert budget X GiB, wired Y GiB)" once the runtime is constructed
  if [ -z "${alloc_checked:-}" ]; then
    alloc=$(grep -m1 -oE "expert budget [0-9.]+ GiB" "$OUT" 2>/dev/null | grep -oE "[0-9.]+")
    if [ -n "$alloc" ]; then
      alloc_checked=1
      echo "guardian: runtime reports expert budget ${alloc} GiB (requested ${BUDGET_GIB} GiB)"
      awk -v a="$alloc" -v b="$BUDGET_GIB" 'BEGIN {exit !(a <= b + 0.5)}' || reason="runtime ignored the budget: expert budget ${alloc} GiB"
    fi
  fi
  if [ -n "$reason" ]; then :
  elif [ "$crit" -ge "$KILL_CRITICAL_SAMPLES" ]; then reason="memory pressure critical for ${crit} s"
  elif [ "$warn" -ge "$KILL_WARN_SAMPLES" ];     then reason="memory pressure warning for ${warn} s"
  elif [ $((sw - SWAP_START)) -ge "$KILL_SWAP_MB" ]; then reason="swap grew ${SWAP_START} -> ${sw} MB"
  elif [ "$disk" -lt "$KILL_DISK_GIB" ];         then reason="boot volume down to ${disk} GiB free"
  elif [ "$now" -ge "$MAX_SECONDS" ];            then reason="timeout after ${now} s"
  fi
  if [ -n "$reason" ]; then
    echo "guardian: KILL pid ${PID}: ${reason} (free $(gib "$pf" "$PS") GiB, wired $(gib "$pw" "$PS") GiB, rss ${rss_gib} GiB)"
    kill -9 "$PID" 2>/dev/null
    pkill -9 -P "$PID" 2>/dev/null
    break
  fi
  sleep 1
done
wait "$PID" 2>/dev/null; rc=$?
trap - INT TERM

# ---- summary ---------------------------------------------------------------
awk -F, 'NR>1 {
  if ($3 > pw) pw = $3; if ($4 > psw) psw = $4; if ($5 > pl) pl = $5
  if (mf == "" || $2 < mf) mf = $2; if ($6 > pr) pr = $6; if ($7 < md || md == "") md = $7
  if ($8 > pa) pa = $8; if ($10 > pc) pc = $10; n = $1 }
  END {printf "footprint: %d s | peak wired %.1f GiB | peak anon %.1f GiB | peak compressor %.1f GiB | min free %.1f GiB | peak swap %d MB | peak pressure %d | peak rss %.1f GiB | min disk %d GiB\n", n, pw, pa, pc, mf, psw, pl, pr, md}' "$CSV"
if [ -n "$reason" ]; then
  echo "result: KILLED (${reason}); last output lines:"; tail -5 "$OUT"; exit 4
fi
echo "result: exit ${rc}"
grep -E "ready in|tok/s|predicted loads|^text:|Error|Traceback" "$OUT" | tail -8
exit "$rc"

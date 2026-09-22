#!/bin/bash
# Sample memory state once a second beside an interactive chat, without killing anything.
#
# Why: on 2026-09-22 a 54 GiB chat decoded at 4.9-6.2 tok/s against 9.4-9.6 at 52 and 50 GiB,
# with only +0.3 points of hit rate, and nothing recorded what the machine was doing. guarded_run.sh
# samples the same counters but only for benchmarks it launches, and it kills on pressure, which a
# chat you are typing into must not. Run this in a SECOND terminal before starting ./chat.sh, stop it
# with Ctrl-C after /exit, and read the summary it prints.
#
#   benchmarks/memwatch.sh 54        # the label goes in the file name
#
# Columns: free_gib is what the OS calls free; avail_gib is free + speculative + purgeable + the
# smaller of file-backed and inactive, the same figure guarded_run.sh's preflight uses. compressor_gib
# growth and any swap growth or pressure above 1 during a slow turn is the memory-cliff signature;
# a flat compressor and pressure 1 with slow decode means the cause is not memory.
set -uo pipefail
cd "$(dirname "$0")/.."
LABEL=${1:-chat}
OUT=benchmarks/results/guarded/memwatch_${LABEL}_$(date +%Y%m%d-%H%M%S).csv
PS=$(vm_stat | awk '/page size of/ {print $8}')
gib() { awk -v p="$1" -v s="$PS" 'BEGIN {printf "%.2f", p * s / 1073741824}'; }
swap_mb() { sysctl -n vm.swapusage | awk '{for (n=1;n<=NF;n++) if ($n=="used") {v=$(n+2); break}
  u=substr(v,length(v)); x=substr(v,1,length(v)-1); if (u=="G") x*=1024; else if (u=="K") x/=1024; printf "%d\n", x}'; }
echo "t_s,free_gib,avail_gib,wired_gib,compressor_gib,anon_gib,file_gib,swap_mb,pressure,pageouts,decompressions,runtime_rss_gib" >"$OUT"
summary() {
  awk -F, 'NR>1 {n++; if (NR==2 || $3<a) a=$3; if ($4>w) w=$4; if ($5>c) c=$5; if ($8>s) s=$8; if ($9>p) p=$9
    if (NR==2) {c0=$5; s0=$8; po0=$10; d0=$11} po=$10; d=$11}
    END {printf "samples %d | min avail %.1f GiB | peak wired %.1f GiB | compressor %.1f -> peak %.1f GiB | swap %d -> peak %d MB | peak pressure %d | pageouts +%d | decompressions +%d\n", n, a, w, c0, c, s0, s, p, po-po0, d-d0}' "$OUT"
  echo "csv: $OUT"
}
trap 'summary; exit 0' INT TERM
START=$(date +%s)
while true; do
  read -r pf psp ppu pin pw pfb pan pc pout dec <<<"$(vm_stat | awk -F'[: .]+' '
    /^Pages free/ {f=$3} /^Pages speculative/ {s=$3} /^Pages purgeable/ {p=$3} /^Pages inactive/ {i=$3}
    /^Pages wired down/ {w=$4} /^File-backed pages/ {fb=$3} /^Anonymous pages/ {an=$3}
    /^Pages occupied by compressor/ {c=$5} /^Pageouts/ {po=$2} /^Decompressions/ {d=$2}
    END {printf "%d %d %d %d %d %d %d %d %d %d\n", f, s, p, i, w, fb, an, c, po, d}')"
  reclaim=$(( pfb < pin ? pfb : pin ))
  rss=$(ps -axo rss=,command= | awk '/deepseek-v41\/bin\/python|cachalot/ && !/awk/ {printf "%.1f", $1/1048576; exit}')
  echo "$(( $(date +%s) - START )),$(gib $pf),$(gib $((pf+psp+ppu+reclaim))),$(gib $pw),$(gib $pc),$(gib $pan),$(gib $pfb),$(swap_mb),$(sysctl -n kern.memorystatus_vm_pressure_level),${pout},${dec},${rss:-0}" >>"$OUT"
  sleep 1
done

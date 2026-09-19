#!/bin/bash
# A/B the non-cumulative lookahead on the FP4 bank.
#
# Arms, both at PREDICT_TOPK=6 so they read the same number of predicted
# experts per layer and differ only in which layer those predictions are for:
#
#   lead1  CACHALOT_PREDICT_LEAD=1  predict L+1 -- the shipped default
#   lead2  CACHALOT_PREDICT_LEAD=2  predict L+2 instead, same bytes
#
# The offline recall table puts lead2 6.5 points of recall below lead1 (65.0 %
# against 71.5 %) and two layers of lead time above it. The FP4 anatomy
# attributes 41.4 ms per token to prefetch timing rather than coverage, so the
# question is whether the lead is worth more than the recall. A recall table is
# a screen; this is the gate. HANDOFF sections 9.12 and 9.13.
#
# lead1 also serves as a regression check on the predicted-load lifetime fix
# (HANDOFF 9.13): at lead 1 every prediction is aimed at the very next layer,
# so the fix should change nothing, and a lead1 median away from the recorded
# 341.5 ms/token would mean it did.
#
# Discipline, from HANDOFF section 5: four runs a side is the minimum for an
# effect smaller than the 7 % run-to-run spread; the arms are interleaved in
# both directions so a drift in machine state cannot be read as an effect; and
# settle.sh runs between every pair because a finished runtime's pages are
# still on the inactive list when the next preflight counts them as free.
#
# Nothing else may be on the GPU while this runs, and only one runtime at a
# time. This script is long-running: start it with nohup and kill it by the PID
# it records, never with a pgrep pattern that would match the shell running it.
#
#   cd /Users/hamedprooshani/Projects/deepseek-v41-mac && \
#     nohup benchmarks/ab_predict_lead.sh > /tmp/ab_predict_lead.log 2>&1 &
#   echo $! > /tmp/ab_predict_lead.pid
set -uo pipefail
cd "$(dirname "$0")/.."

BUDGET_GIB=${BUDGET_GIB:-36}
PAIRS=${PAIRS:-4}
MAX_SECONDS=${MAX_SECONDS:-1800}

MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash
EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts
PYTHON=~/venvs/deepseek-v41/bin/python

echo "budget ${BUDGET_GIB} GiB, ${PAIRS} pairs, interleaved both ways"
echo "started $(date '+%F %T'), pid $$"

run_arm() {
  local lead=$1 pair=$2
  local tag="lead${lead}-p${pair}"
  echo
  echo "=== ${tag} $(date '+%T') ==="
  # The mirror stays on in both arms: it is shipped, and an A/B against a
  # configuration nobody runs answers nothing.
  benchmarks/guarded_run.sh \
    --budget-gib "$BUDGET_GIB" --max-seconds "$MAX_SECONDS" --tag "$tag" -- \
    env CACHALOT_MODEL_PATH="$MODEL_PATH" \
        CACHALOT_EXPERT_BANK="$EXPERT_BANK" \
        CACHALOT_PAGE_CACHE=1 \
        CACHALOT_MIRROR_PATH="$MODEL_PATH" \
        CACHALOT_MIRROR_FRACTION=0.10 \
        CACHALOT_PREDICT_TOPK=6 \
        CACHALOT_PREDICT_AHEAD=1 \
        CACHALOT_PREDICT_LEAD="$lead" \
        PYTHONPATH=src "$PYTHON" \
        benchmarks/decode_anatomy.py --prompt-tokens 512 --decode-tokens 64
  local status=$?
  echo "=== ${tag} exit ${status} ==="
  benchmarks/settle.sh --budget-gib "$BUDGET_GIB"
}

for pair in $(seq 1 "$PAIRS"); do
  # Alternate which arm goes first, so neither one always runs on the colder
  # page cache.
  if [ $((pair % 2)) -eq 1 ]; then
    run_arm 1 "$pair"
    run_arm 2 "$pair"
  else
    run_arm 2 "$pair"
    run_arm 1 "$pair"
  fi
done

echo
echo "finished $(date '+%F %T')"
echo "Every arm must be checked for completion before it is compared: a guarded"
echo "arm killed at its timeout still leaves a result file. Read the exit lines"
echo "above and benchmarks/results/guarded/lead*-p*.out, not only the JSON."

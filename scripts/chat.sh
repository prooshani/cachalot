#!/bin/bash
# Interactive chat in the qualified configuration, so a wrapped copy-paste
# cannot silently drop an environment variable.
#
# A bare VAR=value on its own line sets a *shell* variable, which is never
# exported to a child process. When the documented one-liner wraps in the
# terminal, zsh splits it, and the runtime then falls back to discovering the
# FP4 checkpoint on the USB drive: same flags, a fifth of the speed, and the
# only visible clue is a missing "expert bank:" line. This script cannot be
# split.
#
# Usage:
#   scripts/chat.sh            # machine idle: 44 GiB budget, 72 GiB wired
#   scripts/chat.sh 36 64      # other applications open
#   scripts/chat.sh 28 60      # conservative
#
# This must stay in step with the command in HANDOFF section 4. Until
# 2026-09-19 it did not: it still selected the oQ3e 3-bit bank from the USB
# drive, omitted the frequency penalty without which 62 % of long code replies
# collapse into a loop, and capped the context at 8192.
set -euo pipefail

BUDGET_GIB=${1:-44}
WIRED_GIB=${2:-72}

# The trunk, Engram, head and tokenizer come from the checkpoint on the X10Pro;
# only the routed experts come from the bank. They are independent, which is why
# switching banks is one variable and no code.
MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash

# FP4 is the quality bank and the standing decision (HANDOFF section 2). Set
# CACHALOT_EXPERT_BANK in the environment to override -- the 2-bit bank at
# ~/DeepSeek-V4.1-Flash-q2g128 is the fast alternative.
EXPERT_BANK=${CACHALOT_EXPERT_BANK:-/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts}

# The mirror: the tail fraction of every expert read goes to the X10Pro copy at
# the same time as the head goes to the internal SSD, which cuts the read on the
# critical path. Measured at a 36 GiB budget: cold prefill -6.9 %, decode -5 %.
# 0.08 to 0.12 are indistinguishable; 0.15 is worse than off. The X10Pro is
# opened read-only. HANDOFF section 9.11.
MIRROR_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash
MIRROR_FRACTION=0.10

# Recorded hot experts, read in the background while the runtime finishes
# starting: 5.7 % of a cold prefill and 4.0 points of first-turn hit rate.
# Absent, the runtime behaves exactly as it would without them.
HOTLIST=/Users/hamedprooshani/cachalot-hotlist.json
HOTLIST_GIB=8

PYTHON=~/venvs/deepseek-v41/bin/python

cd "$(dirname "$0")/.."

if pgrep -fl "deepseek-v41/bin/python|cachalot" ; then
  echo "A runtime is already alive. Two at once exhaust memory; exit that one first." >&2
  exit 1
fi

for path in "$MODEL_PATH" "$EXPERT_BANK"; do
  [ -d "$path" ] || { echo "missing: $path" >&2; exit 1; }
done

# The mirror and the hotlist are optimizations, not requirements: drop either
# rather than refusing to start.
MIRROR_ENV=()
if [ -d "$MIRROR_PATH" ]; then
  MIRROR_ENV=(CACHALOT_MIRROR_PATH="$MIRROR_PATH" CACHALOT_MIRROR_FRACTION="$MIRROR_FRACTION")
else
  echo "no mirror at $MIRROR_PATH; running without it (expect ~5 % slower decode)" >&2
fi

HOTLIST_ENV=()
if [ -f "$HOTLIST" ]; then
  HOTLIST_ENV=(CACHALOT_HOTLIST="$HOTLIST" CACHALOT_HOTLIST_GIB="$HOTLIST_GIB")
else
  echo "no hotlist at $HOTLIST; only the first turn is affected" >&2
fi

echo "budget ${BUDGET_GIB} GiB, wired limit ${WIRED_GIB} GiB"
echo "expert bank: $EXPERT_BANK"
echo "expect the next lines to name that bank and the wired limit you asked for."
echo "a bank line naming the USB path means the variable was lost: quit and check."

# set -u makes an empty "${arr[@]}" abort under /bin/bash 3.2, so the arrays are
# expanded with the ${arr[@]+"${arr[@]}"} guard rather than bare.
exec env \
  CACHALOT_MODEL_PATH="$MODEL_PATH" \
  CACHALOT_EXPERT_BANK="$EXPERT_BANK" \
  CACHALOT_PAGE_CACHE=1 \
  CACHALOT_MLX_WIRED_LIMIT_GIB="$WIRED_GIB" \
  ${MIRROR_ENV[@]+"${MIRROR_ENV[@]}"} \
  ${HOTLIST_ENV[@]+"${HOTLIST_ENV[@]}"} \
  PYTHONPATH=src \
  "$PYTHON" -m cachalot.cli chat \
    --expert-budget-gib "$BUDGET_GIB" \
    --max-seq-len 32768 \
    --max-new-tokens 4096 \
    --temperature 0.6 \
    --frequency-penalty 0.2 \
    --penalty-window 128

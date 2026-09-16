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
set -euo pipefail

BUDGET_GIB=${1:-44}
WIRED_GIB=${2:-72}

MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash
EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-oQ3e-mtp
PYTHON=~/venvs/deepseek-v41/bin/python

cd "$(dirname "$0")/.."

if pgrep -fl "deepseek-v41/bin/python|cachalot" ; then
  echo "A runtime is already alive. Two at once exhaust memory; exit that one first." >&2
  exit 1
fi

for path in "$MODEL_PATH" "$EXPERT_BANK"; do
  [ -d "$path" ] || { echo "missing: $path" >&2; exit 1; }
done

echo "budget ${BUDGET_GIB} GiB, wired limit ${WIRED_GIB} GiB"
echo "expect the next lines to name the 3-bit expert bank and the wired limit you asked for."

exec env \
  CACHALOT_MODEL_PATH="$MODEL_PATH" \
  CACHALOT_EXPERT_BANK="$EXPERT_BANK" \
  CACHALOT_PAGE_CACHE=1 \
  CACHALOT_MLX_WIRED_LIMIT_GIB="$WIRED_GIB" \
  PYTHONPATH=src \
  "$PYTHON" -m cachalot.cli chat \
    --expert-budget-gib "$BUDGET_GIB" \
    --max-seq-len 8192 \
    --max-new-tokens 1024 \
    --temperature 0.6

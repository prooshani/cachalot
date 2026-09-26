#!/bin/bash
# MiniMax-M3 in the terminal: the MiniMax counterpart of chat.sh (DeepSeek) and chat-glm.sh (GLM).
#   ./chat-minimax.sh                 interactive
#   ./chat-minimax.sh --thinking      with M3's reasoning shown
#   ./chat-minimax.sh "a question"    one turn
# Only one runtime runs at a time. HANDOFF section 18.
set -euo pipefail
cd "$(dirname "$0")"

if pgrep -fl "deepseek-v41/bin/python|cachalot\.cli" >/dev/null 2>&1; then
    echo "a runtime is already running; not starting a second one:" >&2
    pgrep -fl "deepseek-v41/bin/python|cachalot\.cli" >&2
    exit 1
fi

export CACHALOT_MODEL_PATH=${CACHALOT_MINIMAX_PATH:-/Users/hamedprooshani/MiniMax-M3-MLX-3bit}
export CACHALOT_MODEL_FAMILY=minimax
export CACHALOT_PAGE_CACHE=1
export CACHALOT_MLX_WIRED_LIMIT_GIB=${CACHALOT_MLX_WIRED_LIMIT_GIB:-80}
export PYTHONPATH=src
export MLX_METAL_FAST_SYNCH=${MLX_METAL_FAST_SYNCH:-1}
# The routed experts come from the bias-free bank (HANDOFF 18.4): one contiguous 22.2 MiB record per expert, the
# biases rebuilt exactly from 2-bit codes; the internal checkpoint holds only the non-expert weights since 0.23.0.
# Same outputs, 6 % fewer bytes, ~9 % less read wait per token. The download on the X10Pro is the full original.
export CACHALOT_MINIMAX_BANK=${CACHALOT_MINIMAX_BANK-$HOME/MiniMax-M3-coded-bank}
# Mirror striping (HANDOFF 18.3, 18.4): the tail 10 % of each expert's weight pieces comes from the bank's copy on
# the X10Pro at the same time as the rest from the internal SSD. Off when the X10Pro is not mounted;
# CACHALOT_MINIMAX_MIRROR= (empty) turns it off.
MIRROR=${CACHALOT_MINIMAX_MIRROR-/Volumes/X10Pro/models/MiniMax-M3-coded-bank}
if [ -n "$MIRROR" ] && [ -f "$MIRROR/bank.json" ]; then
    export CACHALOT_MINIMAX_BANK_MIRROR=$MIRROR
    export CACHALOT_MIRROR_FRACTION=${CACHALOT_MIRROR_FRACTION:-0.10}
fi

exec ~/venvs/deepseek-v41/bin/python -m cachalot.cli chat \
    --model "$CACHALOT_MODEL_PATH" \
    --expert-budget-gib 52 \
    --max-seq-len 131072 \
    --max-new-tokens 2000 \
    --temperature 1.0 \
    "$@"

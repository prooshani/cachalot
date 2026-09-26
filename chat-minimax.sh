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
# Mirror striping (HANDOFF 18.3): ~10 % of every expert read (its four smallest pieces) comes from the X10Pro copy
# at the same time as the rest from the internal SSD. Decode -5 %, cold prefill -9 %, same bytes. Off when the
# X10Pro is not mounted; CACHALOT_MINIMAX_MIRROR= (empty) turns it off.
MIRROR=${CACHALOT_MINIMAX_MIRROR-/Volumes/X10Pro/models/MiniMax-M3-MLX-3bit}
if [ -n "$MIRROR" ] && [ -f "$MIRROR/model-00001-of-00036.safetensors" ]; then
    export CACHALOT_MIRROR_PATH=$MIRROR
    export CACHALOT_MIRROR_FRACTION=${CACHALOT_MIRROR_FRACTION:-0.10}
fi

exec ~/venvs/deepseek-v41/bin/python -m cachalot.cli chat \
    --model "$CACHALOT_MODEL_PATH" \
    --expert-budget-gib 52 \
    --max-seq-len 131072 \
    --max-new-tokens 2000 \
    --temperature 1.0 \
    "$@"
